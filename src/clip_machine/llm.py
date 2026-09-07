"""Claude reranks the shortlisted clip candidates.

The heuristics in score.py are good at "is this a usable clip" — clean
start, clean end, sane length, the right vocabulary. They are bad at the
judgment that actually matters: does this stand alone, and does it hook.
That is what this step is for.

Blast radius is deliberately small. Claude only REORDERS and SELECTS from
candidates the heuristics already validated; it never defines cut points.
So a weaker model can rank worse, but it cannot produce a broken clip.

Falls back to the heuristic order on any failure — no key, no network, bad
JSON, rate limit. Selection never depends on the API being up.
"""
import json
import os
import re

# Imported for its side effect: config loads ~/.config/clip-machine/.env into
# the environment. Without it, using this module standalone fails with
# "Could not resolve authentication method" even though the key is on disk.
from . import config  # noqa: F401

# Haiku 4.5, chosen on measurement rather than assumption. A/B'd against
# Sonnet 5 on an identical 21-candidate shortlist:
#
#   Sonnet 5   22.9s  3669 in / 32 out  ~$0.008  8/8 with signals  trap found
#   Haiku 4.5   0.8s  2795 in / 35 out  ~$0.003  7/8 with signals  trap found
#
# 5/8 picks agreed. Sonnet edged signal coverage, but Haiku's top pick was
# the better hook ("have you ever fought in Gaza?" over a mid-argument line),
# and Sonnet broke its own rule once by picking a clip opening "I just want
# to repeat what you just said" - exactly the context-dependent start the
# prompt tells it to reject. No measurable win for 3x the price.
#
# Worth re-checking on a 2-hour source, where the shortlist roughly doubles
# and position bias in long lists is where a smaller model would slip.
# Swap with CLIP_RERANK_MODEL=claude-sonnet-5 (or claude-opus-5).
MODEL = os.environ.get("CLIP_RERANK_MODEL", "claude-haiku-4-5")
EFFORT = os.environ.get("CLIP_RERANK_EFFORT", "medium")

# Titles use a bigger model than ranking, on measurement. Haiku ranks fine -
# that is judgment over short candidates - but it does not hold the title
# format: across two runs it left three of six titles with no ALL-CAPS word,
# used two words in caps on another, and kept writing "Man EXPOSES" instead
# of the host's name despite an explicit rule. Multi-rule formatting is where
# the model difference actually shows. The title call is ~2k tokens once per
# video, so this costs fractions of a cent.
TITLE_MODEL = os.environ.get("CLIP_TITLE_MODEL", "claude-sonnet-5")
MAX_CANDIDATE_CHARS = 700

# What a free-text instruction can actually change. Only two things reach a
# model in this pipeline - which candidates get picked, and how titles are
# written - so those are the only things we accept. Anything mechanical (clip
# length, clip count, layout, captions) is a pipeline parameter no model here
# can touch, and an instruction quietly half-ignored is worse than a UI that
# says up front what it handles.
ACCEPTED_INSTRUCTIONS = [
    'Topics to include — "only the parts about Gaza"',
    'Topics to skip — "skip anything about the Talmud"',
    'Kind of moment — "focus on where he gets challenged", "only funny bits"',
    'Title style — "no hashtags", "titles under 60 characters", "don\'t use my name"',
]
NOT_ACCEPTED = [
    "clip length or how many clips (both scored automatically)",
    "layout (detected from the video)",
    "captions (not generated)",
    "picking a specific person (no speaker labels yet)",
]


def _instruction_block(instructions: str | None, kind: str) -> str:
    """Render the user's instruction into a prompt, or return nothing."""
    text = (instructions or "").strip()
    if not text:
        return ""
    return (
        "\n\nUSER INSTRUCTION — follow it over your own preference where they "
        "conflict, but never at the cost of a clip being unusable:\n"
        f'"{text}"\n'
        f"It may only affect {kind}. If any part of it asks for something you "
        "cannot do from here (clip length, how many clips, layout, captions, or "
        "picking a specific speaker), ignore that part and name it in `ignored`.\n"
    )

_SYSTEM = """You rank candidate clips cut from a longform video transcript.

You are choosing which moments become standalone short-form videos. Judge each
candidate on:

1. HOOK — do the first few seconds make someone stop scrolling? A question, a
   bold claim, an accusation, a number. Not scene-setting or throat-clearing.
2. STANDS ALONE — can a viewer who saw none of the source follow it? Reject
   anything that answers a question the viewer never heard, or refers to "that"
   / "he" / "what you just said" without the referent inside the clip.
3. PAYOFF — does it resolve? A setup with no landing is not a clip.
4. TRAVELS — a hypocrisy trap, an extracted admission, a heated exchange, or a
   genuinely surprising claim will outperform ordinary good talking.

Prefer a tight clip over a long one when both make the same point. Ignore
transcription errors — the transcript is a machine index, not the output, and
these clips carry no captions.

Return ONLY the ranked candidate numbers, best first. Drop candidates that fail
any of 1-3, even if that means returning fewer than asked."""


# Title formula extracted from ~55 real titles across the reference channels
# (@TheSaintandTheSinner, @MARQUETTFORWOMEN, @MarquettOnBusiness).
_TITLE_SYSTEM = """You write titles for short-form podcast clips in one
specific house style.

WHAT A TITLE IS FOR
The title says WHAT THE CLIP TEACHES - the point made, the lesson to take
from it, or the claim being argued. Someone scrolling should be able to tell
what they get by watching. A title that only reports that a topic came up
("Guest discusses friendship") is a failed title. Name the actual takeaway
("Real loyalty takes ten years to prove").

TWO SHAPES - pick whichever fits the clip
  1. THE LESSON.  [the point / rule / hard truth]  -> hashtags
     For teaching, advice, wisdom, an explanation, or a story with a moral.
     This is the DEFAULT - most clips are this.
  2. THE CONTESTED CLAIM.  [group descriptor] + [says / admits / claims]
     + [the claim]  -> hashtags
     ONLY when someone is making an argument another person would dispute.
     Attribute to a DESCRIPTOR, never a personal name: "Muslim says",
     "Privileged white woman says". Never force this shape onto a clip that
     is simply someone explaining something.

CAPS
- ALL CAPS is for emphasis that EARNS it - a reversal, a denial, a hard
  number, the one word the whole clip turns on. Most titles need none.
- Never more than one capitalised word, and never capitalise a word just to
  satisfy a format. A title with no caps at all is completely fine and is
  usually the better choice.
- When it is earned, useful words: NOT, NEVER, LIES, EXPOSES, ADMITS, BROKE,
  DODGES.
- Optional outcome tease ("and gets FLAMED") or comment bait ("Agree?") -
  use sparingly, and only on a contested-claim title.
- 3-6 hashtags, and ONLY from the vocabulary below. These are the tags the
  channels actually use, counted across 120 of their real titles. Do not
  invent tags, and do not borrow from a topic the clip is not about -
  #redpill is real but appears only on dating/gender clips, never on a
  religious or geopolitical one.

  Religion:      #shia #deen #islam #sunni #muslim #nikah #jesus #judaism
                 #talmud #israelite #holyland #holywar #atheist #kufar
  Geopolitics:   #gaza #palestine #iran #geopolitics #saudiarabia #uae #uk
                 #iraq #history #colonialism #activism #maga #news
  Black America: #foundationalblackamericans #blackgirlmagic #africandiaspora
                 #blackhistory #blackwomen #blacklove #blackfamily #race
                 #blackexcellence #blackpower #zulu #capetown #durban
  Gender/dating: #masculinity #manhood #tradwife #relationships #dating
                 #datingadvice #divorce #alimony #childsupport #redpill
                 #hypergamy #femalenature #femaleprivilege #whitewomen
                 #feminine #femininity #poly #polygamy #patriarchy
                 #mansrights #pua #player
  Family:        #family #familyvalues #nuclearfamily #values #parenting
                 #mothers
  Reach:         #debate #comedy #logic #philosophy #advice #wordsofwisdom
                 #crashout #lol #ism #miami #freshandfit
- 40-95 characters BEFORE the hashtags. Sentence case apart from any caps word.
- Name the host ("Marquett") whenever HE is the one doing the exposing,
  asking, or calling out: "Marquett EXPOSES...", "Marquett asks...". Do not
  write "Man EXPOSES" or "Debater EXPOSES" for the host - that is his own
  channel, use his name.
- Vary the tags across a batch and lean on the ones they use most. Frequency
  across their 120 titles: #shia 17, #foundationalblackamericans 11, #iran
  10, #deen 10, #palestine 9, #gaza 8, #geopolitics 7, #islam 6, #family 6.
  #shia is their single most-used tag and belongs on most religious clips.
  Do not put the same three tags on every clip in a batch.
- State what actually happens in the clip. Do not invent claims that are not
  in the transcript, and do not exaggerate beyond it.

Ignore transcription errors - the transcript is a machine index. If a proper
noun is clearly garbled ("designists" for Zionists, "jibriya" for Jibreel),
use the correct spelling.

Return ONLY JSON: {"titles": {"<clip number>": "<title with hashtags>", ...}}"""


# One request covering 22 clips came back at 3830 output tokens against a
# 4096 cap - 266 tokens short of truncation - and on the real run it went
# over. The truncated JSON failed to parse, title_clips returned None, and
# ALL 22 clips silently fell back to a raw transcript line as their "title".
# Nothing raised; the run looked clean. Four defences, because any one of
# them alone still loses a whole batch:
#   small batches, a real ceiling, streaming, and a salvage path.
TITLES_PER_CALL = 8
_TITLE_MAX_TOKENS = 16000


def _salvage_titles(text: str) -> dict:
    """Pull every "<n>": "<title>" pair present, even mid-truncation.

    A cut-off response still contains most of its titles. Discarding all of
    them because the closing brace never arrived is the expensive mistake.
    """
    pairs = re.findall(r'"(\d+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    out = {}
    for k, v in pairs:
        try:
            out[k] = json.loads(f'"{v}"')
        except Exception:
            out[k] = v
    return out


def _title_batch(client, clips: list[dict], offset: int,
                 instructions: str | None) -> tuple[dict[int, str], int, int]:
    """Title one batch. Returns (titles by GLOBAL index, in_tokens, out)."""
    listing = "\n\n".join(
        f"[{offset + n}] ({c['duration']:.0f}s) {c['text'][:900]}"
        for n, c in enumerate(clips)
    )
    numbers = ", ".join(str(offset + n) for n in range(len(clips)))
    prompt = (
        f"Write one title for each of these {len(clips)} clips."
        + _instruction_block(instructions, "how the titles are written")
        + f"\n\n{listing}\n\n"
        + f"Return ONLY JSON, with a title for every one of these clip "
          f"numbers ({numbers}):\n"
        + '{"titles": {"%d": "...", ...}}' % offset
    )

    kwargs = {
        "model": TITLE_MODEL,
        "max_tokens": _TITLE_MAX_TOKENS,
        "system": _TITLE_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    if not TITLE_MODEL.startswith("claude-haiku"):
        kwargs["thinking"] = {"type": "adaptive"}

    # Streamed: with max_tokens this high a blocking request can hit the
    # request timeout before it finishes generating.
    try:
        with client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()
    except TypeError:
        kwargs.pop("thinking", None)
        with client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

    if getattr(response, "stop_reason", None) == "refusal":
        print("      (Claude declined to title one batch)")
        return {}, 0, 0

    text = "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    )

    titles = None
    try:
        titles = json.loads(text).get("titles")
    except Exception:
        if m := re.search(r"\{.*\}", text, re.DOTALL):
            try:
                titles = json.loads(m.group(0)).get("titles")
            except Exception:
                titles = None
    if not isinstance(titles, dict):
        titles = _salvage_titles(text)
        if titles:
            print(f"      (recovered {len(titles)} titles from a truncated reply)")

    lo, hi = offset, offset + len(clips)
    out: dict[int, str] = {}
    for k, v in (titles or {}).items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if lo <= idx < hi and isinstance(v, str) and v.strip():
            out[idx] = v.strip()

    used = getattr(response, "usage", None)
    return out, (used.input_tokens if used else 0), (used.output_tokens if used else 0)


def title_clips(clips: list[dict],
                instructions: str | None = None) -> dict[int, str] | None:
    """Generate a house-style title per clip. Returns {index: title} or None."""
    if not clips:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic()
        out: dict[int, str] = {}
        tin = tout = 0
        for offset in range(0, len(clips), TITLES_PER_CALL):
            batch = clips[offset:offset + TITLES_PER_CALL]
            got, i_tok, o_tok = _title_batch(client, batch, offset, instructions)
            out.update(got)
            tin += i_tok
            tout += o_tok

        calls = (len(clips) + TITLES_PER_CALL - 1) // TITLES_PER_CALL
        print(f"      titled by {TITLE_MODEL} "
              f"({len(out)}/{len(clips)} clips, {calls} call"
              f"{'' if calls == 1 else 's'}, {tin} in / {tout} out)")
        # Say which clips fell back. A transcript line masquerading as a
        # title is the exact failure that shipped 22 bad titles unnoticed.
        missing = [n for n in range(len(clips)) if n not in out]
        if missing:
            print(f"      ! no title for clip"
                  f"{'' if len(missing) == 1 else 's'} "
                  f"{', '.join(str(m + 1) for m in missing)}; "
                  f"falling back to a transcript line")
        return out or None

    except Exception as e:
        detail = " ".join(str(e).split())[:200]
        print(f"      (titles unavailable: {type(e).__name__}: {detail})")
        return None


def _extract_picks(text: str, n: int) -> list[int]:
    """Pull the picks array out of the response, tolerantly."""
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    picks = data.get("picks") if isinstance(data, dict) else data
    if not isinstance(picks, list):
        return []
    return [p for p in picks if isinstance(p, int) and 0 <= p < n]


def rerank(candidates: list[dict], max_clips: int,
           instructions: str | None = None) -> list[dict] | None:
    """Return candidates reordered by Claude, or None to keep heuristic order."""
    if not candidates:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    listing = "\n\n".join(
        f"[{n}] ({c['duration']:.0f}s) {c['text'][:MAX_CANDIDATE_CHARS]}"
        for n, c in enumerate(candidates)
    )
    prompt = (
        f"Rank these {len(candidates)} candidates and return the best "
        f"{max_clips}, best first."
        + _instruction_block(instructions, "which clips you choose")
        + f"\n\n{listing}\n\n"
        + 'Return ONLY JSON, no prose: {"picks": [<candidate numbers, best '
          'first>], "applied": "<what you did with the instruction, or empty>", '
          '"ignored": "<any part you could not act on, or empty>"}'
    )

    try:
        client = anthropic.Anthropic()
        kwargs = {
            "model": MODEL,
            "max_tokens": 4096,
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Haiku 4.5 rejects `effort` and has no adaptive thinking; the
        # Sonnet/Opus families take both. Send them only where supported so
        # the model stays swappable by env var.
        if not MODEL.startswith("claude-haiku"):
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": EFFORT}

        try:
            response = client.messages.create(**kwargs)
        except TypeError:
            # An older SDK rejects `thinking` / `output_config` as unexpected
            # kwargs. The ranking works without them; don't fail over it.
            for k in ("thinking", "output_config"):
                kwargs.pop(k, None)
            response = client.messages.create(**kwargs)

        if getattr(response, "stop_reason", None) == "refusal":
            print("      (Claude declined to rank — keeping heuristic order)")
            return None

        text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
        # Report what the instruction actually did. A silently half-ignored
        # instruction is the main way this feature could mislead someone.
        if instructions:
            try:
                note = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
                if note.get("applied"):
                    print(f"      instruction applied: {note['applied']}")
                if note.get("ignored"):
                    print(f"      instruction IGNORED: {note['ignored']}")
            except Exception:
                pass

        picks = _extract_picks(text, len(candidates))
        if not picks:
            print("      (no usable ranking returned — keeping heuristic order)")
            return None

        used = getattr(response, "usage", None)
        if used is not None:
            print(f"      ranked by {MODEL} "
                  f"({used.input_tokens} in / {used.output_tokens} out)")
        return [candidates[i] for i in picks]

    except Exception as e:
        # Include the message - a bare exception name is useless for
        # diagnosing a 400, and this path is the one that silently degrades.
        detail = " ".join(str(e).split())[:200]
        print(f"      (Claude rerank unavailable: {type(e).__name__}: {detail} "
              f"— keeping heuristic order)")
        return None
