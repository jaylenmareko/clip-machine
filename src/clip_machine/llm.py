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
_TITLE_SYSTEM = """You write titles for short-form debate/podcast clips, in one
specific house style. Match it exactly.

THE FORMULA
  [group descriptor] + [says / admits / claims / LIES] + [the claim]  -> hashtags

RULES
- Attribute the claim to a DESCRIPTOR, never a personal name: "Muslim says",
  "Rabbi says", "Privileged white woman says", "Pakistani man says".
- EXACTLY ONE word in ALL CAPS for emphasis. Pick from: LIES, NOT, EXPOSES,
  FLAMED, BROKE, NEVER, ADMITS, DODGES, TRAITORS, HELL. One. Never two.
- Optional outcome tease ("and gets FLAMED", "gets EXPOSED") or comment bait
  ("Agree?", "Agree or disagree?", "What should we do?") - use sparingly.
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
- 40-95 characters BEFORE the hashtags. Sentence case apart from the caps word.
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


def title_clips(clips: list[dict]) -> dict[int, str] | None:
    """Generate a house-style title per clip. Returns {index: title} or None."""
    if not clips:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    listing = "\n\n".join(
        f"[{n}] ({c['duration']:.0f}s) {c['text'][:900]}"
        for n, c in enumerate(clips)
    )
    prompt = (
        f"Write one title for each of these {len(clips)} clips.\n\n{listing}\n\n"
        'Return ONLY JSON: {"titles": {"0": "...", "1": "...", ...}}'
    )

    try:
        client = anthropic.Anthropic()
        kwargs = {
            "model": TITLE_MODEL,
            "max_tokens": 4096,
            "system": _TITLE_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not TITLE_MODEL.startswith("claude-haiku"):
            kwargs["thinking"] = {"type": "adaptive"}
        try:
            response = client.messages.create(**kwargs)
        except TypeError:
            kwargs.pop("thinking", None)
            response = client.messages.create(**kwargs)
        if getattr(response, "stop_reason", None) == "refusal":
            print("      (Claude declined to write titles)")
            return None

        text = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
        try:
            data = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))

        titles = data.get("titles") if isinstance(data, dict) else None
        if not isinstance(titles, dict):
            return None

        out: dict[int, str] = {}
        for k, v in titles.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(clips) and isinstance(v, str) and v.strip():
                out[idx] = v.strip()

        used = getattr(response, "usage", None)
        if used is not None:
            print(f"      titled by {TITLE_MODEL} "
                  f"({used.input_tokens} in / {used.output_tokens} out)")
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


def rerank(candidates: list[dict], max_clips: int) -> list[dict] | None:
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
        f"{max_clips}, best first.\n\n{listing}\n\n"
        'Return ONLY JSON, no prose: {"picks": [<candidate numbers, best first>]}'
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
