"""Pick clip-worthy moments — fully local.

Scoring is two parts:

  base_quality  — is this a usable clip at all (self-contained, clean
                  ending, sane length). Always applies.
  mode_bonus    — is this the KIND of moment we're hunting for (exciting,
                  controversial, funny, story, educational). See modes.py.

Selection runs in two stages, because feeding a whole transcript to a
CPU-bound LLM is not viable on a GPU-less machine:

  1. Heuristic prefilter over every candidate window (cheap, instant).
     Decides whether a clip is USABLE - clean start, clean end, sane length.
  2. Claude reranks the survivors (see llm.py). Decides whether a clip is
     GOOD - does it hook, does it stand alone, does it pay off. It only ever
     sees the shortlist, never the full transcript, and it reorders rather
     than re-cuts, so it cannot produce a broken clip.
     Falls back to a local Ollama model, then to the heuristic order, so
     selection never depends on the API being reachable.
"""
import json
import re

from . import config, llm, modes

# Openers that prove the clip depends on earlier context.
# The bare replies ("no", "yes", "okay"...) were added after a real debate
# run produced clips opening on "No Was that translation..." and "Yes or
# no?" — an answer to an unheard question is the worst possible hook.
_CONTEXT_DEPENDENT = re.compile(
    r"^(and|so|but|because|then|anyway|also|plus|which|that's why|that is why"
    r"|he|she|they|it|this|that|those|these|him|her|them"
    r"|no|yes|yeah|yep|nope|okay|ok|right|correct|exactly|sure|true|false"
    r"|of course|absolutely|agreed|hold on|wait|well)\b",
    re.IGNORECASE,
)

# Openers that tend to hook.
_HOOK_PATTERNS = [
    r"^(the (truth|reason|problem|biggest|worst|best|number one))",
    r"^(most people|nobody|no one|everyone|everybody)",
    r"^(here'?s (why|how|what|the))",
    r"^(if you|when you|you (should|need|have to|can't|shouldn't|never))",
    r"^(i (never|always|used to|stopped|quit|learned|realized|made|lost|spent))",
    r"^(never|always|stop|don'?t|forget)\b",
    r"^(what (nobody|no one|people|most))",
    r"^\d",
    r"\?$",
]
_HOOK_RE = [re.compile(p, re.IGNORECASE) for p in _HOOK_PATTERNS]

# Concrete-detail markers — specifics clip better than abstractions.
_CONCRETE = re.compile(r"(\$|\d+%|\d{2,}|one time|the first time|last (year|month|week))", re.I)

_TERMINAL = re.compile(r"[.!?]$")

# --------------------------------------------------------------------------
# Does the clip actually FINISH its thought?
#
# Terminal punctuation cannot answer this. faster-whisper closes EVERY
# segment with a full stop whether the speaker finished or not: measured on
# a 67-minute episode, 22 of 22 selected clips "ended" in punctuation, so
# the old +1.5 _TERMINAL reward was a constant applied to everything and
# discriminated nothing.
#
# What does survive sloppy ASR is grammar at the tail:
#   - the last sentence is only a SUBORDINATE clause, so its main clause
#     never arrived  ("Because when you have wealth success status.")
#   - the text trails off on a connective  ("...and", "...that", "...to")
#   - a list was announced and cut off after item one
#     ("The reason I lived there is one, I didn't have money.")
_SUBORDINATE_ONLY = re.compile(
    r"^\s*(because|when|if|since|while|although|though|unless|whereas"
    r"|so that|even though|as if|which|where|after|before|until)\b", re.I)
_TRAILS_OFF = re.compile(
    r"\b(and|but|so|or|because|that|which|to|of|for|with|like|about|into"
    r"|than|then|as|at|in|on|is|was|were|are|would|could|should|gonna"
    r"|gotta)\s*[.!?]?$", re.I)
_LIST_CUT = re.compile(r"\b(is|are|was|were)\s+(one|number one|first)\b\s*[,.]", re.I)


def ends_incomplete(text: str) -> list[str]:
    """Names of the ways this clip's ending is unfinished. Empty is good."""
    t = text.strip()
    if not t:
        return ["empty"]
    parts = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    last = parts[-1] if parts else t
    why = []
    if _SUBORDINATE_ONLY.match(last):
        why.append("subordinate-clause-only")
    if _TRAILS_OFF.search(t):
        why.append("trails-off")
    if _LIST_CUT.search(last):
        why.append("list-cut-short")
    return why


# --------------------------------------------------------------------------
# Virality: exchange STRUCTURE, not topic.
#
# Derived from ~55 real titles across the reference channels. What travels
# in this format is almost never the subject matter — it's the shape of the
# exchange. Their single most-used frame is the hypocrisy trap ("X says
# Saudis should fight for Gaza but he's done nothing"), which needs BOTH a
# demand made of others AND an excuse made for oneself. That pairing is the
# highest-weighted signal here, and it can only be scored on a window that
# contains both halves — which is also why the length cap had to rise.
# --------------------------------------------------------------------------
_VIRAL_PATTERNS = [
    # An opponent conceding something damaging.
    ("concession", r"\b(i should have|i didn'?t|you'?re right|i'?ll agree with that"
                   r"|fair enough|that'?s true|i was wrong|my error|unfortunately my)\b", 3.0),
    # A yes/no demand — creates tension and invites the audience to judge.
    ("pinned-down", r"\b(yes or no|answer the question|i'?m asking you"
                    r"|simple question|don'?t dodge|hold it right there"
                    r"|is that what you just said)\b", 2.5),
    # Direct accusation, the money line in a debate clip.
    ("accusation", r"\b(you'?re a (goddamn |damn )?liar|you damn liar|you lying"
                   r"|you'?re lying|hypocrite|you don'?t even believe)\b", 2.5),
    # Excuses / evasion — half of the hypocrisy pair.
    ("evasion", r"\b(no opportunity|stay within the law|nothing i will do is illegal"
                r"|it'?s very difficult|i have to|i can'?t because)\b", 1.5),
    # Demands placed on others — the other half.
    ("demands-others", r"\b(they should|they don'?t defend|they oppress|why don'?t they"
                       r"|should fight|must protect|they'?re not (real|even))\b", 1.5),
    # Comment bait — an explicit question to the audience.
    ("audience-question", r"(agree\?|agree or disagree|what should we do|who'?s right)", 2.0),
]
_VIRAL_RE = [(n, re.compile(p, re.IGNORECASE), w) for n, p, w in _VIRAL_PATTERNS]

_VIRAL_CAP = 6.0
_HYPOCRISY_BONUS = 2.5   # both halves of the trap present in one clip


def viral_signals(c: dict) -> tuple[float, list[str]]:
    """Score how likely this exchange is to travel. Mode-independent."""
    text = c["text"]
    total, fired = 0.0, []
    for name, rx, w in _VIRAL_RE:
        if rx.search(text):
            total += w
            fired.append(name)

    # The hypocrisy trap: demanding of others while excusing yourself.
    if "evasion" in fired and "demands-others" in fired:
        total += _HYPOCRISY_BONUS
        fired.append("HYPOCRISY-TRAP")

    return min(total, _VIRAL_CAP + _HYPOCRISY_BONUS), fired


# No target duration — a clip runs as long as the point takes. But removing
# the length bonus outright is NOT length-neutral: a longer window contains
# more text, hits more patterns, and wins by accumulation. Measured, that
# pushed every pick to 112-136s (stdev 9) against the channel's real 20-135s
# (stdev 29).
#
# So each second costs a flat amount instead, and extra length has to pay
# for itself in content. Calibrated against their distribution: 0.0 gave
# stdev 9, 0.02 gave 14, 0.03 gave stdev 33 over a 25-113s range — the
# closest fit to their stdev 29 — and 0.08 started compressing everything
# short. This is a neutrality correction, not a preferred length.
#
# 0.03 WAS TOO STRONG AND CUT CLIPS OFF EARLY. Measured on the 67-minute
# episode: across every candidate window, the LONGEST window from a given
# start paid a median 3.63-point penalty against the shortest, while a whole
# fired lexical signal is worth under 1.0 after scaling. So this term was
# 4-5x stronger than any content signal - not a tie-break, a veto. Every
# clip got dragged to the short end of its moment.
#
# Re-swept on that episode, 22 clips, heuristics only:
#
#     k      min  med  mean  max  stdev  ends-cut
#   0.005     25  101   89   116   27.9     0
#   0.008     21   96   82   116   31.1     0
#   0.012     21   76   70   116   35.9     0
#   0.016     21   51   61   110   31.6     0
#   0.020     20   51   60   110   32.1     0     <- chosen
#   0.025     20   40   49   110   29.2     0
#   0.030     20   37   45   110   26.8     0     <- old, the one that felt short
#
# Two things that sweep settles.
#
# First, ends-cut is 0 at EVERY value. Whether a clip finishes its thought is
# now handled by ends_incomplete() and extend_to_finish(), independently of
# length. That decoupling is the real fix; this constant only shapes the
# duration distribution now, and can be tuned without reintroducing
# mid-sentence endings.
#
# Second, MATCHING THE REFERENCE STDEV IS THE WRONG TARGET. Best stdev fit
# against the channel's 29 published shorts (stdev 29) is k=0.025, which
# lands median 40s - i.e. it reproduces the length that read as cut off
# short. A distribution can have the right spread and the wrong centre. So
# this is calibrated to the actual complaint - 40s clips should run "a bit
# longer" - which puts the median at 51s, with the full range still inside
# the reference 20-135s.
#
# 0.016 and 0.020 both give median 51s, so the choice sits on a plateau
# rather than a knife edge; 0.020 is the conservative end of it.
LENGTH_COST_PER_SEC = 0.020

# Cost of ending mid-thought. Deliberately larger than the entire length
# cost swing, so running longer to finish a sentence always wins.
INCOMPLETE_END_PENALTY = 2.5

# Max share of the shorter clip two SHIPPED clips may share. dedupe uses a
# looser 0.4 for the shortlist on purpose; this governs the output.
OUTPUT_OVERLAP = 0.05

TURN_TOLERANCE = 0.6   # seconds; how close a clip start must be to a turn
TURN_BONUS = 1.0       # reward for starting where someone starts talking.
                       # Measured: 3.0 outranked actual content (4/5 picks had
                       # debate signals); 1.0 gives 5/5 with signals AND 5/5
                       # clean starts, still beating no-bonus on turn alignment.
TURN_PENALTY = 1.5     # cost of starting mid-turn
END_TURN_BONUS = 1.5   # reward for ending as the other person takes over


def build_sentences(words: list[dict], turn_starts: list[float] | None = None) -> list[dict]:
    """Group the flat word list into sentences on terminal punctuation.

    Splitting on detected speaker turns was tried and measurably made
    things WORSE (3/5 clean clip starts vs 5/5): the turn detector carries
    ~0.5s of smoothing lag, so its boundaries land mid-sentence and the
    split manufactures fragments like "a simple yes or no?" and "case you
    didn't understand". Splitting on long pauses hurt for the same reason.

    Turns are still valuable — just as a SCORING signal in base_quality,
    where "this clip happens to begin where someone starts talking" is a
    bonus rather than a licence to cut the transcript there. `turn_starts`
    is accepted here only to keep the call sites uniform.
    """
    sentences, current = [], []
    for w in words:
        current.append(w)
        if _TERMINAL.search(w["word"]):
            sentences.append(_pack(current))
            current = []
    if current:
        sentences.append(_pack(current))
    return [s for s in sentences if s["text"].strip()]


def _pack(ws: list[dict]) -> dict:
    return {
        "start": ws[0]["start"],
        "end": ws[-1]["end"],
        "text": " ".join(w["word"] for w in ws).strip(),
        "words": ws,
    }


def build_candidates(sentences: list[dict]) -> list[dict]:
    """Every run of consecutive sentences whose duration is in range."""
    candidates = []
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            start = sentences[i]["start"]
            end = sentences[j]["end"]
            dur = end - start
            if dur < config.CLIP_MIN_SECONDS:
                continue
            if dur > config.CLIP_MAX_SECONDS:
                break
            text = " ".join(s["text"] for s in sentences[i:j + 1])
            candidates.append({
                "start": start, "end": end, "duration": dur,
                "text": text, "i": i, "j": j,
            })
    return candidates


def base_quality(c: dict, turn_starts: list[float] | None = None) -> float:
    """Is this a usable clip at all — regardless of what it's about."""
    text = c["text"].strip()
    first_sentence = re.split(r"(?<=[.!?])\s", text)[0]
    score = 0.0

    # Starting exactly where someone begins talking is the single best
    # predictor of a clip that stands alone. Starting mid-turn is the
    # worst, and punctuation alone cannot tell the difference.
    if turn_starts:
        starts_on_turn = any(abs(c["start"] - t) <= TURN_TOLERANCE for t in turn_starts)
        if starts_on_turn:
            score += TURN_BONUS
            c["starts_on_turn"] = True
        else:
            score -= TURN_PENALTY
            c["starts_on_turn"] = False

        # With no length target, "where does this clip STOP" carries the
        # weight instead. A clip that ends just as the other person starts
        # talking is one where the point landed and the exchange moved on -
        # that is what "as long as it needs to be" actually looks like.
        ends_on_turn = any(abs(c["end"] - t) <= TURN_TOLERANCE for t in turn_starts)
        if ends_on_turn:
            score += END_TURN_BONUS
            c["ends_on_turn"] = True
        else:
            c["ends_on_turn"] = False

    # Self-containedness — the single biggest quality lever.
    if _CONTEXT_DEPENDENT.match(text):
        score -= 4.0

    # Was: +1.5 for terminal punctuation, -2.0 without. Whisper punctuates
    # every segment end, so 22 of 22 shipped clips scored the +1.5 and the
    # branch never once took its else. A constant is not a signal. Grammar
    # at the tail is, so penalise the specific ways a thought is left open.
    unfinished = ends_incomplete(text)
    if unfinished:
        score -= INCOMPLETE_END_PENALTY
        c["ends_incomplete"] = unfinished

    # Hook strength in the opening line.
    for rx in _HOOK_RE:
        if rx.search(first_sentence):
            score += 2.0
            break

    # Concrete detail anywhere in the clip.
    if _CONCRETE.search(text):
        score += 1.0

    # NO length target. Measured 29 of the reference channel's published
    # shorts: 20s to 135s, a 6.8x spread, stdev 29s, with only 31% inside
    # the 59-76s band an earlier 3-clip sample had suggested. They are not
    # aiming at a duration - they run a clip for as long as the point takes
    # and stop when it lands. So duration earns nothing and costs nothing
    # here; what matters is whether the clip is a COMPLETE unit, which the
    # start/end/turn checks above and below score directly.
    #
    # Their shortest published short is 20s, which is also CLIP_MIN_SECONDS,
    # so the floor is validated rather than guessed.
    dur = c["duration"]
    if dur > 140:
        score -= 1.0   # beyond anything they have ever published

    # Speech density — catch dead air or garbled runs.
    wps = len(text.split()) / max(dur, 1)
    if 2.0 <= wps <= 4.0:
        score += 1.0
    elif wps < 1.2:
        score -= 2.0

    return score


def mode_bonus(c: dict, mode, energy=None, median_wps: float = 0.0,
               lexical=None) -> tuple[float, list[str]]:
    """What this mode is hunting for, on top of base quality.

    Returns (bonus, fired signal names) so every pick stays explainable.

    `lexical` accepts an already-computed (score, fired) pair. best_mode
    needs the lexical result for labelling anyway, and re-deriving it here
    ran every mode's regex set twice per candidate - 17,615 candidates x 7
    modes, so it mattered.
    """
    if lexical is None:
        bonus, fired = mode.lexical(c["text"])
    else:
        bonus, fired = lexical[0], list(lexical[1])

    if energy is not None and energy.ok:
        # Loudness is a property of the WINDOW, not of the mode, so it is
        # computed once per candidate and reused. best_mode now evaluates
        # every mode, which was recomputing these seven times each.
        if "_exc" not in c:
            c["_exc"] = energy.excitement(c["start"], c["end"])
            c["_dyn"] = energy.dynamics(c["start"], c["end"])
        if mode.w_excite:
            e = c["_exc"]
            bonus += mode.w_excite * e
            if e > 0.7:
                fired.append("loud")
        if mode.w_dynamics:
            d = c["_dyn"]
            bonus += mode.w_dynamics * d
            if d > 0.7:
                fired.append("animated")

    # Talking faster than this speaker's own norm reads as urgency.
    if mode.w_rate and median_wps > 0:
        wps = len(c["text"].split()) / max(c["duration"], 1)
        rel = (wps - median_wps) / median_wps
        if rel > 0:
            bonus += mode.w_rate * min(rel / 0.25, 1.0)
            if rel > 0.15:
                fired.append("fast")

    return bonus, fired


def _overlaps(a: dict, b: dict, threshold: float = 0.4) -> bool:
    overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if overlap <= 0:
        return False
    return overlap / min(a["duration"], b["duration"]) > threshold


def extend_to_finish(clips: list[dict], sentences: list[dict],
                     max_extra: float = 12.0) -> None:
    """Grow any clip that still ends mid-thought, in place.

    Scoring pressure alone cannot fix every case: sometimes the window that
    finishes the sentence genuinely scores lower, and sometimes the only
    finishing window would exceed CLIP_MAX_SECONDS. So this is a final
    repair pass on the picks themselves - add whole sentences, one at a
    time, until the thought closes.

    Bounded by `max_extra`, by CLIP_MAX_SECONDS, and by the start of the
    next selected clip - growing into a neighbour would ship the same
    seconds of source twice. A clip that cannot be closed inside those
    limits is left alone rather than padded: a longer clip that STILL
    trails off is worse than a short one that does.
    """
    # Nearest selected start after each clip, so extension cannot overlap a
    # sibling. Computed up front because the loop mutates ends as it goes.
    starts = sorted(c["start"] for c in clips)
    for c in clips:
        if not ends_incomplete(c["text"]):
            continue
        j = c.get("j")
        if j is None:
            continue
        ceiling = next((s for s in starts if s > c["start"]), float("inf"))
        text, end, jj = c["text"], c["end"], j
        while jj + 1 < len(sentences):
            nxt = sentences[jj + 1]
            if nxt["end"] - c["start"] > config.CLIP_MAX_SECONDS:
                break
            if nxt["end"] - c["end"] > max_extra:
                break
            if nxt["end"] > ceiling:
                break
            jj += 1
            text = f"{text} {nxt['text']}"
            end = nxt["end"]
            if not ends_incomplete(text):
                break
        if not ends_incomplete(text):
            # Keep score consistent with the clip it now is, or the manifest
            # reports a number that no longer describes it: the ending is
            # complete now, so the penalty comes back off, and the added
            # seconds cost what any other seconds cost.
            c["score"] = (c["score"] + INCOMPLETE_END_PENALTY
                          - LENGTH_COST_PER_SEC * (end - c["end"]))
            c["extended_by"] = round(end - c["end"], 2)
            c["text"], c["end"], c["j"] = text, end, jj
            c["duration"] = end - c["start"]
            c.pop("ends_incomplete", None)


def dedupe(candidates: list[dict], limit: int) -> list[dict]:
    """Greedy non-maximum suppression by score."""
    kept: list[dict] = []
    # Ties break toward the clip that FINISHES ITS THOUGHT, then toward the
    # shorter one. Breaking on duration alone compounded the length-cost
    # bias: the shortest window won every near-tie even when it stopped
    # mid-sentence and a two-second-longer one landed the point.
    for c in sorted(candidates,
                    key=lambda x: (-x["score"],
                                   bool(x.get("ends_incomplete")),
                                   x["duration"])):
        if any(_overlaps(c, k) for k in kept):
            continue
        kept.append(c)
        if len(kept) >= limit:
            break
    return kept


def _rerank_with_ollama(candidates: list[dict], max_clips: int, mode=None) -> list[dict] | None:
    """Return reordered candidates, or None if Ollama isn't available.

    The mode is described to the model too — this is where genuine semantic
    judgment of "controversial" or "exciting" happens, versus the regex
    lexicon which only catches surface phrasing.
    """
    import requests  # local HTTP only

    listing = "\n\n".join(
        f"[{n}] ({c['duration']:.0f}s) {c['text'][:600]}"
        for n, c in enumerate(candidates)
    )
    # mode is None in auto mode, where we hunt every kind at once.
    if mode is not None:
        hunting = (
            f"You are specifically hunting for: {mode.label} — {mode.blurb}.\n\n"
            "Every pick must also hook in its first 3 seconds, stand alone with "
            "zero outside context, and have a clear payoff. Prefer a clip that "
            f"genuinely matches '{mode.label}' over one that merely uses "
            "dramatic words.\n\n"
        )
    else:
        hunting = (
            "Pick the strongest moments of any kind — a hot take, a heated "
            "exchange, a story, a funny beat, a clear explanation.\n\n"
            "Every pick must hook in its first 3 seconds, stand alone with zero "
            "outside context, and have a clear payoff.\n\n"
        )

    prompt = (
        "You are picking short-form video clips from a longform transcript.\n"
        + hunting
        + f"Candidates:\n\n{listing}\n\n"
        + f'Return ONLY JSON: {{"picks": [<up to {max_clips} candidate numbers, best first>]}}'
    )

    try:
        resp = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=600,
        )
        resp.raise_for_status()
        picks = json.loads(resp.json()["response"]).get("picks", [])
    except Exception as e:
        print(f"  (Ollama unavailable - using heuristic ranking. {type(e).__name__})")
        return None

    ordered = [candidates[i] for i in picks if isinstance(i, int) and 0 <= i < len(candidates)]
    return ordered or None


# Scale for the normalised mode match, so it sits in the same range the old
# per-mode raw bonus did and the other weights stay meaningful.
_MODE_SCALE = 6.0
# Minimum normalised lexical match before a mode may put its name on a clip.
# Below this the "type" is noise, and "balanced" is the honest answer.
_LABEL_FLOOR = 0.25

# "balanced" is the generic catch-all; it would win the auto comparison on
# ordinary clips simply by not being specific about anything.
_AUTO_MODES = [k for k in modes.MODES if k != "balanced"]


def best_mode(c: dict, energy=None, median_wps: float = 0.0) -> tuple[float, list[str], str]:
    """Score a candidate against EVERY mode in one pass and label its type.

    The TYPE is decided on lexical match only, normalised against each mode's
    own ceiling. Audio is deliberately excluded from that choice: `exciting`
    carries 4.0 of audio weight, so on loud material (a heated debate is loud
    throughout) it would win the label on volume alone — measured, it took
    5 of 6 clips including an obvious hypocrisy-trap moment. Loudness makes
    any clip more clippable; it does not make it a different KIND of moment.
    Language does.

    The returned score still includes that mode's audio contribution, so a
    loud debate clip still outranks a quiet one.

    SCORE and LABEL are computed separately, and conflating them was a bug.
    A clip full of money talk is genuinely more clippable, so it should keep
    that score - but it is not a "debate" unless somebody actually pushes
    back. So the score takes the best-scoring mode over all of them, while
    the label only considers modes whose IDENTIFYING signals fired. Tying
    the two together meant either mislabelling clips or quietly restoring
    them, and the second is worse because it changes which clips ship.
    """
    best_scaled, best_fired = 0.0, []
    label_norm, label_key = 0.0, "balanced"

    for key in _AUTO_MODES:
        mode = modes.get(key)
        lex, lex_fired = mode.lexical(c["text"])

        # --- scoring: every mode competes, topic vocabulary included ---
        raw, fired = mode_bonus(c, mode, energy=energy, median_wps=median_wps,
                                lexical=(lex, lex_fired))
        ceiling = mode.lexicon_cap + mode.w_excite + mode.w_dynamics + mode.w_rate
        scaled = (raw / ceiling if ceiling > 0 else 0.0) * _MODE_SCALE
        if scaled > best_scaled:
            best_scaled, best_fired = scaled, fired

        # --- labelling: only if something identifying fired ---
        if not mode.identifies(lex_fired):
            continue
        norm = lex / mode.lexicon_cap if mode.lexicon_cap > 0 else 0.0
        # A trace match should not win a label by default just because every
        # other mode scored zero.
        if norm >= _LABEL_FLOOR and norm > label_norm:
            label_norm, label_key = norm, key

    if not best_fired:
        mode = modes.get("balanced")
        raw, best_fired = mode_bonus(c, mode, energy=energy,
                                     median_wps=median_wps)
        ceiling = mode.lexicon_cap + mode.w_excite + mode.w_dynamics + mode.w_rate
        best_scaled = (raw / ceiling if ceiling > 0 else 0.0) * _MODE_SCALE

    return best_scaled, best_fired, label_key


# --------------------------------------------------------------------------
# Quality floor: return FEWER clips rather than filler.
#
# Asking for 8 clips from a 15-minute source scraped the barrel - the run
# produced "I win." (20s) and a mid-argument line with no signals at all.
# The rerank prompt tells the model to drop candidates that fail, but it
# filled the quota instead, so the floor is enforced in code rather than
# trusted to the model.
#
# The observed split was clean. The six good clips scored 6.54-15.12 with
# 2-5 content signals each; both rejects scored ~2.5 with ZERO. Their only
# signal was audio, and loudness is intensity, not substance - a clip whose
# sole evidence is "someone got loud here" has no demonstrated hook.
# --------------------------------------------------------------------------
_AUDIO_SIGNALS = {"loud", "animated", "fast"}

# Relative, not absolute: score magnitudes shift with content, so a fixed
# threshold tuned on one debate would mis-fire elsewhere. 0.35 of the best
# clip keeps 6.54 against a 15.12 top pick and drops 2.68.
FLOOR_FRACTION = 0.35


def passes_floor(c: dict, top_score: float) -> bool:
    """Is this clip worth publishing at all, independent of its rank?"""
    content = [s for s in c.get("signals", []) if s not in _AUDIO_SIGNALS]
    if not content:
        return False
    return c["score"] >= FLOOR_FRACTION * top_score


def _median_wps(candidates: list[dict]) -> float:
    """This speaker's typical words-per-second, for relative rate comparison."""
    rates = sorted(len(c["text"].split()) / max(c["duration"], 1) for c in candidates)
    return rates[len(rates) // 2] if rates else 0.0


def find_clips(
    words: list[dict],
    max_clips: int = 6,
    use_llm: bool = True,
    mode_key: str = "auto",
    energy=None,
    turn_starts: list[float] | None = None,
    instructions: str | None = None,
) -> list[dict]:
    """Pick clips matching `mode_key`, e.g. 'exciting' or 'controversial'."""
    auto = (mode_key or "auto").lower() == "auto"
    mode = None if auto else modes.get(mode_key)
    sentences = build_sentences(words, turn_starts=turn_starts)
    candidates = build_candidates(sentences)
    if not candidates:
        return []

    median_wps = _median_wps(candidates)
    for c in candidates:
        base = base_quality(c, turn_starts=turn_starts)
        if auto:
            # One pass hunts every kind of moment; the clip is labelled with
            # whichever it matched best.
            bonus, fired, kind = best_mode(c, energy=energy, median_wps=median_wps)
        else:
            bonus, fired = mode_bonus(c, mode, energy=energy, median_wps=median_wps)
            kind = mode.key
        vir, vfired = viral_signals(c)
        # Confrontational structure IS a debate moment, even when none of the
        # subject-matter lexicons fired - the HYPOCRISY-TRAP clip was landing
        # as "balanced" purely because it argues without naming a topic.
        if auto and kind == "balanced" and (
                {"accusation", "pinned-down", "HYPOCRISY-TRAP"} & set(vfired)):
            kind = "debate"
        c["base"] = round(base, 2)
        c["bonus"] = round(bonus, 2)
        c["viral"] = round(vir, 2)
        c["kind"] = kind
        c["signals"] = fired + vfired
        c["score"] = base + bonus + vir - LENGTH_COST_PER_SEC * c["duration"]

    shortlist = dedupe(candidates, limit=max(max_clips * 3, 40))

    # Claude first (real judgment), then local Ollama, then heuristics. Each
    # step returns None on failure, so selection always completes.
    ranked = None
    if use_llm:
        ranked = llm.rerank(shortlist, max_clips, instructions=instructions)
        if ranked is None:
            ranked = _rerank_with_ollama(shortlist, max_clips, mode)
    ordered = ranked or shortlist

    # Enforce the floor before truncating, so a weak tail is dropped rather
    # than padding the count. Returning 6 good clips beats 8 with filler.
    if ordered:
        top_score = max(c["score"] for c in ordered)
        kept = [c for c in ordered if passes_floor(c, top_score)]
        dropped = len(ordered) - len(kept)
        if dropped and kept:
            print(f"      (quality floor dropped {dropped} weak candidate"
                  f"{'s' if dropped != 1 else ''})")
        if kept:
            ordered = kept

    # Strict non-overlap on the OUTPUT. dedupe's 0.4 threshold is right for
    # the shortlist - it keeps genuinely different framings of a moment so
    # the reranker has choices - but it is wrong for what ships: two clips
    # may legitimately share up to 40% of the shorter one. Measured on this
    # episode, 6 of 22 pairs overlapped, and longer clips made it worse,
    # so it surfaced when the length cost came down. A viewer seeing the
    # same 30 seconds in two clips reads it as broken.
    final: list[dict] = []
    for c in ordered:
        if len(final) >= max_clips:
            break
        if any(_overlaps(c, k, threshold=OUTPUT_OVERLAP) for k in final):
            continue
        final.append(c)
    extend_to_finish(final, sentences)

    for n, c in enumerate(final, 1):
        c["title"] = f"clip_{n:02d}"
        c["hook"] = re.split(r"(?<=[.!?])\s", c["text"].strip())[0][:120]
        c["mode"] = c.get("kind", "balanced")
    return final
