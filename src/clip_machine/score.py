"""Pick clip-worthy moments — fully local.

Scoring is two parts:

  base_quality  — is this a usable clip at all (self-contained, clean
                  ending, sane length). Always applies.
  mode_bonus    — is this the KIND of moment we're hunting for (exciting,
                  controversial, funny, story, educational). See modes.py.

Selection runs in two stages, because feeding a whole transcript to a
CPU-bound LLM is not viable on a GPU-less machine:

  1. Heuristic prefilter over every candidate window (cheap, instant).
  2. Optional rerank of the ~25 survivors by a local Ollama model, which
     only ever sees short candidate text. If Ollama isn't running, the
     heuristic ranking stands on its own and the pipeline still works.
"""
import json
import re

from . import config, modes

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


TURN_TOLERANCE = 0.6   # seconds; how close a clip start must be to a turn
TURN_BONUS = 1.0       # reward for starting where someone starts talking.
                       # Measured: 3.0 outranked actual content (4/5 picks had
                       # debate signals); 1.0 gives 5/5 with signals AND 5/5
                       # clean starts, still beating no-bonus on turn alignment.
TURN_PENALTY = 1.5     # cost of starting mid-turn


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

    # Self-containedness — the single biggest quality lever.
    if _CONTEXT_DEPENDENT.match(text):
        score -= 4.0
    if _TERMINAL.search(text):
        score += 1.5
    else:
        score -= 2.0

    # Hook strength in the opening line.
    for rx in _HOOK_RE:
        if rx.search(first_sentence):
            score += 2.0
            break

    # Concrete detail anywhere in the clip.
    if _CONCRETE.search(text):
        score += 1.0

    # Length sweet spot: 30-60s.
    dur = c["duration"]
    if 30 <= dur <= 60:
        score += 2.0
    elif dur < 25 or dur > 80:
        score -= 1.0

    # Speech density — catch dead air or garbled runs.
    wps = len(text.split()) / max(dur, 1)
    if 2.0 <= wps <= 4.0:
        score += 1.0
    elif wps < 1.2:
        score -= 2.0

    return score


def mode_bonus(c: dict, mode, energy=None, median_wps: float = 0.0) -> tuple[float, list[str]]:
    """What this mode is hunting for, on top of base quality.

    Returns (bonus, fired signal names) so every pick stays explainable.
    """
    bonus, fired = mode.lexical(c["text"])

    if energy is not None and energy.ok:
        if mode.w_excite:
            e = energy.excitement(c["start"], c["end"])
            bonus += mode.w_excite * e
            if e > 0.7:
                fired.append("loud")
        if mode.w_dynamics:
            d = energy.dynamics(c["start"], c["end"])
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


def dedupe(candidates: list[dict], limit: int) -> list[dict]:
    """Greedy non-maximum suppression by score."""
    kept: list[dict] = []
    for c in sorted(candidates, key=lambda x: -x["score"]):
        if any(_overlaps(c, k) for k in kept):
            continue
        kept.append(c)
        if len(kept) >= limit:
            break
    return kept


def _rerank_with_ollama(candidates: list[dict], max_clips: int, mode) -> list[dict] | None:
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
    prompt = (
        "You are picking short-form video clips from a longform transcript.\n"
        f"You are specifically hunting for: {mode.label} — {mode.blurb}.\n\n"
        "Every pick must also hook in its first 3 seconds, stand alone with zero "
        "outside context, and have a clear payoff. Prefer a clip that genuinely "
        f"matches '{mode.label}' over one that merely uses dramatic words.\n\n"
        f"Candidates:\n\n{listing}\n\n"
        f'Return ONLY JSON: {{"picks": [<up to {max_clips} candidate numbers, best first>]}}'
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


def _median_wps(candidates: list[dict]) -> float:
    """This speaker's typical words-per-second, for relative rate comparison."""
    rates = sorted(len(c["text"].split()) / max(c["duration"], 1) for c in candidates)
    return rates[len(rates) // 2] if rates else 0.0


def find_clips(
    words: list[dict],
    max_clips: int = 6,
    use_llm: bool = True,
    mode_key: str = modes.DEFAULT,
    energy=None,
    turn_starts: list[float] | None = None,
) -> list[dict]:
    """Pick clips matching `mode_key`, e.g. 'exciting' or 'controversial'."""
    mode = modes.get(mode_key)
    sentences = build_sentences(words, turn_starts=turn_starts)
    candidates = build_candidates(sentences)
    if not candidates:
        return []

    median_wps = _median_wps(candidates)
    for c in candidates:
        base = base_quality(c, turn_starts=turn_starts)
        bonus, fired = mode_bonus(c, mode, energy=energy, median_wps=median_wps)
        c["base"] = round(base, 2)
        c["bonus"] = round(bonus, 2)
        c["signals"] = fired
        c["score"] = base + bonus

    shortlist = dedupe(candidates, limit=max(max_clips * 4, 25))

    ranked = _rerank_with_ollama(shortlist, max_clips, mode) if use_llm else None
    final = (ranked or shortlist)[:max_clips]

    for n, c in enumerate(final, 1):
        c["title"] = f"clip_{n:02d}"
        c["hook"] = re.split(r"(?<=[.!?])\s", c["text"].strip())[0][:120]
        c["mode"] = mode.key
    return final
