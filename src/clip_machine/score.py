"""Pick clip-worthy moments — fully local.

Two stages, because feeding a whole transcript to a CPU-bound LLM is not
viable on a GPU-less machine:

  1. Heuristic prefilter over every candidate window (cheap, instant).
  2. Optional rerank of the ~25 survivors by a local Ollama model, which
     only ever sees short candidate text. If Ollama isn't running, the
     heuristic ranking stands on its own and the pipeline still works.
"""
import json
import re

from . import config

# Openers that prove the clip depends on earlier context.
_CONTEXT_DEPENDENT = re.compile(
    r"^(and|so|but|because|then|anyway|also|plus|which|that's why|that is why"
    r"|he|she|they|it|this|that|those|these|him|her|them)\b",
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


def build_sentences(words: list[dict]) -> list[dict]:
    """Group the flat word list into sentences on terminal punctuation."""
    sentences, current = [], []
    for w in words:
        current.append(w)
        gap_next = False
        if _TERMINAL.search(w["word"]) or gap_next:
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


def heuristic_score(c: dict) -> float:
    text = c["text"].strip()
    first_sentence = re.split(r"(?<=[.!?])\s", text)[0]
    score = 0.0

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


def _rerank_with_ollama(candidates: list[dict], max_clips: int) -> list[dict] | None:
    """Return reordered candidates, or None if Ollama isn't available."""
    import requests  # local HTTP only

    listing = "\n\n".join(
        f"[{n}] ({c['duration']:.0f}s) {c['text'][:600]}"
        for n, c in enumerate(candidates)
    )
    prompt = (
        "You are picking short-form video clips from a longform transcript.\n"
        "A good clip: hooks in its first 3 seconds, stands alone with zero outside "
        "context, and has a clear payoff.\n\n"
        f"Candidates:\n\n{listing}\n\n"
        f"Return ONLY JSON: {{\"picks\": [<up to {max_clips} candidate numbers, best first>]}}"
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
        print(f"  (Ollama unavailable — using heuristic ranking. {type(e).__name__})")
        return None

    ordered = [candidates[i] for i in picks if isinstance(i, int) and 0 <= i < len(candidates)]
    return ordered or None


def find_clips(words: list[dict], max_clips: int = 6, use_llm: bool = True) -> list[dict]:
    sentences = build_sentences(words)
    candidates = build_candidates(sentences)
    if not candidates:
        return []

    for c in candidates:
        c["score"] = heuristic_score(c)

    shortlist = dedupe(candidates, limit=max(max_clips * 4, 25))

    ranked = _rerank_with_ollama(shortlist, max_clips) if use_llm else None
    final = (ranked or shortlist)[:max_clips]

    for n, c in enumerate(final, 1):
        c["title"] = f"clip_{n:02d}"
        c["hook"] = re.split(r"(?<=[.!?])\s", c["text"].strip())[0][:120]
    return final
