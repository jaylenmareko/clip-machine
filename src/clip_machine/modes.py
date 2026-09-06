"""Clip selection modes — what KIND of moment to hunt for.

Base quality (self-containedness, clean ending, length) always applies and
lives in score.py. A mode adds a bonus on top of it, so a controversial
clip that starts mid-sentence still loses to a clean one.

Two signal families:
  - lexical: regexes over the transcript
  - acoustic: loudness percentiles from audio_features (excitement is
    physical — a keyword list cannot hear someone raise their voice)

Weights below are reasoned starting points, NOT empirically tuned. Every
pick's fired signals go into manifest.json so they can be tuned against
real output.
"""
import re
from dataclasses import dataclass, field


@dataclass
class Mode:
    key: str
    label: str
    blurb: str
    patterns: list[tuple[str, str, float]]      # (signal, regex, weight)
    w_excite: float = 0.0                       # acoustic loudness
    w_dynamics: float = 0.0                     # acoustic variation
    w_rate: float = 0.0                         # faster-than-usual speech
    lexicon_cap: float = 4.0                    # ceiling on stacked lexical hits
    _compiled: list = field(default_factory=list, repr=False)

    def compiled(self):
        if not self._compiled:
            self._compiled = [
                (name, re.compile(rx, re.IGNORECASE), w) for name, rx, w in self.patterns
            ]
        return self._compiled

    def lexical(self, text: str) -> tuple[float, list[str]]:
        """Return (capped score, names of signals that fired)."""
        total, fired = 0.0, []
        for name, rx, w in self.compiled():
            if rx.search(text):
                total += w
                fired.append(name)
        return min(total, self.lexicon_cap), fired


_CONTROVERSIAL = Mode(
    key="controversial",
    label="Controversial",
    blurb="Contrarian takes, called-out consensus, conflict",
    patterns=[
        ("disagreement", r"\b(i disagree|that'?s (wrong|nonsense|bullshit|garbage)"
                         r"|actually,? no|dead wrong|flat out wrong)\b", 2.5),
        ("against-consensus", r"\b(everyone (thinks|says|believes|assumes)|most people (think|believe)"
                              r"|conventional wisdom|the myth|it'?s a lie|they lied to you"
                              r"|unpopular opinion|hot take|controversial)\b", 2.5),
        ("calling-out", r"\b(the (real )?problem with|what they don'?t (tell|want)"
                        r"|nobody (talks about|wants to (hear|admit))|the truth (is|about)"
                        r"|here'?s what nobody)\b", 2.0),
        ("prescriptive", r"\b(stop (doing|saying|believing)|you should never|never (do|listen|trust)"
                         r"|quit (doing|trying))\b", 1.5),
        ("conflict", r"\b(fired|banned|cancell?ed|sued|scam|fraud|corrupt|rigged"
                     r"|outrage|backlash|attacked|blame)\b", 1.5),
        ("overrated", r"\b(overrated|useless|waste of (time|money)|doesn'?t work|failed)\b", 1.0),
    ],
    w_excite=1.5,
    w_dynamics=0.5,
    lexicon_cap=5.5,
)

_EXCITING = Mode(
    key="exciting",
    label="Exciting",
    blurb="High energy, big claims, raised voice",
    patterns=[
        ("intensity", r"\b(insane|crazy|wild|unbelievable|incredible|amazing|ridiculous"
                      r"|mind[- ]?blow\w*|blew my mind|no way|shocking|nuts|epic)\b", 2.0),
        ("superlative", r"\b(the (most|best|worst|biggest|craziest|greatest|hardest)"
                        r"|never seen|first time ever|world record|all time)\b", 1.5),
        ("stakes", r"\b(million|billion|\$\s?\d|died|almost (died|lost)|life changing"
                   r"|emergency|disaster|survived)\b", 1.5),
        ("exclaim", r"!", 1.0),
    ],
    w_excite=4.0,
    w_dynamics=1.0,
    w_rate=1.5,
    lexicon_cap=3.5,
)

# Tuned to conflict-driven podcast clips (the @TheSaintandTheSinner format:
# money / relationships / masculinity takes, plus on-air beef). Generic
# "controversial" misses this because the heat is in the SUBJECT vocabulary,
# not in dramatic phrasing.
_DEBATE = Mode(
    key="debate",
    label="Debate / Hot Take",
    blurb="Podcast conflict — money, relationships, status takes, pushback",
    patterns=[
        ("pushback", r"\b(hold on|wait wait|let me finish|that'?s not true|you'?re wrong"
                     r"|see that'?s the problem|no no no|first of all|listen to me)\b", 3.0),
        ("gender-take", r"\b(women|men|female|male|wife|husband|marriage|divorce|dating"
                        r"|loyalty|cheat\w*|standards|body count|provider|hypergam\w*"
                        r"|high value|feminine|masculine)\b|50/50", 2.5),
        ("money-take", r"\b(broke|rich|wealth|money|income|salary|paycheck|debt|invest\w*"
                       r"|millionaire|net worth|cheap|financially)\b", 2.0),
        ("status-attack", r"\b(soft|weak|lazy|excuses|delusional|entitled|clown|fraud"
                          r"|beta|simp|washed|overrated)\b", 2.5),
        ("direct-address", r"\b(you need to|if you'?re a (man|woman)|let me tell you"
                           r"|here'?s the reality|real talk|facts)\b", 1.5),
        ("absolutes", r"\b(never|always|every (man|woman)|nobody|everybody)\b", 1.0),
    ],
    w_excite=2.5,
    w_dynamics=1.5,
    w_rate=1.0,
    lexicon_cap=6.0,
)

_FUNNY = Mode(
    key="funny",
    label="Funny",
    blurb="Laughter and comedic beats",
    patterns=[
        ("laughter", r"(\(laugh\w*\)|\[laugh\w*\]|\bhaha+\b|\blol\b|\(chuckl\w*\))", 4.0),
        ("humor-words", r"\b(joke|funny|hilarious|ridiculous|absurd|embarrassing)\b", 1.5),
    ],
    w_excite=2.0,
    w_dynamics=1.5,
    lexicon_cap=5.0,
)

_STORY = Mode(
    key="story",
    label="Story",
    blurb="Personal narrative with a beginning and a payoff",
    patterns=[
        ("narrative-open", r"\b(when i was|one day|the first time|back in|years ago"
                           r"|i remember|it started (when|with)|so i)\b", 2.5),
        ("sequence", r"\b(and then|suddenly|next thing|turns out|ended up|eventually)\b", 1.5),
        ("personal", r"\b(i (was|had|got|went|felt|realized|decided))\b", 1.0),
    ],
    w_excite=0.5,
    w_dynamics=0.5,
    lexicon_cap=4.5,
)

_EDUCATIONAL = Mode(
    key="educational",
    label="Educational",
    blurb="Clear explanation, steps, concrete numbers",
    patterns=[
        ("explains", r"\b(the reason (is|why)|this is how|here'?s how|the key is"
                     r"|which means|what that means)\b", 2.5),
        ("enumerates", r"\b(first,|second,|third,|three (things|ways|reasons|steps)"
                       r"|step \d|number one|two things)\b", 2.0),
        ("data", r"(\b\d+\s?(%|percent)\b|\$\s?\d)", 1.5),
    ],
    lexicon_cap=4.5,
)

_BALANCED = Mode(
    key="balanced",
    label="Balanced",
    blurb="General-purpose — clean, self-contained, decent hook",
    patterns=[
        ("intensity", r"\b(insane|crazy|unbelievable|amazing|shocking)\b", 1.0),
        ("against-consensus", r"\b(most people|everyone (thinks|says)|the truth is"
                              r"|nobody talks about)\b", 1.0),
        ("story", r"\b(when i was|one day|the first time|i remember)\b", 1.0),
    ],
    w_excite=1.0,
    w_dynamics=0.25,
    lexicon_cap=2.5,
)

MODES: dict[str, Mode] = {
    m.key: m for m in (_BALANCED, _EXCITING, _CONTROVERSIAL, _DEBATE, _FUNNY,
                       _STORY, _EDUCATIONAL)
}

DEFAULT = "balanced"


def get(key: str) -> Mode:
    return MODES.get((key or DEFAULT).lower(), MODES[DEFAULT])
