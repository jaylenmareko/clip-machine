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
    # Which of this mode's signals actually make a clip THIS KIND of moment,
    # as opposed to merely being about the same subject matter.
    #
    # Measured on a 67-minute interview: 12 of 22 clips came back labelled
    # "debate" and only 2 of those had a conflict signal. The rest won the
    # label on `money-take`, `gender-take` and `absolutes` - i.e. on the
    # words "money", "broke", "men", "never", "always", which show up in any
    # ordinary conversation about life. One was a story about Whoopi
    # Goldberg being turned away from a restaurant, labelled a debate purely
    # because someone said "never".
    #
    # Subject matter is not moment type. Topic signals still add score - a
    # money argument IS more clippable - but they cannot name the clip.
    # Empty means every signal counts as identifying.
    identifying: tuple[str, ...] = ()
    _compiled: list = field(default_factory=list, repr=False)

    def identifies(self, fired: list[str]) -> bool:
        """Did anything fire that makes this clip genuinely this kind?"""
        if not self.identifying:
            return bool(fired)
        return any(f in self.identifying for f in fired)

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
    # Taking a stance against something makes a clip controversial. Merely
    # mentioning a scandal word ("fired", "sued") does not.
    identifying=("disagreement", "against-consensus", "calling-out",
                 "prescriptive"),
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
    # A debate needs someone pushing back or going after somebody. Talking
    # ABOUT money, gender or absolutes is just conversation.
    identifying=("pushback", "status-attack"),
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
    # "I was", "I felt" is how everyone talks. A story needs an opening or a
    # sequence of events.
    identifying=("narrative-open", "sequence"),
)

_EDUCATIONAL = Mode(
    key="educational",
    label="Lesson",
    blurb="A point taught — explanation, principle, hard-won rule",
    patterns=[
        ("explains", r"\b(the reason (is|why)|this is how|here'?s how|the key is"
                     r"|which means|what that means)\b", 2.5),
        # The original three patterns only caught FORMAL teaching - "here's
        # how", "step 1", "40 percent". A long-form interview teaches in
        # none of those shapes; it teaches in maxims. On a 67-minute episode
        # made almost entirely of life lessons, exactly ONE clip out of 22
        # came back labelled educational, and 12 came back "debate". The
        # lesson vocabulary below is how the wisdom actually sounds.
        ("principle", r"\b(the thing about|what people don'?t (realize|understand|get)"
                      r"|most people (don'?t|never)|you have to understand"
                      r"|understand something|the difference between"
                      r"|that'?s (what|how) (real|true)|at the end of the day)\b", 2.5),
        ("conditional-rule", r"\b(if you (can'?t|don'?t|want)|if somebody|when you"
                             r"|the moment you|once you)\b.{0,80}?"
                             r"\b(you'?ll|you will|you can'?t|it'?s|then)\b", 2.0),
        ("earned", r"\b(i learned|taught me|the lesson|what i realized"
                   r"|took me (years|a long time)|the hard way"
                   r"|i coined the phrase|my (father|mother|pops|dad) (said|told)"
                   r"|there'?s a saying)\b", 2.5),
        ("defines", r"\b(that'?s not|it'?s not (about|a)|isn'?t (about|a))\b"
                    r".{0,60}?\b(it'?s|that'?s)\b", 1.5),
        ("enumerates", r"\b(first,|second,|third,|three (things|ways|reasons|steps)"
                       r"|step \d|number one|two things)\b", 2.0),
        ("data", r"(\b\d+\s?(%|percent)\b|\$\s?\d)", 1.5),
    ],
    lexicon_cap=5.5,
    # Numbers alone are not a lesson - a dollar figure shows up in any story.
    identifying=("explains", "principle", "conditional-rule", "earned",
                 "defines", "enumerates"),
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
