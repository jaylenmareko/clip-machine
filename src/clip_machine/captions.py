r"""Burned-in captions, built from the word timings we already have.

Costs almost nothing to render - measured 0.5s on a 35s clip, +5% - because
the timings come free with the transcript. The real cost of captions is that
they make transcription errors VISIBLE. tiny.en was chosen precisely because
nobody reads the transcript ("Are you open floor? That's beautiful." shipped
in a clip's index and harmed nothing). On screen that is embarrassing, so
captions pull the model up to base.en, which measured 11.0x -> 7.5x realtime
on this machine: a 67-minute source goes from 6 to 9 minutes.

WINDOWS PATHS. ffmpeg treats ':' as an option separator inside a filter, so
a drive letter breaks the subtitles filter. Tested all three candidates:

    subtitles=C:/x/cap.ass       -> "Unable to parse original_size"
    subtitles=C\:/x/cap.ass      -> "Unable to parse original_size"
    subtitles='C\:/x/cap.ass'    -> works

Escaped AND quoted. filter_arg() is the only place that knowledge lives.
"""
import re
from pathlib import Path

# Short-form captions are read in motion on a small screen, so: few words at
# a time, heavy weight, thick outline, and clear of the bottom UI where the
# platform puts its own overlays.
WORDS_PER_LINE = 4
FONT = "Arial"
FONT_SIZE = 62
MARGIN_V = 210          # px up from the bottom edge at 1280 tall
OUTLINE = 4
SHADOW = 1

_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
; WrapStyle 0, not 2. Style 2 means "no wrapping", so a group of four long
; words runs straight off the side of the frame instead of breaking. 0 wraps
; and balances the lines.
WrapStyle: 0
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,60,60,{mv},1

[Events]
Format: Layer, Start, End, Style, Text
"""


def _ts(t: float) -> str:
    """ASS wants h:mm:ss.cc, and refuses a negative."""
    t = max(t, 0.0)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _clean(text: str) -> str:
    # '{' and '}' open an ASS override block; a literal brace from a
    # transcript would silently swallow the rest of the line.
    text = text.replace("{", "(").replace("}", ")")
    # A newline would terminate the event early.
    return re.sub(r"\s+", " ", text).strip()


def build_ass(words: list[dict], start: float, end: float, out_path: Path,
              size: tuple[int, int], per_line: int = WORDS_PER_LINE) -> Path | None:
    """Write an ASS file covering [start, end], timed from the clip's own zero.

    Returns None when there is nothing to show, so the caller can skip the
    filter entirely rather than pass an empty subtitle file.
    """
    w, h = size
    inside = [x for x in words if start <= float(x["start"]) < end]
    if not inside:
        return None

    lines = []
    for i in range(0, len(inside), per_line):
        group = inside[i:i + per_line]
        text = _clean(" ".join(str(g["word"]) for g in group))
        if not text:
            continue
        a = float(group[0]["start"]) - start
        b = min(float(group[-1]["end"]) - start, end - start)
        # A group whose words all share a timestamp would render for 0s.
        if b - a < 0.20:
            b = a + 0.20
        lines.append(f"Dialogue: 0,{_ts(a)},{_ts(b)},Cap,{text}")

    if not lines:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _ASS_HEADER.format(w=w, h=h, font=FONT, size=FONT_SIZE,
                           outline=OUTLINE, shadow=SHADOW,
                           mv=round(MARGIN_V * h / 1280))
        + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return out_path


def filter_arg(ass_path: Path) -> str:
    """The `subtitles=` argument, escaped the one way ffmpeg accepts."""
    posix = ass_path.resolve().as_posix()
    return "'" + posix.replace(":", r"\:", 1) + "'"
