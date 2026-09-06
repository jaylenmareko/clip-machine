"""Word-timed burned-in captions via a generated .ass subtitle file.

Styles are measured from real reference clips, not invented. The
"reference" preset matches the podcast-clip look: sentence case, regular
weight grotesque, a solid box hugging each line, two lines of ~5-6 words,
line-at-a-time (no word-by-word karaoke).

ASS colour format is &HAABBGGRR — alpha first, then BLUE, green, red.
AA=00 is opaque, AA=FF is fully transparent.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CaptionStyle:
    key: str
    label: str
    font: str
    size: int
    primary: str        # text colour
    back: str           # box colour (BorderStyle=3 makes this a filled box)
    border_style: int   # 3 = opaque box, 1 = outline+shadow
    outline: int        # with BorderStyle=3 this is box padding
    shadow: int
    bold: int           # -1 = bold, 0 = normal
    words_per_line: int
    lines: int
    uppercase: bool


# Measured off the reference channel: white box, black text, ~80px regular
# grotesque, 2 lines x ~5-6 words, sitting in the lower third.
REFERENCE = CaptionStyle(
    key="reference", label="Reference (white box, black text)",
    font="Arial", size=78,
    primary="&H00000000", back="&H00FFFFFF",
    border_style=3, outline=8, shadow=0, bold=0,
    words_per_line=5, lines=2, uppercase=False,
)

# Same geometry, inverted for busy/bright footage — this is what the
# reference uses when captions sit over the video rather than over a
# letterbox.
REFERENCE_DARK = CaptionStyle(
    key="reference-dark", label="Reference (dark box, white text)",
    font="Arial", size=78,
    primary="&H00FFFFFF", back="&HA0000000",
    border_style=3, outline=8, shadow=0, bold=0,
    words_per_line=5, lines=2, uppercase=False,
)

# The original punchy style — big, bold, all-caps, 3 words at a time.
POP = CaptionStyle(
    key="pop", label="Pop (bold all-caps)",
    font="Arial Black", size=96,
    primary="&H00FFFFFF", back="&HA0000000",
    border_style=1, outline=7, shadow=3, bold=-1,
    words_per_line=3, lines=1, uppercase=True,
)

STYLES = {s.key: s for s in (REFERENCE, REFERENCE_DARK, POP)}
DEFAULT_STYLE = "reference"

# Vertical placement presets.
#   lower  — lower third, for a single full-frame speaker
#   seam   — vertically centred, straddling a 50/50 split-screen seam
POSITIONS = {"lower": 2, "seam": 5}   # ASS numpad alignment
DEFAULT_POSITION = "lower"

_MARGIN_V = 210  # measured: caption block bottom sits ~210px off the frame bottom


def _header(style: CaptionStyle, alignment: int) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{style.font},{style.size},{style.primary},&H000000FF,{style.back},{style.back},{style.bold},0,0,0,100,100,0,0,{style.border_style},{style.outline},{style.shadow},{alignment},60,60,{_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _clean(word: str) -> str:
    return word.replace("{", "").replace("}", "").replace("\n", " ").strip()


def build_ass(
    words: list[dict],
    clip_start: float,
    clip_end: float,
    out_path: Path,
    style: str = DEFAULT_STYLE,
    position: str = DEFAULT_POSITION,
) -> Path:
    """Write an .ass file with times rebased to the clip's own timeline."""
    st = STYLES.get(style, REFERENCE)
    alignment = POSITIONS.get(position, 2)
    per_event = st.words_per_line * st.lines

    in_clip = [
        w for w in words
        if w["start"] >= clip_start - 0.05 and w["end"] <= clip_end + 0.05
    ]

    events: list[tuple[float, float, str]] = []
    for i in range(0, len(in_clip), per_event):
        group = in_clip[i:i + per_event]
        start = group[0]["start"] - clip_start
        end = group[-1]["end"] - clip_start

        tokens = [_clean(w["word"]) for w in group]
        tokens = [t for t in tokens if t]
        if not tokens:
            continue
        if st.uppercase:
            tokens = [t.upper() for t in tokens]

        # Break into the style's line count rather than relying on wrapping,
        # so line breaks land where we want them.
        rows, step = [], st.words_per_line
        for j in range(0, len(tokens), step):
            rows.append(" ".join(tokens[j:j + step]))
        text = "\\N".join(rows[:st.lines])

        events.append((start, end, text))

    # Two events on screen at once render stacked on top of each other, so
    # clamp every event to end before the next one starts. Also hold each
    # line through the gap to the next, so captions don't flicker off during
    # natural pauses.
    lines = [_header(st, alignment)]
    for n, (start, end, text) in enumerate(events):
        next_start = events[n + 1][0] if n + 1 < len(events) else None
        if next_start is not None:
            end = max(min(next_start - 0.02, next_start), start + 0.05)
        elif end <= start:
            end = start + 0.4
        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,,0,0,0,,{text}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
