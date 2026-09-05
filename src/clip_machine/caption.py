"""Word-timed burned-in captions via a generated .ass subtitle file."""
from pathlib import Path

WORDS_PER_LINE = 3

_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Pop,Arial Black,96,&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,340,1

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


def build_ass(words: list[dict], clip_start: float, clip_end: float, out_path: Path) -> Path:
    """Write an .ass file with times rebased to the clip's own timeline."""
    in_clip = [w for w in words if w["start"] >= clip_start - 0.05 and w["end"] <= clip_end + 0.05]

    lines = [_HEADER]
    for i in range(0, len(in_clip), WORDS_PER_LINE):
        group = in_clip[i:i + WORDS_PER_LINE]
        start = group[0]["start"] - clip_start
        end = group[-1]["end"] - clip_start
        if end <= start:
            end = start + 0.3
        text = " ".join(_clean(w["word"]) for w in group).upper()
        if not text:
            continue
        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Pop,,0,0,0,,{text}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
