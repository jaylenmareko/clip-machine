"""Assemble several segments into ONE video of a target length.

Different operation from the normal output. Normally each clip is a
standalone short. Here the segments are cut from the same source and joined
into a single edit - "turn this 8-minute video into one 2:50 vertical for
Instagram".

Two decisions worth stating:

ORDER. Not chronological, and not purely by rank. The strongest segment
opens (a cold open - the hook has to land in the first seconds or nothing
else matters), and everything after it runs in source order so the argument
still tracks. Pure rank order jumbles the timeline; pure chronological order
buries the hook.

LENGTH. Segments are taken best-first until the target is reached, then the
LAST one in playback order is trimmed to land exactly on target rather than
overshooting.
"""
import subprocess
from pathlib import Path

from . import cut


def choose(clips: list[dict], target: float) -> list[dict]:
    """Pick and order segments to fill `target` seconds."""
    if not clips:
        return []

    chosen: list[dict] = []
    total = 0.0
    for c in clips:                      # already ranked best-first
        if total >= target:
            break
        chosen.append(c)
        total += c["duration"]

    if not chosen:
        return []

    # Cold open: best segment first, the rest in source order.
    opener, rest = chosen[0], sorted(chosen[1:], key=lambda c: c["start"])
    ordered = [opener] + rest

    # Trim the final segment so the edit lands on the target instead of
    # running past it. Anything under 4s left over is not worth a cut.
    running = 0.0
    trimmed: list[dict] = []
    for c in ordered:
        remaining = target - running
        if remaining <= 0:
            break
        seg = dict(c)
        if seg["duration"] > remaining:
            if remaining < 4:
                break
            seg["end"] = seg["start"] + remaining
            seg["duration"] = remaining
        trimmed.append(seg)
        running += seg["duration"]

    return trimmed


def build(video_path: Path, segments: list[dict], out_path: Path,
          layout: str = "center", look: str | None = None,
          radius: int | None = None, work_dir: Path | None = None) -> Path:
    """Render each segment identically, then join them.

    Rendering with identical settings first is what lets the join be a
    stream copy - no re-encode, so the seam costs nothing in quality.
    """
    work = work_dir or out_path.parent
    work.mkdir(parents=True, exist_ok=True)

    parts: list[Path] = []
    for i, seg in enumerate(segments, 1):
        part = work / f"{out_path.stem}_part{i:02d}.mp4"
        cut.render_clip(video_path, seg["start"], seg["end"], part,
                        layout=layout, look=look, radius=radius)
        parts.append(part)

    listing = work / f"{out_path.stem}_parts.txt"
    listing.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in parts),
        encoding="utf-8",
    )

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing.resolve()), "-c", "copy",
         "-movflags", "+faststart", str(out_path.resolve())],
        check=True, capture_output=True,
    )

    for p in parts:
        p.unlink(missing_ok=True)
    listing.unlink(missing_ok=True)
    return out_path


def parse_target(text: str) -> float | None:
    """Read '2:50', '170', '2m50s' as seconds."""
    import re

    t = (text or "").strip().lower()
    if not t:
        return None
    if m := re.fullmatch(r"(\d+):(\d{1,2})", t):
        return int(m.group(1)) * 60 + int(m.group(2))
    if m := re.fullmatch(r"(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?", t):
        if m.group(1) or m.group(2):
            return int(m.group(1) or 0) * 60 + int(m.group(2) or 0)
    if m := re.fullmatch(r"(\d+(?:\.\d+)?)", t):
        return float(m.group(1))
    return None
