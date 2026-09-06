"""Cut + reframe (+ optional caption burn) — one ffmpeg pass, one encode.

Two passes would double encode time, which matters on a CPU-only machine.

Layouts (measured off the reference channel):
  center     — single speaker, 9:16 centre crop of the source
  split      — 50/50 stacked panels, seam at the midpoint
  split-auto — same, but the active speaker is swapped into the top panel
               as the turns change (see speakers.py)

Output size defaults to 720x1280, not 1080x1920. Measured: 1080 costs 2x
the render time (14.1s vs 7.1s per 30s of split-auto) and buys nothing on a
480p-ish source, where each panel is already being upscaled. Override with
CLIP_OUT_WIDTH / CLIP_OUT_HEIGHT if a genuinely high-res source turns up.

Hardware encoding (h264_qsv) was benchmarked and is NOT used: it came in at
9.5s vs libx264 veryfast's 10.1s, because the filter graph runs on the CPU
either way and QSV pays for frame upload/download.
"""
import subprocess
from pathlib import Path

from . import config

_MAX_SWAPS = 60  # keep the enable expression a sane length

LAYOUTS = ("center", "split", "split-auto")


def _center(w: int, h: int) -> str:
    # min() guards sources that are already vertical.
    return f"crop='min(iw,ih*{w}/{h})':ih,scale={w}:{h},setsar=1"


def _split(w: int, h: int) -> str:
    hh = h // 2
    return (
        f"[0:v]crop=iw/2:ih:0:0,scale={w}:{hh}:force_original_aspect_ratio=increase,"
        f"crop={w}:{hh},setsar=1[top];"
        f"[0:v]crop=iw/2:ih:iw/2:0,scale={w}:{hh}:force_original_aspect_ratio=increase,"
        f"crop={w}:{hh},setsar=1[bot];"
        f"[top][bot]vstack=inputs=2[stacked]"
    )


def _split_auto(w: int, h: int, enable: str) -> str:
    hh = h // 2
    return (
        f"[0:v]crop=iw/2:ih:0:0,scale={w}:{hh}:force_original_aspect_ratio=increase,"
        f"crop={w}:{hh},setsar=1,split=2[l1][l2];"
        f"[0:v]crop=iw/2:ih:iw/2:0,scale={w}:{hh}:force_original_aspect_ratio=increase,"
        f"crop={w}:{hh},setsar=1,split=2[r1][r2];"
        f"[l1][r1]vstack=inputs=2[ab];"
        f"[r2][l2]vstack=inputs=2[ba];"
        f"[ab][ba]overlay=enable='{enable}'[stacked]"
    )


def render_clip(
    video_path: Path,
    start: float,
    end: float,
    ass_path: Path | None,
    out_path: Path,
    layout: str = "center",
    swap_intervals: list[tuple[float, float]] | None = None,
    size: tuple[int, int] | None = None,
) -> Path:
    """Render one clip. `ass_path=None` renders without burned-in captions."""
    w, h = size or (config.OUT_WIDTH, config.OUT_HEIGHT)
    duration = end - start

    # ffmpeg's subtitles filter chokes on Windows drive-letter paths, so when
    # captions are on we run from the .ass file's directory and reference it
    # by bare filename.
    subs = f"subtitles={ass_path.name}" if ass_path else None
    cwd = str(ass_path.parent) if ass_path else None

    if layout == "split-auto":
        spans = (swap_intervals or [])[:_MAX_SWAPS]
        # between() returns 1/0; summing them acts as logical OR. '0' means
        # never swap, which degrades cleanly to a plain split.
        enable = "+".join(f"between(t,{s:.2f},{e:.2f})" for s, e in spans) or "0"
        graph = _split_auto(w, h, enable)
    elif layout == "split":
        graph = _split(w, h)
    else:
        graph = None

    if graph is not None:
        tail = f";[stacked]{subs}[v]" if subs else ";[stacked]null[v]"
        filter_args = ["-filter_complex", graph + tail, "-map", "[v]", "-map", "0:a?"]
    else:
        vf = _center(w, h) + (f",{subs}" if subs else "")
        filter_args = ["-vf", vf]

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path.resolve()),
            "-t", f"{duration:.3f}",
            *filter_args,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-r", "30",
            "-movflags", "+faststart",
            str(out_path.resolve()),
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return out_path
