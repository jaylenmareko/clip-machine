"""Cut + reframe — one ffmpeg pass, one encode.

Layouts:
  center — single speaker, 9:16 centre crop of the source
  split  — 50/50 stacked panels, seam at the midpoint. Takes the LEFT half
           of a wide source into the top panel and the RIGHT half into the
           bottom, which is what a two-person podcast recording needs.

Output size defaults to 720x1280, not 1080x1920. Measured: 1080 costs 2x
the render time (14.1s vs 7.1s per 30s) and buys nothing on a 480p-ish
source, where each panel is already being upscaled. Override with
CLIP_OUT_WIDTH / CLIP_OUT_HEIGHT if a genuinely high-res source turns up.

Hardware encoding (h264_qsv) was benchmarked and is NOT used: 9.5s vs
libx264 veryfast's 10.1s, because the filter graph runs on the CPU either
way and QSV pays for frame upload/download.
"""
import subprocess
from pathlib import Path

from . import config, looks

LAYOUTS = ("center", "split")


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


def make_thumbnail(clip_path: Path, out_path: Path, at: float = 1.0) -> Path | None:
    """Grab a poster frame so the grid is scannable before anything is played.

    Without this the library is a wall of black rectangles - <video> shows
    nothing until the user presses play.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.2f}",
             "-i", str(clip_path.resolve()), "-frames:v", "1",
             "-vf", "scale=360:-2", str(out_path.resolve())],
            check=True, capture_output=True, timeout=120,
        )
        return out_path
    except Exception:
        return None


def render_clip(
    video_path: Path,
    start: float,
    end: float,
    out_path: Path,
    layout: str = "center",
    size: tuple[int, int] | None = None,
    look: str | None = None,
) -> Path:
    w, h = size or (config.OUT_WIDTH, config.OUT_HEIGHT)
    duration = end - start

    # Picture treatment rides on the end of the same pass - no second encode.
    treat = looks.filters(look or config.LOOK, w, h)
    tail = f",{treat}" if treat else ""

    if layout == "split":
        stacked = f"[stacked]{treat}[v]" if treat else "[stacked]null[v]"
        filter_args = ["-filter_complex", _split(w, h) + ";" + stacked,
                       "-map", "[v]", "-map", "0:a?"]
    else:
        filter_args = ["-vf", _center(w, h) + tail]

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path.resolve()),
            "-t", f"{duration:.3f}",
            *filter_args,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            # Pinned: colortemperature and friends output yuv444p, which
            # encodes as High 4:4:4 Predictive and will not play in
            # Windows Movies & TV.
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-r", "30",
            "-movflags", "+faststart",
            str(out_path.resolve()),
        ],
        check=True,
        capture_output=True,
    )
    return out_path
