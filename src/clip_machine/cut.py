"""Cut + reframe + treat — one ffmpeg pass, one encode.

Layouts:
  center — single speaker, 9:16 centre crop of the source
  split  — 50/50 stacked panels, seam at the midpoint. Takes the LEFT half
           of a wide source into the top panel and the RIGHT half into the
           bottom, which is what a two-person podcast recording needs.

Output size defaults to 720x1280, not 1080x1920. Measured: 1080 costs 2x
the render time (14.1s vs 7.1s per 30s) and buys nothing on a 480p-ish
source, where each panel is already being upscaled.

Hardware encoding (h264_qsv) was benchmarked and is NOT used: 9.5s vs
libx264 veryfast's 10.1s, because the filter graph runs on the CPU either
way and QSV pays for frame upload/download.
"""
import subprocess
from pathlib import Path

from . import captions, config, looks

LAYOUTS = ("center", "split")


def _center(w: int, h: int) -> str:
    # min() guards sources that are already vertical.
    return f"crop='min(iw,ih*{w}/{h})':ih,scale={w}:{h},setsar=1"


def _split(w: int, h: int, swap: bool = False) -> str:
    # Left half goes on top by default. swap puts the right half on top -
    # the user's only control over who is where, since nothing here knows
    # which side a particular person sits on.
    hh = h // 2
    top_x, bot_x = ("iw/2", "0") if swap else ("0", "iw/2")
    return (
        f"[0:v]crop=iw/2:ih:{top_x}:0,scale={w}:{hh}:force_original_aspect_ratio=increase,"
        f"crop={w}:{hh},setsar=1[top];"
        f"[0:v]crop=iw/2:ih:{bot_x}:0,scale={w}:{hh}:force_original_aspect_ratio=increase,"
        f"crop={w}:{hh},setsar=1[bot];"
        f"[top][bot]vstack=inputs=2[stacked]"
    )


def render_clip(
    video_path: Path,
    start: float,
    end: float,
    out_path: Path,
    layout: str = "center",
    size: tuple[int, int] | None = None,
    look: str | None = None,
    radius: int | None = None,
    caption_file: Path | None = None,
    swap: bool = False,
) -> Path:
    w, h = size or (config.OUT_WIDTH, config.OUT_HEIGHT)
    duration = end - start
    treat = looks.filters(look or config.LOOK, w, h)
    radius = config.CORNER_RADIUS if radius is None else radius

    inputs = ["-ss", f"{start:.3f}", "-i", str(video_path.resolve())]
    parts: list[str] = []

    # Build the picture up to a single labelled stream, then treat and round.
    if layout == "split":
        parts.append(_split(w, h, swap))
        stream = "stacked"
    else:
        parts.append(f"[0:v]{_center(w, h)}[framed]")
        stream = "framed"

    if treat:
        parts.append(f"[{stream}]{treat}[treated]")
        stream = "treated"

    if caption_file is not None:
        # After the look, so the treatment's downsample does not soften the
        # text, and before the corner mask so captions cannot overhang it.
        parts.append(f"[{stream}]subtitles={captions.filter_arg(caption_file)}[capped]")
        stream = "capped"

    if radius > 0:
        # The mask is a still, so it needs -loop or the stream ends after one
        # frame and the overlay stops with it.
        mask = looks.corner_mask(w, h, radius)
        inputs += ["-loop", "1", "-i", str(mask.resolve())]
        parts.append(f"[1:v]format=gray[mask]")
        parts.append(f"[{stream}]format=rgba[rgba]")
        parts.append("[rgba][mask]alphamerge[rounded]")
        parts.append(f"color=c=black:s={w}x{h}[bg]")
        parts.append("[bg][rounded]overlay=0:0:shortest=1,format=yuv420p[v]")
    else:
        parts.append(f"[{stream}]format=yuv420p[v]")

    subprocess.run(
        [
            "ffmpeg", "-y",
            *inputs,
            "-t", f"{duration:.3f}",
            "-filter_complex", ";".join(parts),
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            # Pinned: colortemperature and friends output yuv444p, which
            # encodes as High 4:4:4 Predictive and will not play in Windows
            # Movies & TV. That bug shipped once already.
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
