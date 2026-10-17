"""Cut + reframe + treat — one ffmpeg pass, one encode.

Layouts:
  center — single speaker, 9:16 centre crop of the source
  split  — 50/50 stacked panels, seam at the midpoint. Takes the LEFT half
           of a wide source into the top panel and the RIGHT half into the
           bottom, which is what a two-person podcast recording needs.
  left   — one speaker who sits in the LEFT half of a two-panel source (the
  right    classic camera-on-one-side / live-chat-on-the-other stream). Crops
           to that half, then 9:16 centres within it, so the chat panel is
           dropped instead of being stacked under the speaker. `right` is the
           mirror for a camera on the right.

Output size defaults to 720x1280, not 1080x1920. Measured: 1080 costs 2x
the render time (14.1s vs 7.1s per 30s) and buys nothing on a 480p-ish
source, where each panel is already being upscaled.

Hardware encoding (h264_qsv) was benchmarked and is NOT used: 9.5s vs
libx264 veryfast's 10.1s, because the filter graph runs on the CPU either
way and QSV pays for frame upload/download.
"""
import re
import subprocess
from pathlib import Path

from . import captions, config, looks

LAYOUTS = ("center", "split", "left", "right")


def _detect_content(video_path: Path, start: float, duration: float):
    """The non-black content box of the source at this moment, or None.

    Many of these livestream VODs frame the webcam in a corner with black
    (or a hard letterbox) filling the rest of the 16:9 frame. A straight
    9:16 crop of that padded frame lands half on the speaker, half on black.
    cropdetect returns the bounding box of the real picture so the reframe
    can run on THAT. Returns (w, h, x, y) or None when the frame is already
    full / detection fails - in which case rendering behaves exactly as before.

    Conservative on purpose: the largest box seen across the sample wins, so a
    dark frame (dark skin, dark clothes against black) can't make it crop IN
    past the real edge. limit=24 keeps it to near-black padding, not shadow.
    """
    sample = min(max(duration, 1.0), 10.0)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-ss", f"{start:.3f}", "-i", str(video_path.resolve()),
             "-t", f"{sample:.3f}", "-vf", "cropdetect=24:2:0",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=90,
            creationflags=config.WIN_NO_WINDOW,
        )
    except Exception:
        return None
    boxes = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", proc.stderr or "")
    if not boxes:
        return None
    w, h, x, y = max((tuple(map(int, b)) for b in boxes), key=lambda t: t[0] * t[1])
    if w <= 0 or h <= 0:
        return None
    return w, h, x, y


def _center(w: int, h: int) -> str:
    # min() guards sources that are already vertical.
    return f"crop='min(iw,ih*{w}/{h})':ih,scale={w}:{h},setsar=1"


def _half(w: int, h: int, side: str = "left") -> str:
    # One panel of a two-panel source (camera | chat, or two cameras) taken as
    # a single speaker: crop that half, then 9:16 centre within it. The other
    # panel is dropped, not stacked - the point of left/right vs split.
    x = "iw/2" if side == "right" else "0"
    return (f"crop=iw/2:ih:{x}:0,"
            f"crop='min(iw,ih*{w}/{h})':ih,scale={w}:{h},setsar=1")


def _split(w: int, h: int, swap: bool = False, src: str = "0:v") -> str:
    # Left half goes on top by default. swap puts the right half on top -
    # the user's only control over who is where, since nothing here knows
    # which side a particular person sits on. `src` is the input stream, so
    # a content-cropped source can be fed in place of the raw frame.
    hh = h // 2
    top_x, bot_x = ("iw/2", "0") if swap else ("0", "iw/2")
    return (
        f"[{src}]crop=iw/2:ih:{top_x}:0,scale={w}:{hh}:force_original_aspect_ratio=increase,"
        f"crop={w}:{hh},setsar=1[top];"
        f"[{src}]crop=iw/2:ih:{bot_x}:0,scale={w}:{hh}:force_original_aspect_ratio=increase,"
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

    # Strip any black padding first so the reframe works on the real picture,
    # not on a webcam boxed into a corner. A no-op when the frame is already
    # full (the detected box == the frame), so good sources are untouched.
    src = "0:v"
    box = _detect_content(video_path, start, duration)
    if box:
        cw, ch, cx, cy = box
        parts.append(f"[0:v]crop={cw}:{ch}:{cx}:{cy}[content]")
        src = "content"

    # Build the picture up to a single labelled stream, then treat and round.
    if layout == "split":
        parts.append(_split(w, h, swap, src))
        stream = "stacked"
    elif layout in ("left", "right"):
        parts.append(f"[{src}]{_half(w, h, layout)}[framed]")
        stream = "framed"
    else:
        parts.append(f"[{src}]{_center(w, h)}[framed]")
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
        creationflags=config.WIN_NO_WINDOW,
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
            creationflags=config.WIN_NO_WINDOW,
        )
        return out_path
    except Exception:
        return None
