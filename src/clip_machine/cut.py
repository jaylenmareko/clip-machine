"""Cut + reframe + caption burn — one ffmpeg pass, one encode.

Two passes would double encode time, which matters on a CPU-only machine.

Layouts (measured off the reference channel):
  center — single speaker, 9:16 centre crop of the source
  split  — 50/50 stacked panels, seam at y=960. Takes the LEFT half of a
           wide source into the top panel and the RIGHT half into the
           bottom, which is what a two-person podcast recording usually
           needs. Captions then straddle the seam.
"""
import subprocess
from pathlib import Path

# Centre crop to 9:16, then normalize to 1080x1920. min() guards sources
# that are already vertical (cropping wider than the frame would fail).
_VF_CENTER = "crop='min(iw,ih*9/16)':ih,scale=1080:1920,setsar=1"

# Each source half becomes a 1080x960 panel; stack them.
_VF_SPLIT = (
    "[0:v]crop=iw/2:ih:0:0,scale=1080:960:force_original_aspect_ratio=increase,"
    "crop=1080:960,setsar=1[top];"
    "[0:v]crop=iw/2:ih:iw/2:0,scale=1080:960:force_original_aspect_ratio=increase,"
    "crop=1080:960,setsar=1[bot];"
    "[top][bot]vstack=inputs=2[stacked]"
)

LAYOUTS = ("center", "split")


def render_clip(
    video_path: Path,
    start: float,
    end: float,
    ass_path: Path,
    out_path: Path,
    layout: str = "center",
) -> Path:
    duration = end - start
    # ffmpeg's subtitles filter chokes on Windows drive-letter paths, so run
    # from the .ass file's own directory and reference it by bare filename.
    subs = f"subtitles={ass_path.name}"

    if layout == "split":
        filter_args = ["-filter_complex", f"{_VF_SPLIT};[stacked]{subs}[v]",
                       "-map", "[v]", "-map", "0:a?"]
    else:
        filter_args = ["-vf", f"{_VF_CENTER},{subs}"]

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
        cwd=str(ass_path.parent),
        check=True,
        capture_output=True,
    )
    return out_path
