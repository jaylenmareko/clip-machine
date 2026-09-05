"""Cut + 9:16 crop + caption burn — one ffmpeg pass, one encode.

Two passes would double encode time, which matters on a CPU-only machine.
"""
import subprocess
from pathlib import Path

# Center crop to 9:16, then normalize to 1080x1920. min() guards sources
# that are already vertical (cropping wider than the frame would fail).
_VF = "crop='min(iw,ih*9/16)':ih,scale=1080:1920,setsar=1"


def render_clip(
    video_path: Path,
    start: float,
    end: float,
    ass_path: Path,
    out_path: Path,
) -> Path:
    duration = end - start
    # ffmpeg's subtitles filter chokes on Windows drive-letter paths, so run
    # from the .ass file's own directory and reference it by bare filename.
    vf = f"{_VF},subtitles={ass_path.name}"

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path.resolve()),
            "-t", f"{duration:.3f}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path.resolve()),
        ],
        cwd=str(ass_path.parent),
        check=True,
        capture_output=True,
    )
    return out_path
