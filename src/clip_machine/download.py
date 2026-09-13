"""Get a local video file from a URL (yt-dlp) or pass a local path through."""
import re
import subprocess
from pathlib import Path

from . import config


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "-", text.strip().lower())
    return re.sub(r"-+", "-", text).strip("-")[:60] or "video"


def get_video(source: str) -> tuple[Path, str]:
    """Return (local_mp4_path, video_id). Downloads if `source` looks like a URL."""
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)

    if source.startswith("http://") or source.startswith("https://"):
        # Ask yt-dlp for the id first so we know the output path deterministically.
        video_id = subprocess.run(
            ["yt-dlp", "--print", "%(id)s", "--no-warnings", source],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        video_id = _slugify(video_id)
        out_path = config.WORK_DIR / f"{video_id}.mp4"
        if not out_path.exists():
            # Capped at 720p. Output is 720 wide, so anything taller is
            # downloaded only to be thrown away - and a past livestream's
            # best format is a 1080p "premium" HLS stream: 3h measured at
            # 7.5GB, fetched one fragment at a time, ~50 min of download
            # alone. At 720p the same 3h is ~2.1GB. Single-file https is
            # preferred over HLS fragments, and -N parallelises what's left.
            subprocess.run(
                [
                    "yt-dlp",
                    "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]"
                          "/b[height<=720][ext=mp4]/b[height<=720]/b",
                    # h264 first: unsorted, this picked AV1, which decodes
                    # far slower in ffmpeg on CPU and would slow every render.
                    "-S", "vcodec:h264,proto:https",
                    "-N", "8",
                    "--merge-output-format", "mp4",
                    "-o", str(out_path),
                    "--no-warnings",
                    source,
                ],
                check=True,
            )
        return out_path, video_id

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    return path, _slugify(path.stem)
