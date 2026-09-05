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
            subprocess.run(
                [
                    "yt-dlp", "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
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
