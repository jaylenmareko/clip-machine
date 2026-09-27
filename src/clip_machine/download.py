"""Get a local video file from a URL (yt-dlp) or pass a local path through."""
import re
import subprocess
from pathlib import Path

from . import config


def _slugify(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "-", text.strip().lower())
    return re.sub(r"-+", "-", text).strip("-")[:60] or "video"


def get_video(source: str) -> tuple[Path, str, str, str | None]:
    """Return (local_mp4_path, video_id, title, thumbnail_url).

    The title and thumbnail exist so a project folder can show what the
    video actually IS ("Interviewing Rabbi Weiss") instead of the URL it
    came from - `nwzvdzzcu5w` and a 90-character query string told a user
    nothing about which of five projects was which.
    """
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)

    if source.startswith("http://") or source.startswith("https://"):
        # One call for id + title + thumbnail - not three - so a slow
        # network only costs this once. Tab-separated on one line: a real
        # video title can contain almost anything else printable.
        probe = subprocess.run(
            ["yt-dlp", "--print", "%(id)s	%(title)s	%(thumbnail)s",
             "--no-warnings", source],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        raw_id, _, rest = probe.partition("	")
        title, _, thumb = rest.partition("	")
        video_id = _slugify(raw_id)
        title = title.strip() or video_id
        thumb = thumb.strip() or None
        if thumb in ("NA", "None"):
            thumb = None
        out_path = config.WORK_DIR / f"{video_id}.mp4"
        if not out_path.exists():
            # Capped at 720p. Output is 720 wide, so anything taller is
            # downloaded only to be thrown away - and a past livestream's
            # best format is a 1080p "premium" HLS stream: 3h measured at
            # 7.5GB, fetched one fragment at a time, ~50 min of download
            # alone. At 720p the same 3h is ~2.1GB. Single-file https is
            # preferred over HLS fragments, and -N parallelises what's left.
            # capture_output so a failure carries yt-dlp's OWN stderr on the
            # exception - without it, a CalledProcessError says only "exit
            # status 1", which cannot tell a private video from a dropped
            # connection apart, and that ambiguity is what put "make sure
            # the link is public" on a run that failed from a lost Wi-Fi
            # connection mid-download.
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
                check=True, capture_output=True, text=True,
            )
        return out_path, video_id, title, thumb

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    # A local file's only title IS its filename - prettify the slug rather
    # than show "z0jx2l3u2ew" for something the user picked off their disk.
    pretty = re.sub(r"[_\-]+", " ", path.stem).strip().title() or path.stem
    return path, _slugify(path.stem), pretty, None
