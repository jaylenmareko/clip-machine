"""Config for clip-machine. Everything runs locally — no API keys required.

Optional .env (project-local, or ~/.config/clip-machine/.env) can override
model choices. Nothing here needs a paid service.
"""
import os
from pathlib import Path

# Secrets live OUTSIDE the repo and outside OneDrive - the school tenant syncs
# .env files, so nothing sensitive goes in the project tree.
_ENV_FILE = Path.home() / ".config" / "clip-machine" / ".env"


def _load_env_file() -> None:
    """Minimal .env loader. Real environment variables always win."""
    if not _ENV_FILE.exists():
        return
    try:
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
    except Exception:
        pass


_load_env_file()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = PROJECT_ROOT / "artifacts" / "clips"
WORK_DIR = PROJECT_ROOT / "artifacts" / "_work"

# --- transcription (local, faster-whisper) ---
# tiny.en / base.en / small.en / medium.en — bigger = better + slower.
# tiny.en. There are no captions, so the transcript is an INDEX, not output:
# nobody ever reads these words. Transcription errors ("designists" for
# Zionists) never reach a viewer. The model only has to be good enough to
# locate moments and match scoring phrases.
WHISPER_MODEL = os.environ.get("CLIP_WHISPER_MODEL", "tiny.en")
WHISPER_THREADS = int(os.environ.get("CLIP_WHISPER_THREADS", "8"))
# Batched inference: ~2.5x faster than sequential on CPU, same word
# timestamps. Set to 1 to disable. Larger batches use more RAM, which is
# the binding constraint on this machine.
BATCH_SIZE = int(os.environ.get("CLIP_BATCH_SIZE", "8"))

# --- clip scoring (local, Ollama; falls back to heuristics if absent) ---
OLLAMA_URL = os.environ.get("CLIP_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("CLIP_OLLAMA_MODEL", "llama3.2:3b")

# --- picture treatment ---
# Both OFF by default. The clips ship untreated; the looks are available when
# a particular video wants one, per Jaylen. Neither is a house style yet.
#
# LOOK: none | nostalgia-light | nostalgia | nostalgia-strong  (see looks.py)
#   CLIP_LOOK=nostalgia   or  --look nostalgia
LOOK = os.environ.get("CLIP_LOOK", "none")
# CORNER_RADIUS: rounded corners in pixels at the output size, 0 disables.
#   CLIP_CORNER_RADIUS=48  or  --radius 48
# Untested on a real upload - on a full-screen feed these render as black
# corners, which reads deliberate on a dark UI and broken on a light one.
CORNER_RADIUS = int(os.environ.get("CLIP_CORNER_RADIUS", "0"))

# --- how many clips ---
# OpusClip produces 15-30 clips from a long video (one reported 25 from a
# 30-minute upload), and critically it does NOT charge per clip - it bills
# source length, so there is no reason to be stingy. A fixed 6 was leaving
# most of a 2-hour stream on the floor.
#
# So the count scales with the source: roughly one clip per CLIP_MINUTES_PER
# minutes, floored and capped. A 15-min video gives 8; a 2-hour stream gives
# 40. Each clip costs render time (~0.24s per second of output at 720p) and
# nothing else.
CLIPS_MIN = int(os.environ.get("CLIP_COUNT_MIN", "8"))
CLIPS_MAX = int(os.environ.get("CLIP_COUNT_MAX", "40"))
MINUTES_PER_CLIP = float(os.environ.get("CLIP_MINUTES_PER", "3"))


def auto_clip_count(source_seconds: float) -> int:
    """How many clips to pull from a source of this length."""
    n = int(source_seconds / 60.0 / MINUTES_PER_CLIP)
    return max(CLIPS_MIN, min(CLIPS_MAX, n))


# --- output ---
# 720x1280, not 1080x1920. Measured: 1080 doubles render time (14.1s vs
# 7.1s per 30s of split-auto) and buys nothing when the source is ~480p,
# since each panel is upscaled either way.
OUT_WIDTH = int(os.environ.get("CLIP_OUT_WIDTH", "720"))
OUT_HEIGHT = int(os.environ.get("CLIP_OUT_HEIGHT", "1280"))

# Length: these are OUTER BOUNDS, not a target. Measured 29 of the
# reference channel's published shorts: 20s to 135s, stdev 29 — they run a
# clip for as long as its point takes. The 20s floor is exactly their
# shortest published short. score.LENGTH_COST_PER_SEC handles the rest by
# making each extra second pay for itself; there is no preferred duration.
CLIP_MIN_SECONDS = float(os.environ.get("CLIP_MIN_SECONDS", "20"))
CLIP_MAX_SECONDS = float(os.environ.get("CLIP_MAX_SECONDS", "150"))
