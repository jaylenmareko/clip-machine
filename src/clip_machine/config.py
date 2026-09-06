"""Config for clip-machine. Everything runs locally — no API keys required.

Optional .env (project-local, or ~/.config/clip-machine/.env) can override
model choices. Nothing here needs a paid service.
"""
import os
from pathlib import Path

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
