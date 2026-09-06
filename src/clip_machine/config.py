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
# base.en is the default because this is a 16GB machine that in practice
# runs with ~2GB free (OneDrive + browsers); small.en gets OOM-killed under
# that pressure. Bump to small.en via env when the machine is quiet.
WHISPER_MODEL = os.environ.get("CLIP_WHISPER_MODEL", "base.en")
WHISPER_THREADS = int(os.environ.get("CLIP_WHISPER_THREADS", "8"))

# --- clip scoring (local, Ollama; falls back to heuristics if absent) ---
OLLAMA_URL = os.environ.get("CLIP_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("CLIP_OLLAMA_MODEL", "llama3.2:3b")

# --- output ---
# Length: measured the reference channel's own published shorts at 59-76s,
# so a hard 60s cap is wrong — but so is running to the 3-minute platform
# limit, since completion rate drives distribution. Allow up to 150s and
# let base_quality's length curve make long clips earn it.
CLIP_MIN_SECONDS = float(os.environ.get("CLIP_MIN_SECONDS", "20"))
CLIP_MAX_SECONDS = float(os.environ.get("CLIP_MAX_SECONDS", "150"))
