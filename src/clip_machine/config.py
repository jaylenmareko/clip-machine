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
# small.en is the sweet spot on a CPU-only machine.
WHISPER_MODEL = os.environ.get("CLIP_WHISPER_MODEL", "small.en")
WHISPER_THREADS = int(os.environ.get("CLIP_WHISPER_THREADS", "8"))

# --- clip scoring (local, Ollama; falls back to heuristics if absent) ---
OLLAMA_URL = os.environ.get("CLIP_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("CLIP_OLLAMA_MODEL", "llama3.2:3b")

# --- output ---
CLIP_MIN_SECONDS = float(os.environ.get("CLIP_MIN_SECONDS", "20"))
CLIP_MAX_SECONDS = float(os.environ.get("CLIP_MAX_SECONDS", "90"))
