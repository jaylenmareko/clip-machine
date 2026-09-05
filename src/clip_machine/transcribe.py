"""Local word-level transcription via faster-whisper. No API, no network.

Model files download once from HuggingFace on first run, then live in the
local HF cache and never touch the network again.
"""
import subprocess
from pathlib import Path

from . import config

_model = None


def extract_audio(video_path: Path) -> Path:
    """16kHz mono wav — what Whisper wants, no resampling cost at inference."""
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = config.WORK_DIR / f"{video_path.stem}.audio.wav"
    if not audio_path.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vn", "-ac", "1", "-ar", "16000", str(audio_path)],
            check=True, capture_output=True,
        )
    return audio_path


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        # int8 on CPU is the only sane config on a GPU-less ultrabook.
        _model = WhisperModel(
            config.WHISPER_MODEL, device="cpu", compute_type="int8",
            cpu_threads=config.WHISPER_THREADS,
        )
    return _model


def transcribe(video_path: Path, verbose: bool = True) -> list[dict]:
    """Return a flat list of {word, start, end} for the whole video."""
    audio_path = extract_audio(video_path)
    model = _get_model()

    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,  # skips silence — real speedup on longform
        vad_parameters={"min_silence_duration_ms": 500},
    )

    words: list[dict] = []
    for seg in segments:
        if verbose:
            print(f"  [{seg.start:7.1f}s] {seg.text.strip()[:70]}")
        for w in (seg.words or []):
            words.append({
                "word": w.word.strip(),
                "start": float(w.start),
                "end": float(w.end),
            })
    return words
