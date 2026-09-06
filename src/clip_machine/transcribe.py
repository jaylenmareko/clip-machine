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


def free_ram_gb() -> float | None:
    """Free physical RAM in GB, or None if it can't be determined."""
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullAvailPhys / (1024 ** 3)
    except Exception:
        return None


# Rough working-set needs, measured on this machine.
_MODEL_RAM_GB = {"tiny.en": 0.6, "base.en": 1.0, "small.en": 2.2, "medium.en": 5.0}


def preflight() -> None:
    """Warn before a long run gets OOM-killed partway through."""
    free = free_ram_gb()
    if free is None:
        return
    need = _MODEL_RAM_GB.get(config.WHISPER_MODEL, 2.0)
    if free < need + 0.5:
        print(f"  ! Only {free:.1f}GB RAM free; {config.WHISPER_MODEL} wants "
              f"~{need}GB. Close OneDrive/browser tabs, or set "
              f"CLIP_WHISPER_MODEL=tiny.en, or this may get killed mid-run.")


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
    preflight()
    model = _get_model()

    # Batched inference is ~2.5x faster than sequential on this CPU (measured
    # same-session on 900s: 153.0s -> 60.3s) and still returns word-level
    # timestamps, which the scorer needs for cut points.
    #
    # It transcribes ~4% fewer words, which does NOT matter now that captions
    # are off: the transcript is an index, not output. Verified by comparing
    # selection on both transcripts - the top 3 moments were identical
    # (including the HYPOCRISY-TRAP clip), all picks carried viral signals in
    # both, and batched actually produced MORE sentences (222 vs 140), giving
    # the scorer finer boundaries to cut on.
    vad = {"min_silence_duration_ms": 500}
    segments = None
    if config.BATCH_SIZE > 1:
        try:
            from faster_whisper import BatchedInferencePipeline
            pipeline = BatchedInferencePipeline(model=model)
            segments, _ = pipeline.transcribe(
                str(audio_path), batch_size=config.BATCH_SIZE,
                word_timestamps=True, vad_parameters=vad,
            )
        except Exception as e:
            print(f"  (batched inference unavailable, falling back: {type(e).__name__})")
            segments = None

    if segments is None:
        segments, _ = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,  # skips silence — real speedup on longform
            vad_parameters=vad,
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
