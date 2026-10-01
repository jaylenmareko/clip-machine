"""Local word-level transcription via faster-whisper. No API, no network.

Model files download once from HuggingFace on first run, then live in the
local HF cache and never touch the network again.
"""
import subprocess
from pathlib import Path

from . import config

_model = None
_model_name = None


def extract_audio(video_path: Path) -> Path:
    """16kHz mono wav — what Whisper wants, no resampling cost at inference."""
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = config.WORK_DIR / f"{video_path.stem}.audio.wav"
    if not audio_path.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vn", "-ac", "1", "-ar", "16000", str(audio_path)],
            check=True, capture_output=True,
            creationflags=config.WIN_NO_WINDOW,
        )
    return audio_path


def _mem_status():
    """Raw GlobalMemoryStatusEx, or None."""
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
        return stat
    except Exception:
        return None


def free_ram_gb() -> float | None:
    """Available physical RAM in GB, or None if it can't be determined."""
    stat = _mem_status()
    return None if stat is None else stat.ullAvailPhys / (1024 ** 3)


def commit_ratio() -> float | None:
    """Commit charge as a multiple of physical RAM.

    Above 1.0 the OS has promised more memory than it has and is paging to
    disk to keep the promise. This turned out to be the thing actually
    capping transcription speed on this machine: a 67-min stream measured
    5.6x realtime with the CPU at only 71%, and a 10-min slice of the SAME
    audio later measured 3.59x - slower on a shorter job, which only makes
    sense if the limit is memory, not compute. Commit was 1.94x physical
    (30.6GB promised against 15.8GB, 346 processes) and the standby cache
    was empty, so whisper's weights were being paged in and out rather than
    staying resident.

    Adding parallel workers to a machine in this state makes it worse, not
    faster - which is why split_asr stays off until there is real headroom.
    """
    stat = _mem_status()
    if stat is None or not stat.ullTotalPhys:
        return None
    charge = stat.ullTotalPageFile - stat.ullAvailPageFile
    return charge / stat.ullTotalPhys


# Rough working-set needs, measured on this machine.
_MODEL_RAM_GB = {"tiny.en": 0.6, "base.en": 1.0, "small.en": 2.2, "medium.en": 5.0}


def preflight(model: str | None = None) -> None:
    """Warn before a long run gets OOM-killed or paged into the ground."""
    free = free_ram_gb()
    if free is None:
        return
    model = model or config.WHISPER_MODEL
    need = _MODEL_RAM_GB.get(model, 2.0)
    if free < need + 0.5:
        print(f"  ! Only {free:.1f}GB RAM free; {model} wants "
              f"~{need}GB. Close OneDrive/browser tabs, or set "
              f"CLIP_WHISPER_MODEL=tiny.en, or this may get killed mid-run.")

    # Slow is not the same failure as dead, and it used to be invisible: the
    # run completes, just at half speed, and the only symptom is a wait that
    # looks like a regression. Say it out loud instead.
    ratio = commit_ratio()
    if ratio and ratio > 1.25:
        print(f"  ! Memory oversubscribed: {ratio:.1f}x physical RAM "
              f"committed, {free:.1f}GB free. Windows is paging, so expect "
              f"roughly half speed. Closing browser windows / OneDrive is "
              f"worth more here than any setting.")


def _get_model(name: str | None = None):
    """Load (and cache) a model. Reloads if a DIFFERENT model is asked for -
    otherwise turning captions on mid-session would keep using tiny.en."""
    global _model, _model_name
    name = name or config.WHISPER_MODEL
    if _model is None or _model_name != name:
        _model_name = name
        from faster_whisper import WhisperModel
        # int8 on CPU is the only sane config on a GPU-less ultrabook.
        _model = WhisperModel(
            name, device="cpu", compute_type="int8",
            cpu_threads=config.WHISPER_THREADS,
        )
    return _model


def unload_model() -> None:
    """Drop the cached Whisper model so its RAM goes back to the OS.

    A finished job leaves 0.6-5GB resident in `_model` for no reason - the
    server is idle between runs, and this laptop is already RAM-oversub-
    scribed (reference_machine_ram_oversubscribed.md), so that idle memory
    is exactly what has gotten the server itself killed after a run. The
    next run reloads it in ~1s from the local HF cache, so there is no real
    cost to dropping it.
    """
    global _model, _model_name
    _model = None
    _model_name = None
    import gc
    gc.collect()


def transcribe(video_path: Path, verbose: bool = True,
               model: str | None = None) -> list[dict]:
    """Return a flat list of {word, start, end} for the whole video."""
    audio_path = extract_audio(video_path)
    model = model or config.WHISPER_MODEL
    preflight(model)

    # Chunked parallel path, when asked for and when RAM allows. Returns None
    # for short sources, where per-worker model loading costs more than it
    # saves, and the single-process path below handles those.
    if config.ASR_WORKERS > 1:
        import os as _os

        from . import split_asr
        n = split_asr.plan_workers(
            config.ASR_WORKERS, free_ram_gb(),
            _MODEL_RAM_GB.get(model, 2.0), _os.cpu_count() or 4)
        if n < config.ASR_WORKERS and verbose:
            print(f"      (asking for {config.ASR_WORKERS} ASR workers, "
                  f"only {n} fit in free RAM)")
        words = split_asr.transcribe_parallel(
            audio_path, config.WORK_DIR, model,
            workers=n, threads=config.WHISPER_THREADS,
            batch=config.BATCH_SIZE, verbose=verbose)
        if words is not None:
            return words

    whisper = _get_model(model)

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
            pipeline = BatchedInferencePipeline(model=whisper)
            segments, _ = pipeline.transcribe(
                str(audio_path), batch_size=config.BATCH_SIZE,
                word_timestamps=True, vad_parameters=vad,
            )
        except Exception as e:
            print(f"  (batched inference unavailable, falling back: {type(e).__name__})")
            segments = None

    if segments is None:
        segments, _ = whisper.transcribe(
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
