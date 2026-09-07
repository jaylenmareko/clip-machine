"""Chunked parallel transcription.

Why this exists: on a 67-minute livestream, transcription measured 4.2x
realtime - about 16 minutes - and the CPU sat at 71%, not saturated. This
machine is an i5-1235U: 2 performance cores plus 8 efficiency cores. One
whisper process with cpu_threads=8 spreads a batch across E-cores, and
because the batch only finishes when its slowest thread does, the E-cores
pace the whole thing.

Separate PROCESSES on separate slices of the audio do not share that fate -
each finishes its own work independently and the OS can park them on whatever
cores are free.

The seam cost: chunks are cut on wall-clock boundaries, so a word straddling
a boundary can be lost or clipped. That is acceptable here and nowhere else -
there are no captions, so the transcript is an INDEX used to locate moments,
never output a viewer reads. Two seams on a 67-minute stream is under a
second of audio.

RAM is the binding constraint on this machine, not cores - six OOM kills came
out of ignoring that - so the worker count is clamped by free memory before
anything is launched.
"""
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _duration(audio_path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(audio_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _slice(audio_path: Path, start: float, length: float, out: Path) -> Path:
    """Cut one chunk. -ss before -i seeks without decoding what it skips."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}",
         "-t", f"{length:.3f}", "-i", str(audio_path),
         "-ac", "1", "-ar", "16000", str(out)],
        check=True, capture_output=True,
    )
    return out


def _run_chunk(args: tuple[str, float, str, int, int]) -> str:
    """Transcribe one chunk in a child process. Returns JSON words.

    Each child loads its own model - that is the point, since the win comes
    from independent processes rather than threads inside one. Returns JSON
    rather than objects so nothing unpicklable crosses the boundary.
    """
    chunk_path, offset, model_name, threads, batch = args
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cpu", compute_type="int8",
                         cpu_threads=threads)
    vad = {"min_silence_duration_ms": 500}

    segments = None
    if batch > 1:
        try:
            from faster_whisper import BatchedInferencePipeline
            segments, _ = BatchedInferencePipeline(model=model).transcribe(
                chunk_path, batch_size=batch, word_timestamps=True,
                vad_parameters=vad,
            )
        except Exception:
            segments = None
    if segments is None:
        segments, _ = model.transcribe(
            chunk_path, word_timestamps=True, vad_filter=True,
            vad_parameters=vad,
        )

    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"word": w.word.strip(),
                          "start": float(w.start) + offset,
                          "end": float(w.end) + offset})
    return json.dumps(words)


def plan_workers(requested: int, free_gb: float | None,
                 model_ram_gb: float, cpus: int) -> int:
    """How many workers will actually fit. RAM first, then cores."""
    n = max(1, requested)
    if free_gb is not None:
        # Leave 0.8GB headroom - the parent process, ffmpeg and the OS still
        # need room, and an OOM kill costs the whole run.
        affordable = int(max(0.0, free_gb - 0.8) / model_ram_gb)
        n = min(n, max(1, affordable))
    return max(1, min(n, max(1, cpus // 2)))


def transcribe_parallel(audio_path: Path, work_dir: Path, model_name: str,
                        workers: int, threads: int, batch: int,
                        verbose: bool = True) -> list[dict] | None:
    """Split, transcribe in parallel, stitch. None if it isn't worth it."""
    total = _duration(audio_path)
    # Under ~6 minutes the model-load cost per worker (a few seconds each)
    # eats the win, and chunking adds seams for nothing.
    if workers < 2 or total < 360:
        return None

    work_dir.mkdir(parents=True, exist_ok=True)
    span = total / workers
    per_worker_threads = max(1, threads // workers)

    jobs, chunks = [], []
    for i in range(workers):
        start = i * span
        length = (total - start) if i == workers - 1 else span
        chunk = work_dir / f"{audio_path.stem}.part{i}.wav"
        _slice(audio_path, start, length, chunk)
        chunks.append(chunk)
        jobs.append((str(chunk), start, model_name, per_worker_threads, batch))

    if verbose:
        print(f"      splitting into {workers} chunks of "
              f"{span/60:.0f} min, {per_worker_threads} threads each")

    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_run_chunk, jobs))
    finally:
        for c in chunks:
            c.unlink(missing_ok=True)

    words: list[dict] = []
    for blob in results:
        words.extend(json.loads(blob))
    words.sort(key=lambda w: w["start"])
    return words
