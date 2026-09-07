"""CLI entrypoint: longform video in, vertical clips out.

    python -m clip_machine.pipeline <url-or-path> [--max-clips 6] [--no-llm]
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import audio_features, config, cut, download, llm, modes, score, transcribe


def run(source: str, max_clips: int | None = None, use_llm: bool = True,
        mode: str = "auto", layout: str = "center") -> Path:
    t0 = time.time()

    print(f"[1/6] Fetching video: {source}")
    video_path, video_id = download.get_video(source)
    print(f"      -> {video_path.name}")

    out_dir = config.CLIPS_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/6] Transcribing locally (faster-whisper {config.WHISPER_MODEL}) — "
          f"this is the slow part...")
    words_cache = config.WORK_DIR / f"{video_id}.words.json"
    if words_cache.exists():
        words = json.loads(words_cache.read_text(encoding="utf-8"))
        print(f"      -> reused cached transcript ({len(words)} words)")
    else:
        words = transcribe.transcribe(video_path)
        words_cache.write_text(json.dumps(words), encoding="utf-8")
        print(f"      -> {len(words)} words in {time.time() - t0:.0f}s")

    if not words:
        raise SystemExit("No speech found in that video.")

    label = "all types" if mode == "auto" else modes.get(mode).label
    print(f"[3/6] Reading audio energy (hunting: {label})...")
    energy = audio_features.EnergyProfile(
        audio_features.energy_timeline(transcribe.extract_audio(video_path))
    )
    print(f"      -> {'energy profile ready' if energy.ok else 'no usable audio signal'}")

    if max_clips is None:
        max_clips = config.auto_clip_count(float(words[-1]["end"]))
        print(f"      -> pulling up to {max_clips} clips "
              f"({words[-1]['end']/60:.0f} min of source)")

    print(f"[4/6] Scoring clip candidates{' (+ local LLM rerank)' if use_llm else ''}...")
    clips = score.find_clips(words, max_clips=max_clips, use_llm=use_llm,
                             mode_key=mode, energy=energy)
    print(f"      -> {len(clips)} clips selected")

    # Default 1. Parallel rendering was measured on this machine at 186s /
    # 212s / 157s for 1 / 2 / 3 workers - non-monotonic, i.e. noise rather
    # than speedup, because libx264 already saturates the CPU and extra
    # ffmpeg processes just contend. Raise CLIP_RENDER_WORKERS on a machine
    # with real headroom; here it only adds OOM risk.
    # House-style titles for the final picks; falls back to the hook line.
    titles = llm.title_clips(clips) or {}
    for i, c in enumerate(clips):
        c["title_text"] = titles.get(i) or c["hook"]

    workers = max(1, int(os.environ.get("CLIP_RENDER_WORKERS", "1")))
    print(f"[5/6] Rendering {config.OUT_WIDTH}x{config.OUT_HEIGHT} clips"
          f"{f' ({workers} in parallel)' if workers > 1 else ''}...")

    def render_one(n: int, c: dict) -> dict:
        clip_path = out_dir / f"clip_{n:02d}.mp4"
        cut.render_clip(video_path, c["start"], c["end"], clip_path, layout=layout)

        print(f"      -> clip_{n:02d}.mp4  "
              f"({c['duration']:.0f}s @ {c['start']:.0f}s)  "
              f"{c.get('title_text', c['hook'])[:72]}")
        return {
            "file": clip_path.name,
            "start": round(c["start"], 2),
            "end": round(c["end"], 2),
            "duration": round(c["duration"], 2),
            "score": round(c["score"], 2),
            "title": c.get("title_text", c["hook"]),
            "hook": c["hook"],
            "signals": c.get("signals", []),
            "base": c.get("base"),
            "bonus": c.get("bonus"),
            "viral": c.get("viral"),
            "mode": c.get("mode", mode),
            "transcript": c["text"],
        }

    # ffmpeg already threads internally, so this is a modest win rather than
    # linear — measured below. Kept low by default because RAM is the binding
    # constraint on this machine, not cores.
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(render_one, n, c): n
                       for n, c in enumerate(clips, 1)}
            done = {}
            for f in as_completed(futures):
                done[futures[f]] = f.result()
        manifest = [done[n] for n in sorted(done)]
    else:
        manifest = [render_one(n, c) for n, c in enumerate(clips, 1)]

    print("[6/6] Writing manifest...")
    (out_dir / "manifest.json").write_text(
        json.dumps({"source": source, "video_id": video_id, "mode": mode,
                    "clips": manifest}, indent=2),
        encoding="utf-8",
    )

    print(f"\nDone in {time.time() - t0:.0f}s — {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Local longform -> vertical clips.")
    p.add_argument("source", help="video URL (yt-dlp) or local file path")
    p.add_argument("--max-clips", type=int, default=None,
                   help="default scales with source length (~1 per 3 min, 8-40)")
    p.add_argument("--no-llm", action="store_true",
                   help="skip the Ollama rerank, use heuristics only")
    p.add_argument("--mode", default="auto", choices=["auto"] + list(modes.MODES),
                   help="auto hunts every kind at once (default); name one to force it")
    p.add_argument("--layout", default="center", choices=list(cut.LAYOUTS),
                   help="center crop, or 50/50 split-screen for a two-person source")
    args = p.parse_args()
    run(args.source, max_clips=args.max_clips, use_llm=not args.no_llm,
        mode=args.mode, layout=args.layout)


if __name__ == "__main__":
    main()
