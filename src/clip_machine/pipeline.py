"""CLI entrypoint: longform video in, vertical clips out.

    python -m clip_machine.pipeline <url-or-path> [--max-clips 6] [--no-llm]
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import (assemble, audio_features, config, cut, download,
               layout as layout_detect, llm, looks, modes, score,
               transcribe)


def run(source: str, max_clips: int | None = None, use_llm: bool = True,
        mode: str = "auto", layout: str = "auto",
        instructions: str | None = None,
        look: str | None = None, radius: int | None = None,
        compile_to: float | None = None) -> Path:
    t0 = time.time()

    print(f"[1/6] Fetching video: {source}")
    video_path, video_id = download.get_video(source)
    print(f"      -> {video_path.name}")

    out_dir = config.CLIPS_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if layout == "auto":
        print("[1b/6] Detecting layout from the source...")
        layout = layout_detect.detect(video_path)

    words_cache = config.WORK_DIR / f"{video_id}.words.json"

    # Tell the user how long this will take before they start waiting. Rates
    # are the ones measured on this machine: transcription ~8x realtime with
    # batched tiny.en, energy ~107x, render ~0.24s per second of output.
    source_secs = layout_detect._duration(video_path)
    if source_secs > 0:
        eta = source_secs / 107.0 + 40.0          # energy + rank/title calls
        if not words_cache.exists():
            eta += source_secs / 8.0              # transcription, the slow part
        eta += config.auto_clip_count(source_secs) * 70 * 0.24   # rendering
        print(f"      -> {source_secs/60:.0f} min of source; "
              f"about {max(1, round(eta / 60))} min to process")

    print(f"[2/6] Transcribing locally (faster-whisper {config.WHISPER_MODEL}) — "
          f"this is the slow part...")
    if words_cache.exists():
        words = json.loads(words_cache.read_text(encoding="utf-8-sig"))
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
        if compile_to:
            # Need enough ranked material to fill the target, with slack for
            # whatever the quality floor drops.
            max_clips = max(max_clips, int(compile_to / 25) + 4)
        print(f"      -> pulling up to {max_clips} clips "
              f"({words[-1]['end']/60:.0f} min of source)")

    print(f"[4/6] Scoring clip candidates{' (+ Claude rerank)' if use_llm else ''}...")
    clips = score.find_clips(words, max_clips=max_clips, use_llm=use_llm,
                             mode_key=mode, energy=energy,
                             instructions=instructions)
    print(f"      -> {len(clips)} clips selected")

    if compile_to:
        segments = assemble.choose(clips, compile_to)
        total = sum(c["duration"] for c in segments)
        mins, secs = divmod(int(round(total)), 60)
        print(f"[5/6] Assembling ONE video: {len(segments)} segments, "
              f"{mins}:{secs:02d}...")
        out = out_dir / f"compiled_{mins}m{secs:02d}s.mp4"
        assemble.build(video_path, segments, out, layout=layout, look=look,
                       radius=radius, work_dir=config.WORK_DIR)
        thumb = cut.make_thumbnail(out, out.with_suffix(".jpg"))

        titled = llm.title_clips([{"duration": total,
                                   "text": " ".join(c["text"] for c in segments)}],
                                 instructions=instructions)
        titles, descs = titled if titled else ({}, {})
        manifest = [{
            "file": out.name,
            "thumb": thumb.name if thumb else None,
            "start": round(segments[0]["start"], 2) if segments else 0,
            "end": round(segments[-1]["end"], 2) if segments else 0,
            "duration": round(total, 2),
            "score": round(max((c["score"] for c in segments), default=0), 2),
            "title": titles.get(0) or (segments[0]["hook"] if segments else ""),
            "description": descs.get(0, ""),
            "hook": segments[0]["hook"] if segments else "",
            "signals": sorted({s for c in segments for s in c.get("signals", [])}),
            "mode": "compilation",
            "segments": [{"start": round(c["start"], 2),
                          "end": round(c["end"], 2),
                          "duration": round(c["duration"], 2)}
                         for c in segments],
            "transcript": " ".join(c["text"] for c in segments),
        }]
        print(f"      -> {out.name}  ({total:.0f}s from {len(segments)} segments)")
        print("[6/6] Writing manifest...")
        (out_dir / "manifest.json").write_text(
            json.dumps({"source": source, "video_id": video_id,
                        "mode": "compilation", "clips": manifest}, indent=2),
            encoding="utf-8")
        print(f"\nDone in {time.time() - t0:.0f}s — {out_dir}")
        return out_dir

    # House-style titles for the final picks; falls back to the hook line.
    titled = llm.title_clips(clips, instructions=instructions)
    titles, descs = titled if titled else ({}, {})
    for i, c in enumerate(clips):
        c["title_text"] = titles.get(i) or c["hook"]
        c["description"] = descs.get(i, "")

    # Default 1. Parallel rendering was measured on this machine at 186s /
    # 212s / 157s for 1 / 2 / 3 workers - non-monotonic, i.e. noise rather
    # than speedup, because libx264 already saturates the CPU and extra
    # ffmpeg processes just contend. Raise CLIP_RENDER_WORKERS on a machine
    # with real headroom; here it only adds OOM risk.
    workers = max(1, int(os.environ.get("CLIP_RENDER_WORKERS", "1")))
    print(f"[5/6] Rendering {config.OUT_WIDTH}x{config.OUT_HEIGHT} clips"
          f" [{look or config.LOOK}]"
          f"{f' ({workers} in parallel)' if workers > 1 else ''}...")

    def render_one(n: int, c: dict) -> dict:
        clip_path = out_dir / f"clip_{n:02d}.mp4"
        cut.render_clip(video_path, c["start"], c["end"], clip_path,
                        layout=layout, look=look, radius=radius)
        thumb = cut.make_thumbnail(clip_path, out_dir / f"clip_{n:02d}.jpg")

        print(f"      -> clip_{n:02d}.mp4  "
              f"({c['duration']:.0f}s @ {c['start']:.0f}s)  "
              f"{c.get('title_text', c['hook'])[:72]}")
        return {
            "file": clip_path.name,
            "thumb": thumb.name if thumb else None,
            "start": round(c["start"], 2),
            "end": round(c["end"], 2),
            "duration": round(c["duration"], 2),
            "score": round(c["score"], 2),
            "title": c.get("title_text", c["hook"]),
            "description": c.get("description", ""),
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
                   help="skip Claude reranking, use heuristics only")
    p.add_argument("--compile", default=None, dest="compile_to",
                   help="assemble ONE video of this length, e.g. 2:50")
    p.add_argument("--radius", type=int, default=None,
                   help=f"rounded corner radius in px, 0 = off "
                        f"(default {config.CORNER_RADIUS})")
    p.add_argument("--look", default=None, choices=list(looks.LOOKS),
                   help=f"picture treatment (default {config.LOOK})")
    p.add_argument("--say", default=None, dest="instructions",
                   help="plain-English steer, e.g. \"only the Gaza parts\"")
    p.add_argument("--mode", default="auto", choices=["auto"] + list(modes.MODES),
                   help="auto hunts every kind at once (default); name one to force it")
    p.add_argument("--layout", default="auto",
                   choices=["auto"] + list(cut.LAYOUTS),
                   help="auto detects a side-by-side source (default)")
    args = p.parse_args()
    run(args.source, max_clips=args.max_clips, use_llm=not args.no_llm,
        mode=args.mode, layout=args.layout,
        instructions=args.instructions, look=args.look,
        radius=args.radius,
        compile_to=assemble.parse_target(args.compile_to))


if __name__ == "__main__":
    main()
