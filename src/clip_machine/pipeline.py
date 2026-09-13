"""CLI entrypoint: longform video in, vertical clips out.

    python -m clip_machine.pipeline <url-or-path> [--max-clips 6] [--no-llm]
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import (assemble, audio_features, captions as captions_mod, config,
               cut, download, layout as layout_detect, llm, looks, modes,
               score, transcribe)


def run(source: str, max_clips: int | None = None, use_llm: bool = True,
        mode: str = "auto", layout: str = "auto",
        instructions: str | None = None,
        look: str | None = None, radius: int | None = None,
        compile_to: float | None = None,
        captions: bool | None = None,
        swap: bool = False,
        on_clip=None) -> Path:
    """Longform in, vertical clips out.

    `on_clip(record, done, total, video_id)` is called as each clip finishes
    rendering, so a caller can show results while the rest are still being
    made. On a 67-minute source the old behaviour was 25 minutes of blank
    progress bar followed by 15 clips at once; the render stage is the last
    ~6 minutes of that, and there is no reason to sit on it.
    """
    t0 = time.time()

    print(f"[1/6] Fetching video: {source}")
    video_path, video_id = download.get_video(source)
    print(f"      -> {video_path.name}")

    out_dir = config.CLIPS_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if layout == "auto":
        print("[1b/6] Detecting layout from the source...")
        layout = layout_detect.detect(video_path)
    if swap and layout != "split":
        # Only a side-by-side source has a top and bottom to swap.
        print(f"      swap ignored: layout is {layout}, not side by side")

    want_caps = config.CAPTIONS if captions is None else captions
    asr_model = config.transcribe_model(want_caps)
    # Model in the cache key on purpose: a tiny.en transcript reused for a
    # captioned run would put tiny.en's errors on screen, which is the whole
    # thing the upgrade exists to avoid.
    words_cache = config.WORK_DIR / f"{video_id}.{asr_model}.words.json"
    legacy_cache = config.WORK_DIR / f"{video_id}.words.json"
    if (not words_cache.exists() and legacy_cache.exists()
            and asr_model == "tiny.en"):
        words_cache = legacy_cache      # pre-existing caches were all tiny.en

    # Tell the user how long this will take before they start waiting. Rates
    # are the ones measured on this machine: transcription ~8x realtime with
    # batched tiny.en, energy ~107x, render ~0.24s per second of output.
    source_secs = layout_detect._duration(video_path)
    if source_secs > 0:
        eta = source_secs / 107.0 + 40.0          # energy + rank/title calls
        if not words_cache.exists():
            eta += source_secs / 8.0              # transcription, the slow part
        # Blank count still estimates from source length: that is roughly
        # how many survive the quality floor, whereas the cap of 40 is not.
        likely = config.auto_clip_count(source_secs)
        if max_clips:
            likely = min(max_clips, likely)
        eta += likely * 70 * 0.24                      # rendering
        print(f"      -> {source_secs/60:.0f} min of source; "
              f"about {max(1, round(eta / 60))} min to process")

    print(f"[2/6] Transcribing locally (faster-whisper {asr_model}"
          f"{', captions on' if want_caps else ''}) — "
          f"this is the slow part...")
    if words_cache.exists():
        words = json.loads(words_cache.read_text(encoding="utf-8-sig"))
        print(f"      -> reused cached transcript ({len(words)} words)")
    else:
        words = transcribe.transcribe(video_path, model=asr_model)
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

    # A number the user typed is a ceiling, clamped to what we allow. No
    # number means "as many as we can": ask for the cap and let the quality
    # floor and the no-overlap rule decide how many actually ship, rather
    # than guessing a count from the source length.
    asked = max_clips
    if compile_to:
        if max_clips is None:
            # Need enough ranked material to fill the target, with slack for
            # whatever the quality floor drops.
            max_clips = max(config.auto_clip_count(float(words[-1]["end"])),
                            int(compile_to / 25) + 4)
    elif max_clips is None:
        max_clips = config.CLIPS_MAX
    max_clips = max(1, min(int(max_clips), config.CLIPS_MAX))
    print(f"      -> pulling up to {max_clips} clips "
          f"({words[-1]['end']/60:.0f} min of source)")

    print(f"[4/6] Scoring clip candidates{' (+ Claude rerank)' if use_llm else ''}...")
    clips = score.find_clips(words, max_clips=max_clips, use_llm=use_llm,
                             mode_key=mode, energy=energy,
                             instructions=instructions)
    print(f"      -> {len(clips)} clips selected")
    if asked and len(clips) < min(asked, config.CLIPS_MAX):
        # "!" surfaces this on the page. Without it, 14 clips from a request
        # for 20 looks like the number box was ignored.
        print(f"! You asked for {asked} clips. This video had {len(clips)} "
              f"good moments, so you're getting {len(clips)}.")

    if compile_to:
        segments = assemble.choose(clips, compile_to)
        total = sum(c["duration"] for c in segments)
        mins, secs = divmod(int(round(total)), 60)
        print(f"[5/6] Assembling ONE video: {len(segments)} segments, "
              f"{mins}:{secs:02d}...")
        out = out_dir / f"compiled_{mins}m{secs:02d}s.mp4"
        assemble.build(video_path, segments, out, layout=layout, look=look,
                       radius=radius, work_dir=config.WORK_DIR, swap=swap)
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

    total = len(clips)
    finished = 0
    manifest_lock = threading.Lock()

    def publish(rec: dict) -> None:
        """Hand a finished clip to the caller and persist progress.

        The manifest is rewritten after every clip, not once at the end, so
        a run killed partway through (OOM has done it six times on this
        machine) leaves the clips it did finish usable instead of a
        directory of orphaned mp4s with no titles.
        """
        nonlocal finished
        with manifest_lock:
            done_records.append(rec)
            finished += 1
            ordered = sorted(done_records, key=lambda r: r["file"])
            (out_dir / "manifest.json").write_text(
                json.dumps({"source": source, "video_id": video_id,
                            "mode": mode, "partial": finished < total,
                            "clips": ordered}, indent=2),
                encoding="utf-8")
            if on_clip:
                on_clip(rec, finished, total, video_id)

    done_records: list[dict] = []

    def render_one(n: int, c: dict) -> dict:
        clip_path = out_dir / f"clip_{n:02d}.mp4"
        cap_file = None
        if want_caps:
            cap_file = captions_mod.build_ass(
                words, c["start"], c["end"],
                config.WORK_DIR / f"{video_id}.clip{n:02d}.ass",
                (config.OUT_WIDTH, config.OUT_HEIGHT))
        cut.render_clip(video_path, c["start"], c["end"], clip_path,
                        layout=layout, look=look, radius=radius,
                        caption_file=cap_file, swap=swap)
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

    def render_and_publish(n: int, c: dict) -> dict:
        rec = render_one(n, c)
        publish(rec)
        return rec

    # ffmpeg already threads internally, so this is a modest win rather than
    # linear — measured below. Kept low by default because RAM is the binding
    # constraint on this machine, not cores.
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(render_and_publish, n, c): n
                       for n, c in enumerate(clips, 1)}
            done = {}
            for f in as_completed(futures):
                done[futures[f]] = f.result()
        manifest = [done[n] for n in sorted(done)]
    else:
        manifest = [render_and_publish(n, c) for n, c in enumerate(clips, 1)]

    print("[6/6] Writing manifest...")
    # Final write: same content, minus the `partial` flag.
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
                   help="most clips to make (1-40); default is as many "
                        "good ones as the video has")
    p.add_argument("--no-llm", action="store_true",
                   help="skip Claude reranking, use heuristics only")
    p.add_argument("--compile", default=None, dest="compile_to",
                   help="assemble ONE video of this length, e.g. 2:50")
    p.add_argument("--radius", type=int, default=None,
                   help=f"rounded corner radius in px, 0 = off "
                        f"(default {config.CORNER_RADIUS})")
    p.add_argument("--look", default=None, choices=list(looks.LOOKS),
                   help=f"picture treatment (default {config.LOOK})")
    p.add_argument("--captions", action="store_true",
                   help="burn in captions (forces a better ASR model)")
    p.add_argument("--say", default=None, dest="instructions",
                   help="plain-English steer, e.g. \"only the Gaza parts\"")
    p.add_argument("--mode", default="auto", choices=["auto"] + list(modes.MODES),
                   help="auto hunts every kind at once (default); name one to force it")
    p.add_argument("--layout", default="auto",
                   choices=["auto"] + list(cut.LAYOUTS),
                   help="auto detects a side-by-side source (default)")
    p.add_argument("--swap", action="store_true",
                   help="side-by-side sources: put the right person on top")
    args = p.parse_args()
    run(args.source, max_clips=args.max_clips, use_llm=not args.no_llm,
        mode=args.mode, layout=args.layout,
        instructions=args.instructions, look=args.look,
        radius=args.radius,
        compile_to=assemble.parse_target(args.compile_to),
        captions=True if args.captions else None, swap=args.swap)


if __name__ == "__main__":
    main()
