"""CLI entrypoint: longform video in, vertical captioned clips out.

    python -m clip_machine.pipeline <url-or-path> [--max-clips 6] [--no-llm]
"""
import argparse
import json
import time
from pathlib import Path

from . import (audio_features, caption, config, cut, download, modes, score,
               speakers, transcribe)


def run(source: str, max_clips: int = 6, use_llm: bool = True,
        mode: str = modes.DEFAULT, style: str = caption.DEFAULT_STYLE,
        layout: str = "center") -> Path:
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

    print(f"[3/6] Reading audio energy (mode: {modes.get(mode).label})...")
    energy = audio_features.EnergyProfile(
        audio_features.energy_timeline(transcribe.extract_audio(video_path))
    )
    print(f"      -> {'energy profile ready' if energy.ok else 'no usable audio signal'}")

    print(f"[4/6] Scoring clip candidates{' (+ local LLM rerank)' if use_llm else ''}...")
    clips = score.find_clips(words, max_clips=max_clips, use_llm=use_llm,
                             mode_key=mode, energy=energy)
    print(f"      -> {len(clips)} clips selected")

    print("[5/6] Rendering vertical clips with burned-in captions...")
    manifest = []
    for n, c in enumerate(clips, 1):
        is_split = layout in ("split", "split-auto")
        # Captions straddle the seam on a split, sit in the lower third otherwise.
        ass_path = caption.build_ass(
            words, c["start"], c["end"], out_dir / f"clip_{n:02d}.ass",
            style=style, position="seam" if is_split else "lower",
        )

        swaps = None
        if layout == "split-auto":
            segs = speakers.active_speaker_segments(video_path, c["start"], c["end"])
            swaps = speakers.swap_intervals(segs, c["start"])
            c["speaker_switches"] = max(len(segs) - 1, 0)

        clip_path = out_dir / f"clip_{n:02d}.mp4"
        cut.render_clip(video_path, c["start"], c["end"], ass_path, clip_path,
                        layout=layout, swap_intervals=swaps)
        ass_path.unlink(missing_ok=True)

        print(f"      -> clip_{n:02d}.mp4  "
              f"({c['duration']:.0f}s @ {c['start']:.0f}s)  "
              f"[{','.join(c.get('signals', [])) or 'no signals'}]  "
              f"\"{c['hook'][:44]}\"")
        manifest.append({
            "file": clip_path.name,
            "start": round(c["start"], 2),
            "end": round(c["end"], 2),
            "duration": round(c["duration"], 2),
            "score": round(c["score"], 2),
            "hook": c["hook"],
            "signals": c.get("signals", []),
            "base": c.get("base"),
            "bonus": c.get("bonus"),
            "mode": c.get("mode", mode),
            "speaker_switches": c.get("speaker_switches"),
            "transcript": c["text"],
        })

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
    p.add_argument("--max-clips", type=int, default=6)
    p.add_argument("--no-llm", action="store_true",
                   help="skip the Ollama rerank, use heuristics only")
    p.add_argument("--mode", default=modes.DEFAULT, choices=list(modes.MODES),
                   help="what kind of moment to hunt for")
    p.add_argument("--style", default=caption.DEFAULT_STYLE, choices=list(caption.STYLES),
                   help="caption look")
    p.add_argument("--layout", default="center", choices=list(cut.LAYOUTS),
                   help="center crop, or 50/50 split-screen for a two-person source")
    args = p.parse_args()
    run(args.source, max_clips=args.max_clips, use_llm=not args.no_llm,
        mode=args.mode, style=args.style, layout=args.layout)


if __name__ == "__main__":
    main()
