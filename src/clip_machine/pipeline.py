"""CLI entrypoint: longform video in, vertical captioned clips out.

    python -m clip_machine.pipeline <url-or-path> [--max-clips 6] [--no-llm]
"""
import argparse
import json
import time
from pathlib import Path

from . import caption, config, cut, download, score, transcribe


def run(source: str, max_clips: int = 6, use_llm: bool = True) -> Path:
    t0 = time.time()

    print(f"[1/5] Fetching video: {source}")
    video_path, video_id = download.get_video(source)
    print(f"      -> {video_path.name}")

    out_dir = config.CLIPS_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/5] Transcribing locally (faster-whisper {config.WHISPER_MODEL}) — "
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

    print(f"[3/5] Scoring clip candidates{' (+ local LLM rerank)' if use_llm else ''}...")
    clips = score.find_clips(words, max_clips=max_clips, use_llm=use_llm)
    print(f"      -> {len(clips)} clips selected")

    print("[4/5] Rendering vertical clips with burned-in captions...")
    manifest = []
    for n, c in enumerate(clips, 1):
        ass_path = caption.build_ass(
            words, c["start"], c["end"], out_dir / f"clip_{n:02d}.ass"
        )
        clip_path = out_dir / f"clip_{n:02d}.mp4"
        cut.render_clip(video_path, c["start"], c["end"], ass_path, clip_path)
        ass_path.unlink(missing_ok=True)

        print(f"      -> clip_{n:02d}.mp4  "
              f"({c['duration']:.0f}s @ {c['start']:.0f}s)  \"{c['hook'][:50]}\"")
        manifest.append({
            "file": clip_path.name,
            "start": round(c["start"], 2),
            "end": round(c["end"], 2),
            "duration": round(c["duration"], 2),
            "score": round(c["score"], 2),
            "hook": c["hook"],
            "transcript": c["text"],
        })

    print("[5/5] Writing manifest...")
    (out_dir / "manifest.json").write_text(
        json.dumps({"source": source, "video_id": video_id, "clips": manifest}, indent=2),
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
    args = p.parse_args()
    run(args.source, max_clips=args.max_clips, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
