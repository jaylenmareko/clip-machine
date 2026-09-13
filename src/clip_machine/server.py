"""Local web UI for clip-machine.

    python -m clip_machine.server        -> http://127.0.0.1:8420

Stdlib only — no Flask, no extra install, minimal memory footprint (which
matters on this machine). Binds to loopback only.

One job at a time by design: transcription is the memory bottleneck and
running two at once is how you get OOM-killed.
"""
import http.server
import json
import mimetypes
import os
import re
import socketserver
import threading
import time
import urllib.parse
import webbrowser
from contextlib import redirect_stdout
from pathlib import Path

from . import (config, cut, llm, modes, pipeline, publish, transcribe,
               youtube)

PORT = int(os.environ.get("CLIP_PORT", "8420"))

_jobs: list[dict] = []
_job_lock = threading.Lock()
_worker_alive = False


# --------------------------------------------------------------------------
# job running
# --------------------------------------------------------------------------
class _LogWriter:
    """Fan pipeline print() output into the job's log list, line by line."""

    def __init__(self, job: dict):
        self.job = job
        self.buf = ""

    def write(self, s: str) -> int:
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.job["log"].append(line.rstrip())
        return len(s)

    def flush(self) -> None:
        pass


# The pipeline's own output is developer text - "energy profile ready",
# "quality floor dropped 2 weak candidates", "ranked by claude-haiku-4-5
# (2814 in / 35 out)". Useful in a terminal, alarming in a product. These map
# the stage markers onto something a stranger can read; the raw log stays
# available behind a details toggle.
_STAGES = [
    ("[1/6]", "Downloading the video"),
    ("[1b/6]", "Checking the video"),
    ("[2/6]", "Transcribing the audio"),
    ("[3/6]", "Listening for energy"),
    ("[4/6]", "Finding the best moments"),
    ("titled by", "Writing titles"),
    ("[5/6]", "Making the clips"),
    ("[6/6]", "Finishing up"),
]
_ETA_RE = re.compile(r"about (\d+) min to process")
_SOURCE_RE = re.compile(r"-> (\d+) min of source")
# Transcription prints one line per segment, stamped with where it is in the
# source: "  [  791.8s] So let me just get into it...". That is REAL progress
# through the slow stage - better than any easing curve - and dividing it by
# wall time gives a measured rate the static ETA can be corrected against.
_SEG_RE = re.compile(r"^\s*\[\s*([\d.]+)s\]")


def _progress(log: list[str]) -> dict:
    """Derive a friendly stage, ETA and instruction notices from the output."""
    stage, index = "Starting", 0
    eta = None
    applied = ignored = None
    source_secs = 0.0
    heard = 0.0
    for line in log:
        for i, (marker, label) in enumerate(_STAGES):
            # Never go backwards: compile mode logs "titled by" AFTER [5/6],
            # the reverse of the clip path, and a ring that reverses reads as
            # a bug. Highest stage reached wins.
            if marker in line and i + 1 > index:
                stage, index = label, i + 1
        if m := _ETA_RE.search(line):
            eta = int(m.group(1))
        if m := _SOURCE_RE.search(line):
            source_secs = int(m.group(1)) * 60.0
        if m := _SEG_RE.match(line):
            heard = max(heard, float(m.group(1)))
        # Surfaced rather than left in the log: the whole point of the
        # instruction box is knowing what it did and did not do.
        if "instruction applied:" in line:
            applied = line.split("instruction applied:", 1)[1].strip()
        elif "instruction IGNORED:" in line:
            ignored = line.split("instruction IGNORED:", 1)[1].strip()
    # Lines the pipeline flags with "!" are conditions that change what the
    # user should expect from this run - memory pressure above all. Buried in
    # the Details log they explain nothing; a slow run just looks broken.
    warning = next((l.split("!", 1)[1].strip()
                    for l in reversed(log) if l.strip().startswith("!")), None)
    cached = any("reused cached transcript" in line for line in log)
    heard_frac = round(heard / source_secs, 4) if source_secs and heard else None
    return {"stage": stage, "step": index, "steps": len(_STAGES),
            "eta_min": eta, "applied": applied, "ignored": ignored,
            "cached": cached, "warning": warning,
            "source_secs": source_secs or None,
            "heard_secs": round(heard, 1) or None, "heard_frac": heard_frac}


def _friendly_error(err: Exception, log: list[str]) -> str:
    """Turn an exception into something worth showing a person."""
    text = f"{type(err).__name__}: {err}".lower()
    blob = " ".join(log[-12:]).lower()

    if isinstance(err, FileNotFoundError):
        return "Couldn't find that file. Check the path and try again."
    if "no speech" in text:
        return "No speech was detected in that video."
    if any(k in text + blob for k in
           ("yt-dlp", "unavailable", "private", "sign in", "http error 4",
            "unsupported url", "is not a valid url")):
        return ("Couldn't download that video. Make sure the link is public "
                "and playable without signing in.")
    if "credit balance" in text + blob:
        return ("Claude credit ran out, so clips were ranked locally instead. "
                "Top up at console.anthropic.com for better picks.")
    if any(k in text for k in ("memory", "killed", "oom")):
        return ("Ran out of memory. Close other apps and try again, or use a "
                "shorter video.")
    if "ffmpeg" in text or "calledprocesserror" in text:
        return "The video couldn't be processed. It may be corrupt or an unusual format."
    return "Something went wrong. Open Details for the technical log."


def _run_job(job: dict) -> None:
    """Execute one job. Called only by the single queue worker."""
    try:
        job["status"] = "running"
        job["started"] = time.time()
        def on_clip(rec, done, total, video_id):
            # Clips land in /api/status the moment each one finishes, so the
            # page can show results while the rest are still rendering
            # instead of holding everything back to the final second.
            # video_id has to be set here too: /clips/<id>/<file> is how the
            # page fetches them, and it was previously only known once the
            # whole run finished.
            job["video_id"] = video_id
            job["clips"].append(rec)
            job["made"], job["to_make"] = done, total

        with redirect_stdout(_LogWriter(job)):
            out_dir = pipeline.run(job["source"], max_clips=job["max_clips"],
                                   use_llm=job["use_llm"], layout=job["layout"],
                                   instructions=job.get("instructions"),
                                   captions=job.get("captions"),
                                   swap=bool(job.get("swap")),
                                   on_clip=on_clip)
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8-sig"))
        job["video_id"] = manifest["video_id"]
        job["clips"] = manifest["clips"]
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = _friendly_error(e, job.get("log", []))
        job["log"].append(f"ERROR  {type(e).__name__}: {e}")
    finally:
        # Stop the clock. Without this, elapsed is time.time()-started
        # forever, so a finished run's duration keeps growing and any ETA or
        # rate derived from it after the fact is nonsense.
        job["finished"] = time.time()


def _drain_queue() -> None:
    """Run queued jobs one at a time until the queue is empty.

    Still strictly serial - transcription is the memory bottleneck and this
    machine OOM-kills under two at once. The queue only means a second
    submission waits its turn instead of being rejected outright.
    """
    global _worker_alive
    while True:
        with _job_lock:
            nxt = next((j for j in _jobs if j["status"] == "queued"), None)
            if nxt is None:
                _worker_alive = False
                return
        _run_job(nxt)


def _start_job(source: str, max_clips: int | None, use_llm: bool,
               layout: str, instructions: str | None = None,
               captions: bool | None = None, swap: bool = False) -> dict:
    global _worker_alive
    with _job_lock:
        job = {
            "id": len(_jobs) + 1,
            "status": "queued", "log": [], "clips": [],
            "source": source, "video_id": None, "error": None,
            "max_clips": max_clips, "use_llm": use_llm, "layout": layout,
            "instructions": instructions, "captions": captions,
            "swap": swap,
        }
        _jobs.append(job)
        position = sum(1 for j in _jobs if j["status"] == "queued")
        start_worker = not _worker_alive
        if start_worker:
            _worker_alive = True

    if start_worker:
        threading.Thread(target=_drain_queue, daemon=True).start()
    return {"ok": True, "queued": position}


def _active_job() -> dict | None:
    """The running job, else the most recent finished one."""
    running = next((j for j in _jobs if j["status"] in ("running", "queued")), None)
    if running:
        return running
    return _jobs[-1] if _jobs else None


# --------------------------------------------------------------------------
# keep / reject
#
# Two jobs at once. Immediately, it curates: rejected clips drop out of the
# zip so "download all" means "download the good ones". Longer term it is the
# only real signal for tuning selection - every scoring change so far came
# from Jaylen reacting to output and me measuring against it, which does not
# scale past one user.
# --------------------------------------------------------------------------
def _feedback_path(video_id: str) -> Path:
    return config.CLIPS_DIR / video_id / "feedback.json"


# Every clip is IN by default. The only per-clip action is removing one.
#
# This is the third revision of this decision and the reasoning matters.
# "Keep" went first: a clip stayed unless rejected, so keeping it asserted
# the default and carried no information. Then per-clip "Approve" was added
# for the scheduler, and it has now gone the same way for the same reason -
# approving 15 clips one at a time to publish all 15 is 15 clicks to express
# the default. What is left is: nothing to do if you like them, one click to
# drop one.
#
# Publishing is still an explicit human act, just BATCHED - "Schedule these
# 13" is one deliberate confirmation for the whole set rather than fifteen.
# Nothing reaches a social account without somebody pressing something,
# because that is not an undoable mistake.
#
# "queued" therefore means the batch confirm has happened. "reject" means
# removed, and carries an OPTIONAL reason - optional because a mandatory
# explanation to delete a clip is the most friction in the app for a first
# time user, and the reason is still recorded whenever it is given. Those
# reasons are the only labelled data about what counts as a bad clip.
VERDICTS = ("queued", "reject")


def _manifest_starts(video_id: str) -> dict | None:
    """{file: start_seconds} for a project, or None if unreadable."""
    p = config.CLIPS_DIR / video_id / "manifest.json"
    if not p.exists():
        return None
    try:
        clips = json.loads(p.read_text(encoding="utf-8-sig")).get("clips", [])
    except Exception:
        return None
    return {c["file"]: float(c.get("start", -1)) for c in clips if c.get("file")}


def _read_feedback(video_id: str) -> dict:
    """{file: {"verdict": ..., "reason": ..., "at": epoch, "start": secs}}

    Older files stored a bare string per clip ({"clip_01.mp4": "reject"}).
    Those are normalised on read rather than migrated on disk, so an old
    project keeps working and nothing has to be rewritten.

    VERDICTS ARE VALIDATED AGAINST THE MANIFEST, because clip filenames are
    positional and a re-run overwrites them. Re-running the 67-min source
    turned 18 clips into 16, and the five removals from the first run
    silently re-attached themselves to four unrelated clips of the second -
    clip_15.mp4 went from a clip Jaylen had removed to "Real leadership
    means accepting you're not for everybody", which he had not. A verdict
    that survives into content it was never about is worse than no verdict.
    So a stored `start` must still match the clip at that filename, and a
    verdict with no start (written before this check existed) cannot be
    verified and is dropped.
    """
    p = _feedback_path(video_id)
    if not p.exists():
        return {}
    try:
        # utf-8-sig, not utf-8: a BOM here used to raise, get swallowed by
        # the except below, and silently discard every verdict in the
        # project. Windows tools add one without asking.
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as e:
        print(f"  ! could not read {p.name}: {type(e).__name__}: {e}")
        return {}
    if not isinstance(raw, dict):
        return {}
    starts = _manifest_starts(video_id)
    out = {}
    for file, v in raw.items():
        if starts is not None:
            if file not in starts:
                continue                      # clip no longer exists
            stored = v.get("start") if isinstance(v, dict) else None
            if stored is None or abs(float(stored) - starts[file]) > 1.0:
                continue                      # different clip, same filename
        if isinstance(v, str):
            # "keep" predates approve and meant only "not rejected", which
            # is the unreviewed default - so it is dropped, not upgraded to
            # an approval the user never gave.
            if v == "reject":
                out[file] = {"verdict": "reject", "reason": "", "at": 0}
        elif isinstance(v, dict) and v.get("verdict") in VERDICTS + ("approve",):
            # 'approve' was the per-clip form of today's batch confirm.
            verdict = "queued" if v["verdict"] == "approve" else v["verdict"]
            out[file] = {"verdict": verdict,
                         "reason": str(v.get("reason", ""))[:2000],
                         "at": int(v.get("at", 0) or 0),
                         "start": float(v.get("start", -1))}
    return out


def _write_feedback(video_id: str, file: str, verdict: str | None,
                    reason: str = "") -> dict:
    data = _read_feedback(video_id)
    if verdict in VERDICTS:
        # Stamp WHICH clip this was about, not just which filename.
        starts = _manifest_starts(video_id) or {}
        data[file] = {"verdict": verdict,
                      "reason": (reason or "").strip()[:2000],
                      "at": int(time.time()),
                      "start": starts.get(file, -1)}
    else:
        data.pop(file, None)   # clicking the active verdict clears it
    path = _feedback_path(video_id)
    if path.parent.exists():
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _build_zip(video_id: str) -> Path | None:
    """Bundle the clips (minus anything rejected) plus a titles file."""
    import zipfile

    d = (config.CLIPS_DIR / video_id).resolve()
    if not str(d).startswith(str(config.CLIPS_DIR.resolve())) or not d.is_dir():
        return None
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        clips = json.loads(manifest_path.read_text(encoding="utf-8-sig")).get("clips", [])
    except Exception:
        return None

    verdicts = _read_feedback(video_id)
    keep = [c for c in clips
            if verdicts.get(c["file"], {}).get("verdict") != "reject"]
    if not keep:
        return None

    # Written into the clip dir so repeat downloads are free. Rebuilt when a
    # clip or a verdict is newer than the zip - otherwise rejecting a clip
    # would silently keep shipping it.
    zip_path = d / f"{video_id}-clips.zip"
    newest = max(
        [(d / c["file"]).stat().st_mtime for c in keep if (d / c["file"]).exists()]
        + ([_feedback_path(video_id).stat().st_mtime]
           if _feedback_path(video_id).exists() else [])
        or [0]
    )
    if zip_path.exists() and zip_path.stat().st_mtime >= newest:
        return zip_path

    def mmss(s: float) -> str:
        s = round(s)
        return f"{s // 60}:{s % 60:02d}"

    lines = [f"CLIP MACHINE - {video_id}", ""]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for n, c in enumerate(keep, 1):
            f = d / c["file"]
            if f.exists():
                z.write(f, arcname=c["file"])
            lines += [
                f"{c['file']}   {mmss(c['duration'])}   at {mmss(c['start'])}"
                f"   [{c.get('mode', '')}]",
                f"   {c.get('title') or c.get('hook', '')}",
                "",
            ]
        lines += ["NOTE: the transcript has no speaker labels, so a title can",
                  "misattribute who said what. Check before posting."]
        z.writestr("TITLES.txt", "\n".join(lines))
    return zip_path


def _library() -> list[dict]:
    """Every project, newest first. One entry per source video."""
    out = []
    if not config.CLIPS_DIR.exists():
        return out
    for d in sorted(config.CLIPS_DIR.iterdir(), key=lambda p: -p.stat().st_mtime):
        manifest = d / "manifest.json"
        if not (d.is_dir() and manifest.exists()):
            continue
        try:
            m = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        clips = m.get("clips", [])
        feedback = _read_feedback(d.name)
        out.append({
            "video_id": d.name,
            "count": len(clips),
            "clips": clips,
            "feedback": feedback,
            "published": publish.by_file(d.name),
            "footer": m.get("footer", ""),
            "source": m.get("source", ""),
            "mode": m.get("mode", ""),
            "created": int(d.stat().st_mtime),
            "total_seconds": round(sum(c.get("duration", 0) for c in clips)),
            "rejected": sum(1 for v in feedback.values()
                            if v.get("verdict") == "reject"),
            "queued": sum(1 for v in feedback.values()
                          if v.get("verdict") == "queued"),
        })
    return out


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the console clean for pipeline output
        pass

    # -- helpers --
    def _send_json(self, obj: dict | list, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        """Serve a clip, with Range support so the scrubber works."""
        if not path.exists() or not path.is_file():
            self._send_json({"error": "not found"}, 404)
            return
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        rng = self.headers.get("Range")

        if rng and (m := re.match(r"bytes=(\d+)-(\d*)", rng)):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else size - 1
            end = min(end, size - 1)
            length = max(end - start + 1, 0)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            start, length = 0, size
            self.send_response(200)

        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    # -- routes --
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        if route == "/":
            self._send_html(_render_page())
        elif route == "/api/status":
            free = transcribe.free_ram_gb()
            job = _active_job() or {"status": "idle", "log": [], "clips": []}
            waiting = sum(1 for j in _jobs if j["status"] == "queued")
            if job.get("status") == "queued":
                waiting = max(waiting - 1, 0)   # this one is next, not waiting
            self._send_json({
                "status": job["status"],
                "log": job["log"][-400:],
                "clips": job.get("clips", []),
                "video_id": job.get("video_id"),
                "error": job.get("error"),
                "free_ram": round(free, 2) if free else None,
                "model": config.WHISPER_MODEL,
                "queued": waiting,
                "made": job.get("made", 0),
                "to_make": job.get("to_make", 0),
                "elapsed": round((job.get("finished") or time.time())
                                 - job["started"], 1)
                           if job.get("started") else 0,
                **_progress(job.get("log", [])),
            })
        elif route == "/api/library":
            self._send_json(_library())
        elif route.startswith("/bundle/"):
            vid = urllib.parse.unquote(route[len("/bundle/"):]).strip("/")
            if not vid or "/" in vid or ".." in vid:
                self._send_json({"error": "Couldn't tell which project that was."}, 400)
                return
            mpath = config.CLIPS_DIR / vid / "manifest.json"
            if not mpath.exists():
                self._send_json({"error": "unknown project"}, 404)
                return
            try:
                data = json.loads(mpath.read_text(encoding="utf-8-sig"))
            except Exception:
                self._send_json({"error": "could not read that project"}, 500)
                return
            fb = _read_feedback(vid)
            # Everything except what was removed - same rule as the plain
            # zip. This used to require clips be QUEUED first, which made
            # "Full bundle" fail for anyone who had not scheduled yet.
            # Downloading is not publishing, so there is nothing to gate.
            picked = [c for c in data.get("clips", [])
                      if fb.get(c["file"], {}).get("verdict") != "reject"]
            if not picked:
                self._send_json({"error": "Every clip in this project was removed."}, 400)
                return
            path = publish.bundle(vid, picked, data.get("source", ""),
                                  config.CLIPS_DIR / vid,
                                  footer=data.get("footer", ""))
            self._send_file(path)
            return
        elif route.startswith("/zip/"):
            vid = urllib.parse.unquote(route[len("/zip/"):]).strip("/")
            path = _build_zip(vid)
            if path is None:
                self._send_json({"error": "nothing to zip"}, 404)
                return
            self._send_file(path)
        elif route.startswith("/clips/"):
            rel = urllib.parse.unquote(route[len("/clips/"):])
            # contain the path inside CLIPS_DIR
            target = (config.CLIPS_DIR / rel).resolve()
            if not str(target).startswith(str(config.CLIPS_DIR.resolve())):
                self._send_json({"error": "forbidden"}, 403)
                return
            self._send_file(target)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        if path not in ("/api/run", "/api/feedback",
                        "/api/youtube", "/api/youtube/plan",
                        "/api/clip", "/api/project", "/api/delete"):
            self._send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")

        if path == "/api/feedback":
            vid = (data.get("video_id") or "").strip()
            file = (data.get("file") or "").strip()
            verdict = data.get("verdict")
            if not vid or not file or "/" in file or ".." in file + vid:
                self._send_json({"error": "Couldn't tell which project that was."}, 400)
                return
            reason = data.get("reason") or ""
            self._send_json({"ok": True,
                             "feedback": _write_feedback(vid, file, verdict,
                                                         reason)})
            return

        if path == "/api/delete":
            vid = (data.get("video_id") or "").strip()
            # Path safety matters more here than anywhere else - this is the
            # only endpoint that removes files. Reject separators outright,
            # then resolve and PROVE the result is a direct child of
            # CLIPS_DIR before touching it.
            if not vid or "/" in vid or chr(92) in vid or ".." in vid:
                self._send_json({"error": "Couldn't tell which project that was."}, 400)
                return
            root = config.CLIPS_DIR.resolve()
            target = (root / vid).resolve()
            if target.parent != root or not target.is_dir():
                self._send_json({"error": "unknown project"}, 404)
                return

            import shutil
            try:
                n = len(list(target.glob("clip_*.mp4")))
                shutil.rmtree(target)
            except Exception as e:
                self._send_json(
                    {"error": f"Could not delete it: {type(e).__name__}"}, 500)
                return
            # The publish ledger is deliberately NOT touched: it records what
            # went out, and a video on YouTube does not stop existing because
            # the local copy was deleted.
            self._send_json({"ok": True, "deleted": n})
            return

        if path == "/api/project":
            # One footer per project, appended wherever a description is
            # used. Stored ONCE rather than baked into every clip: editing
            # it updates all of them, and applying it twice cannot
            # duplicate it - which hand-editing 16 descriptions would.
            vid = (data.get("video_id") or "").strip()
            if not vid or "/" in vid or ".." in vid:
                self._send_json({"error": "Couldn't tell which project that was."}, 400)
                return
            mpath = config.CLIPS_DIR / vid / "manifest.json"
            if not mpath.exists():
                self._send_json({"error": "unknown project"}, 404)
                return
            try:
                man = json.loads(mpath.read_text(encoding="utf-8-sig"))
            except Exception:
                self._send_json({"error": "could not read that project"}, 500)
                return
            man["footer"] = (data.get("footer") or "").strip()[:2000]
            mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")
            self._send_json({"ok": True, "footer": man["footer"]})
            return

        if path == "/api/clip":
            vid = (data.get("video_id") or "").strip()
            file = (data.get("file") or "").strip()
            if not vid or not file or "/" in file or ".." in file + vid:
                self._send_json({"error": "Couldn't tell which clip that was."}, 400)
                return
            mpath = config.CLIPS_DIR / vid / "manifest.json"
            if not mpath.exists():
                self._send_json({"error": "unknown project"}, 404)
                return
            try:
                man = json.loads(mpath.read_text(encoding="utf-8-sig"))
            except Exception:
                self._send_json({"error": "could not read that project"}, 500)
                return

            title = (data.get("title") or "").strip()
            tags = " ".join((data.get("tags") or "").split())
            desc = (data.get("description") or "").strip()
            # Tags live glued to the end of the title on an upload, which is
            # how every platform wants them - they are only separate fields
            # while editing, so nobody has to hand-manage the spacing.
            combined = f"{title} {tags}".strip() if tags else title

            hit = None
            for c in man.get("clips", []):
                if c.get("file") == file:
                    c["title"] = combined
                    c["description"] = desc[:5000]
                    c["edited"] = True
                    hit = c
                    break
            if hit is None:
                self._send_json({"error": "no such clip"}, 404)
                return
            mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")
            self._send_json({"ok": True, "clip": hit})
            return

        if path == "/api/youtube/plan":
            vid = (data.get("video_id") or "").strip()
            if not vid or "/" in vid or ".." in vid:
                self._send_json({"error": "Couldn't tell which project that was."}, 400)
                return
            mpath = config.CLIPS_DIR / vid / "manifest.json"
            if not mpath.exists():
                self._send_json({"error": "unknown project"}, 404)
                return
            try:
                clips = json.loads(
                    mpath.read_text(encoding="utf-8-sig")).get("clips", [])
            except Exception:
                self._send_json({"error": "could not read that project"}, 500)
                return
            fb = _read_feedback(vid)
            picked = [c for c in clips
                      if fb.get(c["file"], {}).get("verdict") != "reject"]
            try:
                gap = max(0.5, min(24.0, float(data.get("gap_hours") or 4)))
                hour = max(0, min(23, int(data.get("hour") or 9)))
                minute = max(0, min(59, int(data.get("minute") or 0)))
            except (TypeError, ValueError):
                gap, hour, minute = 4.0, 9, 0
            self._send_json(youtube.plan(picked, vid, gap_hours=gap,
                                         hour=hour, minute=minute))
            return

        if path == "/api/youtube":
            vid = (data.get("video_id") or "").strip()
            if not vid or "/" in vid or ".." in vid:
                self._send_json({"error": "Couldn't tell which project that was."}, 400)
                return
            if not youtube.configured():
                self._send_json({"error": "YouTube is not connected yet."}, 400)
                return
            mpath = config.CLIPS_DIR / vid / "manifest.json"
            if not mpath.exists():
                self._send_json({"error": "unknown project"}, 404)
                return
            try:
                man = json.loads(mpath.read_text(encoding="utf-8-sig"))
                clips = man.get("clips", [])
            except Exception:
                self._send_json({"error": "could not read that project"}, 500)
                return
            fb = _read_feedback(vid)
            # Only clips the user actually confirmed. Anything removed, or
            # never queued, is not this function's business.
            picked = [c for c in clips
                      if fb.get(c["file"], {}).get("verdict") != "reject"]
            if not picked:
                self._send_json({"error": "Every clip in this project was removed."}, 400)
                return
            try:
                gap = max(0.5, min(24.0, float(data.get("gap_hours") or 4)))
                hour = max(0, min(23, int(data.get("hour") or 9)))
                minute = max(0, min(59, int(data.get("minute") or 0)))
            except (TypeError, ValueError):
                gap, hour, minute = 4.0, 9, 0

            # Honour the order the user dragged into. Anything the client
            # did not name keeps its existing position at the end, so a
            # stale or partial order cannot silently drop a clip.
            order = data.get("order") or []
            if isinstance(order, list) and order:
                rank = {f: i for i, f in enumerate(order) if isinstance(f, str)}
                picked.sort(key=lambda c: rank.get(c.get("file"), len(rank)))

            # Uploading is slow and blocking; the browser waits on it. Fine
            # for six clips, and simpler than a second job queue.
            result = youtube.schedule_batch(
                picked, config.CLIPS_DIR / vid, vid,
                gap_hours=gap, hour=hour, minute=minute,
                footer=man.get("footer", ""))
            self._send_json(result)
            return

        source = (data.get("source") or "").strip()
        want_caps = bool(data.get("captions"))
        if not source:
            self._send_json({"error": "Give it a video URL or a local file path."}, 400)
            return
        result = _start_job(
            source,
            # Always auto: the count scales with source length in config.
            None,
            # Reranking is part of the product now, not an optional extra.
            # This defaulted to False when the old checkbox was removed, which
            # silently skipped Claude for every run started from the UI.
            True,
            # Detected from the source; the user is not asked.
            data.get("layout") or "auto",
            (data.get("instructions") or "").strip() or None,
            want_caps,
            data.get("swap") is True,
        )
        self._send_json(result, 409 if "error" in result else 200)


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ClipYourVideos.com</title>
<style>
  :root{
    --bg:#0d0f14; --panel:#151922; --line:#242b38; --text:#e8ecf4;
    --muted:#8b95a8; --accent:#6ea8fe; --good:#3ddc97; --warn:#ffb454; --bad:#ff6b6b;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font:14px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:28px 20px 64px}
  header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px}
  h1{font-size:20px;margin:0;letter-spacing:-.02em}
  /* Wordmark. The .com is set lighter so the name reads first and the
     domain trails it, rather than the two competing at one weight. */
  .brand{font-size:23px;font-weight:700;letter-spacing:-.03em}
  .brand .tld{font-weight:500;color:var(--muted);letter-spacing:-.02em}
  .sub{color:var(--muted);font-size:13px}
  .badge{margin-left:auto;font-size:12px;color:var(--muted);
    border:1px solid var(--line);border-radius:999px;padding:3px 10px}
  .badge.warn{color:var(--warn);border-color:#4a3a1e}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:18px;margin-top:18px}
  label{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}
  input,select{width:100%;background:#0f131b;border:1px solid var(--line);color:var(--text);
    border-radius:8px;padding:10px 12px;font:inherit}
  input:focus,select:focus{outline:2px solid var(--accent);outline-offset:-1px;border-color:transparent}
  .row{display:grid;grid-template-columns:1fr;gap:12px}
  .row-mode{margin-top:12px}
  .modes{display:flex;gap:8px;flex-wrap:wrap}
  .mode{border:1px solid var(--line);background:#0f131b;border-radius:8px;
    padding:8px 12px;cursor:pointer;font-size:13px;color:var(--muted);user-select:none}
  .mode:hover{border-color:#39435a;color:var(--text)}
  .mode.on{border-color:var(--accent);color:var(--text);background:#16223a}
  .mode small{display:block;font-size:11px;opacity:.7;margin-top:2px}
  .chips{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}
  .chip{font-size:10.5px;letter-spacing:.02em;background:#1b2434;border:1px solid var(--line);
    color:#9fb4d8;border-radius:999px;padding:2px 7px}
  .chip.aud{background:#1d2a1f;border-color:#2c4230;color:#8fd3a0}
  .prog{display:flex;align-items:center;gap:18px}
  .ring{width:96px;height:96px;flex:0 0 auto;transform:rotate(-90deg)}
  .ring circle{fill:none;stroke-width:8;stroke-linecap:round}
  .ring .track{stroke:#0f131b}
  /* 2*PI*52 = 326.7 - the full circumference, so dashoffset maps to percent */
  .ring .dial{stroke:var(--accent);stroke-dasharray:326.7;stroke-dashoffset:326.7;
    /* Matches the 1s poll interval, so the arc sweeps continuously
       instead of jerking for .45s then sitting still for .55s. */
    transition:stroke-dashoffset 1s linear}
  .prog.done .dial{stroke:var(--good)}
  .prog.err .dial{stroke:var(--bad)}
  .progtext{min-width:0}
  .pct{font-size:26px;font-weight:600;letter-spacing:-.02em;line-height:1.1}
  .prog.done .pct{color:var(--good)}
  .prog.err .pct{color:var(--bad)}
  #details summary{cursor:pointer;color:var(--muted);font-size:12px;margin-top:6px}
  #details pre{margin-top:8px}
  .acts{display:flex;margin-top:10px}
  /* Download and Reject are a stacked pair now, so they have to match
     exactly: same padding, and a transparent 1px border to answer Reject's
     real one, or the two sit 2px different in height. */
  .acts .dl{flex:1 1 auto;text-align:center;padding:6px;
    border:1px solid transparent}
  /* Deliberately NOT line-clamped. A fixed line count only aligns at one
     viewport width - descriptions run 234-291 chars, which is 6 lines on a
     wide card and 8 on a narrow one, so any clamp that fits one truncates
     the other. Instead .stats carries margin-top:auto, so everything from
     the stats row down is pinned to the bottom of the card; the grid
     already stretches sibling cards to equal height, so those rows line up
     across the row at every width and the description absorbs the slack. */
  .desc{margin:0 0 9px;font-size:12.5px;line-height:1.5;color:#a8b6cc;
    min-height:calc(2 * 1.5em)}
  .vote{display:flex;gap:8px;margin-top:8px}
  .vote button{flex:1;background:transparent;border:1px solid var(--line);
    color:var(--muted);border-radius:7px;padding:6px;font:600 12px/1.4 inherit;
    cursor:pointer}
  .vote button:hover{color:var(--text);border-color:#39435a}
  /* Keep was removed - a clip is kept unless it is rejected, so the button
     asserted a state that was already the default. Reject is the only verdict
     now, and it reads red rather than neutral-grey. */
  .vote button.edit{color:var(--muted);border-color:transparent;font-weight:500}
  .vote button.edit:hover{color:var(--text);border-color:var(--line);
    background:#161d2a}
  .vote button.rej{color:var(--muted);border-color:transparent;font-weight:500}
  .vote button.rej:hover{color:var(--bad);border-color:#4d2c2b;background:#2a1a1a}
  .vote button.on-rej{color:var(--bad);border-color:#4d2c2b;background:#2a1a1a;
    font-weight:600}
  .clip.rejected{opacity:.42}
  .clip.queued{border-color:#2c4a37;box-shadow:inset 0 0 0 1px #2c4a3755}
  /* Already on YouTube: recede it, because the decision is made and the
     card is now a record rather than something to act on. Not as faded as
     a removed clip - this one is a success. */
  .clip.onair{border-color:#2b3c5c}
  .media{position:relative}
  /* Chrome drops the native time readout on a video narrower than about
     300px, and at 4 cards across these are 248px. So the timecode is ours:
     same information, same place, independent of the browser's width
     heuristics. Sits above the controls bar and ignores clicks. */
  .tc{position:absolute;right:7px;bottom:38px;pointer-events:none;
    background:#0a0d14c8;color:#eaf0fa;font-size:11px;font-weight:600;
    font-variant-numeric:tabular-nums;padding:2px 6px;border-radius:5px;
    letter-spacing:.01em}
  /* A stamp across the middle of the frame, because a small line of text
     in the metadata was missable - and missing "this already went out" is
     how a clip gets posted twice. pointer-events:none so the player still
     works underneath, and it fades on hover so it never blocks watching. */
  .stamp{position:absolute;inset:0;display:flex;flex-direction:column;
    align-items:center;justify-content:center;gap:6px;pointer-events:none;
    background:#070a12b0;transition:opacity .18s;text-align:center;padding:14px}
  .clip:hover .stamp{opacity:.12}
  .stamp b{font-size:19px;font-weight:800;letter-spacing:.06em;
    line-height:1.15;color:#fff;text-shadow:0 2px 10px #000a}
  .stamp span{font-size:12px;color:#dfe6f2;text-shadow:0 1px 6px #000a}
  .stamp.soon b{color:#7fb2ff}
  .stamp.done b{color:#5fd08a}
  .stamp.bad  b{color:#e8615e}
  .stamp.priv b{color:#d9a13a}
  /* The reason box only appears once Reject is pressed - the verdict is not
     sent until Submit, so a mis-click costs nothing. */
  .why{margin-top:8px}
  .why textarea{width:100%;background:#0f131b;border:1px solid var(--line);
    color:var(--text);border-radius:7px;padding:7px 9px;font:inherit;
    font-size:12px;resize:vertical;box-sizing:border-box}
  .why textarea:focus{outline:none;border-color:var(--accent)}
  .whyacts{display:flex;gap:8px;margin-top:6px}
  .whyacts .btn{padding:6px;border:1px solid transparent}
  .whyacts .btn:disabled{opacity:.4;cursor:not-allowed}
  .proj{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    margin-top:14px;overflow:hidden}
  .proj>summary{cursor:pointer;padding:14px 16px;display:flex;align-items:baseline;
    gap:14px;flex-wrap:wrap;list-style:none}
  .proj>summary::-webkit-details-marker{display:none}
  .proj>summary::before{content:'B8';color:var(--muted);font-size:11px;
    transition:transform .15s ease;display:inline-block}
  .proj[open]>summary::before{transform:rotate(90deg)}
  .proj>summary:hover{background:#181d27}
  .pname{font-weight:600;font-size:13px;word-break:break-all;flex:1 1 auto}
  .pmeta{display:flex;align-items:center;gap:9px;white-space:nowrap;font-size:12px}
  .pcount{color:var(--text);font-weight:600}
  .pdate{color:var(--muted)}
  .ptime{color:#6c7a90}
  .pkindx{color:var(--accent)}
  .phow{margin:0 16px 12px;padding:10px 13px;font-size:13.5px;line-height:1.5;
    color:var(--text);background:#16223a;border:1px solid #2b3c5c;
    border-left:3px solid var(--accent);border-radius:8px}
  .phow b{color:var(--accent);font-weight:600}
  .pbar{display:flex;align-items:center;gap:12px;padding:0 16px 4px;
    flex-wrap:wrap}
  .pbar .btn{flex:0 0 auto;padding:7px 14px}
  /* Deletion is the one irreversible action in the bar, so it reads as
     one: red, iconic, and pushed away from the buttons you use daily. */
  .pbar .trash{margin-left:auto;flex:0 0 auto;background:transparent;
    border:1px solid transparent;color:#8a6a6a;font-size:15px;
    padding:6px 10px;line-height:1}
  .pbar .trash:hover{color:#fff;background:var(--bad);border-color:var(--bad)}
  .btn.danger{background:var(--bad);color:#fff;font-weight:600}
  .btn.danger:hover{background:#c2413f}
  .delsum{margin:0 0 12px;font-size:13.5px;color:var(--text);line-height:1.5}
  .pbar .sched{background:var(--good);color:#08130c;font-weight:600}
  .pbar .sched:hover{background:#3f9e6a}
  .pbar .sched.done{background:transparent;color:var(--good);
    border:1px solid #2c4a37}
  .pbar .sched:disabled{opacity:.5;cursor:default}
  /* Cadence lives next to the button that uses it. 6/day is the ceiling
     because that is the API's upload quota, not a taste decision. */
  .ytset{display:flex;align-items:center;gap:6px;font-size:12px;
    color:var(--muted)}
  .ytset select,.ytset input{width:auto;padding:5px 7px;font-size:12px;
    background:#0f131b;border:1px solid var(--line);color:var(--text);
    border-radius:6px}
  /* A confirm sheet, because uploading is irreversible. The point of it is
     the LIST of real dates - "Fri 11 Sep 9:00 AM" explains the cadence
     better than any label describing what "3 a day" means. */
  .modal{position:fixed;inset:0;background:#05070bd0;display:flex;
    align-items:center;justify-content:center;padding:24px;z-index:50}
  /* The `hidden` attribute is only display:none from the browser's default
     sheet, so the display:flex above outranked it and the sheet could never
     close - it was on screen from page load. Any element toggled with
     `hidden` needs this whenever author CSS sets display. */
  .modal[hidden]{display:none}
  .sheet{background:var(--panel);border:1px solid var(--line);
    border-radius:14px;padding:20px 22px;max-width:440px;width:100%;
    max-height:86vh;overflow:auto}
  .sheet h3{margin:0 0 14px;font-size:16px;letter-spacing:-.01em}
  .ytrow{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px}
  .ytrow label{font-size:13px;color:var(--muted);display:flex;
    align-items:center;gap:7px}
  .ytrow select,.ytrow input{width:auto;padding:6px 8px;font-size:13px}
  .ytsum{margin:0 0 10px;font-size:13.5px;color:var(--text);
    line-height:1.5}
  .ytleft{margin:0 0 14px;padding:10px 12px;background:#161b26;
    border-left:3px solid #d9a13a;border-radius:6px;font-size:12px;
    color:#c6cfdd;line-height:1.55}
  .ytsum b{color:var(--text);font-weight:600}
  .ytsum .ok{color:var(--good)}
  .ytsum .wait{color:#d9a13a}
  .draghint{margin:0 0 7px;font-size:11.5px;color:var(--muted)}
  .ytlist{margin:0 0 12px;padding:0;list-style:none;font-size:12.5px;
    color:#a8b6cc}
  .ytlist li{display:flex;align-items:center;gap:9px;padding:7px 8px;
    border:1px solid transparent;border-radius:7px;cursor:grab;
    background:#0f131b;margin-bottom:4px}
  .ytlist li:hover{border-color:var(--line)}
  .ytlist li.dragging{opacity:.55;border-color:var(--accent);cursor:grabbing}
  .grip{color:#4d5a70;letter-spacing:-3px;font-size:15px;flex:0 0 auto;
    line-height:1}
  .slot{color:var(--muted);font-variant-numeric:tabular-nums;flex:0 0 14px}
  .sl-when{color:var(--text);font-weight:600;white-space:nowrap;
    font-variant-numeric:tabular-nums}
  .sl-title{color:#8fa0bb;overflow:hidden;text-overflow:ellipsis;
    white-space:nowrap;min-width:0}
  .ytnote{margin:0 0 16px;font-size:12px;color:var(--muted);line-height:1.5}
  .ytacts{display:flex;gap:9px;justify-content:flex-end}
  .ytacts .btn{flex:0 0 auto;padding:8px 16px}
  .ytwarn{color:var(--bad)}
  .pick{display:block;width:100%;text-align:left;background:#0f131b;
    border:1px solid var(--line);border-radius:10px;padding:13px 15px;
    margin-bottom:10px;cursor:pointer;color:var(--text);font:inherit}
  .pick:hover{border-color:var(--accent);background:#141b28}
  .pick b{display:block;font-size:14px;margin-bottom:4px}
  .pick span{display:block;font-size:12px;color:var(--muted);line-height:1.5}
  .foot{display:block;margin-top:6px;color:#7d8ea8}
  .fld{display:block;margin-bottom:12px;font-size:12px;color:var(--muted)}
  .fld textarea{width:100%;margin-top:5px;background:#0f131b;
    border:1px solid var(--line);color:var(--text);border-radius:7px;
    padding:8px 10px;font:inherit;font-size:13px;resize:vertical;
    box-sizing:border-box;line-height:1.45}
  .fld textarea:focus{outline:none;border-color:var(--accent)}
  .pnote{color:var(--muted);font-size:12px}
  .proj .grid{padding:12px 16px 16px}
  .allbar{display:flex;align-items:center;gap:12px;margin:16px 0 0}
  .allbar .btn{flex:0 0 auto;padding:9px 16px}
  .allbar span{color:var(--muted);font-size:12px}
  .btn{flex:1;text-align:center;background:var(--accent);color:#0a1020;border:0;
    border-radius:7px;padding:7px 10px;font:600 12px/1.4 inherit;cursor:pointer;
    text-decoration:none}
  .btn.ghost{background:transparent;color:var(--muted);border:1px solid var(--line)}
  .btn.ghost:hover{color:var(--text);border-color:#39435a}
  .kind{background:#26304a;border:1px solid #35406090;color:#a9c0ee;
    border-radius:5px;padding:1px 7px;font-size:11px;letter-spacing:.03em}
  /* The warning is not decoration: with captions on, ASR errors go on
     screen, so the review step is real work. Saying so next to the
     checkbox is cheaper than a surprise. */
  .cap{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--text);
    cursor:pointer;flex:1 1 auto}
  .cap input{width:auto;flex:0 0 auto;accent-color:var(--accent)}
  .cap:has(.hint){flex-wrap:wrap;row-gap:2px}
  .cap .hint{flex-basis:100%;padding-left:24px;color:var(--muted);font-size:12px}
  .row2{display:flex;align-items:center;gap:16px;margin-top:14px;flex-wrap:wrap}
  button{background:var(--accent);color:#0a1020;border:0;border-radius:8px;
    padding:10px 20px;font:600 14px/1 inherit;cursor:pointer}
  button:disabled{opacity:.45;cursor:not-allowed}
  .check{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:13px}
  .check input{width:auto}
  .hint{color:var(--muted);font-size:12px;margin-top:10px}
  /* Label left, one-line description right, input beneath both. */
  .labelrow{display:flex;justify-content:space-between;align-items:baseline;
    gap:20px;flex-wrap:wrap}
  .labelrow label{margin-bottom:6px}
  .labelrow .hint{margin:0 0 6px;text-align:right;flex:1 1 260px}
  .notice{font-size:12px;color:var(--good);margin-top:8px;line-height:1.5}
  .notice.warn{color:var(--warn)}
  .ask{margin-top:14px}
  .ask label em{color:var(--muted);font-style:normal;font-weight:400}
  textarea{width:100%;background:#0f131b;border:1px solid var(--line);
    color:var(--text);border-radius:8px;padding:10px 12px;font:inherit;resize:vertical}
  textarea:focus{outline:2px solid var(--accent);outline-offset:-1px;border-color:transparent}
  .can{margin-top:8px;font-size:12px;color:var(--muted);line-height:1.6}
  .can p{margin:0}
  .can b{color:#aebbd0;font-weight:600}
  .can ul{margin:2px 0 6px;padding-left:18px}
  .can li{margin:1px 0}
  pre{background:#0a0d13;border:1px solid var(--line);border-radius:8px;padding:14px;
    max-height:320px;overflow:auto;font:12px/1.6 ui-monospace,Consolas,monospace;
    color:#c6d0e0;white-space:pre-wrap;margin:0}
  .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}
  @media (max-width:1180px){ .grid{grid-template-columns:repeat(3,minmax(0,1fr))} }
  @media (max-width:860px){  .grid{grid-template-columns:repeat(2,minmax(0,1fr))} }
  @media (max-width:560px){  .grid{grid-template-columns:1fr} }
  .clip{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    overflow:hidden;display:flex;flex-direction:column}
  .clip video{width:100%;display:block;background:#000;aspect-ratio:9/16;object-fit:contain}
  .meta{padding:12px 13px;display:flex;flex-direction:column;flex:1 1 auto}
  /* Fixed-height text rows: this is what makes every card's stats and
     buttons land on the same line as its neighbours'. */
  .clamp{display:-webkit-box;-webkit-box-orient:vertical;overflow:hidden}
  /* The title is the point of the card - it is what gets pasted into the
     upload form - so it gets the size and weight, and the tags recede. */
  .hook{font-weight:600;font-size:15px;margin:0 0 5px;line-height:1.35;
    letter-spacing:-.01em;color:var(--text);
    height:calc(3 * 1.35em);-webkit-line-clamp:3}
  .tags{margin:0 0 8px;font-size:11.5px;line-height:1.45;color:#7d8ea8;
    word-break:break-word;height:calc(2 * 1.45em);-webkit-line-clamp:2}
  .stats{color:var(--muted);font-size:12px;display:flex;gap:10px;
    align-items:center;min-height:22px;margin-top:auto}
  .stats b{color:var(--good);font-weight:600}
  /* No duration badge here: the player's own controls already show it,
     and two places showing the same number meant reconciling them. */
  .from{color:var(--muted)}
  details{margin-top:8px}
  summary{cursor:pointer;color:var(--muted);font-size:12px}
  details p{color:var(--muted);font-size:12px;margin:8px 0 0;max-height:150px;overflow:auto}
  h2{font-size:14px;margin:26px 0 0;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
  .status{font-size:13px;color:var(--muted);margin-bottom:10px}
  .status.run{color:var(--accent)} .status.done{color:var(--good)} .status.err{color:var(--bad)}
  .empty{color:var(--muted);font-size:13px;padding:8px 0}
</style></head><body><div class="wrap">

<header>
  <!-- The wordmark IS the domain, per Jaylen. No separate mark: the name
       already says what the product does, so a logo would only decorate it. -->
  <h1 class="brand">ClipYourVideos<span class="tld">.com</span></h1>
</header>

<div class="panel">
  <div class="row">
    <div>
      <div class="labelrow">
        <label for="src">Video URL or local file path</label>
      </div>
      <input id="src" placeholder="https://youtube.com/watch?v=…  or  C:\path\to\video.mp4">
    </div>
  </div>

  <div class="ask">
    <label for="say">Anything specific? <em>(optional)</em></label>
    <textarea id="say" rows="2" placeholder="e.g. only the parts about Gaza"></textarea>
    <div class="can">
      <p><b>It can act on:</b></p>
      <ul>__ACCEPTED__</ul>
      <p><b>It cannot act on:</b></p>
      <ul>__NOT_ACCEPTED__</ul>
    </div>
  </div>

  <div class="row2">
    <label class="cap"><input type="checkbox" id="caps"> Add captions</label>
    <label class="cap"><input type="checkbox" id="swap"> Put the person on the right on top
      <span class="hint">Only for videos with two people side by side.
        Leave off otherwise.</span></label>
    <button id="go">Generate clips</button>
  </div>
</div>

<div class="panel" id="runpanel" hidden>
  <div class="prog" id="prog">
    <svg class="ring" viewBox="0 0 120 120" aria-hidden="true">
      <circle class="track" cx="60" cy="60" r="52"></circle>
      <circle class="dial" id="dial" cx="60" cy="60" r="52"></circle>
    </svg>
    <div class="progtext">
      <div class="pct" id="pct">0%</div>
      <div class="status" id="status"></div>
    </div>
  </div>
  <div class="notice warn" id="warning" hidden></div>
  <div class="notice" id="applied" hidden></div>
  <div class="notice warn" id="ignored" hidden></div>
  <details id="details"><summary>Details</summary><pre id="log"></pre></details>
  <div class="grid" id="livegrid"></div>
</div>

<div class="modal" id="delmodal" hidden>
  <div class="sheet">
    <h3>Delete this project?</h3>
    <p class="delsum" id="delsum"></p>
    <p class="ytleft" id="delkept" hidden></p>
    <p class="ytnote">This cannot be undone.</p>
    <div class="ytacts">
      <button class="btn ghost" id="delcancel">Cancel</button>
      <button class="btn danger" id="delgo">Delete</button>
    </div>
  </div>
</div>

<div class="modal" id="ftmodal" hidden>
  <div class="sheet">
    <h3>Add to all descriptions</h3>
    <label class="fld">Goes at the end of every clip's description
      <textarea id="fttext" rows="4"
        placeholder="Watch the full episode at peacetothesaints.com"></textarea></label>
    <p class="ytnote">Stored once for the project, so editing it here
      changes every clip. Clips already uploaded keep what they were
      posted with.</p>
    <div class="ytacts">
      <button class="btn ghost" id="ftcancel">Cancel</button>
      <button class="btn" id="ftsave">Save</button>
    </div>
  </div>
</div>

<div class="modal" id="edmodal" hidden>
  <div class="sheet">
    <h3>Edit clip</h3>
    <label class="fld">Title
      <textarea id="edtitle" rows="2"></textarea></label>
    <label class="fld">Hashtags
      <textarea id="edtags" rows="2" placeholder="#advice #wordsofwisdom"></textarea></label>
    <label class="fld">Description
      <textarea id="eddesc" rows="5"></textarea></label>
    <p class="ytnote" id="ednote" hidden></p>
    <div class="ytacts">
      <button class="btn ghost" id="edcancel">Cancel</button>
      <button class="btn" id="edsave">Save</button>
    </div>
  </div>
</div>

<div class="modal" id="dlmodal" hidden>
  <div class="sheet">
    <h3>Download</h3>
    <button class="pick" id="dlfull">
      <b>Full bundle</b>
      <span>Videos + titles and descriptions</span>
    </button>
    <button class="pick" id="dlvids">
      <b>Just the videos</b>
      <span>Video files only</span>
    </button>
    <div class="ytacts">
      <button class="btn ghost" id="dlcancel">Cancel</button>
    </div>
  </div>
</div>

<div class="modal" id="ytmodal" hidden>
  <div class="sheet">
    <h3>Schedule on YouTube</h3>
    <div class="ytrow">
      <label>Every <select id="ytgap">
        <option value="2">2 hours</option>
        <option value="3">3 hours</option>
        <option value="4" selected>4 hours</option>
        <option value="6">6 hours</option>
        <option value="8">8 hours</option>
        <option value="24">day</option>
      </select></label>
      <label>starting at <input id="ythour" type="time" value="09:00"></label>
    </div>
    <p class="ytsum" id="ytsum">Working it out…</p>
    <p class="draghint" id="draghint" hidden>Drag to change the order</p>
    <ol class="ytlist" id="ytlist"></ol>
    <p class="ytleft" id="ytleft" hidden></p>
    <p class="ytnote">Each video uploads private and YouTube makes it public
      at its time. Nothing is public before then.</p>
    <div class="ytacts">
      <button class="btn ghost" id="ytcancel">Cancel</button>
      <button class="btn" id="ytgo" disabled>Upload</button>
    </div>
  </div>
</div>

<h2>Library</h2>
<div id="projects"></div>
<div class="empty" id="noclips">Paste a link to a long video above. You'll get
  vertical clips, each with a title and description ready to post.</div>

<script>
const $ = id => document.getElementById(id);
let running = false;

const AUDIO_SIGNALS = new Set(['loud','animated','fast']);
const FEEDBACK = {};
const PUBLISHED = {};
const FOOTERS = {};   // video_id -> {file: 'keep'|'reject'}
let CURRENT_VID = null;

document.addEventListener('click', async e => {
  const b = e.target.closest('button[data-vote]');
  if (!b) return;
  const card = b.closest('.clip');
  const vid = card.dataset.vid, file = card.dataset.file;
  const want = b.dataset.vote;
  const now = ((FEEDBACK[vid]||{})[file]||{}).verdict || '';
  // Removing is INSTANT and the note comes after. Demanding an explanation
  // before you may delete a clip is the most friction in the app for a
  // first-time user; the note still gets recorded whenever it is given.
  await sendVerdict(card, now === want ? null : want, '');
});

async function sendVerdict(card, verdict, reason){
  const vid = card.dataset.vid, file = card.dataset.file;
  try {
    const r = await fetch('/api/feedback', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({video_id: vid, file, verdict, reason})
    });
    const j = await r.json();
    FEEDBACK[vid] = j.feedback || {};
    paintVerdict(card, vid, file);
    updateAllBar(vid);
  } catch (err) {}
}

function paintVerdict(card, vid, file){
  const fb = (FEEDBACK[vid]||{})[file] || {};
  const v = fb.verdict || '';
  const rejBtn = card.querySelector('[data-vote=reject]');
  rejBtn.className = v==='reject' ? 'rej on-rej' : 'rej';
  rejBtn.textContent = v==='reject' ? 'Removed · undo' : 'Remove';
  card.classList.toggle('rejected', v==='reject');
  card.classList.toggle('queued', v==='queued');

  // The note box only exists for a removed clip, and keeps whatever was
  // written so a decision made last week still explains itself.
  const why = card.querySelector('.why');
  why.hidden = v !== 'reject';
  why.querySelector('textarea').value = fb.reason || '';
}

// "Save note" attaches a reason to an already-removed clip.
document.addEventListener('click', async e => {
  const submit = e.target.closest('.why .wsubmit');
  if (!submit) return;
  const why = submit.closest('.why');
  const was = submit.textContent;
  await sendVerdict(submit.closest('.clip'), 'reject',
                    why.querySelector('textarea').value.trim());
  submit.textContent = 'Saved';
  setTimeout(() => { submit.textContent = was; }, 1200);
});

// Upload the queued clips to YouTube, each with a publish time. YouTube
// itself flips them public at that moment, which is why none of this needs
// a server: the laptop only has to be awake long enough to upload.
document.addEventListener('click', async e => {
  const b = e.target.closest('button[data-yt]');
  if (!b) return;
  YT_VID = b.dataset.yt;
  $('ytmodal').hidden = false;
  ytRefresh();
});

// Deleting removes files. The sheet names exactly what goes, and what
// does NOT - clips already on YouTube are not ours to remove.
let DEL_VID = null;
document.addEventListener('click', e => {
  const b = e.target.closest('button[data-del]');
  if (!b) return;
  DEL_VID = b.dataset.del;
  const cards = document.querySelectorAll(`.clip[data-vid="${DEL_VID}"]`).length;
  const posted = Object.values(PUBLISHED[DEL_VID] || {})
    .filter(x => x.status === 'posted').length;
  $('delsum').textContent =
    `${cards} clip${cards===1?'':'s'} and their titles and descriptions will `
    + `be deleted from this computer.`;
  const kept = $('delkept');
  if (posted) {
    kept.textContent = `${posted} of them ${posted===1?'is':'are'} already on `
      + `YouTube. Those stay up — delete them in YouTube Studio if you `
      + `want them gone.`;
    kept.hidden = false;
  } else { kept.hidden = true; }
  $('delmodal').hidden = false;
});
$('delcancel').onclick = () => { $('delmodal').hidden = true; };
$('delmodal').onclick = e => {
  if (e.target === $('delmodal')) $('delmodal').hidden = true;
};
$('delgo').onclick = async () => {
  const btn = $('delgo');
  btn.disabled = true; btn.textContent = 'Deleting…';
  try {
    const r = await fetch('/api/delete', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({video_id: DEL_VID})
    });
    const j = await r.json();
    if (j.ok) { $('delmodal').hidden = true; library(); }
    else { $('delsum').textContent = j.error || 'Could not delete it.'; }
  } catch (err) {}
  btn.disabled = false; btn.textContent = 'Delete';
};

// One footer for the whole project. Bulk-editing sixteen descriptions to
// paste the same line into each is the wrong shape for the actual job -
// and it would duplicate the moment you ran it twice.
let FT_VID = null;
document.addEventListener('click', e => {
  const b = e.target.closest('button[data-ft]');
  if (!b) return;
  FT_VID = b.dataset.ft;
  $('fttext').value = FOOTERS[FT_VID] || '';
  $('ftmodal').hidden = false;
  $('fttext').focus();
});
$('ftcancel').onclick = () => { $('ftmodal').hidden = true; };
$('ftmodal').onclick = e => {
  if (e.target === $('ftmodal')) $('ftmodal').hidden = true;
};
$('ftsave').onclick = async () => {
  const btn = $('ftsave');
  btn.disabled = true; btn.textContent = 'Saving…';
  try {
    const r = await fetch('/api/project', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({video_id: FT_VID, footer: $('fttext').value})
    });
    const j = await r.json();
    if (j.ok){ FOOTERS[FT_VID] = j.footer; $('ftmodal').hidden = true;
               library(); }
  } catch (err) {}
  btn.disabled = false; btn.textContent = 'Save';
};

// Edit title / hashtags / description. The card's own text is what gets
// edited, so the sheet is seeded from the live manifest values rather than
// from anything cached.
let ED = {vid: null, file: null};
document.addEventListener('click', e => {
  const b = e.target.closest('button[data-edit]');
  if (!b) return;
  ED = {vid: b.dataset.vid, file: b.dataset.edit};
  const card = b.closest('.clip');
  const full = card.querySelector('.hook')?.title || '';
  const tags = card.querySelector('.tags')?.title || '';
  $('edtitle').value = full;
  $('edtags').value = tags;
  $('eddesc').value = card.querySelector('.desc')?.title || '';

  // Editing does not reach a video that has already been uploaded - say so
  // here rather than let someone discover it after the fact.
  const pub = (PUBLISHED[ED.vid] || {})[ED.file];
  const note = $('ednote');
  if (pub && pub.status === 'posted') {
    note.textContent = 'This clip is already on YouTube. Editing here will '
      + 'not change the video that was uploaded — change that in YouTube Studio.';
    note.hidden = false;
  } else { note.hidden = true; }

  $('edmodal').hidden = false;
  $('edtitle').focus();
});
$('edcancel').onclick = () => { $('edmodal').hidden = true; };
$('edmodal').onclick = e => {
  if (e.target === $('edmodal')) $('edmodal').hidden = true;
};
$('edsave').onclick = async () => {
  const btn = $('edsave');
  btn.disabled = true; btn.textContent = 'Saving…';
  try {
    const r = await fetch('/api/clip', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({video_id: ED.vid, file: ED.file,
        title: $('edtitle').value, tags: $('edtags').value,
        description: $('eddesc').value})
    });
    const j = await r.json();
    if (j.ok) { $('edmodal').hidden = true; library(); }
    else { $('ednote').textContent = j.error || 'Could not save.';
           $('ednote').hidden = false; }
  } catch (err) {}
  btn.disabled = false; btn.textContent = 'Save';
};

// Download: one button, two choices in a sheet. Two buttons side by side
// made the user weigh a decision before knowing what either meant.
let DL_VID = null;
document.addEventListener('click', e => {
  const b = e.target.closest('button[data-dl]');
  if (!b) return;
  DL_VID = b.dataset.dl;
  $('dlmodal').hidden = false;
});
$('dlcancel').onclick = () => { $('dlmodal').hidden = true; };
$('dlmodal').onclick = e => {
  if (e.target === $('dlmodal')) $('dlmodal').hidden = true;
};
$('dlfull').onclick = () => {
  $('dlmodal').hidden = true;
  window.location = `/bundle/${encodeURIComponent(DL_VID)}`;
};
$('dlvids').onclick = () => {
  $('dlmodal').hidden = true;
  window.location = `/zip/${encodeURIComponent(DL_VID)}`;
};

// The sheet asks first. Uploading cannot be undone from here - the token
// only carries the upload scope - so the confirm step is the only place a
// mistake can still be caught.
let YT_VID = null;
let YT_SLOTS = [];      // publish times, fixed
let YT_ORDER = [];      // which clip sits in each slot, draggable

function ytDraw(){
  $('ytlist').innerHTML = YT_ORDER.map((c, i) => {
    const d = new Date(YT_SLOTS[i]);
    const when = d.toLocaleString([], {weekday:'short', day:'numeric',
      month:'short', hour:'numeric', minute:'2-digit'});
    return `<li draggable="true" data-i="${i}">`
      + `<span class="grip" aria-hidden="true">&#8942;&#8942;</span>`
      + `<span class="slot">${i + 1}</span>`
      + `<span class="sl-when">${when}</span>`
      + `<span class="sl-title">${c.title.replace(/[<>&]/g,'').slice(0,60)}</span>`
      + `</li>`;
  }).join('');
}

// Native HTML5 drag - no library, and the list is at most six rows.
let DRAG_FROM = null;
document.addEventListener('dragstart', e => {
  const li = e.target.closest('#ytlist li');
  if (!li) return;
  DRAG_FROM = +li.dataset.i;
  li.classList.add('dragging');
  e.dataTransfer.effectAllowed = 'move';
});
document.addEventListener('dragover', e => {
  const li = e.target.closest('#ytlist li');
  if (!li || DRAG_FROM === null) return;
  e.preventDefault();                      // required or drop never fires
  const to = +li.dataset.i;
  if (to === DRAG_FROM) return;
  const [moved] = YT_ORDER.splice(DRAG_FROM, 1);
  YT_ORDER.splice(to, 0, moved);
  DRAG_FROM = to;
  ytDraw();
  const el = $('ytlist').querySelector(`li[data-i="${to}"]`);
  if (el) el.classList.add('dragging');
});
document.addEventListener('dragend', () => {
  DRAG_FROM = null;
  $('ytlist').querySelectorAll('li').forEach(l => l.classList.remove('dragging'));
});

function ytOpts(){
  const [h, m] = ($('ythour').value || '09:00').split(':').map(Number);
  return {gap_hours: parseFloat($('ytgap').value) || 4,
          hour: h || 0, minute: m || 0};
}

async function ytRefresh(){
  const go = $('ytgo');
  go.disabled = true;
  $('ytlist').innerHTML = '';
  $('ytsum').className = 'ytsum';
  $('ytsum').textContent = 'Working it out…';
  try {
    const r = await fetch('/api/youtube/plan', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({video_id: YT_VID, ...ytOpts()})
    });
    const j = await r.json();
    if (j.error){ $('ytsum').textContent = j.error; return; }

    if (!j.will_upload){
      $('ytsum').className = 'ytsum ytwarn';
      // Same plain language as the footnote - "midnight Pacific" is a
      // clock the user does not keep, so say when in THEIR time.
      const reset = j.quota_resets
        ? new Date(j.quota_resets).toLocaleString([], {weekday:'long',
            hour:'numeric', minute:'2-digit'})
        : 'tomorrow';
      $('ytsum').textContent = j.used_today >= j.max_per_day
        ? `You've scheduled ${j.used_today} clips today, which is YouTube's `
          + `daily limit. Open this again after ${reset}.`
        : 'Every clip in this project is already on YouTube.';
      $('ytleft').hidden = true;
      return;
    }

    // Lead with what IS happening. Everything else - what already went
    // out, what did not fit - is a footnote under the list, not part of
    // the headline. The old version opened with two exceptions and made
    // the user assemble the actual answer from them.
    $('ytsum').innerHTML =
      `<b>These ${j.will_upload} clip${j.will_upload===1?'':'s'} will be scheduled:</b>`;

    // The real reset time, in the user's own clock. "tomorrow" is not
    // actionable and is often wrong - the quota rolls over at midnight
    // US/Pacific, which lands at 2 or 3am here depending on DST, so
    // "tomorrow" could mean tonight.
    const reset = j.quota_resets
      ? new Date(j.quota_resets).toLocaleString([], {weekday:'long',
          month:'short', day:'numeric', hour:'numeric', minute:'2-digit'})
      : 'tomorrow';
    const notes = [];
    if (j.deferred)
      notes.push(`${j.deferred} clip${j.deferred===1?'':'s'} won’t fit today. `
        + `YouTube caps uploads at ${j.max_per_day} a day — open this again `
        + `after <b>${reset}</b> to schedule the rest.`);
    if (j.already_posted)
      notes.push(`${j.already_posted} clip${j.already_posted===1?'':'s'} `
        + `already went to YouTube and ${j.already_posted===1?'is':'are'} skipped here.`);
    $('ytleft').innerHTML = notes.join('<br><br>');
    $('ytleft').hidden = !notes.length;

    // The TIMES are the schedule and stay put; dragging moves a clip
    // between slots. So slot 1 is always the earliest and reordering is
    // "what opens", not "when does this one go" - one idea, not two.
    YT_SLOTS = j.items.map(it => it.when);
    YT_ORDER = j.items.map(it => ({file: it.file, title: it.title || ''}));
    $('draghint').hidden = YT_ORDER.length < 2;
    ytDraw();
    go.textContent = `Upload ${j.will_upload}`;
    go.disabled = false;
  } catch (err) { $('ytsum').textContent = 'Could not work out a schedule.'; }
}

$('ytgap').onchange = ytRefresh;
$('ythour').onchange = ytRefresh;
$('ytcancel').onclick = () => { $('ytmodal').hidden = true; };
$('ytmodal').onclick = e => {
  if (e.target === $('ytmodal')) $('ytmodal').hidden = true;
};

$('ytgo').onclick = async () => {
  const go = $('ytgo');
  go.disabled = true; go.textContent = 'Uploading…';
  try {
    const r = await fetch('/api/youtube', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({video_id: YT_VID, ...ytOpts(),
                            order: YT_ORDER.map(c => c.file)})
    });
    const j = await r.json();
    const note = document.querySelector(`.pnote[data-vid="${YT_VID}"]`);
    const n = (j.scheduled || []).length;
    const bits = [];
    if (n) bits.push(`${n} scheduled on YouTube`);
    if (j.deferred && j.deferred.length) bits.push(`${j.deferred.length} waiting for tomorrow`);
    if (j.failed && j.failed.length) bits.push(`${j.failed.length} failed`);
    if (note) note.textContent = bits.join(' · ') || (j.error || 'Nothing uploaded');
    console.log('youtube', j);
  } catch (err) {}
  $('ytmodal').hidden = true;
  go.textContent = 'Upload';
};

function updateAllBar(vid){
  // Refresh just this project's note - how many clips the zip will skip.
  const fb = FEEDBACK[vid] || {};
  const vals = Object.values(fb);
  const removed = vals.filter(x => x.verdict==='reject').length;
  const queued = vals.filter(x => x.verdict==='queued').length;
  const cards = document.querySelectorAll(`.clip[data-vid="${vid}"]`).length;
  const keeping = Math.max(cards - removed, 0);



  const note = document.querySelector(`.pnote[data-vid="${vid}"]`);
  if (!note) return;
  const bits = [];
  if (removed) bits.push(`${removed} removed`);
  note.textContent = bits.join(' · ');
}

$('go').onclick = async () => {
  const source = $('src').value.trim();
  if (!source) { $('src').focus(); return; }
  $('go').disabled = true;
  $('log').textContent = 'Starting…';
  const r = await fetch('/api/run', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ source, instructions: $('say').value,
                           captions: $('caps').checked,
                           swap: $('swap').checked })
  });
  const j = await r.json();
  if (j.error) { $('log').textContent = j.error; $('go').disabled = false; }
};

// The title and its hashtags are one string in the manifest, and rendering
// them as one blob is why the title didn't read as a title - it arrived as
// "Why a four-star hotel will never have a laundry room #advice
// #wordsofwisdom #lol" wrapped over three lines of small text. Posting needs
// them separately anyway, so split here rather than changing the manifest -
// this way old runs render correctly too.
function splitTitle(s){
  const raw = (s || '').trim();
  const i = raw.search(/(^|\s)#\S/);
  if (i < 0) return {text: raw, tags: ''};
  return {text: raw.slice(0, i).trim(), tags: raw.slice(i).trim()};
}

// A clip already on YouTube must read differently at a glance. Without
// that the grid gives no way to tell what has gone out, which is how six
// accidental uploads sat unnoticed among sixteen identical cards.
function pubState(p){
  if (!p || p.status !== 'posted') return 'bad';
  if (!p.publish_at) return 'priv';
  return new Date(p.publish_at) > new Date() ? 'soon' : 'done';
}
function pubWord(p){
  return {bad:'FAILED', priv:'ON YOUTUBE', soon:'SCHEDULED',
          done:'PUBLISHED'}[pubState(p)];
}
function pubWhen(p){
  const st = pubState(p);
  if (st === 'bad') return 'upload did not complete';
  if (st === 'priv') return 'private, no publish date';
  const d = new Date(p.publish_at);
  return d.toLocaleString([], {weekday:'short', day:'numeric', month:'short',
    hour:'numeric', minute:'2-digit'});
}

document.addEventListener('timeupdate', e => {
  const v = e.target;
  if (!v.matches || !v.matches('.media video')) return;
  const tc = v.parentElement.querySelector('.tc');
  if (!tc) return;
  const m = n => `${Math.floor(n/60)}:${String(Math.floor(n%60)).padStart(2,'0')}`;
  // Trust the file's real duration once known; fall back to the manifest.
  const total = isFinite(v.duration) && v.duration
    ? m(v.duration) : tc.dataset.total;
  tc.textContent = `${m(v.currentTime)} / ${total}`;
}, true);

function card(c, vid){
  const src = `/clips/${encodeURIComponent(vid)}/${encodeURIComponent(c.file)}`;
  const esc = s => (s||'').replace(/[<>&]/g,'');
  const clock = s => {
    s = Math.round(s||0);
    return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
  };
  const fb = (FEEDBACK[vid]||{})[c.file] || {};
  const v = fb.verdict || '', r = fb.reason || '';
  const pub = (PUBLISHED[vid]||{})[c.file] || null;
  const live = pub && pub.status === 'posted';
  const full = (c.title || c.hook || '').trim();
  const t = splitTitle(full);
  return `<div class="clip${v==='reject'?' rejected':''}${v==='queued'?' queued':''}${live?' onair':''}" data-vid="${esc(vid)}" data-file="${esc(c.file)}">
    <div class="media">
      <video src="${src}" controls preload="metadata"
        ${c.thumb?`poster="/clips/${encodeURIComponent(vid)}/${encodeURIComponent(c.thumb)}"`:''}></video>
      ${live?`<div class="stamp ${pubState(pub)}">
        <b>${pubWord(pub)}</b><span>${pubWhen(pub)}</span></div>`:''}
      <span class="tc" data-total="${clock(c.duration)}">0:00 / ${clock(c.duration)}</span>
    </div>
    <div class="meta">
      <p class="hook clamp" title="${esc(t.text)}">${esc(t.text)}</p>
      <p class="tags clamp" title="${esc(t.tags)}">${esc(t.tags)}</p>
      <p class="desc" title="${esc(c.description||'')}">${esc(c.description||'')}${
        FOOTERS[vid] ? `<span class="foot">${esc(FOOTERS[vid])}</span>` : ''}</p>
      <div class="stats">
        <span class="from">from ${clock(c.start)}</span>
        ${c.mode?`<span class="kind">${esc(c.mode)}</span>`:''}</div>
      <div class="acts">
        <a class="btn dl" href="${src}" download="${esc(c.file)}">Download</a>
      </div>
      <div class="vote">
        <button class="edit" data-edit="${esc(c.file)}" data-vid="${esc(vid)}">Edit</button>
        <button data-vote="reject" class="rej${v==='reject'?' on-rej':''}">${v==='reject'?'Removed · undo':'Remove'}</button>
      </div>
      <div class="why"${v==='reject'?'':' hidden'}>
        <textarea rows="2" placeholder="Why? (optional)">${esc(r)}</textarea>
        <div class="whyacts">
          <button class="btn ghost wsubmit">Save note</button>
        </div>
      </div>
      <details><summary>transcript</summary><p>${esc(c.transcript)}</p></details>
    </div></div>`;
}

// A project is one source video. Native <details> does the collapsing, so it
// stays keyboard-accessible; open/closed is remembered in renderLibrary.
function projectFolder(s, open){
  const esc = t => (t||'').replace(/[<>&]/g,'');
  const clock = n => {
    n = Math.round(n||0);
    return `${Math.floor(n/60)}:${String(n%60).padStart(2,'0')}`;
  };
  const d = new Date((s.created||0) * 1000);
  const when = {
    date: d.toLocaleDateString([], {month:'short', day:'numeric'}),
    time: d.toLocaleTimeString([], {hour:'numeric', minute:'2-digit'}),
  };
  // A URL is the useful name; a local path is only useful as its filename.
  const src = s.source || '';
  const name = /^https?:/i.test(src) ? src : (src.split(/[\\/]/).pop() || s.video_id);
  const rejected = Object.values(s.feedback||{}).filter(x => x==='reject').length;

  return `<details class="proj" data-vid="${esc(s.video_id)}" ${open?'open':''}>
    <summary>
      <span class="pname">${esc(name)}</span>
      <span class="pmeta">
        <b class="pcount">${s.count} clip${s.count===1?'':'s'}</b>
        <span class="pdate">${esc(when.date)}</span>
        <span class="ptime">${esc(when.time)}</span>
        ${s.mode==='compilation'?'<span class="pkindx">compilation</span>':''}
      </span>
    </summary>
    <p class="phow">Every clip below is included. <b>Remove</b> any you don't
      want, then <b>schedule</b> the rest.</p>
    <div class="pbar">
      <button class="btn yt" data-yt="${esc(s.video_id)}">Schedule on YouTube…</button>
      <button class="btn ghost dl" data-dl="${esc(s.video_id)}">Download…</button>
      <button class="btn ghost ft" data-ft="${esc(s.video_id)}">Edit All</button>
      <button class="btn trash" data-del="${esc(s.video_id)}" title="Delete this project"
        aria-label="Delete this project">&#128465;</button>
      <span class="pnote" data-vid="${esc(s.video_id)}"></span>
    </div>
    <div class="grid">${s.clips.map(c => card(c, s.video_id)).join('')}</div>
  </details>`;
}

// Folders start closed and stay however the user last left them. Every edit,
// delete and feedback click re-renders the whole library, so without this
// the folder a user was working in snapped shut under them.
const OPEN_KEY = 'cyv.openProjects';
function openSet(){
  try { return new Set(JSON.parse(localStorage.getItem(OPEN_KEY) || '[]')); }
  catch { return new Set(); }
}
function saveOpen(vid, isOpen){
  const s = openSet();
  isOpen ? s.add(vid) : s.delete(vid);
  try { localStorage.setItem(OPEN_KEY, JSON.stringify([...s])); } catch {}
}
// 'toggle' does not bubble, so listen in the capture phase.
document.addEventListener('toggle', e => {
  const el = e.target;
  if (el.classList && el.classList.contains('proj') && el.dataset.vid)
    saveOpen(el.dataset.vid, el.open);
}, true);

function renderLibrary(sets, expand){
  sets.forEach(s => { FEEDBACK[s.video_id] = s.feedback || {};
                      PUBLISHED[s.video_id] = s.published || {};
                      FOOTERS[s.video_id] = s.footer || ''; });
  // expand is only passed for a run that just finished - the user asked for
  // those clips, so that one opens.
  if (expand) saveOpen(expand, true);
  const open = openSet();
  $('projects').innerHTML = sets
    .map(s => projectFolder(s, open.has(s.video_id)))
    .join('');
  $('noclips').style.display = sets.length ? 'none' : 'block';
  sets.forEach(s => updateAllBar(s.video_id));
}

async function library(expand){
  const lib = await (await fetch('/api/library')).json();
  renderLibrary(lib, expand);
}

// --- the ring ---------------------------------------------------------
// Eight stages is only eight jumps, and transcription alone can be minutes,
// so a percentage driven by step/steps sits frozen for most of the run -
// exactly the "feels stuck" problem this replaces. Two fixes:
//
//  1. WEIGHTS, not equal steps. Downloading and transcribing and rendering
//     are the run; "Checking the video" and "Finishing up" are instants.
//     Weighted, 33% means roughly a third of the WORK is done.
//  2. TIME INSIDE A STAGE. Each stage owns a band of the circle, and the
//     dial eases across its own band while the stage runs: 1-exp(-t/tau).
//     Asymptotic, so it always creeps forward, never stalls, and can never
//     spill into the next stage's band no matter how long the stage takes.
const WEIGHTS = [15, 2, 40, 6, 12, 5, 35, 1];   // parallel to _STAGES
const CIRC = 326.7;                             // 2*pi*52, matches the CSS
let stageAt = 0, stageWas = -1;

function mmss(n){
  n = Math.round(n || 0);
  return `${Math.floor(n/60)}:${String(n%60).padStart(2,'0')}`;
}

function ringPct(s){
  const w = WEIGHTS.slice();
  // A cached transcript skips the slow part entirely; leaving it weighted at
  // 40 would park the ring at 15% through a run that is nearly done.
  if (s.cached) w[2] = 2;
  const total = w.reduce((a,b) => a+b, 0);
  const step = Math.min(s.step || 0, w.length);

  let before = 0;
  for (let i = 0; i < step - 1; i++) before += w[i];
  const band = step > 0 ? w[step-1] : w[0];

  // Transcription is the one stage that reports REAL progress - it prints
  // where it is in the source. Use that instead of a curve; on a 67-min
  // stream the difference is a dial that means something.
  if (step === 3 && s.heard_frac != null){
    return (before + band * Math.min(1, s.heard_frac)) / total;
  }

  // Rendering reports real progress too, now that clips publish one by one:
  // made/to_make is an exact fraction, so the biggest remaining band stops
  // being a guess as well.
  if (step === 7 && s.to_make){
    return (before + band * Math.min(1, (s.made || 0) / s.to_make)) / total;
  }

  // Every other stage: ease across its own band on elapsed time. Expected
  // duration comes from the pipeline's ETA, split by this stage's weight.
  const runSecs = (s.eta_min ? s.eta_min * 60 : 300);
  const expect = Math.max(3, runSecs * band / total);
  const inStage = Math.max(0, (s.elapsed || 0) - stageAt);
  const eased = 1 - Math.exp(-inStage / (expect * 0.6));

  return (before + band * eased) / total;
}

// "about 16 min" was a static guess baked into the pipeline, and it was ~40%
// low on a 67-min stream because it assumed transcription runs at 8x
// realtime. Once there is a real fraction and real elapsed time, t/p beats
// any constant and corrects itself as the run goes.
function etaText(s, frac){
  const t = s.elapsed || 0;
  if (frac > 0.04 && t > 25){
    const left = Math.round((t / frac - t) / 60);
    if (left >= 1) return ` · about ${left} min left`;
    return ' · nearly done';
  }
  return s.eta_min ? ` · about ${s.eta_min} min total` : '';
}

function paintRing(frac, label, state){
  const p = Math.max(0, Math.min(1, frac));
  $('dial').style.strokeDashoffset = (CIRC * (1 - p)).toFixed(1);
  $('pct').textContent = Math.round(p * 100) + '%';
  const st = $('status');
  st.textContent = label;
  st.className = 'status ' + state;
  $('prog').className = 'prog' +
    (state === 'done' ? ' done' : state === 'err' ? ' err' : '');
}

// Clips appear as they finish rendering rather than all at once at the end.
// APPEND ONLY - rewriting the grid's innerHTML each second would restart any
// video the user had started playing and throw away their scroll position,
// which is worse than the wait it replaces.
let liveShown = 0, liveVid = null;

function liveClips(s){
  const grid = $('livegrid');
  const clips = s.clips || [];
  if (s.status !== 'running' || !clips.length){
    // The finished run is rendered by the library below, so the live grid
    // clears out rather than showing every clip twice.
    if (s.status !== 'running' && liveShown){ grid.innerHTML = ''; liveShown = 0; }
    return;
  }
  if (s.video_id !== liveVid){ grid.innerHTML = ''; liveShown = 0; liveVid = s.video_id; }
  // A clip's own directory is not known until the manifest lands, so during
  // the run the files are served from the job's video_id.
  const vid = s.video_id || liveVid;
  for (let i = liveShown; i < clips.length; i++){
    grid.insertAdjacentHTML('beforeend', card(clips[i], vid));
  }
  liveShown = clips.length;
}

async function poll(){
  try{
    const s = await (await fetch('/api/status')).json();
    if (s.status !== 'idle') $('runpanel').hidden = false;
    // Elapsed-at-stage-entry, so interpolation measures time in THIS stage.
    if ((s.step || 0) !== stageWas){ stageWas = s.step || 0; stageAt = s.elapsed || 0; }

    if (s.status === 'queued'){
      running = true; $('go').disabled = true;
      paintRing(0, `Waiting for ${s.queued||1} run${(s.queued||1)===1?'':'s'} ahead…`, 'run');
    } else if (s.status === 'running'){
      running = true; $('go').disabled = true;
      const frac = ringPct(s);
      const q = s.queued ? ` · ${s.queued} more queued` : '';
      const hd = (s.step === 3 && s.heard_secs)
        ? ` · ${mmss(Math.round(s.heard_secs))} of ${mmss(Math.round(s.source_secs||0))} heard`
        : (s.step === 7 && s.to_make)
        ? ` · ${s.made||0} of ${s.to_make} clips ready` : '';
      paintRing(frac, (s.stage || 'Starting') + '…' + etaText(s, frac) + hd + q, 'run');
    } else if (s.status === 'done'){
      paintRing(1, `Done — ${s.clips.length} clip${s.clips.length===1?'':'s'} ready`, 'done');
      $('go').disabled = false;
      // Reload the whole library so the finished run appears as a project
      // folder, expanded, rather than as a detached grid.
      if (running){ library(s.video_id); running = false; }
    } else if (s.status === 'error'){
      paintRing(1, s.error || 'Something went wrong.', 'err');
      $('details').open = true;
      $('go').disabled = false; running = false;
    }
    const setNote = (id, label, val) => {
      const el = $(id);
      if (val){ el.textContent = label + val; el.hidden = false; }
      else { el.hidden = true; }
    };
    setNote('warning', '', s.warning);
    setNote('applied', 'Applied: ', s.applied);
    setNote('ignored', "Couldn't apply: ", s.ignored);

    if (s.log && s.log.length) $('log').textContent = s.log.join('\n');
    const pre = $('log'); if (running) pre.scrollTop = pre.scrollHeight;
    liveClips(s);
  }catch(e){ console.error('poll', e); }   // silent here once killed the
  setTimeout(poll, 1000);  // whole UI after a removed element left a dead ref

}
library(); poll();
</script>
</div></body></html>
"""


def _render_page() -> str:
    """Render the page."""
    accepted = "".join(f"<li>{x}</li>" for x in llm.ACCEPTED_INSTRUCTIONS)
    refused = "".join(f"<li>{x}</li>" for x in llm.NOT_ACCEPTED)
    return (PAGE.replace("__ACCEPTED__", accepted)
                .replace("__NOT_ACCEPTED__", refused))


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    config.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{PORT}"
    print(f"ClipYourVideos.com  ->  {url}")
    free = transcribe.free_ram_gb()
    if free:
        print(f"Free RAM: {free:.1f} GB  (model default: {config.WHISPER_MODEL})")
    print("Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    with _Server(("127.0.0.1", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
