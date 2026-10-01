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
import sys
import threading
import time
import urllib.parse
import webbrowser
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import studio

from . import (config, cut, llm, modes, pipeline, publish, transcribe,
               youtube)

PORT = int(os.environ.get("CLIP_PORT", "8420"))

# Where main() logs to when it isn't attached to a visible console - the
# whole reason the Desktop launcher can start it hidden (pythonw, no
# window) without losing the diagnostics a crash needs.
LOG_FILE = Path.home() / ".config" / "clip-machine" / "server.log"

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
    """Turn an exception into something worth showing a person.

    yt-dlp failures used to be one bucket - "make sure the link is public" -
    whether the video really was private or the Wi-Fi just dropped mid-
    download. Both raise the same CalledProcessError with the same generic
    "exit status 1" text, so telling them apart needs yt-dlp's own stderr,
    which download.py now captures onto the exception (see there).
    """
    import subprocess as _sp

    text = f"{type(err).__name__}: {err}".lower()
    blob = " ".join(log[-12:]).lower()
    # yt-dlp's real diagnostic text, when this is a yt-dlp failure - "exit
    # status 1" tells a person nothing; this does.
    yt_stderr = (getattr(err, "stderr", "") or "").lower()

    if isinstance(err, FileNotFoundError):
        return "Couldn't find that file. Check the path and try again."
    if "no speech" in text:
        return "No speech was detected in that video."

    is_ytdlp = (isinstance(err, _sp.CalledProcessError)
                and err.cmd and "yt-dlp" in str(err.cmd[0]))
    if is_ytdlp or any(k in text + blob for k in
                       ("yt-dlp", "unsupported url", "is not a valid url")):
        # Network trouble: yt-dlp already retries internally and resumes
        # from the partial file it kept, so the honest instruction really
        # is just "try again" - not "check the link", which was never the
        # problem and sent Jaylen debugging the wrong thing on 2026-09-13.
        if any(k in yt_stderr for k in
               ("getaddrinfo failed", "connection reset", "network is unreachable",
                "read timed out", "timed out", "connection aborted",
                "remote end closed", "errno 11001", "errno 111")):
            return ("Lost the internet connection partway through. Click "
                    "Retry - it picks up from where it stopped.")
        if any(k in yt_stderr + blob for k in
               ("private video", "video unavailable", "sign in to confirm",
                "members-only", "this video is unavailable", "removed by",
                "account associated", "http error 4")):
            return ("Couldn't download that video. Make sure the link is "
                    "public and playable without signing in.")
        # A real yt-dlp failure we don't have a specific message for -
        # surface its own first line rather than guess at a cause.
        if yt_stderr:
            first = yt_stderr.strip().splitlines()[-1][:160]
            return f"Couldn't download that video. yt-dlp said: {first}"
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
        _stay_awake(True)
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
        _stay_awake(False)
        # The Whisper model is the biggest thing this process holds - free
        # it now rather than let it sit resident until the next run, which
        # is what made the idle server a memory-pressure target.
        transcribe.unload_model()


def _stay_awake(on: bool) -> None:
    """Best-effort: stop Windows idle-sleeping while a run is in progress.

    Idle sleep pauses this whole process - indistinguishable from a hang -
    and has already cost one run its download mid-transfer. Windows-only,
    silently does nothing elsewhere, and must never be allowed to break a
    run by raising. Does NOT stop a lid-close sleep; that needs an admin
    policy change on a school-managed laptop, out of reach here.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


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


def _start_date(raw) -> str | None:
    """A "YYYY-MM-DD" from the sheet, never earlier than today.

    None means "tomorrow", which is what scheduling did before the sheet
    had a date picker.
    """
    try:
        want = datetime.fromisoformat(str(raw)).date()
    except (TypeError, ValueError):
        return None
    return max(want, datetime.now().astimezone().date()).isoformat()


# --------------------------------------------------------------------------
# single-clip upload - the card's Upload button
#
# One at a time, in the background. The API is used while today's quota
# lasts; after that, Studio in a visible Chrome window. The person clicks the
# same buttons either way. Studio uploads are recorded in the ledger here,
# tagged via=studio so they are not counted against the API quota.
# --------------------------------------------------------------------------
_upload_lock = threading.Lock()
_upload: dict = {"status": "idle"}


def _upload_worker(vid: str, clip: dict, footer: str, publish_at) -> None:
    path = config.CLIPS_DIR / vid / clip["file"]
    title = publish.youtube_title(clip.get("title", ""))
    desc = publish.full_description(clip, footer)
    tags = publish.hashtag_list(clip.get("title", ""))
    cid = publish.clip_id(vid, clip.get("start", -1))

    def step(msg: str) -> None:
        _upload["message"] = msg

    try:
        if (youtube.configured()
                and youtube.uploads_today() < youtube.MAX_UPLOADS_PER_DAY):
            _upload["via"] = "api"
            step("Uploading to YouTube…")
            # upload() writes its own ledger row, success or failure.
            res = youtube.upload(path, title, desc, privacy="public",
                                 video_id=vid, clip_id=cid, tags=tags,
                                 publish_at=publish_at)
        else:
            _upload["via"] = "studio"
            step("Opening YouTube Studio in Chrome…")
            res = studio.upload(path, title, desc, tags,
                                publish_at=publish_at, on_step=step)
            note = "via=studio privacy=" + ("private" if publish_at else "public")
            if publish_at:
                note += " publishAt=" + publish_at.astimezone(
                    timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            publish.record(vid, cid, clip["file"], "youtube", "posted",
                           url=res.get("url"), note=note)
        _upload.update(status="done", url=res.get("url"), message="Done")
    except Exception as e:
        if _upload.get("via") == "studio":
            publish.record(vid, cid, clip["file"], "youtube", "failed",
                           note=f"via=studio {type(e).__name__}: {e}"[:300])
        msg = (str(e) if isinstance(e, studio.StudioError)
               else f"Upload didn't finish: {type(e).__name__}: {e}")
        _upload.update(status="error", message=msg[:300])


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
        published = publish.by_file(d.name)
        out.append({
            "video_id": d.name,
            "count": len(clips),
            "clips": clips,
            "feedback": feedback,
            "published": published,
            "posted": sum(1 for p in published.values()
                         if p.get("status") == "posted"),
            "footer": m.get("footer", ""),
            "source": m.get("source", ""),
            # What the video actually IS, not the URL it came from - a
            # project used to be named by a 90-character query string.
            # video_title is missing on projects made before this existed;
            # the page falls back to source/video_id for those.
            "video_title": m.get("video_title") or "",
            "video_thumb": m.get("video_thumb") or None,
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
        elif route == "/api/views":
            # One flat call across every project's posted clips, not one
            # per project - view_counts already batches into 1-unit calls,
            # so this is 1-2 Google requests total no matter how many
            # projects or clips are on screen.
            if not youtube.views_configured():
                self._send_json({})
                return
            ids = {}
            for r in publish.history():
                if (r.get("platform") == "youtube" and r.get("status") == "posted"
                        and r.get("url")):
                    vid = r["url"].rstrip("/").split("/")[-1]
                    if vid:
                        ids[vid] = None
            self._send_json(youtube.view_counts(list(ids)))
        elif route == "/api/upload/status":
            left = max(0, youtube.MAX_UPLOADS_PER_DAY - youtube.uploads_today())
            self._send_json({**_upload, "api_left": left})
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
                        "/api/clip", "/api/project", "/api/delete",
                        "/api/upload", "/api/retitle"):
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

        if path == "/api/retitle":
            # Redo titles/descriptions/tags without re-rendering anything -
            # the fix for a run that went out with junk titles because
            # Anthropic credit was out, or that just needs a second pass.
            # Blocking, like /api/youtube: same order of cost (a few Claude
            # calls), same "the browser waits" precedent already in place.
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

            clips = man.get("clips", [])
            published = publish.by_file(vid)
            # Never touch a clip that's already on YouTube - editing here
            # cannot reach the uploaded video, so it would just be a lie -
            # or one a person hand-edited, which was deliberate.
            todo = [(i, c) for i, c in enumerate(clips)
                    if not c.get("edited")
                    and published.get(c.get("file"), {}).get("status") != "posted"]
            if not todo:
                self._send_json({"error": "Every clip here is already on "
                                          "YouTube or was edited by hand - "
                                          "nothing left to rewrite."}, 400)
                return

            result = llm.title_clips(
                [{"duration": c.get("duration", 0), "text": c.get("transcript", "")}
                 for _, c in todo],
                video_title=man.get("video_title"))
            if result is None:
                self._send_json(
                    {"error": "Titles are unavailable right now - check "
                              "Anthropic credit at console.anthropic.com "
                              "and try again."}, 502)
                return
            titles, descs, _tags = result

            changed = 0
            for pos, (_, c) in enumerate(todo):
                new_title = titles.get(pos)
                if not new_title:
                    continue
                c["title"] = new_title
                c["description"] = descs.get(pos, c.get("description", ""))
                changed += 1
            mpath.write_text(json.dumps(man, indent=2), encoding="utf-8")
            self._send_json({"ok": True, "changed": changed,
                             "asked": len(todo),
                             "skipped": len(clips) - len(todo)})
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

        if path == "/api/upload":
            vid = (data.get("video_id") or "").strip()
            file = (data.get("file") or "").strip()
            if (not vid or not file or "/" in file + vid
                    or chr(92) in file + vid or ".." in file + vid):
                self._send_json({"error": "Couldn't tell which clip that was."}, 400)
                return
            mpath = config.CLIPS_DIR / vid / "manifest.json"
            try:
                man = json.loads(mpath.read_text(encoding="utf-8-sig"))
            except Exception:
                self._send_json({"error": "unknown project"}, 404)
                return
            clip = next((c for c in man.get("clips", [])
                         if c.get("file") == file), None)
            if clip is None:
                self._send_json({"error": "no such clip"}, 404)
                return
            if (publish.by_file(vid).get(file) or {}).get("status") == "posted":
                self._send_json({"error": "This clip is already on YouTube."}, 409)
                return

            publish_at = None
            if data.get("mode") == "schedule":
                try:
                    publish_at = datetime.fromisoformat(str(data.get("when")))
                except (TypeError, ValueError):
                    publish_at = None
                if publish_at is None or publish_at.tzinfo is None:
                    self._send_json({"error": "Pick a date and time."}, 400)
                    return
                # Both Studio and the API reject a time too close to now.
                if publish_at < datetime.now(timezone.utc) + timedelta(minutes=15):
                    self._send_json(
                        {"error": "Pick a time at least 15 minutes from now."}, 400)
                    return

            with _upload_lock:
                if _upload.get("status") == "running":
                    self._send_json({"error": "Another clip is uploading. "
                                              "Wait for it to finish."}, 409)
                    return
                _upload.clear()
                _upload.update(status="running", vid=vid, file=file, via=None,
                               url=None, message="Starting…",
                               started=time.time())
            threading.Thread(target=_upload_worker,
                             args=(vid, clip, man.get("footer", ""), publish_at),
                             daemon=True).start()
            self._send_json({"ok": True})
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
            # Same ordering rule as the real upload, so the preview and the
            # upload agree about which clips take the slots.
            order = data.get("order") or []
            if isinstance(order, list) and order:
                rank = {f: i for i, f in enumerate(order) if isinstance(f, str)}
                picked.sort(key=lambda c: rank.get(c.get("file"), len(rank)))
            self._send_json(youtube.plan(picked, vid, gap_hours=gap,
                                         hour=hour, minute=minute,
                                         start_date=_start_date(
                                             data.get("start_date"))))
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
                footer=man.get("footer", ""),
                start_date=_start_date(data.get("start_date")))
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
  .prog{display:flex;align-items:center;gap:18px}
  .retrybtn{display:block;margin:12px 0 0;padding:7px 18px;width:auto}
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
  /* Deliberately NOT line-clamped. A fixed line count only aligns at one
     viewport width - descriptions run 234-291 chars, which is 6 lines on a
     wide card and 8 on a narrow one, so any clamp that fits one truncates
     the other. Instead .stats carries margin-top:auto, so everything from
     the stats row down is pinned to the bottom of the card; the grid
     already stretches sibling cards to equal height, so those rows line up
     across the row at every width and the description absorbs the slack. */
  .desc{margin:0 0 9px;font-size:12.5px;line-height:1.5;color:#a8b6cc;
    min-height:calc(2 * 1.5em)}
  /* One obvious button per card (Upload, or Download once it's posted);
     Edit/Remove/the second Download were equal-weight next to it on all 32
     cards at once - now they only show on hover or keyboard focus. */
  .cardacts{display:flex;flex-direction:column;gap:6px;margin-top:10px}
  .mainrow{display:flex;gap:6px}
  .mainact{display:block;flex:1 1 auto;min-width:0}
  .up.mainact{color:#0a1020;background:var(--accent);border:1px solid var(--accent);
    border-radius:7px;padding:6px;font:600 12px/1.4 inherit;cursor:pointer}
  .up.mainact:hover{filter:brightness(1.12)}
  /* Always-visible way to reach Edit/Remove, not just hover/keyboard-focus -
     a first-time user has no reason to try hovering a card, so hover-only
     made those two actions invisible to anyone who hadn't already learned
     they exist. */
  .more{flex:0 0 auto;width:34px;background:transparent;border:1px solid var(--line);
    color:var(--muted);border-radius:7px;cursor:pointer;font-size:16px;line-height:1}
  .more:hover, .more[aria-expanded="true"]{color:var(--text);border-color:#39435a}
  .morevote{display:flex;gap:8px;opacity:0;pointer-events:none;
    transition:opacity .12s ease}
  .clip:hover .morevote, .clip:focus-within .morevote, .morevote.pinned,
  .morevote.open{opacity:1;pointer-events:auto}
  .morevote a, .morevote button{flex:1;background:transparent;
    border:1px solid var(--line);color:var(--muted);border-radius:7px;padding:6px;
    font:600 12px/1.4 inherit;cursor:pointer;text-align:center;text-decoration:none}
  .morevote a:hover, .morevote button:hover{color:var(--text);border-color:#39435a}
  .postsub{margin:-4px 0 16px;font-size:14px;line-height:1.45;color:var(--muted)}
  .stepq{margin:0 0 10px;font-weight:600;font-size:13px}
  .ytrow input[type=date]{color-scheme:dark}
  .upwhen input{width:100%;margin:4px 0 12px;background:#0f131b;color:var(--text);
    border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit;
    color-scheme:dark}
  .morevote button.edit{color:var(--muted);border-color:transparent;font-weight:500}
  .morevote button.edit:hover{color:var(--text);border-color:var(--line);
    background:#161d2a}
  .morevote button.rej{color:var(--muted);border-color:transparent;font-weight:500}
  .morevote button.rej:hover{color:var(--bad);border-color:#4d2c2b;background:#2a1a1a}
  .morevote button.on-rej{color:var(--bad);border-color:#4d2c2b;background:#2a1a1a;
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
  .stamp .views{display:block;font-weight:700;color:#fff;margin-top:1px}
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
  .proj>summary{cursor:pointer;padding:14px 16px;display:flex;align-items:center;
    gap:14px;flex-wrap:wrap;list-style:none}
  .proj>summary::-webkit-details-marker{display:none}
  .proj>summary::before{content:'\25B8';color:var(--muted);font-size:11px;
    transition:transform .15s ease;display:inline-block}
  .proj[open]>summary::before{transform:rotate(90deg)}
  .proj>summary:hover{background:#181d27}
  .pthumb{width:44px;height:44px;border-radius:6px;object-fit:cover;
    flex:0 0 auto;background:#0a0d13}
  .ptitles{display:flex;flex-direction:column;gap:2px;flex:1 1 auto;min-width:0}
  .pname{font-weight:600;font-size:13px;word-break:break-word}
  .pstatus{font-size:12px;color:var(--muted)}
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
  .ytlist li.waiting{opacity:.6}
  .ytlist li.waiting .sl-when{font-style:italic}
  .ytlist li.sl-split{justify-content:center;background:transparent;border:0;
    color:var(--muted);font-size:11.5px;padding:8px 0 4px;cursor:default}
  .sl-move{margin-left:auto;flex:0 0 auto;background:transparent;
    border:1px solid var(--line);border-radius:6px;color:var(--muted);
    font:inherit;font-size:11.5px;padding:3px 9px;cursor:pointer}
  .sl-move:hover{color:var(--text);border-color:#39435a}
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
  .opts{display:flex;flex-direction:column;gap:10px}
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
  .askhelp{margin-top:8px;display:flex;align-items:center;flex-wrap:wrap;
    gap:6px 8px;font-size:12px}
  .askmore{margin-left:auto;flex:0 0 auto;background:transparent;border:0;
    color:var(--accent);font:inherit;font-size:11.5px;cursor:pointer;
    text-decoration:underline;text-underline-offset:2px;padding:3px 0}
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
    <div class="askhelp">
      <button type="button" class="askmore" id="askmore" aria-expanded="false"
        aria-controls="can">What it can't do</button>
    </div>
    <div class="can" id="can" hidden>
      <p><b>It can act on:</b></p>
      <ul>__ACCEPTED__</ul>
      <p><b>It cannot act on:</b></p>
      <ul>__NOT_ACCEPTED__</ul>
    </div>
  </div>

  <div class="row2">
    <div class="opts">
      <label class="cap"><input type="checkbox" id="caps"> Add captions</label>
      <label class="cap">Framing
        <select id="layout">
          <option value="auto" selected>Let it guess</option>
          <option value="split">Two people, side by side</option>
          <option value="center">One person, single camera</option>
        </select>
      </label>
      <label class="cap"><input type="checkbox" id="swap"> Put the person on the right on top
        <span class="hint">Only for videos with two people side by side.
          Leave unchecked otherwise.</span></label>
    </div>
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
  <button id="retry" class="btn ghost retrybtn" hidden>Retry</button>
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

<div class="modal" id="rtmodal" hidden>
  <div class="sheet">
    <h3>Rewrite titles?</h3>
    <p class="delsum">Asks Claude for a new title, description and tags for
      every clip here that isn't posted yet and hasn't been hand-edited.</p>
    <p class="ytnote">Replaces what's there now. The old titles and
      descriptions can't be recovered.</p>
    <div class="ytacts">
      <button class="btn ghost" id="rtcancel">Cancel</button>
      <button class="btn" id="rtgo">Rewrite</button>
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

<!-- One sheet for both scopes: a card's Upload opens #postsingle (this one
     clip), the project bar's Post to YouTube opens #postbatch (everything
     not yet posted). Same header, same backdrop/Cancel behaviour, same
     place in the user's head - "post to YouTube" is one idea whether it's
     one clip or twelve, and having it be two unrelated-looking sheets was
     the actual confusion, not the scheduling logic underneath either one. -->
<div class="modal" id="postmodal" hidden>
  <div class="sheet">
    <h3>Post to YouTube</h3>
    <p class="postsub" id="postsub" hidden></p>

    <div id="postsingle">
      <div id="upstep1">
        <div class="ytacts">
          <button class="btn ghost" id="upcancel">Cancel</button>
          <button class="btn" id="upnext">Yes, continue</button>
        </div>
      </div>
      <div id="upstep2" hidden>
        <p class="stepq">When should it go live?</p>
        <button class="pick" id="upnow">
          <b>Post now</b>
          <span>Public as soon as it's uploaded</span>
        </button>
        <button class="pick" id="upsched">
          <b>Schedule</b>
          <span>Pick a date and time</span>
        </button>
        <div class="upwhen" id="upwhenrow" hidden>
          <input type="datetime-local" id="upwhen">
          <button class="btn" id="upgo">Schedule it</button>
        </div>
        <p class="ytnote" id="upvia" hidden></p>
        <p class="ytleft" id="uperr" hidden></p>
        <div class="ytacts">
          <button class="btn ghost" id="upback">Back</button>
        </div>
      </div>
    </div>

    <div id="postbatch" hidden>
      <div class="ytrow">
        <label>Every <select id="ytgap">
          <option value="2">2 hours</option>
          <option value="3">3 hours</option>
          <option value="4" selected>4 hours</option>
          <option value="6">6 hours</option>
          <option value="8">8 hours</option>
          <option value="24">day</option>
        </select></label>
        <label>starting <input id="ytdate" type="date"></label>
        <label>at <input id="ythour" type="time" value="09:00"></label>
      </div>
      <p class="ytsum" id="ytsum">Working it out…</p>
      <p class="draghint" id="draghint" hidden>Drag to change the order, or use <b>Take off</b> and <b>Add</b> to choose which ones go today</p>
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

// The "..." button: an always-visible way to reach Edit/Remove, on top of
// the existing hover/keyboard-focus reveal. One open at a time - clicking
// a second card's "..." closes the first, and clicking anywhere else on
// the page closes whichever is open, so cards never get stuck expanded.
function closeMore(except){
  document.querySelectorAll('.morevote.open').forEach(el => {
    if (el === except) return;
    el.classList.remove('open');
    const btn = el.closest('.cardacts')?.querySelector('[data-more]');
    if (btn) btn.setAttribute('aria-expanded', 'false');
  });
}
document.addEventListener('click', e => {
  const btn = e.target.closest('button[data-more]');
  if (!btn){ closeMore(); return; }
  const menu = btn.closest('.cardacts').querySelector('.morevote');
  const opening = !menu.classList.contains('open');
  closeMore(opening ? menu : null);
  menu.classList.toggle('open', opening);
  btn.setAttribute('aria-expanded', String(opening));
});

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
  // Default TODAY, not tomorrow. Scheduling used to jump to tomorrow with
  // nothing on screen saying so, which put a batch meant for today a day out.
  const now = new Date();
  const p2 = n => String(n).padStart(2, '0');
  const today = `${now.getFullYear()}-${p2(now.getMonth()+1)}-${p2(now.getDate())}`;
  $('ytdate').min = today;
  if (!$('ytdate').value || $('ytdate').value < today) $('ytdate').value = today;
  $('postsub').hidden = true;              // ytsum carries the real headline
  $('postbatch').hidden = false; $('postsingle').hidden = true;
  $('postmodal').hidden = false;
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

// Rewrite titles: redo titles/descriptions/tags without re-rendering any
// video - but it overwrites what's there now with no way back, so it asks
// first, the same as Delete. A clip already on YouTube or edited by hand is
// skipped server-side; the note says how many.
let RT_VID = null;
document.addEventListener('click', e => {
  const b = e.target.closest('button[data-rt]');
  if (!b) return;
  RT_VID = b.dataset.rt;
  $('rtmodal').hidden = false;
});
$('rtcancel').onclick = () => { $('rtmodal').hidden = true; };
$('rtmodal').onclick = e => {
  if (e.target === $('rtmodal')) $('rtmodal').hidden = true;
};
$('rtgo').onclick = async () => {
  const go = $('rtgo');
  const note = document.querySelector(`.pnote[data-vid="${CSS.escape(RT_VID)}"]`);
  go.disabled = true; go.textContent = 'Rewriting…';
  if (note) note.textContent = 'Asking Claude for new titles…';
  try {
    const r = await fetch('/api/retitle', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({video_id: RT_VID})
    });
    const j = await r.json();
    $('rtmodal').hidden = true;
    if (j.ok) {
      if (note) note.textContent = `Rewrote ${j.changed} of ${j.asked}`
        + (j.skipped ? ` · ${j.skipped} already posted or edited, left alone` : '');
      setTimeout(() => library(), 1800);   // let the note be read before it's gone
    } else if (note) {
      note.textContent = j.error || 'Could not rewrite titles.';
    }
  } catch (err) {
    $('rtmodal').hidden = true;
    if (note) note.textContent = 'Could not reach the app.';
  }
  go.disabled = false; go.textContent = 'Rewrite';
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

// Upload one clip: confirm first, then choose now or scheduled. Runs in the
// background on the server; the card shows UPLOADING until it finishes.
let UP = {vid: null, file: null};
let UP_TIMER = null;
let UP_WATCHING = false;     // only report a finish this page actually started
function upStep(n){ $('upstep1').hidden = n !== 1; $('upstep2').hidden = n !== 2; }
document.addEventListener('click', async e => {
  const b = e.target.closest('button[data-up]');
  if (!b) return;
  UP = {vid: b.dataset.vid, file: b.dataset.up};
  const card = b.closest('.clip');
  $('postsub').textContent = card.querySelector('.hook')?.title || UP.file;
  $('postsub').hidden = false;
  $('uperr').hidden = true;
  $('upwhenrow').hidden = true;
  $('upvia').hidden = true;
  upStep(1);
  $('postbatch').hidden = true; $('postsingle').hidden = false;
  $('postmodal').hidden = false;
  try {
    const s = await (await fetch('/api/upload/status')).json();
    if (s.api_left === 0) {
      $('upvia').textContent = "Today's automatic uploads are used up, so a Chrome "
        + "window will open and upload it through YouTube Studio. Leave that "
        + "window alone until it's done.";
      $('upvia').hidden = false;
    }
  } catch (err) {}
});
$('upcancel').onclick = () => { $('postmodal').hidden = true; };
$('upnext').onclick = () => upStep(2);
$('upback').onclick = () => { $('upwhenrow').hidden = true; upStep(1); };
$('upsched').onclick = () => {
  // Default: the next whole hour after an hour from now.
  const d = new Date(Date.now() + 60 * 60 * 1000);
  d.setMinutes(0, 0, 0); d.setHours(d.getHours() + 1);
  const p = n => String(n).padStart(2, '0');
  $('upwhen').value = `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`;
  $('upwhenrow').hidden = false;
};
$('upnow').onclick = () => startUpload('now');
$('upgo').onclick = () => {
  if (!$('upwhen').value) { $('upwhen').focus(); return; }
  startUpload('schedule', new Date($('upwhen').value).toISOString());
};
async function startUpload(mode, when){
  $('uperr').hidden = true;
  let j = {};
  try {
    const r = await fetch('/api/upload', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({video_id: UP.vid, file: UP.file, mode, when})
    });
    j = await r.json();
  } catch (err) { j = {error: 'Could not reach the app.'}; }
  if (!j.ok) {
    $('uperr').textContent = j.error || 'Could not start the upload.';
    $('uperr').hidden = false;
    return;
  }
  $('postmodal').hidden = true;
  UP_WATCHING = true;
  pollUpload();
}
async function pollUpload(){
  clearTimeout(UP_TIMER);
  let s;
  try { s = await (await fetch('/api/upload/status')).json(); }
  catch (err) { UP_TIMER = setTimeout(pollUpload, 3000); return; }
  const card = s.vid && document.querySelector(
    `.clip[data-vid="${CSS.escape(s.vid)}"][data-file="${CSS.escape(s.file)}"]`);
  const media = card && card.querySelector('.media');
  let tag = media && media.querySelector('.stamp.upl');
  if (s.status === 'running') {
    if (media && !tag) {
      tag = document.createElement('div');
      tag.className = 'stamp soon upl';
      tag.innerHTML = '<b>UPLOADING</b><span></span>';
      media.appendChild(tag);
    }
    if (tag) tag.querySelector('span').textContent = s.message || '';
    UP_TIMER = setTimeout(pollUpload, 2000);
    return;
  }
  if (tag) tag.remove();
  if (!UP_WATCHING) return;
  UP_WATCHING = false;
  if (s.status === 'error') alert(s.message || 'The upload did not finish.');
  library();
}

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

// Every clip that COULD go out is listed: the first few sit in today's
// slots, the rest wait below. A clip moves between the two groups by drag
// or by its own button, so picking a different six no longer means removing
// clips from the project.
function ytDraw(){
  const n = YT_SLOTS.length;
  const rows = YT_ORDER.map((c, i) => {
    const inSlot = i < n;
    const when = inSlot
      ? new Date(YT_SLOTS[i]).toLocaleString([], {weekday:'short', day:'numeric',
          month:'short', hour:'numeric', minute:'2-digit'})
      : 'waiting';
    return `<li draggable="true" data-i="${i}"${inSlot?'':' class="waiting"'}>`
      + `<span class="grip" aria-hidden="true">&#8942;&#8942;</span>`
      + `<span class="slot">${inSlot ? i + 1 : '&middot;'}</span>`
      + `<span class="sl-when">${when}</span>`
      + `<span class="sl-title">${c.title.replace(/[<>&]/g,'').slice(0,60)}</span>`
      + `<button class="sl-move" data-move="${i}">${inSlot ? 'Take off' : 'Add'}</button>`
      + `</li>`;
  });
  if (n && YT_ORDER.length > n)
    rows.splice(n, 0, `<li class="sl-split">Not today &mdash; `
      + `${YT_ORDER.length - n} waiting for the next reset</li>`);
  $('ytlist').innerHTML = rows.join('');
}

// Take off / Add: one click moves a clip out of today's slots or into the
// last one, pushing whatever was there into the waiting group.
document.addEventListener('click', e => {
  const b = e.target.closest('#ytlist button[data-move]');
  if (!b) return;
  const i = +b.dataset.move, n = YT_SLOTS.length;
  const [c] = YT_ORDER.splice(i, 1);
  if (i < n) YT_ORDER.push(c);
  else YT_ORDER.splice(Math.max(0, n - 1), 0, c);
  ytDraw();
});

// Native HTML5 drag - no library, and the list is at most six rows.
let DRAG_FROM = null;
document.addEventListener('dragstart', e => {
  const li = e.target.closest('#ytlist li');
  if (!li || li.dataset.i === undefined) return;
  DRAG_FROM = +li.dataset.i;
  li.classList.add('dragging');
  e.dataTransfer.effectAllowed = 'move';
});
document.addEventListener('dragover', e => {
  const li = e.target.closest('#ytlist li');
  if (!li || li.dataset.i === undefined || DRAG_FROM === null) return;
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
          hour: h || 0, minute: m || 0,
          start_date: $('ytdate').value || null};
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
    YT_ORDER = j.items.concat(j.waiting || [])
      .map(it => ({file: it.file, title: it.title || ''}));
    $('draghint').hidden = YT_ORDER.length < 2;
    ytDraw();
    go.textContent = `Upload ${j.will_upload}`;
    go.disabled = false;
  } catch (err) { $('ytsum').textContent = 'Could not work out a schedule.'; }
}

$('ytgap').onchange = ytRefresh;
$('ythour').onchange = ytRefresh;
$('ytdate').onchange = ytRefresh;
$('ytcancel').onclick = () => { $('postmodal').hidden = true; };
// One shared backdrop-click-to-close for both bodies, now that they share
// a single overlay - a click on either #upmodal or #ytmodal's own backdrop
// used to close only that one; there's only one backdrop now.
$('postmodal').onclick = e => {
  if (e.target === $('postmodal')) $('postmodal').hidden = true;
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
  $('postmodal').hidden = true;
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

// Named (not an inline arrow on #go) so Retry can call the exact same
// thing - the form fields are never cleared on error, so a retry after a
// dropped connection is this, verbatim, not a second form to fill in.
async function startRun(){
  const source = $('src').value.trim();
  if (!source) { $('src').focus(); return; }
  $('go').disabled = true; $('retry').hidden = true;
  $('log').textContent = 'Starting…';
  document.title = DEFAULT_TITLE;
  // Ask now, on a real click - browsers refuse to prompt for Notification
  // permission from anywhere else, and this is the one moment guaranteed
  // to be a run someone actually wants telling about.
  try {
    if (window.Notification && Notification.permission === 'default')
      Notification.requestPermission();
  } catch (err) {}
  const r = await fetch('/api/run', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ source, instructions: $('say').value,
                           captions: $('caps').checked,
                           layout: $('layout').value,
                           swap: $('swap').checked })
  });
  const j = await r.json();
  if (j.error) { $('log').textContent = j.error; $('go').disabled = false; }
}
$('go').onclick = startRun;
$('retry').onclick = startRun;

$('askmore').onclick = () => {
  const open = $('can').hidden;
  $('can').hidden = !open;
  $('askmore').setAttribute('aria-expanded', String(open));
  $('askmore').textContent = open ? 'Hide' : "What it can't do";
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
  // A "Post now" upload is public with no publish time.
  if (!p.publish_at) return p.privacy === 'public' ? 'done' : 'priv';
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
  const d = new Date(p.publish_at || p.at);
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
      ${live?`<div class="stamp ${pubState(pub)}"${pub.url?` data-yt-id="${esc(pub.url.replace(/\/+$/,'').split('/').pop())}"`:''}>
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
      <div class="cardacts">
        <!-- One obvious action per card: Upload while it isn't posted yet,
             Download once it is (there is nothing left to upload). Edit and
             Remove are real but rare next to that, so they stay out of the
             way by default - but "out of the way" used to mean hover-only,
             which a first-time user has no reason to try. The "..." button
             is the always-visible way in; hover/focus-within still work too
             for anyone who's already learned the shortcut. -->
        <div class="mainrow">
          ${live
            ? `<a class="btn ghost mainact" href="${src}" download="${esc(c.file)}">Download</a>`
            : `<button class="up mainact" data-up="${esc(c.file)}" data-vid="${esc(vid)}">Upload</button>`}
          <button class="more" data-more aria-label="More actions" aria-expanded="false">&#8942;</button>
        </div>
        <div class="morevote${v==='reject'?' pinned':''}">
          ${live ? '' : `<a class="btn ghost dl" href="${src}" download="${esc(c.file)}">Download</a>`}
          <button class="edit" data-edit="${esc(c.file)}" data-vid="${esc(vid)}">Edit</button>
          <button data-vote="reject" class="rej${v==='reject'?' on-rej':''}">${v==='reject'?'Removed · undo':'Remove'}</button>
        </div>
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
  // Prefer what the video actually IS. Projects made before this existed
  // have no video_title, so fall back to the URL/filename/id exactly as
  // before - never blank, and never worse than it already was.
  const src = s.source || '';
  const fallback = /^https?:/i.test(src) ? src : (src.split(/[\\/]/).pop() || s.video_id);
  const name = s.video_title || fallback;
  const rejected = Object.values(s.feedback||{}).filter(x => x==='reject').length;
  const posted = s.posted || 0;
  const left = Math.max(0, s.count - posted - rejected);

  return `<details class="proj" data-vid="${esc(s.video_id)}" ${open?'open':''}>
    <summary>
      ${s.video_thumb ? `<img class="pthumb" src="${esc(s.video_thumb)}" alt="" loading="lazy">` : ''}
      <span class="ptitles">
        <span class="pname">${esc(name)}</span>
        <span class="pstatus">${s.count} clip${s.count===1?'':'s'}${
          posted ? ` &middot; ${posted} posted` : ''}${
          left && posted ? ` &middot; ${left} left` : ''}</span>
      </span>
      <span class="pmeta">
        <span class="pdate">${esc(when.date)}</span>
        <span class="ptime">${esc(when.time)}</span>
        ${s.mode==='compilation'?'<span class="pkindx">compilation</span>':''}
      </span>
    </summary>
    <p class="phow">Every clip below is included. <b>Remove</b> any you don't
      want, then <b>schedule</b> the rest.</p>
    <div class="pbar">
      <button class="btn yt" data-yt="${esc(s.video_id)}">Post to YouTube…</button>
      <button class="btn ghost dl" data-dl="${esc(s.video_id)}">Download…</button>
      <button class="btn ghost ft" data-ft="${esc(s.video_id)}">Edit All</button>
      <button class="btn ghost rt" data-rt="${esc(s.video_id)}"
        title="Redo titles, descriptions and tags for every clip not yet posted">Rewrite titles</button>
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
  // innerHTML above just wiped out any view counts a previous loadViews()
  // wrote into the stamps - ask again rather than try to preserve them
  // across the rebuild, which every edit/upload/retitle triggers.
  loadViews();
}

async function library(expand){
  const lib = await (await fetch('/api/library')).json();
  renderLibrary(lib, expand);
}

// View counts only exist for clips that have actually gone public - see
// youtube.py:view_counts for why this reads with a plain API key rather
// than the upload-scoped OAuth token. One request covers every posted
// clip across every project (server-side batches into 1-unit calls), and
// a clip with no entry in the response just keeps showing its plain
// PUBLISHED/SCHEDULED stamp - that is the correct state for something
// still private, not a failure to report.
async function loadViews(){
  let views;
  try { views = await (await fetch('/api/views')).json(); }
  catch (err) { return; }
  for (const [id, n] of Object.entries(views || {})){
    const stamp = document.querySelector(`.stamp[data-yt-id="${CSS.escape(id)}"]`);
    if (!stamp) continue;
    let vs = stamp.querySelector('.views');
    if (!vs){ vs = document.createElement('span'); vs.className = 'views';
              stamp.appendChild(vs); }
    vs.textContent = `${n.toLocaleString()} view${n===1?'':'s'}`;
  }
}
setInterval(loadViews, 90000);   // refresh in place without a full re-render

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
let stageAt = 0, stageWas = -1, downloadDoneAt = null;

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
//
// EXCLUDING THE DOWNLOAD. Download is network-bound - a resumed or slow
// download can burn ten minutes that say nothing about the CPU-bound
// stages after it. Blending that into one t/p ratio is what turned a real
// ~30-minute run into "about 53 min left": ten slow download minutes got
// extrapolated across the whole job. So both t and frac here are rebased
// to start at zero the moment download finishes - downloadDoneAt is set
// once, in poll(), the first time step reaches 2.
function etaText(s, frac){
  if (downloadDoneAt == null)
    return s.eta_min ? ` · about ${s.eta_min} min total` : '';
  const w = WEIGHTS.slice();
  if (s.cached) w[2] = 2;
  const total = w.reduce((a,b) => a+b, 0);
  const dl = w[0] + w[1];                     // download + checking the video
  const t = Math.max(0, (s.elapsed || 0) - downloadDoneAt);
  const postFrac = Math.max(0, Math.min(1, (frac * total - dl) / (total - dl)));
  if (postFrac > 0.04 && t > 25){
    const left = Math.round((t / postFrac - t) / 60);
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

// A 40-minute run has nobody watching it the whole time. Say so where
// someone who tabbed away will actually see it: the tab title (checked on
// alt-tab even with the tab itself not visible), a system notification
// (only offered while the tab is hidden - no point popping one over the
// page the person is already looking at), and a short chime.
const DEFAULT_TITLE = document.title;
function chime(ok){
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    const ctx = new AC();
    const notes = ok ? [523.25, 659.25, 783.99] : [246.94, 196.00];
    let t = ctx.currentTime;
    notes.forEach(freq => {
      const osc = ctx.createOscillator(), gain = ctx.createGain();
      osc.type = 'sine'; osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(0.16, t + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);
      osc.connect(gain); gain.connect(ctx.destination);
      osc.start(t); osc.stop(t + 0.24);
      t += 0.15;
    });
    setTimeout(() => { try { ctx.close(); } catch (err) {} }, notes.length * 150 + 400);
  } catch (err) {}
}
function notifyFinish(ok, info){
  const msg = ok ? `${info} clip${info===1?'':'s'} ready` : (info || 'Run failed');
  document.title = (ok ? '✓ ' : '✗ ') + msg;
  try {
    if (document.hidden && window.Notification && Notification.permission === 'granted')
      new Notification(ok ? 'Clips ready' : 'Clip run failed', {body: msg});
  } catch (err) {}
  chime(ok);
}
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) document.title = DEFAULT_TITLE;
});

async function poll(){
  try{
    const s = await (await fetch('/api/status')).json();
    if (s.status !== 'idle') $('runpanel').hidden = false;
    // Elapsed-at-stage-entry, so interpolation measures time in THIS stage.
    if ((s.step || 0) !== stageWas){
      stageWas = s.step || 0; stageAt = s.elapsed || 0;
      // Freeze the moment download ends, once, for etaText - see there for
      // why. < 2 means a new run just started (or is still downloading);
      // reset so the NEXT run gets its own mark instead of reusing this one.
      if (stageWas < 2) downloadDoneAt = null;
      else if (downloadDoneAt === null) downloadDoneAt = stageAt;
    }

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
      // folder, expanded, rather than as a detached grid. Guarded by
      // `running` so this (and the notification) fires exactly once, not
      // on every poll tick while status stays 'done'.
      if (running){ notifyFinish(true, s.clips.length); library(s.video_id); running = false; }
    } else if (s.status === 'error'){
      paintRing(1, s.error || 'Something went wrong.', 'err');
      $('details').open = true;
      $('go').disabled = false;
      $('retry').hidden = false;
      if (running){ notifyFinish(false, s.error); running = false; }
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
library(); poll(); pollUpload();
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


class _Tee:
    """Write to every stream given, ignoring one that's already broken.

    Lets `main()` log to a file AND to a real console when one exists,
    without changing behaviour for anyone who still runs this by hand.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def _open_log():
    """Append-mode log file, trimmed once it gets past ~2MB.

    This is the ONLY diagnostics a server started hidden (the Desktop
    launcher runs it via pythonw, no console at all) leaves behind, so it
    has to survive on its own - trimmed to stay small forever, not rotated
    away or left to grow without bound.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 2_000_000:
            LOG_FILE.write_bytes(LOG_FILE.read_bytes()[-500_000:])
    except Exception:
        pass
    return open(LOG_FILE, "a", encoding="utf-8", buffering=1)


def main() -> None:
    config.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = _open_log()
    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)
    print(f"\n=== started {datetime.now().isoformat(timespec='seconds')} ===")
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
