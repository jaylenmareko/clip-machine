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
import urllib.parse
import webbrowser
from contextlib import redirect_stdout
from pathlib import Path

from . import config, cut, llm, modes, pipeline, transcribe

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


def _progress(log: list[str]) -> dict:
    """Derive a friendly stage, ETA and instruction notices from the output."""
    stage, index = "Starting", 0
    eta = None
    applied = ignored = None
    for line in log:
        for i, (marker, label) in enumerate(_STAGES):
            if marker in line:
                stage, index = label, i + 1
        if m := _ETA_RE.search(line):
            eta = int(m.group(1))
        # Surfaced rather than left in the log: the whole point of the
        # instruction box is knowing what it did and did not do.
        if "instruction applied:" in line:
            applied = line.split("instruction applied:", 1)[1].strip()
        elif "instruction IGNORED:" in line:
            ignored = line.split("instruction IGNORED:", 1)[1].strip()
    return {"stage": stage, "step": index, "steps": len(_STAGES),
            "eta_min": eta, "applied": applied, "ignored": ignored}


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
        with redirect_stdout(_LogWriter(job)):
            out_dir = pipeline.run(job["source"], max_clips=job["max_clips"],
                                   use_llm=job["use_llm"], layout=job["layout"],
                                   instructions=job.get("instructions"))
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        job["video_id"] = manifest["video_id"]
        job["clips"] = manifest["clips"]
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = _friendly_error(e, job.get("log", []))
        job["log"].append(f"ERROR  {type(e).__name__}: {e}")


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
               layout: str, instructions: str | None = None) -> dict:
    global _worker_alive
    with _job_lock:
        job = {
            "id": len(_jobs) + 1,
            "status": "queued", "log": [], "clips": [],
            "source": source, "video_id": None, "error": None,
            "max_clips": max_clips, "use_llm": use_llm, "layout": layout,
            "instructions": instructions,
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


def _read_feedback(video_id: str) -> dict:
    p = _feedback_path(video_id)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_feedback(video_id: str, file: str, verdict: str | None) -> dict:
    data = _read_feedback(video_id)
    if verdict in ("keep", "reject"):
        data[file] = verdict
    else:
        data.pop(file, None)   # a third click clears the verdict
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
        clips = json.loads(manifest_path.read_text(encoding="utf-8")).get("clips", [])
    except Exception:
        return None

    verdicts = _read_feedback(video_id)
    keep = [c for c in clips if verdicts.get(c["file"]) != "reject"]
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
    """Previously generated clip sets, newest first."""
    out = []
    if not config.CLIPS_DIR.exists():
        return out
    for d in sorted(config.CLIPS_DIR.iterdir(), key=lambda p: -p.stat().st_mtime):
        manifest = d / "manifest.json"
        if d.is_dir() and manifest.exists():
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
                out.append({"video_id": d.name, "count": len(m.get("clips", [])),
                            "clips": m.get("clips", []),
                            "feedback": _read_feedback(d.name)})
            except Exception:
                continue
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
                **_progress(job.get("log", [])),
            })
        elif route == "/api/library":
            self._send_json(_library())
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
        if path not in ("/api/run", "/api/feedback"):
            self._send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")

        if path == "/api/feedback":
            vid = (data.get("video_id") or "").strip()
            file = (data.get("file") or "").strip()
            verdict = data.get("verdict")
            if not vid or not file or "/" in file or ".." in file + vid:
                self._send_json({"error": "bad request"}, 400)
                return
            self._send_json({"ok": True, "feedback": _write_feedback(vid, file, verdict)})
            return

        source = (data.get("source") or "").strip()
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
        )
        self._send_json(result, 409 if "error" in result else 200)


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clip Machine</title>
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
  .bar{height:5px;background:#0f131b;border-radius:3px;overflow:hidden;margin:10px 0 4px}
  .fill{height:100%;width:0;background:var(--accent);transition:width .4s ease}
  #details summary{cursor:pointer;color:var(--muted);font-size:12px;margin-top:6px}
  #details pre{margin-top:8px}
  .acts{display:flex;gap:8px;margin-top:10px}
  .vote{display:flex;gap:8px;margin-top:8px}
  .vote button{flex:1;background:transparent;border:1px solid var(--line);
    color:var(--muted);border-radius:7px;padding:6px;font:600 12px/1.4 inherit;
    cursor:pointer}
  .vote button:hover{color:var(--text);border-color:#39435a}
  .vote button.on-keep{background:#173226;border-color:#2c4a37;color:var(--good)}
  .vote button.on-rej{background:#33201f;border-color:#4d2c2b;color:var(--bad)}
  .clip.rejected{opacity:.42}
  .allbar{display:flex;align-items:center;gap:12px;margin:16px 0 0}
  .allbar .btn{flex:0 0 auto;padding:9px 16px}
  .allbar span{color:var(--muted);font-size:12px}
  .btn{flex:1;text-align:center;background:var(--accent);color:#0a1020;border:0;
    border-radius:7px;padding:7px 10px;font:600 12px/1.4 inherit;cursor:pointer;
    text-decoration:none}
  .btn.ghost{background:transparent;color:var(--muted);border:1px solid var(--line)}
  .btn.ghost:hover{color:var(--text);border-color:#39435a}
  .btn.ok{background:var(--good);color:#0a1020}
  .kind{background:#26304a;border:1px solid #35406090;color:#a9c0ee;
    border-radius:5px;padding:1px 7px;font-size:11px;letter-spacing:.03em}
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
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px;margin-top:16px}
  .clip{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
  .clip video{width:100%;display:block;background:#000;aspect-ratio:9/16;object-fit:contain}
  .meta{padding:12px 13px}
  .hook{font-weight:600;font-size:13px;margin:0 0 6px;line-height:1.4}
  .stats{color:var(--muted);font-size:12px;display:flex;gap:10px;flex-wrap:wrap}
  .stats b{color:var(--good);font-weight:600}
  details{margin-top:8px}
  summary{cursor:pointer;color:var(--muted);font-size:12px}
  details p{color:var(--muted);font-size:12px;margin:8px 0 0;max-height:150px;overflow:auto}
  h2{font-size:14px;margin:26px 0 0;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
  .status{font-size:13px;color:var(--muted);margin-bottom:10px}
  .status.run{color:var(--accent)} .status.done{color:var(--good)} .status.err{color:var(--bad)}
  .empty{color:var(--muted);font-size:13px;padding:8px 0}
</style></head><body><div class="wrap">

<header>
  <h1>Clip Machine</h1>
  <span class="sub">longform &rarr; vertical clips, fully local</span>
  <span class="badge" id="ram">&nbsp;</span>
</header>

<div class="panel">
  <div class="row">
    <div>
      <div class="labelrow">
        <label for="src">Video URL or local file path</label>
        <span class="hint">Finds hot takes, heated exchanges, admissions, stories, funny beats and clear explanations — in one pass.</span>
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
    <button id="go">Generate clips</button>
  </div>
</div>

<div class="panel" id="runpanel" hidden>
  <div class="status" id="status"></div>
  <div class="bar"><div class="fill" id="fill"></div></div>
  <div class="notice" id="applied" hidden></div>
  <div class="notice warn" id="ignored" hidden></div>
  <details id="details"><summary>Details</summary><pre id="log"></pre></details>
</div>

<h2>Clips</h2>
<div class="allbar" id="allbar" hidden>
  <a class="btn" id="dlall" href="#">Download all</a>
  <span id="allnote"></span>
</div>
<div class="grid" id="clips"></div>
<div class="empty" id="noclips">Nothing yet. Run a video above, or previous runs will show here.</div>

<script>
const $ = id => document.getElementById(id);
let running = false;

const AUDIO_SIGNALS = new Set(['loud','animated','fast']);
const FEEDBACK = {};   // video_id -> {file: 'keep'|'reject'}
let CURRENT_VID = null;

document.addEventListener('click', async e => {
  const b = e.target.closest('button[data-vote]');
  if (!b) return;
  const card = b.closest('.clip');
  const vid = card.dataset.vid, file = card.dataset.file;
  const want = b.dataset.vote;
  const now = (FEEDBACK[vid]||{})[file];
  const verdict = now === want ? null : want;   // clicking again clears it
  try {
    const r = await fetch('/api/feedback', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({video_id: vid, file, verdict})
    });
    const j = await r.json();
    FEEDBACK[vid] = j.feedback || {};
    const keepBtn = card.querySelector('[data-vote=keep]');
    const rejBtn  = card.querySelector('[data-vote=reject]');
    const v = FEEDBACK[vid][file] || '';
    keepBtn.className = v==='keep' ? 'on-keep' : '';
    rejBtn.className  = v==='reject' ? 'on-rej' : '';
    card.classList.toggle('rejected', v==='reject');
    updateAllBar(vid);
  } catch (err) {}
});

function updateAllBar(vid){
  if (!vid) return;
  CURRENT_VID = vid;
  const fb = FEEDBACK[vid] || {};
  const rejected = Object.values(fb).filter(x => x==='reject').length;
  $('dlall').href = `/zip/${encodeURIComponent(vid)}`;
  $('allnote').textContent = rejected
    ? `${rejected} rejected clip${rejected===1?'':'s'} excluded from the zip`
    : 'zips every clip plus a titles file';
  $('allbar').hidden = false;
}

$('go').onclick = async () => {
  const source = $('src').value.trim();
  if (!source) { $('src').focus(); return; }
  $('go').disabled = true;
  $('log').textContent = 'Starting…';
  const r = await fetch('/api/run', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ source, instructions: $('say').value })
  });
  const j = await r.json();
  if (j.error) { $('log').textContent = j.error; $('go').disabled = false; }
};

function card(c, vid){
  const src = `/clips/${encodeURIComponent(vid)}/${encodeURIComponent(c.file)}`;
  const esc = s => (s||'').replace(/[<>&]/g,'');
  const clock = s => {
    s = Math.round(s||0);
    return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
  };
  const v = (FEEDBACK[vid]||{})[c.file] || '';
  return `<div class="clip${v==='reject'?' rejected':''}" data-vid="${esc(vid)}" data-file="${esc(c.file)}">
    <video src="${src}" controls preload="metadata"
      ${c.thumb?`poster="/clips/${encodeURIComponent(vid)}/${encodeURIComponent(c.thumb)}"`:''}></video>
    <div class="meta">
      <p class="hook">${esc(c.title || c.hook)}</p>
      <div class="stats"><span>${clock(c.duration)}</span><span>at ${clock(c.start)}</span>
        ${c.mode?`<span class="kind">${esc(c.mode)}</span>`:''}</div>
      <div class="acts">
        <a class="btn" href="${src}" download="${esc(c.file)}">Download</a>
        <button class="btn ghost" data-copy="${encodeURIComponent(c.title||c.hook||'')}">Copy title</button>
      </div>
      <div class="vote">
        <button data-vote="keep" class="${v==='keep'?'on-keep':''}">Keep</button>
        <button data-vote="reject" class="${v==='reject'?'on-rej':''}">Reject</button>
      </div>
      <details><summary>transcript</summary><p>${esc(c.transcript)}</p></details>
    </div></div>`;
}

document.addEventListener('click', async e => {
  const b = e.target.closest('button[data-copy]');
  if (!b) return;
  try {
    await navigator.clipboard.writeText(decodeURIComponent(b.dataset.copy));
    const was = b.textContent;
    b.textContent = 'Copied'; b.classList.add('ok');
    setTimeout(() => { b.textContent = was; b.classList.remove('ok'); }, 1200);
  } catch (err) { b.textContent = 'Copy failed'; }
});

function render(clips, vid, feedback){
  if (feedback) FEEDBACK[vid] = feedback;
  $('clips').innerHTML = clips.map(c => card(c, vid)).join('');
  $('noclips').style.display = clips.length ? 'none' : 'block';
  if (clips.length) updateAllBar(vid);
}

async function library(){
  const lib = await (await fetch('/api/library')).json();
  if (!lib.length) return;
  lib.forEach(s => { FEEDBACK[s.video_id] = s.feedback || {}; });
  $('clips').innerHTML = lib.flatMap(s => s.clips.map(c => card(c, s.video_id))).join('');
  $('noclips').style.display = 'none';
  updateAllBar(lib[0].video_id);
}

async function poll(){
  try{
    const s = await (await fetch('/api/status')).json();
    if (s.free_ram != null){
      const low = s.free_ram < 2.5;
      $('ram').textContent = `${s.free_ram} GB RAM free · ${s.model}`;
      $('ram').className = 'badge' + (low ? ' warn' : '');
    }
    const st = $('status');
    if (s.status !== 'idle') $('runpanel').hidden = false;
    const pct = s.steps ? Math.round(100 * (s.step||0) / s.steps) : 0;
    if (s.status === 'running' || s.status === 'queued'){
      running = true; $('go').disabled = true;
      const eta = s.eta_min ? ` · about ${s.eta_min} min total` : '';
      const q = s.queued ? ` · ${s.queued} more queued` : '';
      st.textContent = (s.stage || 'Starting') + '…' + eta + q;
      st.className = 'status run';
      $('fill').style.width = Math.max(pct, 4) + '%';
    } else if (s.status === 'done'){
      st.textContent = `Done — ${s.clips.length} clip${s.clips.length===1?'':'s'} ready`;
      st.className = 'status done';
      $('fill').style.width = '100%';
      $('go').disabled = false;
      if (running){ render(s.clips, s.video_id); running = false; }
    } else if (s.status === 'error'){
      st.textContent = s.error || 'Something went wrong.';
      st.className = 'status err';
      $('fill').style.width = '100%';
      $('details').open = true;
      $('go').disabled = false; running = false;
    }
    const setNote = (id, label, val) => {
      const el = $(id);
      if (val){ el.textContent = label + val; el.hidden = false; }
      else { el.hidden = true; }
    };
    setNote('applied', 'Applied: ', s.applied);
    setNote('ignored', "Couldn't apply: ", s.ignored);

    if (s.log && s.log.length) $('log').textContent = s.log.join('\n');
    const pre = $('log'); if (running) pre.scrollTop = pre.scrollHeight;
  }catch(e){}
  setTimeout(poll, 1000);
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
    print(f"Clip Machine UI  ->  {url}")
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
