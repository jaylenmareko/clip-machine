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

from . import config, cut, modes, pipeline, transcribe

PORT = int(os.environ.get("CLIP_PORT", "8420"))

_job: dict | None = None
_job_lock = threading.Lock()


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


def _worker(job: dict, source: str, max_clips: int | None, use_llm: bool,
            layout: str) -> None:
    try:
        job["status"] = "running"
        with redirect_stdout(_LogWriter(job)):
            out_dir = pipeline.run(source, max_clips=max_clips, use_llm=use_llm,
                                   layout=layout)
        manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
        job["video_id"] = manifest["video_id"]
        job["clips"] = manifest["clips"]
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        job["log"].append(f"ERROR  {type(e).__name__}: {e}")


def _start_job(source: str, max_clips: int | None, use_llm: bool,
               layout: str) -> dict:
    global _job
    with _job_lock:
        if _job and _job["status"] in ("queued", "running"):
            return {"error": "A job is already running."}
        _job = {
            "status": "queued", "log": [], "clips": [],
            "source": source, "video_id": None, "error": None,
        }
        threading.Thread(
            target=_worker,
            args=(_job, source, max_clips, use_llm, layout),
            daemon=True,
        ).start()
        return {"ok": True}


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
                            "clips": m.get("clips", [])})
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
            job = _job or {"status": "idle", "log": [], "clips": []}
            self._send_json({
                "status": job["status"],
                "log": job["log"][-400:],
                "clips": job.get("clips", []),
                "video_id": job.get("video_id"),
                "error": job.get("error"),
                "free_ram": round(free, 2) if free else None,
                "model": config.WHISPER_MODEL,
            })
        elif route == "/api/library":
            self._send_json(_library())
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
        if urllib.parse.urlparse(self.path).path != "/api/run":
            self._send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length) or b"{}")
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
  .acts{display:flex;gap:8px;margin-top:10px}
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
      <label for="src">Video URL or local file path</label>
      <input id="src" placeholder="https://youtube.com/watch?v=…  or  C:\path\to\video.mp4">
    </div>
  </div>

  <div class="row2">
    <button id="go">Generate clips</button>
  </div>
  <div class="hint">Finds hot takes, heated exchanges, admissions, stories, funny beats and clear explanations — in one pass. Layout is detected from the video.</div>
</div>

<div class="panel" id="runpanel" hidden>
  <div class="status" id="status"></div>
  <pre id="log"></pre>
</div>

<h2>Clips</h2>
<div class="grid" id="clips"></div>
<div class="empty" id="noclips">Nothing yet. Run a video above, or previous runs will show here.</div>

<script>
const $ = id => document.getElementById(id);
let running = false;

const AUDIO_SIGNALS = new Set(['loud','animated','fast']);

$('go').onclick = async () => {
  const source = $('src').value.trim();
  if (!source) { $('src').focus(); return; }
  $('go').disabled = true;
  $('log').textContent = 'Starting…';
  const r = await fetch('/api/run', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ source })
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
  return `<div class="clip">
    <video src="${src}" controls preload="metadata"></video>
    <div class="meta">
      <p class="hook">${esc(c.title || c.hook)}</p>
      <div class="stats"><span>${clock(c.duration)}</span><span>at ${clock(c.start)}</span>
        ${c.mode?`<span class="kind">${esc(c.mode)}</span>`:''}</div>
      <div class="acts">
        <a class="btn" href="${src}" download="${esc(c.file)}">Download</a>
        <button class="btn ghost" data-copy="${encodeURIComponent(c.title||c.hook||'')}">Copy title</button>
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

function render(clips, vid){
  $('clips').innerHTML = clips.map(c => card(c, vid)).join('');
  $('noclips').style.display = clips.length ? 'none' : 'block';
}

async function library(){
  const lib = await (await fetch('/api/library')).json();
  if (!lib.length) return;
  $('clips').innerHTML = lib.flatMap(s => s.clips.map(c => card(c, s.video_id))).join('');
  $('noclips').style.display = 'none';
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
    if (s.status === 'running' || s.status === 'queued'){
      running = true; $('go').disabled = true;
      st.textContent = 'Running…'; st.className = 'status run';
    } else if (s.status === 'done'){
      st.textContent = `Done — ${s.clips.length} clips`; st.className = 'status done';
      $('go').disabled = false;
      if (running){ render(s.clips, s.video_id); running = false; }
    } else if (s.status === 'error'){
      st.textContent = s.error || 'Failed'; st.className = 'status err';
      $('go').disabled = false; running = false;
    }
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
    return PAGE


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
