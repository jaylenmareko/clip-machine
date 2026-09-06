"""Who is talking right now — local, no ML models, no GPU.

For a two-person source (both people in one wide frame), we find each
person's face once, then watch the mouth region for movement.

Raw motion is NOT enough: measured against a live listener it was a coin
flip (4.7s vs 5.2s), because a nodding head moves as much as a talking
one. So mouth motion is correlated against the speech envelope instead —
a talking mouth moves WITH the audio, a nodding head does not. Verified
with a controlled swap (same video, speaker A's audio vs speaker B's).

Deliberately not using diarization models (pyannote/speechbrain): they
want torch, a HuggingFace token, and more RAM than this machine has spare.

Output is a list of (start, end, "left"|"right") segments in source time.
"""
import os
import subprocess
from dataclasses import dataclass

import cv2

_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

SAMPLE_FPS = 6           # motion sampling rate
_SMOOTH_WINDOW = 5       # samples averaged before deciding (~0.8s)
_MIN_SEGMENT = 1.2       # seconds; shorter switches look like flicker
_DOMINANCE = 1.25        # motion-only fallback: ratio needed to take over
_ANALYSIS_W = 480        # downscale width for analysis - detail is not needed
_CORR_WINDOW = 18        # samples (~3s) correlated against the speech envelope
_CORR_MARGIN = 0.08      # correlation lead needed to steal the active slot

# Frames are held in memory while a window is analysed, so long inputs are
# processed in bounded chunks. 120s at 6fps / 480px gray is ~75MB.
_CHUNK_SECONDS = 120


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    def mouth_roi(self) -> tuple[int, int, int, int]:
        """Lower ~40% of the face box, inset horizontally."""
        my = self.y + int(self.h * 0.58)
        mh = max(int(self.h * 0.40), 4)
        mx = self.x + int(self.w * 0.18)
        mw = max(int(self.w * 0.64), 4)
        return mx, my, mw, mh


def _decode_gray(video_path, start: float, duration: float, width: int):
    """Decode a window to downscaled grayscale frames."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    sw, sh = int(probe[0]), int(probe[1])
    w = width
    h = max(2, int(round(sh * (w / sw) / 2)) * 2)

    proc = subprocess.run(
        ["ffmpeg", "-v", "error",
         "-ss", f"{start:.3f}", "-i", str(video_path), "-t", f"{duration:.3f}",
         "-vf", f"fps={SAMPLE_FPS},scale={w}:{h},format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    )
    buf = proc.stdout
    fs = w * h
    frames = [buf[i:i + fs] for i in range(0, len(buf) - fs + 1, fs)]
    return frames, w, h


def _find_face(frame_bytes: bytes, w: int, h: int, x0: int, x1: int) -> FaceBox | None:
    """Largest frontal face inside the x0..x1 column band."""
    import numpy as np

    img = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(h, w)
    band = img[:, x0:x1]
    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    faces = cascade.detectMultiScale(band, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(int(w * 0.05), int(w * 0.05)))
    if len(faces) == 0:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return FaceBox(int(x) + x0, int(y), int(fw), int(fh))


def _roi_motion(frames: list[bytes], w: int, h: int,
                roi: tuple[int, int, int, int]) -> list[float]:
    """Mean absolute frame-to-frame difference inside an ROI."""
    import numpy as np

    x, y, rw, rh = roi
    x, y = max(x, 0), max(y, 0)
    rw, rh = min(rw, w - x), min(rh, h - y)
    if rw <= 0 or rh <= 0:
        return [0.0] * max(len(frames) - 1, 0)

    prev, out = None, []
    for fb in frames:
        img = np.frombuffer(fb, dtype=np.uint8).reshape(h, w)
        patch = img[y:y + rh, x:x + rw].astype(np.int16)
        if prev is not None:
            out.append(float(np.abs(patch - prev).mean()))
        prev = patch
    return out


def _audio_envelope(video_path, start: float, duration: float, n: int) -> list[float]:
    """Speech loudness resampled to exactly n points."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video_path),
             "-t", f"{duration:.3f}", "-af",
             f"aresample=8000,asetnsamples={8000 // SAMPLE_FPS},"
             f"astats=metadata=1:reset=1,"
             f"ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=600,
        ).stdout
    except Exception:
        return []

    vals = []
    for line in out.splitlines():
        if "RMS_level=" in line:
            raw = line.split("=")[-1].strip()
            vals.append(-90.0 if raw == "-inf" else float(raw))
    if not vals:
        return []
    return [vals[min(int(i * len(vals) / max(n, 1)), len(vals) - 1)] for i in range(n)]


def _pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 1e-9 or vb <= 1e-9:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va ** 0.5 * vb ** 0.5)


def _smooth(vals: list[float], window: int) -> list[float]:
    if not vals:
        return []
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window // 2)
        hi = min(len(vals), i + window // 2 + 1)
        chunk = vals[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def _decide(lm: list[float], rm: list[float], env: list[float],
            carry: str | None) -> tuple[list[str], bool]:
    """Per-sample active side. Returns (decisions, used_correlation)."""
    use_corr = len(env) == len(lm) and len(lm) >= _CORR_WINDOW
    decisions: list[str] = []

    if use_corr:
        current = carry
        for i in range(len(lm)):
            lo = max(0, i - _CORR_WINDOW // 2)
            hi = min(len(lm), i + _CORR_WINDOW // 2 + 1)
            cl = _pearson(lm[lo:hi], env[lo:hi])
            cr = _pearson(rm[lo:hi], env[lo:hi])
            if current is None:
                current = "left" if cl >= cr else "right"
            elif current == "left" and cr > cl + _CORR_MARGIN:
                current = "right"
            elif current == "right" and cl > cr + _CORR_MARGIN:
                current = "left"
            decisions.append(current)
    else:
        current = carry or ("left" if sum(lm) >= sum(rm) else "right")
        for a, b in zip(lm, rm):
            if current == "left" and b > a * _DOMINANCE:
                current = "right"
            elif current == "right" and a > b * _DOMINANCE:
                current = "left"
            decisions.append(current)

    return decisions, use_corr


def active_speaker_segments(
    video_path, start: float, end: float, verbose: bool = False
) -> list[tuple[float, float, str]]:
    """Return [(seg_start, seg_end, 'left'|'right'), ...] in SOURCE time.

    Long ranges are analysed in bounded chunks so memory stays flat, and
    faces are detected once and reused across chunks.

    Empty list means "couldn't tell" — callers should fall back to a fixed
    layout rather than guessing.
    """
    total = end - start
    if total <= 0:
        return []

    left_roi = right_roi = None
    faces_found = (False, False)
    decisions: list[str] = []
    carry: str | None = None
    used_corr = False
    step = 1.0 / SAMPLE_FPS
    offset = 0.0

    while offset < total - step:
        dur = min(_CHUNK_SECONDS, total - offset)
        try:
            frames, w, h = _decode_gray(video_path, start + offset, dur, _ANALYSIS_W)
        except Exception:
            break
        if len(frames) < 4:
            offset += dur
            continue

        half = w // 2
        if left_roi is None:
            lf = rf = None
            for idx in (0, len(frames) // 3, len(frames) // 2, 2 * len(frames) // 3):
                lf = lf or _find_face(frames[idx], w, h, 0, half)
                rf = rf or _find_face(frames[idx], w, h, half, w)
                if lf and rf:
                    break

            def fallback(x0: int) -> tuple[int, int, int, int]:
                return (x0 + half // 4, int(h * 0.42), half // 2, int(h * 0.30))

            left_roi = lf.mouth_roi() if lf else fallback(0)
            right_roi = rf.mouth_roi() if rf else fallback(half)
            faces_found = (bool(lf), bool(rf))

        lm = _smooth(_roi_motion(frames, w, h, left_roi), _SMOOTH_WINDOW)
        rm = _smooth(_roi_motion(frames, w, h, right_roi), _SMOOTH_WINDOW)
        del frames  # free the window before the next chunk
        if not lm:
            offset += dur
            continue

        env = _audio_envelope(video_path, start + offset, dur, len(lm))
        chunk_decisions, uc = _decide(lm, rm, env, carry)
        used_corr = used_corr or uc
        if chunk_decisions:
            carry = chunk_decisions[-1]
        decisions.extend(chunk_decisions)
        offset += dur

    if not decisions:
        return []
    if verbose:
        print(f"      faces: left={'yes' if faces_found[0] else 'no'} "
              f"right={'yes' if faces_found[1] else 'no'}; "
              f"method: {'audio-visual correlation' if used_corr else 'motion only'}")

    # Collapse to segments, dropping ones too short to be worth a cut.
    segments: list[list] = []
    for i, who in enumerate(decisions):
        t = start + i * step
        if segments and segments[-1][2] == who:
            segments[-1][1] = t + step
        else:
            segments.append([t, t + step, who])

    merged: list[list] = []
    for seg in segments:
        if merged and (seg[1] - seg[0] < _MIN_SEGMENT):
            merged[-1][1] = seg[1]
        elif merged and merged[-1][2] == seg[2]:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    return [(round(s, 2), round(e, 2), who) for s, e, who in merged]


def turn_starts(segments: list[tuple[float, float, str]]) -> list[float]:
    """Times where the active speaker changes — good places to start a clip."""
    return [s for s, _e, _w in segments]


def swap_intervals(segments: list[tuple[float, float, str]], clip_start: float,
                   clip_end: float | None = None,
                   active_on_top: str = "left") -> list[tuple[float, float]]:
    """Intervals (in CLIP time) where the panels should be swapped.

    `active_on_top` names which source half occupies the top panel by
    default; the swap applies while the OTHER half is talking.
    """
    other = "right" if active_on_top == "left" else "left"
    out = []
    for s, e, who in segments:
        if who != other:
            continue
        if clip_end is not None and (e <= clip_start or s >= clip_end):
            continue
        lo = max(s, clip_start)
        hi = min(e, clip_end) if clip_end is not None else e
        if hi - lo <= 0.05:
            continue
        out.append((round(lo - clip_start, 2), round(hi - clip_start, 2)))
    return out
