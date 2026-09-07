"""Decide the output layout by looking at the source.

The question is simple: does this recording already have two people side by
side? If it does, each half is its own camera feed and the right output is a
50/50 stacked split. If there's one speaker, a centre crop is right.

Detected rather than asked, because a viewer of the source can tell at a
glance and so can a face detector. Samples a handful of frames and checks
whether BOTH halves of the frame contain a face.

Cheap by construction: ~8 single-frame decodes, no full-video pass.
"""
import os
import subprocess
from pathlib import Path

_SAMPLES = 8
_ANALYSIS_W = 640

# Face detection was tried first and does NOT work for this. Measured on a
# real two-person debate, faces appeared in both halves of the same frame on
# only 2 of 8 samples - the guest wears a cap and aviator glasses, which
# defeats Haar most of the time, and single-speaker sources scored as high as
# 50% on one half simply because the speaker moves across frame.
#
# The seam is the honest signal. A side-by-side composite is two independent
# camera feeds butted together, so there is a hard vertical discontinuity at
# the exact midpoint that persists in every single frame. Measured ratio of
# midpoint gradient to average frame gradient:
#
#   two-person debate      23.96x   <- split
#   single TED speaker      3.14x
#   vertical short          1.12x
#   two people STACKED      1.55x   <- not side by side, so centre is right
#
# A threshold of 6 sits in the middle of a very wide gap. Sampling several
# frames and taking the median matters: a real seam persists, while an
# incidental vertical edge (a door frame at centre) does not.
_SEAM_THRESHOLD = 6.0


def _duration(video_path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _frame(video_path: Path, at: float, width: int):
    """Decode a single grayscale frame as a numpy array, or None."""
    import numpy as np

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(video_path)],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout.strip().split(",")
        sw, sh = int(probe[0]), int(probe[1])
        h = max(2, int(round(sh * (width / sw) / 2)) * 2)

        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{at:.2f}", "-i", str(video_path),
             "-frames:v", "1", "-vf", f"scale={width}:{h},format=gray",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True, timeout=120,
        ).stdout
        if len(raw) < width * h:
            return None
        return np.frombuffer(raw[: width * h], dtype=np.uint8).reshape(h, width)
    except Exception:
        return None


def _seam_ratio(img) -> float:
    """Midpoint vertical-gradient energy vs the frame average."""
    import numpy as np

    gx = np.abs(np.diff(img.astype(np.int16), axis=1)).mean(axis=0)
    half = img.shape[1] // 2
    mid = gx[max(half - 2, 0):half + 2].max()
    return float(mid / max(gx.mean(), 1e-6))


def detect(video_path: Path, verbose: bool = True) -> str:
    """Return "split" for a two-person side-by-side source, else "center"."""
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return "center"

    dur = _duration(video_path)
    if dur <= 0:
        return "center"

    # Sample across the middle 80% - intros and outros are often title cards.
    times = [dur * (0.1 + 0.8 * i / max(_SAMPLES - 1, 1)) for i in range(_SAMPLES)]
    ratios = []
    for t in times:
        img = _frame(video_path, t, _ANALYSIS_W)
        if img is not None:
            ratios.append(_seam_ratio(img))

    if not ratios:
        return "center"

    ratios.sort()
    median = ratios[len(ratios) // 2]
    choice = "split" if median >= _SEAM_THRESHOLD else "center"
    if verbose:
        print(f"      -> centre seam {median:.1f}x frame average "
              f"(threshold {_SEAM_THRESHOLD:.0f}x) -> {choice}")
    return choice
