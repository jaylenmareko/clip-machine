"""Picture treatment — a nostalgic, slightly-degraded film/tape feel.

The brief was audio distortion translated to picture: a mic yelled into,
slightly broken up, old-school and a bit cool.

Two earlier attempts failed and both are worth recording so they are not
repeated:

  1. Pushing contrast UP read harsh and modern. Analog media COMPRESSES
     dynamic range - milky blacks, rolled-off highlights. A clipped mic loses
     dynamic range too; that is the real shared quality with the audio
     reference, not "more aggressive".
  2. `chromashift` plus a colour mixer that dropped blue while lifting red
     produced a purple cast. Warmth comes from `colortemperature` instead,
     which works in Kelvin and cannot throw magenta.

What actually carries the feeling is LOST DETAIL, not colour: old footage is
soft and low-resolution before it is any particular hue (VHS was ~240 lines).
Each look downsamples and scales back up, so real detail is discarded rather
than blurred over.

Every chain ends in format=yuv420p. Filters like colortemperature output
yuv444p, which encodes as H.264 High 4:4:4 Predictive and will not play in
Windows Movies & TV. That bug shipped once already.
"""

_SAFE = "format=yuv420p"


# --------------------------------------------------------------------------
# Rounded corners
#
# ffmpeg has no rounded-rectangle primitive. `geq` can express one, but it is
# a per-pixel expression VM - fine for a single frame, hopeless for 30fps
# (720x1280x30x24s is ~660M evaluations). So the mask is rendered ONCE to a
# cached PNG and composited per clip, which is effectively free.
# --------------------------------------------------------------------------
def corner_mask(width: int, height: int, radius: int):
    """Path to a cached rounded-rect mask (white inside, black outside)."""
    import subprocess

    from . import config

    path = config.WORK_DIR / f"mask_{width}x{height}_r{radius}.png"
    if path.exists():
        return path
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Distance from the nearest corner-circle centre, but only inside the
    # corner squares - elsewhere dx and dy clamp to 0 so the test passes.
    dx = f"max(max({radius}-X,X-({width}-1-{radius})),0)"
    dy = f"max(max({radius}-Y,Y-({height}-1-{radius})),0)"
    expr = f"if(lte(hypot({dx},{dy}),{radius}),255,0)"

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=black:s={width}x{height}",
         "-vf", f"geq=lum='{expr}':cb=128:cr=128,format=gray",
         "-frames:v", "1", str(path)],
        check=True, capture_output=True, timeout=120,
    )
    return path


def _chain(width: int, temp: int, sat: float, black: float, white: float,
           grain: int, lines: bool, out_w: int, out_h: int) -> str:
    """One recipe at a given intensity.

    width  - how far down the picture is resampled before coming back up.
             Lower means more real detail thrown away.
    temp   - colour temperature in Kelvin. Below ~5000 reads warm.
    black/white - the faded curve: lifted black, rolled highlight.
    lines  - halve vertical resolution and double it back with nearest
             neighbour, duplicating scanlines the way tape does.
    """
    if lines:
        resample = (f"scale={width}:-2:flags=bilinear,"
                    f"scale={out_w}:{out_h // 2}:flags=bilinear,"
                    f"scale={out_w}:{out_h}:flags=neighbor")
    else:
        resample = (f"scale={width}:-2:flags=bilinear,"
                    f"scale={out_w}:{out_h}:flags=bilinear")

    return (
        f"{resample},"
        f"colortemperature=temperature={temp},"
        f"eq=saturation={sat},"
        f"curves=all='0/{black} 1/{white}',"
        f"noise=alls={grain}:allf=t+u,"
        f"{_SAFE}"
    )


# Ordered light -> strong. Default is "none" - the treatments are options, not
# a house style. "nostalgia" is the one Jaylen picked, dialled back a step.
LOOKS = ("none", "nostalgia-light", "nostalgia", "nostalgia-strong")
DEFAULT = "none"

_PRESETS = {
    #                     width temp   sat  black white grain lines
    "nostalgia-light":   (600,  4900, 0.94, 0.03, 0.97,   7, False),
    "nostalgia":         (500,  4650, 0.90, 0.05, 0.95,  11, False),
    "nostalgia-strong":  (400,  4300, 0.86, 0.07, 0.93,  15, True),
}


def filters(look: str, out_w: int, out_h: int) -> str | None:
    """Filter chain for a look, or None for no treatment."""
    preset = _PRESETS.get(look)
    if look == "none" or preset is None:
        return None
    width, temp, sat, black, white, grain, lines = preset
    return _chain(width, temp, sat, black, white, grain, lines, out_w, out_h)
