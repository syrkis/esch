"""Colors: CSS names, hex strings and RGB triplets, and blends toward white."""

import numpy as np
import webcolors


def color(value):
    """Convert one RGB triplet to SVG hex, preserving existing string colors."""
    if isinstance(value, str):
        return value
    rgb = np.asarray(value)
    if rgb.shape != (3,) or rgb.dtype.kind not in "iuf" or not np.isfinite(rgb).all():
        raise ValueError("Color must be a string or a finite RGB triplet")
    if np.any(rgb < 0) or np.any(rgb > 255):
        raise ValueError("RGB channels must be between 0 and 255")
    # Float unit-interval colors are common in scientific Python palettes.
    if rgb.dtype.kind == "f" and np.all(rgb <= 1):
        rgb = rgb * 255
    return webcolors.rgb_to_hex(tuple(np.rint(rgb).astype(int)))


def blend(value, weights):
    """Blend an opaque CSS3 name or hex color toward white in sRGB, one hex per weight."""
    try:
        rgb = webcolors.hex_to_rgb(value) if value.startswith("#") else webcolors.name_to_rgb(value)
    except (ValueError, AttributeError) as error:
        raise ValueError("Magnitude colors must be CSS3 color names or #RGB/#RRGGBB hex colors") from error
    weights = np.asarray(weights, dtype=float)[:, None]
    mixed = np.rint(255 * (1 - weights) + np.asarray(rgb) * weights).astype(int)
    return [webcolors.rgb_to_hex(tuple(channels)) for channels in mixed]
