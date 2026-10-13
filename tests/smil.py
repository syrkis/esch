"""Evaluate SMIL <animate> elements at loop fractions, as browsers do.

Tests check what an animation shows rather than how its keyframes are encoded.
"""

import numpy as np
import webcolors


def keyframes(animation):
    """Values and key times, applying SMIL's default spacing when keyTimes is absent."""
    values = animation.get("values").split(";")
    if animation.get("keyTimes"):
        times = np.array([float(time) for time in animation.get("keyTimes").split(";")])
    elif animation.get("calcMode") == "discrete":
        times = np.arange(len(values)) / len(values)
    else:
        times = np.linspace(0, 1, len(values))
    return values, times


def numbers(value):
    """A number, point list or color as a float array."""
    if value.startswith("#") or value.isalpha():
        return np.array(webcolors.hex_to_rgb(value) if value.startswith("#") else webcolors.name_to_rgb(value), float)
    return np.array([float(part) for part in value.replace(",", " ").split()])


def sample(animation, t):
    """The value shown at fraction t of the loop: a string when discrete, else floats."""
    values, times = keyframes(animation)
    index = int(np.searchsorted(times, t, side="right")) - 1
    if animation.get("calcMode") == "discrete":
        return values[index]
    index = min(index, len(values) - 2)
    span = times[index + 1] - times[index]
    weight = 1.0 if span == 0 else (t - times[index]) / span
    start, end = numbers(values[index]), numbers(values[index + 1])
    return start + weight * (end - start)
