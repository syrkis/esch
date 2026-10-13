"""Looping SMIL animations that keep only the keyframes interpolation needs."""

import numpy as np


def number(value):
    """Format a coordinate to 0.001 grid units, without redundant zeros."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def keyframes(track, times, tolerance):
    """Greedy keyframe indices whose linear interpolation stays within tolerance.

    track has shape (frame, point, coordinate); error is the Euclidean distance of
    each point from the interpolation of the surrounding keyframes. Each step
    depends on the previous keyframe, so this stays a sequential search.
    """
    keep, start, last = [0], 0, len(track) - 1
    while start < last:
        end = start + 1
        while end < last:
            span = slice(start + 1, end + 2)
            weight = ((times[span] - times[start]) / (times[end + 1] - times[start]))[:, None, None]
            line = track[start] + weight * (track[end + 1] - track[start])
            if np.max(np.linalg.norm(line - track[span], axis=-1)) > tolerance:
                break
            end += 1
        keep.append(end)
        start = end
    return keep


def _kept(values, times, mode, track, tolerance):
    """Indices of the keyframes that change what is shown."""
    values = np.asarray(values)
    if mode == "discrete":
        # A discrete value at the loop's end time is never shown.
        changes = (values[1:] != values[:-1]) & (times[1:] < 1)
        return [0, *(np.flatnonzero(changes) + 1)]
    if track is not None:
        return keyframes(np.asarray(track, dtype=float), times, tolerance)
    repeated = (values[:-2] == values[1:-1]) & (values[1:-1] == values[2:])
    return [0, *(np.flatnonzero(~repeated) + 1), len(values) - 1]


def animate(e, node, name, values, times, duration, *, mode=None, track=None, tolerance=5e-4):
    """Add a looping SMIL animation of formatted values at times in [0, 1].

    Keyframes that interpolation reproduces are dropped: numeric tracks within
    tolerance, discrete runs and repeated linear values exactly. keyTimes are
    omitted where they equal the SMIL default spacing.
    """
    times = np.asarray(times, dtype=float)
    keep = _kept(values, times, mode, track, tolerance)
    values, times = [values[i] for i in keep], times[keep]
    if len(set(values)) <= 1:
        return
    count = len(values)
    uniform = np.arange(count) / count if mode == "discrete" else np.linspace(0, 1, count)
    attributes = {"attributeName": name, "values": ";".join(values), "dur": f"{duration:g}s", "repeatCount": "indefinite"}
    if mode is not None:
        attributes["calcMode"] = mode
    if not np.allclose(times, uniform, rtol=0, atol=1e-9):
        attributes["keyTimes"] = ";".join(f"{time:.6g}" for time in times)
    node.add(e.dwg.animate(**attributes))
