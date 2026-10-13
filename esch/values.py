"""Validated numeric inputs: value arrays and playback rates."""

import numpy as np


def values_array(arr, *, normalize=False):
    """Convert real, finite, nonempty array-like data without mutating it."""
    arr = np.asarray(arr)
    if arr.dtype.kind not in "biuf" or not arr.size or not np.isfinite(arr).all():
        raise ValueError("Expected a nonempty array of finite real numbers")
    arr = arr.astype(float)
    if normalize:
        maximum = float(np.max(np.abs(arr)))
        if maximum:
            arr /= maximum
    return arr


def rate(fps):
    """Check a playback rate in frames per second."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    return fps
