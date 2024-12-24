# %% data.py
#   hinton plots data processing
# by: Noah Syrkis

# %% Imports
import numpy as np
from esch.util import Array


# %% Constants
MIN_ABS_VALUE = 0.05
MAX_ABS_VALUE = 0.95


# %% Functions
def norm(x: Array) -> Array:
    """Normalize data to the range [0, 1]."""
    return x / np.max(np.abs(x))


def clip(x: Array):
    """Get rid of zeros"""
    return np.where(np.abs(x) < MIN_ABS_VALUE, np.where(x < 0, -MIN_ABS_VALUE, MIN_ABS_VALUE), x)


def prep(x: Array):
    """Preprocess data for visualization."""
    return clip(norm(x))
