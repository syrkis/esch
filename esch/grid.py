# %% grid.py
#    esch grid functions
# by: Noah Syrkis

# Imports
import numpy as np
from esch.plot import plot

# from esch.util import BASE_SIZE
from einops import rearrange


# Functions
def grid(act, **kwargs):
    """Create grid plot for activations."""
    act = act.reshape((1,) * (2 - act.ndim) + act.shape + (1,) * (3 - act.ndim))
    act, pos = pos_fn(act)
    dwg = plot(act, pos, **kwargs)
    return dwg


def pos_fn(act):  # n points x 2
    n_plots, n_rows, n_cols, n_steps = act.shape
    width = n_rows / max(n_rows, n_cols)
    height = n_cols / max(n_rows, n_cols)
    x = np.linspace(0, width, n_rows)
    y = np.linspace(0, height, n_cols)
    x, y = np.meshgrid(x, y)
    pos = np.stack((x.flatten(), y.flatten()), axis=1)
    act = rearrange(act, "n_plots n_rows n_cols n_frames -> n_plots (n_rows n_cols) n_frames")
    act = act / np.max(np.abs(act)) / max(n_rows, n_cols)
    return act, pos
