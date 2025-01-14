# %% grid.py
#    esch grid functions
# by: Noah Syrkis

# Imports
import numpy as np
from esch.plot import draw
from esch.util import setup_drawing
from einops import rearrange
from esch.util import display_fn


# Functions
def grid(ink, path=None, **kwargs):
    """Create grid plot for activations."""
    ink, pos = grid_pos_fn(ink)
    dwg = draw(setup_drawing(ink, pos), ink, pos)
    dwg.saveas(path) if path else display_fn(dwg)
    return dwg


def grid_pos_fn(ink):  # n points x 2
    reshape_dict = {1: lambda x: x[None, None, ..., None], 2: lambda x: x[None, ..., None], 3: lambda x: x[None, ...]}
    ink = reshape_dict[ink.ndim](ink) if ink.ndim < 4 else ink
    n_plots, n_rows, n_cols, n_steps = ink.shape
    width = n_rows / max(n_rows, n_cols)
    height = n_cols / max(n_rows, n_cols)
    x = np.linspace(0, width, n_rows)
    y = np.linspace(0, height, n_cols)
    x, y = np.meshgrid(x, y)
    pos = np.stack((y.flatten(), x.flatten()), axis=1)
    ink = rearrange(ink, "n_plots n_rows n_cols n_frames -> n_plots (n_rows n_cols) n_frames")
    ink = ink / np.max(np.abs(ink)) / max(n_rows, n_cols)
    return ink, pos
