# %% grid.py
#    esch grid functions
# by: Noah Syrkis

# Imports
import numpy as np
from esch.plot import draw
from esch.util import setup_drawing
from einops import rearrange
from esch.util import display_fn

grid_reshape_dict = {
    1: lambda x: x[None, None, ..., None],
    2: lambda x: x[None, ..., None],
    3: lambda x: x[None, ...],
    4: lambda x: x,
}

## TODO: REMOVE ZERO ELEMENTS


# Functions
def grid(act, **kwargs):
    """Create grid plot for activations."""
    act = grid_reshape_dict[act.ndim](act)
    act, pos = grid_pos_fn(act)
    dwg = draw(setup_drawing(act, pos), act, pos)
    path = kwargs.get("path", None)
    dwg.saveas(path) if path else display_fn(dwg)
    return dwg


def grid_pos_fn(act):  # n points x 2
    n_plots, n_rows, n_cols, n_steps = act.shape
    width = n_rows / max(n_rows, n_cols)
    height = n_cols / max(n_rows, n_cols)
    x = np.linspace(0, width, n_rows)
    y = np.linspace(0, height, n_cols)
    x, y = np.meshgrid(x, y)
    pos = np.stack((y.flatten(), x.flatten()), axis=1)
    act = rearrange(act, "n_plots n_rows n_cols n_frames -> n_plots (n_rows n_cols) n_frames")
    act = act / np.max(np.abs(act)) / max(n_rows, n_cols)
    return act, pos
