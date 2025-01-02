# %% mesh.py
#   esch mesh plot
# by: Noah Syrkis

# imports
from esch.plot import draw
from esch.util import display_fn, setup_drawing


# %% Functions
def mesh(act, pos, **kwargs):
    """Create mesh plot for activations."""
    pos = pos - pos.min(axis=0)
    pos /= pos.max()
    act = act.reshape((1,) * (3 - act.ndim) + act.shape + (1,) * (2 - act.ndim))
    act = act / act.max() ** 4  # this is severly suboptimal
    dwg = setup_drawing(act, pos)
    path = kwargs.get("path", None)
    dwg = draw(dwg, act, pos)
    dwg.saveas(path) if path else display_fn(dwg)
    return dwg
