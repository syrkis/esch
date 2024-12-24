# %% mesh.py
#   esch mesh plot
# by: Noah Syrkis

# imports
from esch.plot import plot


# %% Functions
def mesh(act, pos, **kwargs):
    """Create mesh plot for activations."""
    act = act.reshape((1,) * (3 - act.ndim) + act.shape)
    pos = pos.reshape((1,) * (3 - pos.ndim) + pos.shape)
    assert act.shape[-1] == pos.shape[-2], "we need more (or less) coords"
    dwg = plot(act, pos, **kwargs)
    return dwg
