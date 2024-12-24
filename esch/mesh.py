# %% mesh.py
#   esch mesh plot
# by: Noah Syrkis

# imports
from esch.plot import plot


# %% Functions
def mesh(act, pos, **kwargs):
    """Create mesh plot for activations."""
    pos = pos - pos.min(axis=0)
    act = act.reshape((1,) * (3 - act.ndim) + act.shape + (1,) * (2 - act.ndim))
    act = act / 10
    dwg = plot(act, pos, **kwargs)
    return dwg
