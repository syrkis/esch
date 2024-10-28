# %% data.py
#   hinton plots data processing
# by: Noah Syrkis

# %% Imports
from jax import jit, vmap, Array
import jax.numpy as jnp
import numpy as np


# %% Constants
MIN_ABS_VALUE = 0.05
MAX_ABS_VALUE = 0.95


# %% Functions
@jit
def norm(x: Array) -> Array:
    """Normalize data to the range [0, 1]."""
    return x / jnp.max(jnp.abs(x))


@jit
def clip(x: Array) -> Array:
    """Get rid of zeros"""
    return jnp.where(jnp.abs(x) < MIN_ABS_VALUE, jnp.where(x < 0, -MIN_ABS_VALUE, MIN_ABS_VALUE), x)


@jit
def prep(x: Array) -> Array:
    """Preprocess data for visualization."""
    return clip(norm(x))
