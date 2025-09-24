# %% main.py
#   esch main file for testing
# by: Noah Syrkis

# Imports
import esch
import numpy as np
from numpy import random

# row column depth channel time
N, C, H, W, T = int(2), int(4), int(8), int(16), int(32)


# grid tests
grid_test = [
    ((N, C, H, W, T), "n c h w t"),
    ((H, W, T), "h w t"),
    ((N, H, W), "n h w"),
    ((H, W), "h w"),
    ((H, T), "h t"),
    ((W, T), "w t"),
    ((T,), "t"),
]

for shape, pattern in grid_test:
    arr = random.normal(0, 1, shape) * 0.5
    e = esch.draw(pattern, arr, debug=True)
    e.save(f"figs_{pattern}.svg")
