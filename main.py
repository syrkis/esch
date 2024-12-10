# %% main.py
#   esch main file for testing
# by: Noah Syrkis

# Imports
import numpy as np
import esch

x = np.random.randn(10)
img = esch.mesh(x, path="test.svg")
esch.util.display_fn(img)
