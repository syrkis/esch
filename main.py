# %% main.py
#   esch main file for testing
# by: Noah Syrkis

# Imports
import numpy as np
import esch

x = np.abs(np.random.randn(2, 20, 100))
img = esch.mesh(x)
esch.util.display_fn(img)
