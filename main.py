# %% main.py
#   esch main file for testing
# by: Noah Syrkis

# Imports
import numpy as np
import esch

# %%
# act = np.abs(np.random.randn(5, 20, 10, 10)) ** 0.1 # activation
act = np.random.randn(5, 400, 5, 5)
dwg = esch.mesh(act, path="test.svg", fps=1)
pos = np.random.randn(5, 2)
act = np.random.randn(400, 5)
dwg = esch.mesh(act, pos, path="test.svg", fps=1)
esch.util.display_fn(dwg)
