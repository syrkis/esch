# %% main.py
#   esch main file for testing
# by: Noah Syrkis

# Imports
import numpy as np
import esch


# %% GRID TESTS
act = np.abs(np.random.randn(10))  # 1 x 1 x 1 x 10
esch.grid(act, path="paper/figs/grid_1d.svg")

act = np.abs(np.random.randn(6, 7))
esch.grid(act, path="paper/figs/grid_2d.svg")

act = np.abs(np.random.randn(3, 20, 10))
esch.grid(act, path="paper/figs/grid_3d.svg")

act = np.abs(np.random.randn(100, 3, 20, 10))
esch.grid(act, path="paper/figs/grid_4d.svg")

# %% MESH TEST
act = np.abs(np.random.randn(10))
pos = np.abs(np.random.randn(10, 2))
esch.mesh(act, pos, path="paper/figs/mesh_2d.svg")

act = np.abs(np.random.randn(3, 10))
pos = np.abs(np.random.randn(10, 2))
esch.mesh(act, pos, path="paper/figs/mesh_3d.svg")

act = np.abs(np.random.randn(100, 1, 450))
pos = np.random.uniform(0, 1, (450, 2)) * np.array([1, 2])[None, :]
esch.mesh(act, pos, path="paper/figs/mesh_4d.svg")


# %% SIMS TEST
pos = np.abs(np.random.normal(0, 1, (100, 10, 2)).cumsum(axis=1))
esch.sims(pos, path="paper/figs/sims_3d.svg")

# pos = np.random.normal(0, 1, (100, 3, 20, 2)).cumsum(axis=1)
# esch.sims(pos, path="paper/figs/sims_4d.svg")
