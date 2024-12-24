# %% main.py
#   esch main file for testing
# by: Noah Syrkis

# Imports
import numpy as np
import esch

# GRID TESTS
# act = np.random.randn(10)
# esch.grid(act, path="1d.svg")
#
# act = np.random.randn(10, 20)
# esch.grid(act, path="2d.svg")

# act = np.random.randn(20, 10, 5)
# esch.grid(act, path="3d.svg")

act = np.random.randn(5, 10, 40, 10)
esch.grid(act, path="4d.svg")

# %% MESH TEST
# act = np.random.randn(10)
# pos = np.random.randn(10, 2)
# esch.mesh(act, pos)

# exit()
# act = np.random.randn(10, 100).T
# pos = np.random.randn(10, 100, 2).transpose(1, 0, 2)
# print(act[:2], pos[:2])
# esch.mesh(act, pos, path="test.svg")
#
#
# act = np.array(jnp.load("bolds.npy").T)
# pos = np.array(jnp.load("coords.npy").transpose(1, 0, 2))
# pos = (pos - pos.mean()) / pos.std()
# print(act[:2], pos[:2])
# esch.mesh(act, pos, path="test.svg")
