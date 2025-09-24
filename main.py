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
    arr = random.normal(0, 1, shape)
    e = esch.draw(pattern, arr, debug=True)
    e.save(f"figs_{pattern}.svg")
exit()


# %% mesh tests
arr, pos = random.random((units, time)), random.random(((points, 2, time)))  #  "plots nums time, plots points xy time"
arr, pos = random.random((units,)), random.random(((points, 2, time)))  #  "plots nums time, plots points xy time"
arr, pos = random.random((units, time)), random.random(((points, 2)))  #  "plots nums time, plots points xy time"
arr, pos = random.random((units, time)), random.random(((points, 2, time)))  #  "nums time, points xy time"
arr, pos = random.random((units, time)), random.random(((points, 2)))  #  "points time, nums xy"
arr, pos = random.random((units,)), random.random(((points, 2, time)))  #  "points, points xy time"
arr, pos = random.random((units,)), random.random(((points, 2)))  #  "points, points xy"


# %% GRID ######################################################################################
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=1)
# arr = np.ones((h, w))[None, ...] * 0.8
arr = np.random.uniform(0, 1, (h, w))[None, ...] * 0.8
print(arr.shape)
esch.grid_fn(e, arr, shape="square")
esch.save(e.dwg, f"{folder}/grid.svg")


# %% ANIM GRID
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=1)
arr = np.absolute(np.random.randn(h, w, 100)[None, ...].cumsum(3))
esch.grid_fn(e, arr / arr.max() / 2, shape="square", fps=1)
esch.save(e.dwg, f"{folder}/anim_grid.svg")


# %% MESH #####################################################################################
e = esch.Drawing(h=h, w=w, row=1, col=1)
pos = np.stack((np.random.uniform(0, h, 100), np.random.uniform(0, w, 100))).T[None, ...]
arr = np.random.uniform(0, 1, 100)[None, ...]
esch.mesh_fn(e, pos, arr)
esch.save(e.dwg, f"{folder}/mesh.svg")


# %% ANIM MESH
e = esch.Drawing(h=h, w=w, row=1, col=1)
pos = np.stack((np.random.uniform(0, h, 1000), np.random.uniform(0, w, 1000))).T[None, ...]
arr = np.abs(np.random.randn(1000, 20)[None, ...].cumsum(2))
esch.mesh_fn(e, pos, arr / arr.max())
esch.save(e.dwg, f"{folder}/anim_mesh.svg")


# MULTI GRID ####################################################################################
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=n, pad=4)
arr = np.random.uniform(0, 1, (n, h, w)) * 0.8
esch.grid_fn(e, arr, shape="square")
esch.save(e.dwg, f"{folder}/multi_grid.svg")
print(arr.shape)


# MULTI ANIM GRID
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=n)
arr = np.absolute(np.random.randn(n, h, w, 100).cumsum(3))
esch.grid_fn(e, arr / arr.max(), shape="square", fps=1)
esch.save(e.dwg, f"{folder}/multi_anim_grid.svg")

# %% MULTI MESH ###################################################################################
e = esch.Drawing(h=h, w=w, row=n, col=1)
pos = np.stack((np.random.uniform(0, h, (100, n)), np.random.uniform(0, w, (100, n)))).T
arr = np.random.uniform(0, 1, (n, 100))
esch.mesh_fn(e, pos, arr)
esch.save(e.dwg, f"{folder}/multi_mesh.svg")


# %% MULTI ANIM MESH
e = esch.Drawing(h=h, w=w, row=1, col=n)
pos = np.stack((np.random.uniform(0, h, (1000, n)), np.random.uniform(0, w, (1000, n)))).T
arr = np.abs(np.random.randn(n, 1000, 20).cumsum(2))
esch.mesh_fn(e, pos, arr / arr.max())
esch.save(e.dwg, f"{folder}/multi_anim_mesh.svg")
