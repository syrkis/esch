# %% main.py
#   esch main file for testing  #
# by: Noah Syrkis

# Imports
import esch
import numpy as np

# Constants
folder: str = "/Users/nobr/desk/s3/esch"
w, h, n = int(16), int(16), int(4)


# %% GRID ######################################################################################
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=1)
# arr = np.ones((h, w))[None, ...] * 0.8
arr = np.random.uniform(0, 1, (h, w))[None, ...] * 0.8
esch.grid_fn(arr, e, shape="square")
esch.save(e.dwg, f"{folder}/grid.svg")


# %% ANIM GRID
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=1)
arr = np.absolute(np.random.randn(h, w, 100)[None, ...].cumsum(3))
esch.grid_fn(arr / arr.max() / 2, e, shape="square", fps=1)
esch.save(e.dwg, f"{folder}/anim_grid.svg")


# %% MESH #####################################################################################
e = esch.Drawing(h=h, w=w, row=1, col=1)
pos = np.stack((np.random.uniform(0, h, 100), np.random.uniform(0, w, 100))).T[None, ...]
arr = np.random.uniform(0, 1, 100)[None, ...]
esch.mesh_fn(pos, arr, e)
esch.save(e.dwg, f"{folder}/mesh.svg")


# %% ANIM MESH
e = esch.Drawing(h=h, w=w, row=1, col=1)
pos = np.stack((np.random.uniform(0, h, 1000), np.random.uniform(0, w, 1000))).T[None, ...]
arr = np.abs(np.random.randn(1000, 20)[None, ...].cumsum(2))
esch.mesh_fn(pos, arr / arr.max(), e)
esch.save(e.dwg, f"{folder}/anim_mesh.svg")


# MULTI GRID ####################################################################################
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=n, pad=4)
arr = np.random.uniform(0, 1, (n, h, w)) * 0.8
esch.grid_fn(arr, e, shape="square")
esch.save(e.dwg, f"{folder}/multi_grid.svg")
print(arr.shape)


# MULTI ANIM GRID
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=n)
arr = np.absolute(np.random.randn(n, h, w, 100).cumsum(3))
esch.grid_fn(arr / arr.max(), e, shape="square", fps=1)
esch.save(e.dwg, f"{folder}/multi_anim_grid.svg")

# %% MULTI MESH ###################################################################################
e = esch.Drawing(h=h, w=w, row=n, col=1)
pos = np.stack((np.random.uniform(0, h, (100, n)), np.random.uniform(0, w, (100, n)))).T
arr = np.random.uniform(0, 1, (n, 100))
esch.mesh_fn(pos, arr, e)
esch.save(e.dwg, f"{folder}/multi_mesh.svg")


# %% MULTI ANIM MESH
e = esch.Drawing(h=h, w=w, row=1, col=n)
pos = np.stack((np.random.uniform(0, h, (1000, n)), np.random.uniform(0, w, (1000, n)))).T
arr = np.abs(np.random.randn(n, 1000, 20).cumsum(2))
esch.mesh_fn(pos, arr / arr.max(), e)
esch.save(e.dwg, f"{folder}/multi_anim_mesh.svg")


#################################################################################################
# NOT WORKING YET
# %% SIMS #######################################################################################
e = esch.Drawing(h=h, w=w, row=1, col=1)
pos = np.stack((np.random.uniform(0, h, (1, 88, 1000)), np.random.uniform(0, w, (1, 88, 1000)))).transpose((1, 3, 0, 2))
esch.sims_fn(pos, e)
esch.save(e.dwg, f"{folder}/sims.svg")

# %% MULTI SIMS
e = esch.Drawing(h=h, w=w, row=1, col=n)
pos = np.stack((np.random.uniform(0, h, (n, 88, 500)), np.random.uniform(0, w, (n, 88, 500)))).transpose((1, 3, 0, 2))
esch.sims_fn(pos, e)
esch.save(e.dwg, f"{folder}/multi_sims.svg")
