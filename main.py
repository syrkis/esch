# %% main.py  ###################
#   esch main file for testing  #
# by: Noah Syrkis  ##############

# Imports  ##########
import esch
import numpy as np


# %% grid test
dwg = esch.init(10, 5)
arr = np.ones((10, 5))
esch.grid_fn(arr, dwg, shape="square")
esch.save(dwg, "paper/figs/grid.svg")

# %% anim grid test
dwg = esch.init(10, 5)
arr = np.absolute(np.random.randn(10, 5, 10).cumsum(2))
esch.anim_grid_fn(arr / arr.max(), dwg, shape="square")
esch.save(dwg, "paper/figs/anim_grid.svg")

# %% test mesh
dwg = esch.init(10, 5)
pos = np.stack((np.random.uniform(0, 10, 10), np.random.uniform(0, 5, 10))).T
arr = np.random.uniform(0, 1, 10)
pos, arr = np.random.uniform(0, 9, (10, 2)), np.random.uniform(0, 1, 9)
esch.mesh_fn(pos, arr, dwg)
esch.save(dwg, "paper/figs/mesh.svg")

# %% test anim mesh
dwg = esch.init(10, 5)
pos = np.stack((np.random.uniform(0, 10, 10), np.random.uniform(0, 5, 10))).T
arr = np.abs(np.random.randn(10, 20).cumsum(1))
esch.anim_mesh_fn(pos, arr / arr.max(), dwg)
esch.save(dwg, "paper/figs/anim_mesh.svg")

# %% test sims
dwg = esch.init(100, 50)
pos = np.random.randn(100, 2, 200).cumsum(axis=2) * 0.1 + np.array((45, 22.5))[..., None]
esch.anim_sims_fn(pos, dwg)
esch.save(dwg, "paper/figs/anim_sims.svg")

# %% test mix
dwg = esch.init(10, 5)
pos = np.random.randn(100, 2, 200).cumsum(axis=2) * 0.1 + np.array((4.5, 2.25))[..., None]
esch.anim_sims_fn(pos, dwg)
arr = np.random.uniform(0, 1, (10, 5))
esch.grid_fn(arr, dwg)
esch.save(dwg, "paper/figs/anim_mix.svg")


# %% test multi
dwg = esch.init(10, 5, cols=3)
arr = np.absolute(np.random.randn(3, 10, 5))
for i in range(len(arr)):
    group = dwg.g()
    group.translate(0, (5 + 1) * i)
    esch.grid_fn(arr[i] / arr[i].max(), dwg, group)
    dwg.add(group)
esch.save(dwg, "paper/figs/multi.svg")


# %% test anim multi
dwg = esch.init(10, 5, cols=3)
arr = np.absolute(np.random.randn(3, 10, 5, 10).cumsum(3))
for i in range(len(arr)):
    group = dwg.g()
    group.translate(0, (5 + 1) * i)
    esch.anim_grid_fn(arr[i] / arr[i].max(), dwg, group)
    dwg.add(group)
esch.save(dwg, "paper/figs/anim_multi.svg")
