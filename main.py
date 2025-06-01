# %% main.py  ###################
#   esch main file for testing  #
# by: Noah Syrkis  ##############

# Imports  ##########
import esch
import numpy as np
import random


# %% grid test
dwg = esch.init(10, 5)
arr = np.ones((10, 5))
esch.grid_fn(arr, dwg, shape="square")
esch.save(dwg, "paper/figs/grid.svg")

# %% anim grid test
dwg = esch.init(10, 5)
arr = np.absolute(np.random.randn(10, 5, 100).cumsum(2))
esch.anim_grid_fn(arr / arr.max(), dwg, shape="square", fps=1)
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
dwg = esch.init(100, 100)
pos = np.random.randn(100, 2, 1000).cumsum(axis=2) + np.array((50, 50))[..., None]
fill = (["black"] * 50) + (["none"] * 50)
size = [random.randint(1, 4) for _ in range(100)]
random.shuffle(fill)
esch.anim_sims_fn(pos, dwg, fill=fill, size=size)

start_positions = np.random.uniform(0, 100, (100, 2))
end_positions = np.random.uniform(0, 100, (100, 2))
shot_times = np.random.uniform(0, 10, 100)
size = [random.randint(1, 10) for _ in range(100)]

esch.anim_shot_fn(start_positions, end_positions, shot_times, size=size, dwg=dwg)
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
