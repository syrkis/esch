# %% main.py
#   esch main file for testing  #
# by: Noah Syrkis

# Imports
import esch
import numpy as np

# Constants
folder: str = "/Users/nobr/desk/s3/esch"
w, h, n = int(4), int(4), int(4)


# %% GRID
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=1)
arr = np.ones((h, w))[None, ...] * 0.8
esch.grid_fn(arr, e, shape="square")
esch.save(e.dwg, f"{folder}/grid.svg")


# %% ANIM GRID
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=1)
arr = np.absolute(np.random.randn(h, w, 100)[None, ...].cumsum(3))
esch.grid_fn(arr / arr.max(), e, shape="square", fps=1)
esch.save(e.dwg, f"{folder}/anim_grid.svg")


# %% MESH
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


# MULTI GRID
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=n)
arr = np.ones((n, h, w)) / 2
esch.grid_fn(arr, e, shape="square")
esch.save(e.dwg, f"{folder}/multi_grid.svg")


# MULTI ANIM GRID
e = esch.Drawing(h=h - 1, w=w - 1, row=1, col=n)
arr = np.absolute(np.random.randn(n, h, w, 100).cumsum(3))
esch.grid_fn(arr / arr.max(), e, shape="square", fps=1)
esch.save(e.dwg, f"{folder}/multi_anim_grid.svg")

# %% MULTI MESH
e = esch.Drawing(h=h, w=w, row=n, col=1)
pos = np.stack((np.random.uniform(0, h - 1, (100, n)), np.random.uniform(0, w - 1, (100, n)))).T
arr = np.random.uniform(0, 1, (n, 100))
esch.mesh_fn(pos, arr, e)
esch.save(e.dwg, f"{folder}/multi_mesh.svg")


# %% MULTI ANIM MESH
e = esch.Drawing(h=h, w=w, row=1, col=n)
pos = np.stack((np.random.uniform(0, h - 1, (1000, n)), np.random.uniform(0, w - 1, (1000, n)))).T
arr = np.abs(np.random.randn(n, 1000, 20).cumsum(2))
esch.mesh_fn(pos, arr / arr.max(), e)
esch.save(e.dwg, f"{folder}/multi_anim_mesh.svg")


# %% MULTI SIMS
e = esch.Drawing(h=h, w=w, row=1, col=n)
pos = np.stack((np.random.uniform(0, h - 1, (100, n)), np.random.uniform(0, w - 1, (100, n)))).T
arr = np.random.uniform(0, 1, (n, 100))
esch.mesh_fn(pos, arr, e)
esch.save(e.dwg, f"{folder}/multi_mesh.svg")


# exit()
# %% test sims
# dwg = esch.init(100, 100)
# pos = np.random.randn(100, 2, 1000).cumsum(axis=2) + np.array((50, 50))[..., None]
# fill = (["black"] * 50) + (["none"] * 50)
# size = [random.randint(1, 4) for _ in range(100)]
# random.shuffle(fill)
# esch.anim_sims_fn(pos, dwg, fill=fill, size=size)

# start_positions = np.random.uniform(0, 100, (100, 2))
# end_positions = np.random.uniform(0, 100, (100, 2))
# shot_times = np.random.uniform(0, 10, 100)
# size = [random.randint(1, 10) for _ in range(100)]

# esch.anim_shot_fn(start_positions, end_positions, shot_times, size=size, dwg=dwg)
# esch.save(dwg, f"{folder}/anim_sims.svg")

# %% test mix
# dwg = esch.init(10, 5)
# pos = np.random.randn(100, 2, 200).cumsum(axis=2) * 0.1 + np.array((4.5, 2.25))[..., None]
# esch.anim_sims_fn(pos, dwg)
# arr = np.random.uniform(0, 1, (10, 5))
# esch.grid_fn(arr, dwg)
# esch.save(dwg, f"{folder}/anim_mix.svg")


# %% test anim multi
# dwg, gs = esch.init(5, 10, rows=3)
# arr = np.absolute(np.random.randn(3, 5, 10, 100).cumsum(3))
# for idx, g in enumerate(gs):
# esch.anim_grid_fn(arr[idx] / arr[idx].max(), dwg, g)
# dwg.add(g)
# esch.save(dwg, f"{folder}/anim_multi.svg")

# %% test sims with shots
# dwg = esch.init(100, 100)
# pos = np.random.randn(100, 2, 1000).cumsum(axis=2) + np.array((50, 50))[..., None]
# fill = (["black"] * 50) + (["none"] * 50)
# size = [random.randint(1, 4) for _ in range(100)]
# random.shuffle(fill)
# esch.anim_sims_fn(pos, dwg, fill=fill, size=size)
#
# start_positions = np.random.uniform(0, 100, (100, 2))
# end_positions = np.random.uniform(0, 100, (100, 2))
# shot_times = np.random.uniform(0, 10, 100)
# size = [random.randint(1, 10) for _ in range(100)]
#
# esch.anim_shot_fn(start_positions, end_positions, shot_times, size=size, dwg=dwg)
# esch.save(dwg, f"{folder}/anim_sims.svg")
