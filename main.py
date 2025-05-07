# %% main.py  ###################
#   esch main file for testing  #
# by: Noah Syrkis  ##############

# Imports  ##########
import esch
import numpy as np
from itertools import product


# %% GRID TEST
def grid_fn(shape):
    x = np.random.random(shape)
    e = esch.init(*shape[:-1])
    e = esch.grid(e, x)
    esch.save(e, path=f"paper/figs/grid_{x.shape}.svg")
    # print()


options = [(3, 1), (5, 1), (7, 1), (101, 1)]
list(map(grid_fn, list(product(*options))))

# %% MESH TEST
# def mesh_fn(shape):
# x, y = np.abs(np.random.randn(*shape)), np.random.uniform(size=shape)


# options = [(3, 1), (5, 1), (7, 1), (101, 1)]
# map(mesh_fn, list(product(*options)))


# %% SIMS TEST
# pos = np.random.uniform(size=(100, 42, 2))
# pos = np.abs(pos / pos.max())
# dwg = esch.init()
# dwg = esch.sims(dwg, pos)
# esch.save(dwg, path="paper/figs/sims_3d.svg")

# pos = np.random.uniform(size=(100, 3, 20, 2)).cumsum(axis=1)
# pos = np.abs(pos / pos.max())
# dwg = esch.init(num=3)
# dwg = esch.sims(dwg, pos)
# esch.save(dwg, path="paper/figs/sims_4d.svg")
