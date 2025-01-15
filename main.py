# %% main.py
#   esch main file for testing
# by: Noah Syrkis

# Imports
import jax.numpy as jnp
import numpy as np
import esch
import pickle
from jaxtyping import Array
from chex import dataclass

# %% GRID TESTS
# act = np.abs(np.random.randn(3, 4))  # 1 x 1 x 1 x 10
# esch.grid(act, path="paper/figs/1d.svg")

# act = np.random.randn(6, 7)
# esch.grid(act, path="paper/figs/2d.svg")

act = np.abs(np.random.randn(100, 3, 20, 10))
esch.grid(act, path="paper/figs/3d.svg")
exit()


act = np.random.randn(100, 3, 40, 10)
esch.grid(act, path="paper/figs/4d.svg")

exit()

# %% MESH TEST
# act = np.random.randn(10)
# pos = np.random.randn(10, 2)

# act = np.random.randn(10, 100).T
# pos = np.random.randn(10, 100, 2).transpose(1, 0, 2)
# print(act[:2], pos[:2])
# esch.mesh(act, pos, path="test.svg")
#
#
act = np.array(jnp.load("data/bolds.npy")).transpose(1, 0)
pos = np.array(jnp.load("data/coords.npy"))[0][:, [1, 0]]
# print(act.shape, pos.shape)
#
esch.mesh(act, pos, shp="dot")
# pos = (pos - pos.mean()) / pos.std()
# print(act[:2], pos[:2])
esch.mesh(act, pos, path="paper/figs/mesh.svg")


exit()


@dataclass
class State:
    pos: Array
    types: Array
    teams: Array
    health: Array


with open("state.pkl", "rb") as f:
    state = pickle.load(f)


poss = [state.pos]
for i in range(10):
    poss.append(poss[-1] + np.random.randn(*poss[-1].shape) * 0.1)
poss = np.stack(poss).transpose(1, 2, 0)

# esch.sims(np.eye(100), poss, path="paper/figs/sims.svg")
#
