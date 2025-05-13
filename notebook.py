# %% Imports
import svgwrite
import numpy as np


# %% Globals
x_range, y_range = 5, 10


def init_fn(x_range, y_range):
    dwg = svgwrite.Drawing(preserveAspectRatio="xMidYMid meet")
    dwg.viewbox(-1, -1, x_range + 1, y_range + 1)
    return dwg


# # %% Main
def grid_fn(dwg, arr):
    assert arr.ndim == 2
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            circle = dwg.circle(center=(i, j), r=arr[i, j] / arr.shape[1] ** 0.5)
            dwg.add(circle)


def anim_grid_fn(dwg, arr):
    assert arr.ndim == 3
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            circle = dwg.circle(center=(i, j), r=arr[i, j, -1] / arr.shape[1] ** 0.5)
            radii = ";".join([f"{s.item() / 2:.3f}" for s in arr[i, j]])  # anim sizes
            anim = dwg.animate(attributeName="r", values=radii, dur=f"{arr.shape[2]}s", repeatCount="indefinite")
            circle.add(anim)
            dwg.add(circle)


def mesh_fn(dwg, pos, arr):
    assert arr.ndim == 1
    for (x, y), r in zip(pos, arr):
        circle = dwg.circle(center=(x, y), r=r / len(arr) ** 0.5)
        dwg.add(circle)


def anim_mesh_fn(dwg, pos, arr):
    assert arr.ndim == 2
    for (x, y), r in zip(pos, arr):
        circle = dwg.circle(center=(x, y), r=r[0] / len(pos) ** 0.5)
        radii = ";".join([f"{s.item() / 2:.3f}" for s in r])  # anim sizes
        anim = dwg.animate(attributeName="r", values=radii, dur=f"{arr.shape[1]}s", repeatCount="indefinite")
        circle.add(anim)
        dwg.add(circle)


def anim_sims_fn(dwg, pos):
    assert pos.ndim == 3
    for x, y in pos:
        circle = dwg.circle(center=(x[0], y[0]), r=1 / len(pos) ** 0.5)
        xs = ";".join([f"{x.item():.3f}" for x in x])
        ys = ";".join([f"{y.item():.3f}" for y in y])
        animcx = dwg.animate(attributeName="cx", values=xs, dur=f"{len(xs)}s", repeatCount="indefinite")
        animcy = dwg.animate(attributeName="cy", values=ys, dur=f"{len(ys)}s", repeatCount="indefinite")
        circle.add(animcx)
        circle.add(animcy)
        dwg.add(circle)


# %% grid test
dwg = init_fn(5, 10)
arr = np.random.uniform(0, 1, (5, 10))
grid_fn(dwg, arr)
dwg.saveas("grid.svg")

# %% anim grid test
dwg = init_fn(10, 5)
arr = np.random.uniform(0, 1, (10, 5, 10))
anim_grid_fn(dwg, arr)
dwg.saveas("anim_grid.svg")

# %% test mesh
dwg = init_fn(10, 5)
pos = np.stack((np.random.uniform(0, 10, 10), np.random.uniform(0, 5, 10))).T
arr = np.random.uniform(0, 1, 10)
pos, arr = np.random.uniform(0, 9, (10, 2)), np.random.uniform(0, 1, 9)
mesh_fn(dwg, pos, arr)
dwg.saveas("mesh.svg")


# %% test anim mesh
dwg = init_fn(10, 5)
pos = np.stack((np.random.uniform(0, 10, 10), np.random.uniform(0, 5, 10))).T
arr = np.random.uniform(0, 1, (10, 20))
anim_mesh_fn(dwg, pos, arr)
dwg.saveas("anim_mesh.svg")
# pos, arr = np.random.uniform(0, 9, (10, 2)), np.random.uniform(0, 1, (9, 10))
# anim_mesh_fn(dwg, pos, arr)


# %% test sims
dwg = init_fn(10, 5)
pos = np.random.randn(100, 2, 10).cumsum(axis=2) + np.array((5, 2.5))[..., None]
anim_sims_fn(dwg, pos)
dwg.saveas("anim_sims.svg")
