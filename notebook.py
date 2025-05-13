# %% Imports
import svgwrite
import numpy as np


# %% Globals
steps_per_sec = 10


def init_fn(x_range, y_range, rows=1, cols=1):
    dwg = svgwrite.Drawing(size=None, preserveAspectRatio="xMidYMid meet")
    dwg.viewbox(-1, -1, (x_range + 1) * rows, (y_range + 1) * cols)
    return dwg


def save_fn(dwg, filename, scale=80):
    dwg.saveas(filename)


# # %% Main
def grid_fn(arr, dwg, group=None):
    group = dwg if group is None else group
    assert arr.ndim == 2
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            circle = dwg.circle(center=(i, j), r=arr[i, j] ** 0.5 / (min(arr.shape) ** 0.5))
            group.add(circle)


def anim_grid_fn(arr, dwg, group=None):
    group = dwg if group is None else group
    assert arr.ndim == 3
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            circle = dwg.circle(center=(i, j), r=arr[i, j, -1] ** 0.5 / min(arr[:, :, -1].shape) ** 0.5)
            radii = ";".join([f"{s.item() ** 0.5 / min(arr[:, :, -1].shape) ** 0.5}" for s in arr[i, j]])
            anim = dwg.animate(
                attributeName="r", values=radii, dur=f"{arr.shape[2] / steps_per_sec}s", repeatCount="indefinite"
            )
            circle.add(anim)
            group.add(circle)


def mesh_fn(pos, arr, dwg, group=None):
    group = dwg if group is None else group
    assert arr.ndim == 1
    for (x, y), r in zip(pos, arr):
        circle = dwg.circle(center=(x, y), r=r / len(arr) ** 0.5)
        group.add(circle)


def anim_mesh_fn(pos, arr, dwg, group=None):
    group = dwg if group is None else group
    assert arr.ndim == 2
    for (x, y), r in zip(pos, arr):
        circle = dwg.circle(center=(x, y), r=r[-1] / len(pos) ** 0.5)
        radii = ";".join([f"{s.item() / len(pos) ** 0.5:.3f}" for s in r])  # anim sizes
        anim = dwg.animate(
            attributeName="r", values=radii, dur=f"{arr.shape[1] / steps_per_sec}s", repeatCount="indefinite"
        )
        circle.add(anim)
        group.add(circle)


def anim_sims_fn(arr, dwg, group=None):
    group = dwg if group is None else group
    assert pos.ndim == 3
    for x, y in pos:
        circle = dwg.circle(center=(x[0], y[0]), r=1 / len(pos) ** 0.5)
        xs = ";".join([f"{x.item():.3f}" for x in x])
        ys = ";".join([f"{y.item():.3f}" for y in y])
        animcx = dwg.animate(attributeName="cx", values=xs, dur=f"{len(xs) / steps_per_sec}s", repeatCount="indefinite")
        animcy = dwg.animate(attributeName="cy", values=ys, dur=f"{len(ys) / steps_per_sec}s", repeatCount="indefinite")
        circle.add(animcx)
        circle.add(animcy)
        group.add(circle)


# %%
def tick_fn(xticks, yticks, dwg, group=None):
    group = dwg if group is None else group
    tick_len = 0.15  # length of tick marks
    label_offset = 0.25  # offset for labels from ticks

    # x-axis ticks (assume y=0 for axis)
    for x, label in xticks:
        # Tick mark
        line = dwg.line(start=(x, -tick_len / 2), end=(x, tick_len / 2), stroke="black")
        group.add(line)
        # Label
        text = dwg.text(
            str(label),
            insert=(x, tick_len / 2 + label_offset),
            text_anchor="middle",
            alignment_baseline="hanging",
            # font_size="1pt",
        )
        group.add(text)

    # y-axis ticks (assume x=0 for axis)
    for y, label in yticks:
        # Tick mark
        line = dwg.line(start=(-tick_len / 2, y), end=(tick_len / 2, y), stroke="black")
        group.add(line)
        # Label
        text = dwg.text(
            str(label),
            insert=(tick_len / 2 + label_offset, y),
            text_anchor="start",
            alignment_baseline="middle",
            # font_size="1pt",
        )
        group.add(text)


# %% grid test
dwg = init_fn(5, 10)
arr = np.ones((5, 10))
grid_fn(arr, dwg)
# tick_fn([(2, "l")], [(1, "0")], dwg)
save_fn(dwg, "paper/figs/grid.svg")

# %% anim grid test
dwg = init_fn(10, 5)
arr = np.absolute(np.random.randn(10, 5, 10).cumsum(2))
anim_grid_fn(arr / arr.max(), dwg)
save_fn(dwg, "paper/figs/anim_grid.svg")

# %% test mesh
dwg = init_fn(10, 5)
pos = np.stack((np.random.uniform(0, 10, 10), np.random.uniform(0, 5, 10))).T
arr = np.random.uniform(0, 1, 10)
pos, arr = np.random.uniform(0, 9, (10, 2)), np.random.uniform(0, 1, 9)
mesh_fn(pos, arr, dwg)
save_fn(dwg, "paper/figs/mesh.svg")

# %% test anim mesh
dwg = init_fn(10, 5)
pos = np.stack((np.random.uniform(0, 10, 10), np.random.uniform(0, 5, 10))).T
arr = np.abs(np.random.randn(10, 20).cumsum(1))
anim_mesh_fn(pos, arr / arr.max(), dwg)
save_fn(dwg, "paper/figs/anim_mesh.svg")

# %% test sims
dwg = init_fn(10, 5)
pos = np.random.randn(100, 2, 200).cumsum(axis=2) * 0.1 + np.array((4.5, 2.25))[..., None]
anim_sims_fn(pos, dwg)
save_fn(dwg, "paper/figs/anim_sims.svg")

# %% test mix
dwg = init_fn(10, 5)
pos = np.random.randn(100, 2, 200).cumsum(axis=2) * 0.1 + np.array((4.5, 2.25))[..., None]
anim_sims_fn(pos, dwg)
arr = np.random.uniform(0, 1, (10, 5))
grid_fn(arr, dwg)
save_fn(dwg, "paper/figs/anim_mix.svg")


# %% test multi
dwg = init_fn(10, 5, cols=3)
arr = np.absolute(np.random.randn(3, 10, 5))
for i in range(len(arr)):
    group = dwg.g()
    group.translate(0, (5 + 1) * i)
    grid_fn(arr[i] / arr[i].max(), dwg, group)
    dwg.add(group)
save_fn(dwg, "paper/figs/multi.svg")


# %% test anim multi
dwg = init_fn(10, 5, cols=3)
arr = np.absolute(np.random.randn(3, 10, 5, 10).cumsum(3))
for i in range(len(arr)):
    group = dwg.g()
    group.translate(0, (5 + 1) * i)
    anim_grid_fn(arr[i] / arr[i].max(), dwg, group)
    dwg.add(group)
save_fn(dwg, "paper/figs/anim_multi.svg")
