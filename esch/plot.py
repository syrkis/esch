# %% plot.py
#     main hinton plot interface
# by: Noah Syrkis

# Imports
import numpy as np

# import svgwrite
from esch import util
from einops import rearrange


# Interfaces
def sims(pos, path=None):
    ink, pos = sims_fn(pos)
    dwg = draw(ink, pos, path)
    dwg.saveas(path) if path else util.display_fn(dwg)
    return dwg


def grid(ink, path=None):
    ink, pos = grid_fn(ink)
    dwg = draw(ink, pos, path)
    dwg.saveas(path) if path else util.display_fn(dwg)
    return dwg


def mesh(ink, pos, path=None):  # mesh plot
    ink, pos = mesh_fn(ink, pos)
    dwg = draw(ink, pos, path)
    dwg.saveas(path) if path else util.display_fn(dwg)
    return dwg


# Functions
def sims_fn(pos):
    assert pos.ndim >= 3 and pos.ndim <= 4, "pos must be 3 or 4 dimensions"
    pos = pos[:, None, ...] if pos.ndim == 3 else pos
    pos = pos - pos.min(axis=1, keepdims=True)
    ink = np.ones(pos.shape[1:-1])[None, ...] / np.sqrt(pos.shape[-2]) / 10
    pos = rearrange(pos, "n_steps n_plots n_points n_dims -> n_plots n_points n_dims n_steps")
    return ink, pos


def grid_fn(ink):
    reshape = {1: lambda x: x[None, None, None, ...], 2: lambda x: x[None, None, ...], 3: lambda x: x[None, ...]}
    ink = reshape.get(ink.ndim, lambda x: x)(ink)
    m = max(ink.shape[2], ink.shape[3])
    x, y = np.meshgrid(np.linspace(0, ink.shape[2] / m, ink.shape[2]), np.linspace(0, ink.shape[3] / m, ink.shape[3]))
    pos = np.stack((y.flatten(), x.flatten()), axis=1)
    ink = rearrange(ink, "n_steps n_plots n_rows n_cols -> n_steps n_plots (n_rows n_cols)") / np.max(np.abs(ink)) / m
    return ink, pos


# %% Functions
def mesh_fn(ink, pos):
    pos = pos - pos.min(axis=0)
    pos = pos / pos.max(axis=0)
    reshape_dict = {1: lambda x: x[None, None, ...], 2: lambda x: x[None, ...], 3: lambda x: x}
    ink = reshape_dict[ink.ndim](ink)
    ink = ink / ink.max(axis=(-1), keepdims=True) / np.sqrt(len(pos))
    return ink, pos


# Functions
def draw(ink, pos, path=None):
    ink = rearrange(ink, "n_steps n_plots n_points -> n_plots n_points n_steps")
    # pos = rearrange(pos, "n_plots n_points n_dims -> n_plots n_points n_dims")
    dwg, offset = util.setup_drawing(ink, pos)
    n_plots, n_points, n_steps = ink.shape
    for i in range(n_plots):  # for every subplot (usually just one)
        # print(offset.shape, pos.shape)
        # exit()
        p = pos + i * offset + util.PADDING  # util.subplot_offset(i, pos)
        group = dwg.g()
        for j in range(n_points):  # for every point
            if pos.ndim == 4:
                group = circle_fn(group, dwg, ink[i][j], p[i][j][0], p[i][j][1])
            else:
                # print(p.shape)
                group = circle_fn(group, dwg, ink[i][j], p[j][0], p[j][1])
        dwg.add(group)
    dwg.saveas(path) if path else util.display_fn(dwg)
    print()
    return dwg


def circle_fn(group, dwg, point, x, y):
    point /= 2  # radius, not diameter
    start_x, start_y = (x, y) if x.ndim == 0 else (x[-1], y[-1])  # start pos
    circle = dwg.circle(center=(f"{start_x:.3f}", f"{start_y:.3f}"), r=f"{point[-1] / 2}")

    if point.ndim > 0:  # animate sizes
        ss = ";".join([f"{s.item():.3f}" for s in point])  # anim sizes
        circle.add(dwg.animate(attributeName="r", values=ss, dur=f"{point.shape[0]}s", repeatCount="indefinite"))

    if x.ndim > 0:  # animate x and y
        xo, yo = ";".join([f"{x}" for s in x]), ";".join([f"{y}" for s in y])  # account for sims
        circle.add(dwg.animate(attributeName="cx", values=xo, dur=f"{point.shape[0]}s", repeatCount="indefinite"))
        circle.add(dwg.animate(attributeName="cy", values=yo, dur=f"{point.shape[0]}s", repeatCount="indefinite"))

    group.add(circle)  # add shape
    return group


# def rect_fn(group, dwg, ink: util.Array, x, y):
#     ink, col = np.abs(ink), np.sign(ink)  # noqa  TODO: color for white or black fill
#     fill = "black" if col[0] > 0 else "white"
#     pos = (f"{x - ink[0] / 2}", f"{y - ink[0] / 2}")
#     rect = dwg.rect(size=(f"{ink[0]}", f"{ink[0]}"), insert=pos, fill=fill, stroke="black", stroke_width="0.002")

#     ss = ";".join([f"{s.item():.4f}" for s in ink])
#     xo, yo = ";".join([f"{x-s.item()/2:.4f}" for s in ink]), ";".join([f"{y-s.item()/2:.4f}" for s in ink])
#     col = ";".join([f"{'black' if c.item() > 0 else 'white'}" for c in col])
#     # rect.add(dwg.animate(attributeName="fill", values=col, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))
#     rect.add(dwg.animate(attributeName="width", values=ss, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))
#     rect.add(dwg.animate(attributeName="height", values=ss, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))
#     rect.add(dwg.animate(attributeName="x", values=xo, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))
#     rect.add(dwg.animate(attributeName="y", values=yo, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))
#     group.add(rect)  # add shape
#     return group
# t)  # add shape
#     return group
