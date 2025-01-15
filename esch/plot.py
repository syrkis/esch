# %% plot.py
#     main hinton plot interface
# by: Noah Syrkis

# Imports
import numpy as np
import svgwrite
from esch import util
from einops import rearrange


# Interfaces
def sims(ink, pos, path=None):
    raise NotImplementedError


def grid(ink, path=None):
    ink, pos = grid_fn(ink)
    dwg = draw(ink, pos, path)
    return dwg


def mesh(ink, pos, path=None):  # mesh plot
    dwg = mesh_fn(ink, pos)
    dwg.saveas(path) if path else util.display_fn(dwg)
    return dwg


# Functions
def sims_fn(pos) -> svgwrite.Drawing:
    raise NotImplementedError


def grid_fn(ink):
    reshape = {1: lambda x: x[None, None, None, ...], 2: lambda x: x[None, None, ...], 3: lambda x: x[None, ...]}
    ink = reshape.get(ink.ndim, lambda x: x)(ink)
    m = max(ink.shape[2], ink.shape[3])
    x, y = np.meshgrid(np.linspace(0, ink.shape[2] / m, ink.shape[2]), np.linspace(0, ink.shape[3] / m, ink.shape[3]))
    pos = np.stack((y.flatten(), x.flatten()), axis=1)
    ink = rearrange(ink, "n_steps n_plots n_rows n_cols -> n_steps n_plots (n_rows n_cols)") / np.max(np.abs(ink)) / m
    return ink, pos


# %% Functions
def mesh_fn(ink, pos) -> svgwrite.Drawing:
    reshape_dict = {1: lambda x: x[None, ..., None], 2: lambda x: x[None, ..., None], 3: lambda x: x}
    ink = reshape_dict[ink.ndim](ink) / ink.max() ** 4  # this is severly suboptimal
    pos = (pos - pos.min(axis=0)) / (pos - pos.min(axis=0)).max()
    dwg = draw(ink, pos)  # shp)
    return dwg


# Functions
def draw(ink, pos, path=None):
    ink = rearrange(ink, "n_steps n_plots n_points -> n_plots n_points n_steps")
    dwg, offset = util.setup_drawing(ink, pos)
    n_plots, n_points, n_steps = ink.shape
    for i in range(n_plots):  # for every subplot (usually just one)
        p = pos + i * offset + util.PADDING  # util.subplot_offset(i, pos)
        group = dwg.g()
        for j, (x, y) in zip(range(n_points), p):  # for every point
            group = circle_fn(group, dwg, ink[i][j], x, y)
        dwg.add(group)
    dwg.saveas(path) if path else util.display_fn(dwg)
    return dwg


def circle_fn(group, dwg, point, x, y):
    # print(x, y)
    point /= 2
    circle = dwg.circle(center=(f"{x:.3f}", f"{y:.3f}"), r=f"{point[-1] / 2}")  # create circle
    ss = ";".join([f"{s.item():.3f}" for s in point])  # anim sizes
    circle.add(dwg.animate(attributeName="r", values=ss, dur=f"{point.shape[0]}s", repeatCount="indefinite"))
    # xo, yo = ";".join([f"{x}" for s in a]), ";".join([f"{y}" for s in a])  # account for sims
    # circle.add(dwg.animate(attributeName="cx", values=xo, dur=f"{a.shape[0]}s", repeatCount="indefinite"))
    # circle.add(dwg.animate(attributeName="cy", values=yo, dur=f"{a.shape[0]}s", repeatCount="indefinite"))
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
