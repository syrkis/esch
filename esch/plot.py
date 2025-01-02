# %% plot.py
#     main hinton plot interface
# by: Noah Syrkis

# Imports
import numpy as np
import svgwrite
from esch.util import Array, subplot_offset, display_fn, setup_drawing


# Constants
DEBUG = True


# %% Functions
def mesh(ink, pos, shp="circle", path=None):  # mesh plot
    reshape_dict = {1: lambda x: x[None, ..., None], 2: lambda x: x[None, ...], 3: lambda x: x}
    ink = reshape_dict[ink.ndim](ink) / ink.max() ** 4  # this is severly suboptimal
    pos = (pos - pos.min(axis=0)) / (pos - pos.min(axis=0)).max()
    print(ink.shape, pos.shape)
    dwg = draw(setup_drawing(ink, pos), ink, pos, shp)
    dwg.saveas(path) if path else display_fn(dwg)
    return dwg


# Functions
def draw(dwg, ink: Array, pos: Array, shp: str = "rect"):
    # print(f"Drawing {ink.shape} {pos.shape}", end="\n\n") if DEBUG else None
    for i in range(len(ink)):  # for every subplot (usually just one)
        p = pos + subplot_offset(i, pos)
        dwg = add_shp(dwg, ink[i], p, shp)
    return dwg


def add_shp(dwg: svgwrite.Drawing, ink: Array, pos: Array, shp: str = "rect"):
    # print(f"Drawing {ink.shape} {pos.shape}", end="\n\n") if DEBUG else None
    group = dwg.g()
    shp_fn = rect_fn if shp == "rect" else circle_fn
    for i, (x, y) in enumerate(pos):
        group = shp_fn(group, dwg, ink[i], x, y)
    dwg.add(group)
    return dwg


def rect_fn(group, dwg, ink: Array, x, y):
    ink, col = np.abs(ink), np.sign(ink)  # noqa  TODO: color for white or black fill
    fill = "black" if col[0] > 0 else "white"
    pos = (f"{x - ink[0] / 2}", f"{y - ink[0] / 2}")
    rect = dwg.rect(size=(f"{ink[0]}", f"{ink[0]}"), insert=pos, fill=fill, stroke="black", stroke_width="0.002")

    ss = ";".join([f"{s.item():.4f}" for s in ink])
    xo, yo = ";".join([f"{x-s.item()/2:.4f}" for s in ink]), ";".join([f"{y-s.item()/2:.4f}" for s in ink])
    col = ";".join([f"{'black' if c.item() > 0 else 'white'}" for c in col])
    # rect.add(dwg.animate(attributeName="fill", values=col, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))
    rect.add(dwg.animate(attributeName="width", values=ss, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))
    rect.add(dwg.animate(attributeName="height", values=ss, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))
    rect.add(dwg.animate(attributeName="x", values=xo, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))
    rect.add(dwg.animate(attributeName="y", values=yo, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))
    group.add(rect)  # add shape
    return group


def circle_fn(group, dwg, act, x, y):
    a = np.abs(act)
    ss = ";".join([f"{s.item()}" for s in a])  # anim sizes
    xo, yo = ";".join([f"{x}" for s in a]), ";".join([f"{y}" for s in a])  # anim offsets
    circle = dwg.circle(center=(f"{x}", f"{y}"), r=f"{a[0] / 2}")  # create circle
    circle.add(dwg.animate(attributeName="r", values=ss, dur=f"{a.shape[0]}s", repeatCount="indefinite"))
    circle.add(dwg.animate(attributeName="cx", values=xo, dur=f"{a.shape[0]}s", repeatCount="indefinite"))
    circle.add(dwg.animate(attributeName="cy", values=yo, dur=f"{a.shape[0]}s", repeatCount="indefinite"))
    group.add(circle)  # add shape
    return group
