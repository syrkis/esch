# Imports
import numpy as np


# %% Primitives
def circle_fn(s, x, y, dwg, g, fps):  # size, x, y (possible switch x, and y pos)
    shp = dwg.circle(center=(x, y), r=s) if s.size == 1 else sphere_fn(s, x, y, dwg, g, fps=30)
    g.add(shp)


def square_fn(s, x, y, dwg, g, fps):  # size, x, y (possible switch x, and y pos)
    shp = dwg.rect(insert=(x - s / 2, y - s / 2), size=(s, s)) if s.size == 1 else cube_fn(s, x, y, dwg, g, fps=30)
    g.add(shp)


# %% Animations
def sphere_fn(size, x, y, dwg, group, fps):
    size = np.concat((size[-1][..., None], size))
    circle = dwg.circle(center=(x, y), r=size[0] ** 0.5 / 2.1)
    radii = ";".join([f"{elm.item() ** 0.5 / 2.1}" for elm in size])
    anim = dwg.animate(attributeName="r", values=radii, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    circle.add(anim)
    return circle


def cube_fn(size, x, y, dwg, group, fps):
    size = np.concat((size[-1][..., None], size))
    size *= 2
    square = dwg.rect(insert=(x - size[0] / 2, y - size[0] / 2), size=(size[0], size[0]))
    sizes = ";".join([f"{s.item()}" for s in size])
    xs = ";".join([f"{x - s.item() / 2}" for s in size])
    ys = ";".join([f"{y - s.item() / 2}" for s in size])
    animw = dwg.animate(attributeName="width", values=sizes, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animh = dwg.animate(attributeName="height", values=sizes, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animx = dwg.animate(attributeName="x", values=xs, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animy = dwg.animate(attributeName="y", values=ys, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    [square.add(x) for x in [animw, animh, animx, animy]]
    return square
