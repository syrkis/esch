# Imports
import numpy as np


def square_fn(s, x, y, e, g, fps):  # size, x, y (possible switch x, and y pos)
    x, y = y, x  # i, j versus x, y
    if s.size == 1:
        s = round(s.item(), 3)
        shp = e.dwg.rect(insert=(x - s / 2, y - s / 2), size=(s, s), fill="black")
    else:
        shp = cube_fn(s, x, y, e, g, fps=fps)
    g.add(shp)


def cube_fn(size, x, y, e, group, fps):
    size = np.abs(np.concatenate((size[-1][None, ...], size)).round(3))
    square = e.dwg.rect(insert=(x - size[0] / 2, y - size[0] / 2), size=(size[0], size[0]), fill="black", stroke="black")
    sizes = ";".join([f"{round(s.item(), 3)}" for s in size])
    xs = ";".join([f"{round(x - s.item() / 2, 3)}" for s in size])
    ys = ";".join([f"{round(y - s.item() / 2, 3)}" for s in size])
    animw = e.dwg.animate(attributeName="width", values=sizes, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animh = e.dwg.animate(attributeName="height", values=sizes, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animx = e.dwg.animate(attributeName="x", values=xs, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animy = e.dwg.animate(attributeName="y", values=ys, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    # fill_values = ";".join(["white" if s < 0 else "black" for s in size])
    # animf = e.dwg.animate(
    # attributeName="fill", values=fill_values, dur=f"{len(size) / fps}s", repeatCount="indefinite", calcMode="discrete"
    # )
    [square.add(x) for x in [animw, animh, animx, animy]]
    return square


# %% Primitives
def circle_fn(s, x, y, e, g, fps, col):  # size, x, y (possible switch x, and y pos)
    x, y = y, x  # i, j versus x, y
    if s.size == 1:
        s = round(s, 3)
        shp = e.dwg.circle(center=(x, y), r=s, fill=col)
    else:
        shp = sphere_fn(s, x, y, e, g, fps=fps, col=col)
    g.add(shp)


# %% Animations
def sphere_fn(size, x, y, e, group, fps, col):
    size = np.concatenate((size[-1][..., None], size))
    circle = e.dwg.circle(center=(x, y), r=size[0] ** 0.5 / 2.1, fill=col)
    radii = ";".join([f"{round(elm.item() ** 0.5 / 2.1, 3)}" for elm in size])
    anim = e.dwg.animate(attributeName="r", values=radii, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    circle.add(anim)
    return circle
