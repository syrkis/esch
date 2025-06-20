# Imports
import numpy as np


# %% Primitives
def circle_fn(s, x, y, e, g, fps):  # size, x, y (possible switch x, and y pos)
    if s.size == 1:
        s = round(s, 3)
        shp = e.dwg.circle(center=(x, y), r=s)
    else:
        shp = sphere_fn(s, x, y, e, g, fps=30)
    g.add(shp)


def square_fn(s, x, y, e, g, fps):  # size, x, y (possible switch x, and y pos)
    if s.size == 1:
        s = round(s, 3)
        shp = e.dwg.rect(insert=(x - s / 2, y - s / 2), size=(s, s))
    else:
        shp = cube_fn(s, x, y, e, g, fps=30)
    g.add(shp)


def agent_fn(size, xs, ys, e, g, fps):
    xs = np.concat((xs[-1][..., None], xs))
    ys = np.concat((ys[-1][..., None], ys))
    agent = e.dwg.circle(center=(xs[0], ys[0]), r=size)
    xs_str = ";".join([f"{round(x, 3)}" for x in xs])
    ys_str = ";".join([f"{round(y, 3)}" for y in ys])
    # print(len(xs))
    animx = e.dwg.animate(attributeName="cx", values=xs, dur=f"{len(xs_str) / fps}s", repeatCount="indefinite")
    animy = e.dwg.animate(attributeName="cy", values=ys, dur=f"{len(ys_str) / fps}s", repeatCount="indefinite")
    agent.add(animx)
    agent.add(animy)
    g.add(agent)


# %% Animations
def sphere_fn(size, x, y, e, group, fps):
    size = np.concatenate((size[-1][..., None], size))
    circle = e.dwg.circle(center=(x, y), r=size[0] ** 0.5 / 2.1)
    radii = ";".join([f"{round(elm.item() ** 0.5 / 2.1, 3)}" for elm in size])
    anim = e.dwg.animate(attributeName="r", values=radii, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    circle.add(anim)
    return circle


def cube_fn(size, x, y, e, group, fps):
    size = np.concat((size[-1][None, ...], size)).round(3)
    size *= 2
    square = e.dwg.rect(insert=(x - size[0] / 2, y - size[0] / 2), size=(size[0], size[0]))
    sizes = ";".join([f"{round(s.item(), 3)}" for s in size])
    xs = ";".join([f"{round(x - s.item() / 2, 3)}" for s in size])
    ys = ";".join([f"{round(y - s.item() / 2, 3)}" for s in size])
    animw = e.dwg.animate(attributeName="width", values=sizes, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animh = e.dwg.animate(attributeName="height", values=sizes, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animx = e.dwg.animate(attributeName="x", values=xs, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animy = e.dwg.animate(attributeName="y", values=ys, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    [square.add(x) for x in [animw, animh, animx, animy]]
    return square
