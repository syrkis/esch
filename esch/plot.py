# %% plot.py  ###################
#     main esch plot interface  #
# by: Noah Syrkis  ##############

# Imports  ######################
import numpy as np  #############
from einops import rearrange, repeat
from functools import reduce  ###

from esch import util


# %% Constants
DEBUG = True

# %% Lambdas  ####################################################################################
sims = lambda e, pos: plot(e, sims_fn, pos)  # noqa  # particles that change position  ###########
grid = lambda e, ink: plot(e, grid_fn, ink)  # noqa  # matrix visualization  #####################
mesh = lambda e, ink, pos: plot(e, mesh_fn, ink, pos)  # noqa  # mesh visualization  #############
plot = lambda e, fn, *args: fig_fn(e, *fn(e, *args))  # noqa  # plot function  ##################


# %% Drawing  ####################################################################################
def fig_fn(e: util.Esch, ink, pos):  ############################################################
    print("plot_fn", "ink", ink.shape, "pos", pos.shape)
    e = reduce(lambda g, i: sub_fn(g, ink[i], pos[i] + i * e.offset + util.PAD), range(ink.shape[0]), e)
    return e


def sub_fn(e: util.Esch, ink, pos):
    print("subplot_fn", "ink", ink.shape, "pos", pos.shape)
    e.dwg.add(reduce(lambda g, j: circle_fn(g, e, ink[j], pos[j]), range(ink.shape[0]), e.dwg.g()))
    return e


# %% Shapes #####################################################################
def circle_fn(group, e, ink, pos):
    # print("circle_fn", ink.shape, pos.shape)
    # exit()
    start_ink = ink if ink.ndim == 0 else ink[-1]  # start size
    circle = e.dwg.circle(center=(f"{pos[0, -1]:.3f}", f"{pos[1, -1]:.3f}"), r=f"{start_ink / 2}")

    # if ink.ndim > 0:  # animate sizes
    ss = ";".join([f"{s.item() / 2:.3f}" for s in ink])  # anim sizes
    circle.add(e.dwg.animate(attributeName="r", values=ss, dur=f"{ink.shape[0]}s", repeatCount="indefinite"))

    # print(x.ndim, y.ndim, ink.shape)
    # if x.ndim > 0:  # animate x and y
    xo, yo = ";".join([f"{s}" for s in pos[0]]), ";".join([f"{s}" for s in pos[1]])  # account for sims
    circle.add(e.dwg.animate(attributeName="cx", values=xo, dur=f"{pos[0].size}s", repeatCount="indefinite"))
    circle.add(e.dwg.animate(attributeName="cy", values=yo, dur=f"{pos[1].size}s", repeatCount="indefinite"))

    group.add(circle)  # add shape
    return group


# %% Functions
def grid_fn(e: util.Esch, ink):
    # prep ink
    # ink = np.array(np.concat((x[..., -1][..., None], x), axis=-1))  # prepend end state
    ink = rearrange(ink, "num row col time -> num (row col) time") / np.max(np.abs(ink)) / e.step

    # prep pos
    deltas = np.linspace(0, e.row / e.step, e.row), np.linspace(0, e.col / e.step, e.col)
    x_point, y_point = np.meshgrid(*deltas)

    # prep pos
    pos = np.stack((y_point.flat, x_point.flat), axis=1)[None, ...]  # point coords
    pos = repeat(pos, "1 point coord -> num point coord", num=e.num)[..., None]  # pos does not animate
    return ink, pos


def mesh_fn(ink, pos):
    ink, pos = np.array(ink), np.array(pos)
    pos = pos - pos.min(axis=0)
    pos = pos / pos.max(axis=0)
    reshape_dict = {1: lambda x: x[None, None, ...], 2: lambda x: x[None, ...], 3: lambda x: x}
    ink = reshape_dict[ink.ndim](ink)
    ink = ink / ink.max(axis=(-1), keepdims=True) / np.sqrt(len(pos))
    ink = rearrange(ink, "n_steps n_plots n_points -> n_plots n_points n_steps")
    return ink, pos[None, ...]


def sims_fn(pos):
    pos = np.array(pos)
    assert pos.ndim >= 3 and pos.ndim <= 4, "pos must be 3 or 4 dimensions"
    pos = pos[:, None, ...] if pos.ndim == 3 else pos
    ink = np.ones(pos.shape[1:-1])[None, ...] / 100
    pos = rearrange(pos, "n_steps n n_points n_dims ->  n n_points n_dims n_steps")
    ink = rearrange(ink, "n_steps n n_points -> n n_points n_steps")
    return ink, pos
