# temp %% plot.py
#     main esch plot interface  #
# by: Noah Syrkis


# imports
from esch.atom import square_fn, circle_fn

# Config
fps = 1


def grid_fn(arr, e, shape="sphere", fps=fps, ticks=None):
    for idx, g in enumerate(e.gs):
        for x in range(arr.shape[1]):
            for y in range(arr.shape[2]):
                # add shap
                size = arr[idx, x, y] ** 0.5 / 2.1
                # x, y = x + e.pad, y + e.pad
                (circle_fn if shape == "sphere" else square_fn)(size, x, y, e, g, fps)

                # potentially add ticks
                tick_fn(g) if ticks is not None else None
        e.dwg.add(g)


def mesh_fn(pos, arr, e, shape="sphere", fps=fps):
    for idx, g in enumerate(e.gs):
        for (x, y), r in zip(pos[idx], arr[idx]):
            # add shape
            size = r / len(arr[idx]) ** 0.5 / 2.1
            (circle_fn if shape == "sphere" else square_fn)(size, x, y, e, g, fps)
        e.dwg.add(g)


def tick_fn(g):
    pass


# def anim_sims_fn(pos, dwg, shots=None, fill=None, edge=None, size=None, group=None, fps=fps):
# group = dwg if group is None else group
# assert pos.ndim == 3
# for idx, (x, y) in enumerate(pos):  # loop through units
# f = fill[idx] if fill is not None else "black"
# e = edge[idx] if edge is not None else "black"
# s = size[idx] if size is not None else 1
# circle = dwg.circle(center=(float(x[0]), float(y[0])), r=s / 2, fill=f, stroke=e, stroke_width="0.1")
# xs = ";".join([f"{x.item():.3f}" for x in x])
# ys = ";".join([f"{y.item():.3f}" for y in y])
# animcx = dwg.animate(attributeName="cx", values=xs, dur=f"{pos.shape[-1] / fps}s", repeatCount="indefinite")
# animcy = dwg.animate(attributeName="cy", values=ys, dur=f"{pos.shape[-1] / fps}s", repeatCount="indefinite")
# circle.add(animcx)
# circle.add(animcy)
# group.add(circle)
#
# if shots and shots[idx]:  # if unit has shots, animate the shots
# shots is a list of tuples: [(time, (x_coord, y_coord)), ...] f
# the source of the shot is x[t], y[t] and the target is x_coord, y_coord.
# one shape will represent the shots of each units. The shot will move from (x[t], y[t]) to x_cord, y_coord, during the first time step.
# When the shot arrives at the target, it will become invisible. The invisible shot will then move back to the
# pass
