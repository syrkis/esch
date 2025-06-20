# temp %% plot.py
#     main esch plot interface  #
# by: Noah Syrkis


# imports
from esch.atom import square_fn, circle_fn, agent_fn

# Config
fps = 1


def grid_fn(arr, e, shape="sphere", fps=fps, ticks=None):
    for idx, g in enumerate(e.gs):
        for x in range(arr.shape[1]):
            for y in range(arr.shape[2]):
                # add shap
                size = arr[idx, x, y]
                (circle_fn if shape == "sphere" else square_fn)(size, x, y, e, g, fps)

                # potentially add ticks
        tick_fn(e, g) if ticks is not None else None
        e.dwg.add(g)


def mesh_fn(pos, arr, e, shape="sphere", fps=fps):
    for idx, g in enumerate(e.gs):
        for (x, y), r in zip(pos[idx], arr[idx]):
            # add shape
            size = r / len(arr[idx]) ** 0.5 / 2.1
            (circle_fn if shape == "sphere" else square_fn)(size, x, y, e, g, fps)
        e.dwg.add(g)


def sims_fn(pos, e, fps=fps):
    for idx, g in enumerate(e.gs):
        for jdx, (xs, ys) in enumerate(pos[idx]):
            agent_fn(size=0.02, xs=xs, ys=ys, e=e, g=g, fps=fps)
        e.dwg.add(g)


def tick_fn(e, g):
    for i in range(e.w + 1):
        # horizontal ticks
        g.add(e.dwg.line(start=(i, e.tick - 1), end=(i, -1)))
        g.add(e.dwg.line(start=(i, e.h + 1 - e.tick), end=(i, e.h + 1)))

        # vertical ticks
        g.add(e.dwg.line(start=(e.tick - 1, i), end=(-1, i)))
        g.add(e.dwg.line(start=(e.w + 1 - e.tick, i), end=(e.w + 1, i)))
