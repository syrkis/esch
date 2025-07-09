# temp %% plot.py
#     main esch plot interface  #
# by: Noah Syrkis


# imports
from esch.atom import square_fn, circle_fn, agent_fn

# Config
fps = 1


def grid_fn(e, arr, shape="sphere", fps=fps, ticks=None, col="black"):
    for idx, g in enumerate(e.gs):
        for x in range(arr.shape[1]):
            for y in range(arr.shape[2]):
                # add shap
                size = arr[idx, x, y]
                (circle_fn if shape == "sphere" else square_fn)(size, x, y, e, g, fps, col)

                # potentially add ticks
        tick_fn(e, g) if ticks is not None else None
        e.dwg.add(g)


def mesh_fn(e, pos, arr, col="black", shape="sphere", fps=fps):
    for idx, g in enumerate(e.gs):
        for (x, y), r in zip(pos[idx], arr[idx]):
            # add shape
            size = r / len(arr[idx]) ** 0.5 / 2.1
            (circle_fn if shape == "sphere" else square_fn)(size, x, y, e, g, fps, col=col)
        e.dwg.add(g)


def sims_fn(e, pos, action, fps=fps, col="black", size=0.1, blast=0.1, stroke="black"):
    for i, g in enumerate(e.gs):
        for j, (xs, ys) in enumerate(pos[i]):
            shots = (
                {kdx: coord for kdx, coord in enumerate(action.pos[i, :, j]) if action.cast[i, kdx, j]}
                if action is not None
                else None
            )
            agent_fn(size=size, xs=xs, ys=ys, shots=shots, e=e, g=g, fps=fps, col=col, stroke=stroke, blast=blast)
        e.dwg.add(g)


def tick_fn(e, g):
    for i in range(e.w + 1):
        # horizontal ticks
        g.add(e.dwg.line(start=(i, e.tick - 1), end=(i, -1)))
        g.add(e.dwg.line(start=(i, e.h + 1 - e.tick), end=(i, e.h + 1)))

        # vertical ticks
        g.add(e.dwg.line(start=(e.tick - 1, i), end=(-1, i)))
        g.add(e.dwg.line(start=(e.w + 1 - e.tick, i), end=(e.w + 1, i)))
