# %% plot.py
#     main esch plot interface  #
# by: Noah Syrkis

# Config
steps_per_sec = 10


# # %% Main
def grid_fn(arr, dwg, group=None):
    group = dwg if group is None else group
    assert arr.ndim == 2
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            circle = dwg.circle(center=(i, j), r=arr[i, j] ** 0.5 / 2)  # (max(arr.shape) ** 0.5))
            group.add(circle)


def anim_grid_fn(arr, dwg, group=None):
    group = dwg if group is None else group
    assert arr.ndim == 3
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            circle = dwg.circle(center=(i, j), r=arr[i, j, -1] ** 0.5 / 2)  # / min(arr[:, :, -1].shape) ** 0.5)
            radii = ";".join([f"{s.item() ** 0.5 / min(arr[:, :, -1].shape) ** 0.5}" for s in arr[i, j]])
            anim = dwg.animate(
                attributeName="r", values=radii, dur=f"{arr.shape[2] / steps_per_sec}s", repeatCount="indefinite"
            )
            circle.add(anim)
            group.add(circle)


def mesh_fn(pos, arr, dwg, group=None):
    group = dwg if group is None else group
    assert arr.ndim == 1
    for (x, y), r in zip(pos, arr):
        circle = dwg.circle(center=(x, y), r=r / len(arr) ** 0.5)
        group.add(circle)


def anim_mesh_fn(pos, arr, dwg, group=None):
    group = dwg if group is None else group
    assert arr.ndim == 2
    for (x, y), r in zip(pos, arr):
        circle = dwg.circle(center=(x, y), r=r[-1] / len(pos) ** 0.5)
        radii = ";".join([f"{s.item() ** 0.5 / len(pos) ** 0.5:.3f}" for s in r])  # anim sizes
        anim = dwg.animate(
            attributeName="r", values=radii, dur=f"{arr.shape[1] / steps_per_sec}s", repeatCount="indefinite"
        )
        circle.add(anim)
        group.add(circle)


def anim_sims_fn(pos, dwg, group=None):
    group = dwg if group is None else group
    assert pos.ndim == 3
    for x, y in pos:
        circle = dwg.circle(center=(float(x[0]), float(y[0])), r=1 / 2)
        xs = ";".join([f"{x.item():.3f}" for x in x])
        ys = ";".join([f"{y.item():.3f}" for y in y])
        animcx = dwg.animate(attributeName="cx", values=xs, dur=f"{len(xs) / steps_per_sec}s", repeatCount="indefinite")
        animcy = dwg.animate(attributeName="cy", values=ys, dur=f"{len(ys) / steps_per_sec}s", repeatCount="indefinite")
        circle.add(animcx)
        circle.add(animcy)
        group.add(circle)
