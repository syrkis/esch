# %% plot.py
#     main esch plot interface  #
# by: Noah Syrkis

# Config
fps = 24


# # %% Functions
def sphere_fn(size, x, y, dwg, group):
    size *= 2
    circle = dwg.circle(center=(x, y), r=size)
    group.add(circle)


def square_fn(size, x, y, dwg, group):
    size *= 2
    square = dwg.rect(insert=(x - size / 2, y - size / 2), size=(size, size))
    group.add(square)


def anim_sphere_fn(size, x, y, dwg, group):
    circle = dwg.circle(center=(x, y), r=size[0] ** 0.5 / 2.1)  # / min(arr[:, :, -1].shape) ** 0.5)
    radii = ";".join([f"{elm.item() ** 0.5 / 2.1}" for elm in size])
    anim = dwg.animate(attributeName="r", values=radii, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    circle.add(anim)
    group.add(circle)


def anim_square_fn(size, x, y, dwg, group):
    square = dwg.rect(insert=(x - size[0] / 2, y - size[0] / 2), size=(size[0], size[0]))
    sizes = ";".join([f"{s.item()}" for s in size])
    xs = ";".join([f"{x - s.item() / 2}" for s in size])
    ys = ";".join([f"{y - s.item() / 2}" for s in size])
    animw = dwg.animate(attributeName="width", values=sizes, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animh = dwg.animate(attributeName="height", values=sizes, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animx = dwg.animate(attributeName="x", values=xs, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    animy = dwg.animate(attributeName="y", values=ys, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    square.add(animw)
    square.add(animh)
    square.add(animx)
    square.add(animy)
    group.add(square)


def grid_fn(arr, dwg, group=None, shape="sphere"):
    group = dwg if group is None else group
    assert arr.ndim == 2
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            size = arr[x, y] ** 0.5 / 2.1
            (sphere_fn if shape == "sphere" else square_fn)(size, x, y, dwg, group)


def anim_grid_fn(arr, dwg, group=None, shape="sphere"):
    group = dwg if group is None else group
    assert arr.ndim == 3
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            size = arr[x, y] ** 0.5 / 2.1
            (anim_sphere_fn if shape == "sphere" else anim_square_fn)(size, x, y, dwg, group)


def mesh_fn(pos, arr, dwg, group=None, shape="sphere"):
    group = dwg if group is None else group
    assert arr.ndim == 1
    for (x, y), r in zip(pos, arr):
        size = r / len(arr) ** 0.5 / 2.1
        (sphere_fn if shape == "sphere" else square_fn)(size, x, y, dwg, group)


def anim_mesh_fn(pos, arr, dwg, group=None, shape="sphere"):
    group = dwg if group is None else group
    assert arr.ndim == 2
    for (x, y), r in zip(pos, arr):
        size = r / len(arr) ** 0.5 / 2.1
        (anim_sphere_fn if shape == "sphere" else anim_square_fn)(size, x, y, dwg, group)
        # circle = dwg.circle(center=(x, y), r=r[-1] / len(pos) ** 0.5)
        # radii = ";".join([f"{s.item() ** 0.5 / len(pos) ** 0.5:.3f}" for s in r])  # anim sizes
        # anim = dwg.animate(attributeName="r", values=radii, dur=f"{arr.shape[0] / fps}s", repeatCount="indefinite")
        # circle.add(anim)
        # group.add(circle)


def anim_sims_fn(pos, dwg, group=None, shape="sphere"):
    group = dwg if group is None else group
    assert pos.ndim == 3
    for x, y in pos:
        circle = dwg.circle(center=(float(x[0]), float(y[0])), r=1 / 2)
        xs = ";".join([f"{x.item():.3f}" for x in x])
        ys = ";".join([f"{y.item():.3f}" for y in y])
        animcx = dwg.animate(attributeName="cx", values=xs, dur=f"{len(xs) / fps}s", repeatCount="indefinite")
        animcy = dwg.animate(attributeName="cy", values=ys, dur=f"{len(ys) / fps}s", repeatCount="indefinite")
        circle.add(animcx)
        circle.add(animcy)
        group.add(circle)
