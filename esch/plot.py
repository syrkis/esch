# temp %% plot.py
#     main esch plot interface  #
# by: Noah Syrkis


# imports
import numpy as np

# Config
fps = 1


# # %% Functions
def sphere_fn(size, x, y, dwg, group):
    circle = dwg.circle(center=(x, y), r=size)
    group.add(circle)


def square_fn(size, x, y, dwg, group):
    size *= 2
    square = dwg.rect(insert=(x - size / 2, y - size / 2), size=(size, size))
    group.add(square)


def anim_sphere_fn(size, x, y, dwg, group, fps):
    size = np.concat((size[-1][..., None], size))
    circle = dwg.circle(center=(x, y), r=size[0] ** 0.5 / 2.1)  # / min(arr[:, :, -1].shape) ** 0.5)
    radii = ";".join([f"{elm.item() ** 0.5 / 2.1}" for elm in size])
    anim = dwg.animate(attributeName="r", values=radii, dur=f"{len(size) / fps}s", repeatCount="indefinite")
    circle.add(anim)
    group.add(circle)


def anim_square_fn(size, x, y, dwg, group, fps):
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


def anim_grid_fn(arr, dwg, group=None, shape="sphere", fps=fps):
    group = dwg if group is None else group
    assert arr.ndim == 3
    for x in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            size = arr[x, y] ** 0.5 / 2.1
            (anim_sphere_fn if shape == "sphere" else anim_square_fn)(size, x, y, dwg, group, fps)


def mesh_fn(pos, arr, dwg, group=None, shape="sphere"):
    group = dwg if group is None else group
    assert arr.ndim == 1
    for (x, y), r in zip(pos, arr):
        size = r / len(arr) ** 0.5 / 2.1
        (sphere_fn if shape == "sphere" else square_fn)(size, x, y, dwg, group)


def anim_mesh_fn(pos, arr, dwg, group=None, shape="sphere", fps=fps):
    group = dwg if group is None else group
    assert arr.ndim == 2
    for (x, y), r in zip(pos, arr):
        size = r / len(arr) ** 0.5 / 2.1
        (anim_sphere_fn if shape == "sphere" else anim_square_fn)(size, x, y, dwg, group, fps)


def anim_sims_fn(pos, dwg, fill=None, edge=None, size=None, group=None, fps=fps):
    group = dwg if group is None else group
    assert pos.ndim == 3
    # print(pos.shape)
    for idx, (x, y) in enumerate(pos):
        f = fill[idx] if fill is not None else "black"
        e = edge[idx] if edge is not None else "black"
        s = size[idx] if size is not None else 1
        circle = dwg.circle(center=(float(x[0]), float(y[0])), r=s / 2, fill=f, stroke=e, stroke_width="0.1")
        xs = ";".join([f"{x.item():.3f}" for x in x])
        ys = ";".join([f"{y.item():.3f}" for y in y])
        animcx = dwg.animate(attributeName="cx", values=xs, dur=f"{pos.shape[-1] / fps}s", repeatCount="indefinite")
        animcy = dwg.animate(attributeName="cy", values=ys, dur=f"{pos.shape[-1] / fps}s", repeatCount="indefinite")
        circle.add(animcx)
        circle.add(animcy)
        group.add(circle)


# TODO: add gun shots, that move from a to b at time t
def anim_shot_fn(start_pos, end_pos, start_times, dwg, group=None, fps=fps, bullet_size=0.5, color=None, size=None):
    group = dwg if group is None else group

    # Calculate total number of timesteps
    max_time = max(start_times) + 1
    total_duration = max_time / fps

    for i, ((start_x, start_y), (end_x, end_y), shot_time) in enumerate(zip(start_pos, end_pos, start_times)):
        # Determine bullet size and color
        final_size = size[i] if size is not None else bullet_size
        c = color[i] if color is not None else "red"

        # Create value sequences for each timestep
        cx_values = []
        cy_values = []
        opacity_values = []
        r_values = []

        for timestep in range(max_time):
            if timestep == shot_time:
                # Start of shot: at start position
                cx_values.append(f"{start_x:.3f}")
                cy_values.append(f"{start_y:.3f}")
                opacity_values.append("1")
                r_values.append(f"{bullet_size if size is None else 0:.3f}")
            elif timestep == shot_time + 1:
                # End of shot: at end position but invisible
                cx_values.append(f"{end_x:.3f}")
                cy_values.append(f"{end_y:.3f}")
                opacity_values.append("0")
                r_values.append(f"{final_size:.3f}")
            else:
                # Invisible: use start position
                cx_values.append(f"{start_x:.3f}")
                cy_values.append(f"{start_y:.3f}")
                opacity_values.append("0")
                r_values.append(f"{bullet_size if size is None else 0:.3f}")

        # Create bullet circle
        bullet = dwg.circle(
            center=(float(start_x), float(start_y)), r=bullet_size if size is None else 0, fill=c, opacity="0"
        )

        # Add animations
        animcx = dwg.animate(
            attributeName="cx", values=";".join(cx_values), dur=f"{total_duration}s", repeatCount="indefinite"
        )
        animcy = dwg.animate(
            attributeName="cy", values=";".join(cy_values), dur=f"{total_duration}s", repeatCount="indefinite"
        )
        anim_opacity = dwg.animate(
            attributeName="opacity", values=";".join(opacity_values), dur=f"{total_duration}s", repeatCount="indefinite"
        )
        anim_size = dwg.animate(
            attributeName="r", values=";".join(r_values), dur=f"{total_duration}s", repeatCount="indefinite"
        )

        bullet.add(animcx)
        bullet.add(animcy)
        bullet.add(anim_opacity)
        bullet.add(anim_size)
        group.add(bullet)
