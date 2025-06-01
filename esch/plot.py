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
        animcx = dwg.animate(attributeName="cx", values=xs, dur=f"{pos.shape[0] / fps}s", repeatCount="indefinite")
        animcy = dwg.animate(attributeName="cy", values=ys, dur=f"{pos.shape[0] / fps}s", repeatCount="indefinite")
        circle.add(animcx)
        circle.add(animcy)
        group.add(circle)


# TODO: add gun shots, that move from a to b at time t
def anim_shot_fn(start_pos, end_pos, start_times, dwg, group=None, fps=fps, bullet_size=0.5, color=None, size=None):
    """
    Add animated gun shots that move from start_pos to end_pos over discrete time steps.

    Args:
        start_pos: array of shape (n_shots, 2) with starting (x, y) positions
        end_pos: array of shape (n_shots, 2) with ending (x, y) positions
        start_times: array of shape (n_shots,) with start time steps for each shot
        dwg: SVG drawing object
        group: SVG group to add elements to (optional)
        fps: frames per second for animation
        bullet_size: radius of the bullet circle (used when size is None)
        color: color of the bullet
        size: optional array of shape (n_shots,) with target sizes. If provided, bullets grow from 0 to target size
    """
    group = dwg if group is None else group

    for i, ((start_x, start_y), (end_x, end_y), t) in enumerate(zip(start_pos, end_pos, start_times)):
        # Determine bullet size - use size array if provided, otherwise use bullet_size
        if size is not None:
            initial_size = 0
            final_size = size[i]
        else:
            initial_size = bullet_size
            final_size = bullet_size
        if color is None:
            c = "red"
        else:
            c = color[i]

        # Create bullet circle
        bullet = dwg.circle(center=(float(start_x), float(start_y)), r=initial_size, fill=c, opacity="0")

        # Duration is exactly 1 time step (1/fps seconds per frame)
        step_duration = 1.0 / fps
        begin_time = t / fps

        # Animate position from start to end over one time step
        animcx = dwg.animate(
            attributeName="cx",
            values=f"{start_x};{end_x}",
            dur=f"{step_duration}s",
            begin=f"{begin_time}s",
            # fill="freeze",
        )

        animcy = dwg.animate(
            attributeName="cy",
            values=f"{start_y};{end_y}",
            dur=f"{step_duration}s",
            begin=f"{begin_time}s",
            # fill="freeze",
        )

        # Make bullet visible only during the time step (t to t+1)
        anim_opacity = dwg.animate(
            attributeName="opacity", values="0;1;0", dur=f"{step_duration}s", begin=f"{begin_time}s"
        )  # , fill="freeze"
        # )

        # Animate size if size array is provided
        if size is not None:
            anim_size = dwg.animate(
                attributeName="r",
                values=f"{initial_size};{final_size}",
                dur=f"{step_duration}s",
                begin=f"{begin_time}s",
                fill="freeze",
            )
            bullet.add(anim_size)

        bullet.add(animcx)
        bullet.add(animcy)
        bullet.add(anim_opacity)
        group.add(bullet)
