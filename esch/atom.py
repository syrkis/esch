# Imports
import numpy as np


# %% Primitives
def circle_fn(s, x, y, e, g, fps):  # size, x, y (possible switch x, and y pos)
    x, y = y, x  # i, j versus x, y
    if s.size == 1:
        s = round(s, 3)
        shp = e.dwg.circle(center=(x, y), r=s)
    else:
        shp = sphere_fn(s, x, y, e, g, fps=30)
    g.add(shp)


def square_fn(s, x, y, e, g, fps):  # size, x, y (possible switch x, and y pos)
    x, y = y, x  # i, j versus x, y
    if s.size == 1:
        s = round(s, 3)
        shp = e.dwg.rect(insert=(x - s / 2, y - s / 2), size=(s, s))
    else:
        shp = cube_fn(s, x, y, e, g, fps=30)
    g.add(shp)


def agent_fn(size, xs, ys, shots, e, g, fps, col, stroke):
    xs, ys = ys, xs
    agent = e.dwg.circle(center=(xs[0], ys[0]), r=size, fill=col, stroke=stroke)
    animx = e.dwg.animate(attributeName="cx", values=xs.round(3), dur=f"{len(xs) / fps}s", repeatCount="indefinite")
    animy = e.dwg.animate(attributeName="cy", values=ys.round(3), dur=f"{len(xs) / fps}s", repeatCount="indefinite")
    agent.add(animx)
    agent.add(animy)
    shots_fn(e, g, xs, ys, shots, fps) if shots is not None else None
    g.add(agent)


def agent_fn(size, xs, ys, shots, e, g, fps, col, stroke):
    xs, ys = ys, xs
    agent = e.dwg.circle(center=(xs[0], ys[0]), r=size, fill=col, stroke=stroke)
    animx = e.dwg.animate(attributeName="cx", values=xs.round(3), dur=f"{len(xs) / fps}s", repeatCount="indefinite")
    animy = e.dwg.animate(attributeName="cy", values=ys.round(3), dur=f"{len(xs) / fps}s", repeatCount="indefinite")
    agent.add(animx)
    agent.add(animy)
    shots_fn(e, g, xs, ys, shots, fps, size) if shots is not None else None
    g.add(agent)


def shots_fn(e, g, xs, ys, shots, fps, size):
    # shots is a dict of {time_step: coord.ndarray}.
    # A key-value pair is only present if at the time_step there was a shooting from the unit
    # (currently placed at xs[time_step], ys[time_step] to the coord.ndarray).
    # Bullets should reach the target halfway into the time step, expand from size to size*3,
    # become invisible at target, and stay invisible as they return to source.
    bullet = e.dwg.circle(center=(xs[0], ys[0]), r=size, fill="green")

    # Create keyframes for position, size, and opacity animation
    # Each time step gets 3 keyframes: start, halfway (target), end (back to source)
    bullet_xs = []
    bullet_ys = []
    bullet_sizes = []
    opacities = []

    for i in range(len(xs)):
        if i in shots:
            # Shot exists at this time step
            target_x, target_y = shots[i]

            # Start of time step: bullet visible at unit position with original size
            bullet_xs.append(xs[i])
            bullet_ys.append(ys[i])
            bullet_sizes.append(size)
            opacities.append(1.0)

            # Halfway through time step: bullet reaches target and expands
            bullet_xs.append(xs[i] + target_x)
            bullet_ys.append(ys[i] + target_y)
            bullet_sizes.append(size * 3)
            opacities.append(0.0)  # Becomes invisible when reaching target

            # End of time step: bullet returns to unit position (invisible)
            bullet_xs.append(xs[i])
            bullet_ys.append(ys[i])
            bullet_sizes.append(size)
            opacities.append(0.0)
        else:
            # No shot - bullet stays invisible at unit position throughout time step
            for _ in range(3):  # Three keyframes per time step
                bullet_xs.append(xs[i])
                bullet_ys.append(ys[i])
                bullet_sizes.append(size)
                opacities.append(0.0)

    # Convert to numpy arrays and round
    bullet_xs = np.array(bullet_xs).round(3)
    bullet_ys = np.array(bullet_ys).round(3)
    bullet_sizes = np.array(bullet_sizes).round(3)
    opacities = np.array(opacities)

    # Create animations with triple the keyframes (3 per time step)
    total_duration = len(xs) / fps
    animx = e.dwg.animate(attributeName="cx", values=bullet_xs, dur=f"{total_duration}s", repeatCount="indefinite")
    animy = e.dwg.animate(attributeName="cy", values=bullet_ys, dur=f"{total_duration}s", repeatCount="indefinite")
    anim_size = e.dwg.animate(
        attributeName="r", values=bullet_sizes, dur=f"{total_duration}s", repeatCount="indefinite"
    )
    anim_opacity = e.dwg.animate(
        attributeName="opacity", values=opacities, dur=f"{total_duration}s", repeatCount="indefinite"
    )

    bullet.add(anim_opacity)
    bullet.add(animx)
    bullet.add(animy)
    bullet.add(anim_size)
    g.add(bullet)


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
