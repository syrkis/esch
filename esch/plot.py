# %% plot.py
#   hinton plots — turns arrays into .svg or .gif
# by: Noah Syrkis


# %% Imports
import jax.numpy as jnp
from jax import Array, random
import numpy as np
import svgwrite
from tqdm import tqdm  # type: ignore


def hinton_frame(x: Array, normalizer=None, scale=1) -> svgwrite.Drawing:
    # make hinton svg plot
    # the fill is black if positive and white if negative.
    # edge color is black.
    # the size of each rect is based on x[i, j]
    # scale = 1 / (max(x[1:].shape) ** 0.5)
    x = (x / x.max()).T
    dwg = svgwrite.Drawing("temp.svg", profile="full", size=(f"{x.shape[0] * 10}", f"{x.shape[1]* 10}"))
    group = dwg.g()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            size = jnp.abs(x[i, j])
            _x = i - size / 2 + scale / 2
            _y = j - size / 2 + scale / 2
            group.add(
                dwg.rect(
                    insert=(f"{_x * 10:.1f}", f"{_y * 10:.1f}"),
                    size=(f"{size * 10:.1f}", f"{size * 10:.1f}"),
                    # fill="#000" if x[i, j] > 0 else "#fff",
                    # stroke="#000",
                )
            )
    dwg.add(group)
    return dwg

    # merge into one group
    group = svgwrite.container.Group()
    for e in dwg.elements:
        if type(e) is svgwrite.shapes.Rect:
            group.add(e)
    dwg.add(group)
    # return the group
    return dwg  # type: ignore


def hinton_animation(x: Array, normalizer=None, scale=1) -> svgwrite.Drawing:
    # frames per second is 48
    frame_duration_seconds = 1 / 20
    frame_groups = []
    dwg = None
    for i, state in tqdm(enumerate(x)):
        # dwg = v.get_dwg(states=state)
        dwg = hinton_frame(state, normalizer, scale)
        assert (
            len([e for e in dwg.elements if type(e) is svgwrite.container.Group]) == 1  # type: ignore
        ), "Drawing must contain only one group"
        group: svgwrite.container.Group = dwg.elements[-1]  # type: ignore
        group["id"] = f"_fr{i:x}"  # hex frame number
        group["class"] = "frame"
        frame_groups.append(group)

    assert dwg is not None
    del dwg.elements[-1]
    total_seconds = frame_duration_seconds * len(frame_groups)

    style = f".frame{{visibility:hidden; animation:{total_seconds}s linear _k infinite; stroke: black; fill:black;}}"
    style += f"@keyframes _k{{0%,{100/len(frame_groups)}%{{visibility:visible}}{100/len(frame_groups) * 1.000001}%,100%{{visibility:hidden}}}}"

    for i, group in enumerate(frame_groups):
        dwg.add(group)
        style += f"#{group['id']}{{animation-delay:{i * frame_duration_seconds}s}}"
    dwg.defs.add(svgwrite.container.Style(style))  # type: ignore
    return dwg


# %% Main
rng = random.PRNGKey(0)
x = random.normal(rng, shape=(100, 37, 37)).cumsum(axis=0)
hinton_animation(x, normalizer=1).saveas("temp.svg")
