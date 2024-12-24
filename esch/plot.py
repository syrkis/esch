# %% plot.py
#     main hinton plot interface
# by: Noah Syrkis

# Imports
import numpy as np
import svgwrite

from esch.edge import EdgeConfigs
from esch.util import Array, display_fn, PADDING


# Functions
def plot(act: Array, pos: Array, shp: str = "rect", path: str | None = None, text: EdgeConfigs = EdgeConfigs()):
    dwg = play(act, pos, shp, text)
    dwg.saveas(path) if path else display_fn(dwg)
    return dwg


def play(acts, pos, shp, edge: EdgeConfigs):  # -> svgwrite.Drawing:
    n_plots, n_shapes, n_steps = acts.shape
    dwg = setup_drawing(acts, pos)

    for idx, act in enumerate(acts):
        group = dwg.g()

        for a, (x, y) in zip(act, pos + subplot_offset(idx, pos)):
            group = rect_fn(group, dwg, a, x, y) if shp == "rect" else circle_fn(group, dwg, a, x, y)

        # add group
        dwg.add(group)
    return dwg


def setup_drawing(act: Array, pos: Array) -> svgwrite.Drawing:
    n_plots, n_shapes, n_steps = act.shape
    width, height = pos[:, 0].max().item() + 0 * PADDING, pos[:, 1].max().item() + 0 * PADDING
    total_width = width if height < width else (width + 2 * PADDING) * n_plots
    total_height = height if height > width else (height + 2 * PADDING) * n_plots
    dwg = svgwrite.Drawing(size=(f"{total_width}px", f"{total_height}px"))
    dwg["width"], dwg["height"] = "100%", "100%"
    dwg["preserveAspectRatio"] = "xMidYMid meet"
    print(width, height, total_height, total_width)
    dwg.viewbox(0, 0, total_width, total_height)  # TODO: add padding
    dwg.defs.add(dwg.style("text {font-family: 'Computer Modern', 'serif';}"))
    return dwg


def subplot_offset(idx: int, pos: Array):
    width, height = pos[:, 1].max().item(), pos[:, 0].max().item()
    x_offset = 0 if width > height else ((width + 2 * PADDING) * idx + PADDING)
    y_offset = 0 if x_offset != 0 else ((height + 2 * PADDING) * idx + PADDING)
    offset = np.array([x_offset, y_offset])
    return offset[::-1]


def rect_fn(group, dwg, a, x, y):
    a = np.abs(a)
    rect = dwg.rect(insert=(f"{x - a[0] / 2}", f"{y - a[0] / 2}"), size=(f"{a[0]}", f"{a[0]}"))

    # anim seqs
    sizes = ";".join([f"{s.item()}" for s in a])
    x_offsets = ";".join([f"{x - s.item() / 2}" for s in a])
    y_offsets = ";".join([f"{y - s.item() / 2}" for s in a])

    # size anim
    rect.add(dwg.animate(attributeName="width", values=sizes, dur=f"{a.shape[0]}s", repeatCount="indefinite"))
    rect.add(dwg.animate(attributeName="height", values=sizes, dur=f"{a.shape[0]}s", repeatCount="indefinite"))

    # pos anim
    rect.add(dwg.animate(attributeName="x", values=x_offsets, dur=f"{a.shape[0]}s", repeatCount="indefinite"))
    rect.add(dwg.animate(attributeName="y", values=y_offsets, dur=f"{a.shape[0]}s", repeatCount="indefinite"))

    # add shape
    group.add(rect)
    return group


def circle_fn(group, dwg, a, x, y):
    a = np.abs(a)
    circle = dwg.circle(center=(f"{x}", f"{y}"), r=f"{a[0] / 2}")

    # anim seqs
    sizes = ";".join([f"{s.item()}" for s in a])
    x_offsets = ";".join([f"{x}" for s in a])
    y_offsets = ";".join([f"{y}" for s in a])

    # size anim
    circle.add(dwg.animate(attributeName="r", values=sizes, dur=f"{a.shape[0]}s", repeatCount="indefinite"))

    # pos anim
    circle.add(dwg.animate(attributeName="cx", values=x_offsets, dur=f"{a.shape[0]}s", repeatCount="indefinite"))
    circle.add(dwg.animate(attributeName="cy", values=y_offsets, dur=f"{a.shape[0]}s", repeatCount="indefinite"))

    # add shape
    group.add(circle)
    return group
