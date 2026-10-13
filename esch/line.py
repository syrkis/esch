"""Lines: points joined in order, moving with their per-frame positions."""

import numpy as np

from .animation import animate, number
from .color import color
from .glyph import STROKE
from .motion import trajectories
from .values import rate


def _points(vertices):
    """(row, column) vertices as an SVG (x, y) point list."""
    return " ".join(f"{number(x)},{number(y)}" for y, x in vertices)


def _line(e, points, stroke, fps, *, closed):
    """One panel's line; points are (point, frame, [row, column])."""
    frames = points.shape[1]
    # (frame, point, 2), repeating the first frame to close the loop.
    track = np.concatenate([points, points[:, :1]], axis=1).swapaxes(0, 1)
    shape = e.dwg.polygon if closed else e.dwg.polyline
    node = shape(fill="none", stroke=stroke, stroke_width=STROKE, stroke_linejoin="round", stroke_linecap="round")
    node.points = [(number(x), number(y)) for y, x in track[0]]
    animate(
        e, node, "points", [_points(vertices) for vertices in track], np.linspace(0, 1, frames + 1), frames / fps, track=track
    )
    return node


def line_fn(e, positions, col="black", fps=1, *, closed=False):
    """Join points in order; positions are (panel, point[, time], [row, column]).

    Vertices move linearly between frames like mesh glyphs, looping back to the
    first frame. closed=True also joins the last point to the first. Draw lines
    before meshes to keep glyphs on top.
    """
    positions = trajectories(e, positions)
    rate(fps)
    stroke = color(col)
    lines = [_line(e, points, stroke, fps, closed=closed) for points in positions]
    for group, node in zip(e.gs, lines, strict=True):
        group.add(node)
    return e
