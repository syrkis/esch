"""Layer grids or fixed-position meshes onto an existing Drawing."""

import numpy as np

from .glyph import glyph, shape_name
from .motion import point_timelines
from .values import rate, values_array


def _settings(shape, fps):
    shape = shape_name(shape)
    rate(fps)
    return shape


def grid_fn(
    e,
    arr,
    shape="sphere",
    fps=1,
    ticks=None,
    col="black",
    *,
    pos=None,
    neg=None,
    color_by_magnitude=False,
    normalize=True,
    dot=False,
):
    """Add (panel, row, column[, time]) values and return the drawing.

    dot=True draws zero as an outline-sized glyph instead of nothing.
    """
    arr = values_array(arr, normalize=normalize)
    shape = _settings(shape, fps)
    if arr.ndim not in (3, 4) or arr.shape[:3] != (len(e.gs), e.h, e.w):
        raise ValueError("grid values must have shape (panels, h, w[, time]) matching Drawing")
    color_max = float(np.max(np.abs(arr)))
    for group, panel in zip(e.gs, arr, strict=True):
        for row, column in np.ndindex(panel.shape[:2]):
            glyph(
                e,
                group,
                panel[row, column],
                column,
                row,
                shape,
                fps,
                col,
                pos=pos,
                neg=neg,
                color_by_magnitude=color_by_magnitude,
                color_max=color_max,
                dot=dot,
            )
        if ticks is not None and ticks is not False:
            tick_fn(e, group)
    return e


def _mesh_scale(values, count, *, scaled):
    """The historical mesh scaling, 1 / (sqrt(point count) * 2.1), when scaled."""
    return values / np.sqrt(count) / 2.1 if scaled else values


def mesh_fn(
    e,
    positions,
    arr,
    col="black",
    shape="sphere",
    fps=1,
    *,
    pos=None,
    neg=None,
    color_by_magnitude=False,
    normalize=True,
    dot=False,
    scaled=True,
):
    """Add a mesh; positions are (panel, point[, time], [row, column]).

    Values have shape (panel, point[, time]). scaled=True keeps the original
    mesh scaling, dividing values by sqrt(point count) * 2.1; scaled=False sizes
    glyphs as draw does. dot=True draws zero as an outline-sized glyph.
    """
    positions, arr = point_timelines(e, positions, values_array(arr, normalize=normalize))
    shape = _settings(shape, fps)
    count = arr.shape[1]
    color_max = _mesh_scale(float(np.max(np.abs(arr))), count, scaled=scaled)
    for group, points, panel in zip(e.gs, positions, arr, strict=True):
        for trajectory, value in zip(points, panel, strict=True):
            glyph(
                e,
                group,
                _mesh_scale(value, count, scaled=scaled),
                trajectory[:, 1],
                trajectory[:, 0],
                shape,
                fps,
                col,
                pos=pos,
                neg=neg,
                color_by_magnitude=color_by_magnitude,
                color_max=color_max,
                dot=dot,
            )
    return e


def tick_fn(e, g):
    """Place ticks around all four sides, including rectangular grids."""
    for column in range(e.w):
        for edge, direction in ((-0.5, -1), (e.h - 0.5, 1)):
            g.add(e.dwg.line(start=(column, edge), end=(column, edge + direction * e.tick), stroke="black", stroke_width=0.04))
    for row in range(e.h):
        for edge, direction in ((-0.5, -1), (e.w - 0.5, 1)):
            g.add(e.dwg.line(start=(edge, row), end=(edge + direction * e.tick, row), stroke="black", stroke_width=0.04))
