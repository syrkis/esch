"""Signed glyphs: sizes, positions and colors of one value over time."""

import numpy as np

from .animation import animate, number
from .color import blend, color
from .values import rate, values_array

STROKE = 0.04  # Outline width in grid units, shared by glyphs and Voronoi edges.


def shape_name(shape):
    aliases = {"square": "square", "cube": "square", "circle": "circle", "sphere": "circle"}
    if shape not in aliases:
        raise ValueError("shape must be 'square' or 'circle' ('cube' and 'sphere' are aliases)")
    return aliases[shape]


def _crossings(values):
    """Frames with zeros inserted at sign changes, their loop times, and signs.

    Signed values interpolate through zero, including the closing segment, so
    +a -> -a shrinks and regrows; taking abs first would keep a constant size.
    At zero the glyph is invisible and takes the outgoing segment's sign.
    """
    count = len(values)
    if count == 1:
        return values, np.zeros(1), values < 0
    left, right = values, np.roll(values, -1)
    crossing = np.flatnonzero((left < 0) & (right > 0) | (right < 0) & (left > 0))
    scale = np.maximum(np.abs(left[crossing]), np.abs(right[crossing]))
    fraction = (np.abs(left[crossing]) / scale) / (np.abs(left[crossing]) / scale + np.abs(right[crossing]) / scale)
    frames = np.concatenate([values[:1], np.insert(right, crossing, 0.0)])
    ends = (np.arange(count) + 1) / count
    times = np.concatenate([[0.0], np.insert(ends, crossing, (crossing + fraction) / count)])
    negative = frames < 0
    negative[:-1] = np.where(frames[:-1] == 0, frames[1:] < 0, negative[:-1])
    negative[-1] = negative[0]
    return frames, times, negative


def _coordinates(sequence, times, count):
    """A position per frame (or fixed), sampled at the inserted zero crossings too."""
    sequence = np.broadcast_to(np.asarray(sequence), (count,))
    return np.interp(times, np.linspace(0, 1, count + 1), np.append(sequence, sequence[0]))


def _geometry(e, shape, magnitude, x, y, *, vertices, marker, dot):
    """The SVG element and its per-frame numeric attributes.

    With dot, a glyph never shrinks below its own outline: the square side and
    the circle diameter are at least STROKE, so no interior (or fill) shows.
    """
    if shape == "voronoi":
        node = e.dwg.polygon(points=vertices)
        node.update({"stroke-linejoin": "round"})
        return node, {}
    floor = STROKE if dot else 0.0
    if shape == "square":
        side = np.maximum(np.sqrt(magnitude), floor)
        return e.dwg.rect(), {"width": side, "height": side, "x": x - side / 2, "y": y - side / 2}
    radius = np.full(len(magnitude), 0.04) if marker else np.maximum(np.sqrt(magnitude) / 2.1, floor / 2)
    return e.dwg.circle(), {"r": radius, "cx": x, "cy": y}


def _paint(negative, magnitude, positive_color, negative_color, *, color_by_magnitude, color_max, voronoi, marker):
    """Fill and stroke per frame; Voronoi regions have white borders, site markers white centers."""
    fills, strokes = _signed_paint(
        negative, magnitude, positive_color, negative_color, color_by_magnitude=color_by_magnitude, color_max=color_max
    )
    white = ["white"] * len(negative)
    return white if marker else fills, white if voronoi else strokes


def _signed_paint(negative, magnitude, positive_color, negative_color, *, color_by_magnitude, color_max):
    """Fill and stroke per frame; an omitted negative color stays hollow."""
    hollow = "white" if negative_color is None else negative_color
    if not color_by_magnitude:
        fills = np.where(negative, hollow, positive_color).tolist()
        strokes = np.where(negative, negative_color or positive_color, positive_color).tolist()
        return fills, strokes
    maximum = float(np.max(magnitude)) if color_max is None else color_max
    weights = magnitude / maximum if maximum else np.zeros_like(magnitude)
    positives, negatives = blend(positive_color, weights), blend(hollow, weights)
    fills = np.where(negative, negatives, positives).tolist()
    return fills, fills if negative_color is not None else positives


def glyph(
    e,
    g,
    values,
    x,
    y,
    shape="square",
    fps=1,
    col="black",
    *,
    pos=None,
    neg=None,
    color_by_magnitude=False,
    color_max=None,
    vertices=None,
    marker=False,
    dot=False,
):
    """Draw sign-dependent fills and outlines; omitted negative color stays hollow.

    marker draws a fixed-size Voronoi site marker. dot keeps zero (and small)
    values visible as an outline-sized glyph.
    """
    col = color(col if pos is None else pos)
    neg = None if neg is None else color(neg)
    if shape != "voronoi":
        shape = shape_name(shape)
    elif vertices is None:
        raise ValueError("Voronoi glyphs require polygon vertices")
    rate(fps)
    values = values_array(values).reshape(-1)
    frames, times, negative = _crossings(values)
    magnitude = np.abs(frames)
    x, y = _coordinates(x, times, len(values)), _coordinates(y, times, len(values))
    node, attributes = _geometry(e, shape, magnitude, x, y, vertices=vertices, marker=marker, dot=dot)
    fills, strokes = _paint(
        negative,
        magnitude,
        col,
        neg,
        color_by_magnitude=color_by_magnitude,
        color_max=color_max,
        voronoi=shape == "voronoi",
        marker=marker,
    )
    node.update({name: number(seq[0]) for name, seq in attributes.items()})
    node.update({"fill": fills[0], "stroke": strokes[0], "stroke-width": 0.01 if marker else STROKE})
    if len(values) > 1:
        duration = len(values) / fps
        for name, seq in attributes.items():
            animate(e, node, name, [number(v) for v in seq], times, duration, track=seq[:, None, None])
        for attribute, colors in (("fill", fills), ("stroke", strokes)):
            animate(e, node, attribute, colors, times, duration, mode="linear" if color_by_magnitude else "discrete")
    g.add(node)
    return node
