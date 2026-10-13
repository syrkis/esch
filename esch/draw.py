"""Array dimensions to SVG panels, using einops axis conventions."""

import re
from math import ceil, sqrt
from numbers import Integral

import numpy as np
from einops import EinopsError, rearrange
from einops.parsing import ParsedExpression

from .drawing import Drawing
from .glyph import shape_name
from .plot import grid_fn
from .values import rate, values_array

TIME_AXIS_TWICE = "Use either pattern's t axis or time_axis, not both"


def _output_groups(pattern, ndim):
    """(axis index, names) of each parenthesized output group of an arrow pattern."""
    output = ParsedExpression(pattern.split("->")[1])
    index = 0
    for group in output.composition:
        if isinstance(group, str):  # Unparenthesized ellipsis expands to axes.
            index += ndim - len(output.composition) + 1
        else:
            yield index, group
            index += 1


def _named_roles(axes, roles, ndim):
    """Place named row/col/h/w/t output axes; fill h and w from the remaining axes."""
    if any(len(names) > 1 for names in roles.values()):
        raise ValueError("Keep row, col, h, w, and t in separate output axes")
    named = {index: next(iter(names)) for index, names in roles.items() if names}
    if "t" in axes:
        time_index = axes.index("t")
        if time_index in named and named[time_index] != "t":
            raise ValueError("time_axis conflicts with a named layout axis")
        named[time_index] = "t"
    axes = [named.get(i, f"batch{i}") for i in range(ndim)]
    available = [i for i in range(ndim) if i not in named]
    for name in ("w", "h"):
        if name not in axes and available:
            axes[available.pop()] = name
    return axes


def _arrow(arr, pattern, time_axis, lengths):
    """Rearrange with einops, then read layout roles from the output axes."""
    # Parse with einops rather than maintaining a second grouping grammar.
    arr = rearrange(arr, pattern, **lengths)
    roles = {index: set(group) & {"row", "col", "h", "w", "t"} for index, group in _output_groups(pattern, arr.ndim)}
    named_time = [index for index, names in roles.items() if "t" in names]
    if named_time:
        if time_axis is not None:
            raise ValueError(TIME_AXIS_TWICE)
        time_axis = named_time[-1]
    axes = _default_axes(arr.ndim, time_axis)
    if any(names & {"row", "col"} for names in roles.values()):
        axes = _named_roles(axes, roles, arr.ndim)
    return arr, axes


def _shorthand(arr, pattern, lengths):
    """Split packed dimensions of a pattern without an arrow, keeping its axis names."""
    axes = []
    for group in ParsedExpression(pattern).composition:
        if isinstance(group, str):
            axes.append("...")
        else:
            axes.extend("..." if name == "…" else str(name) for name in group)
    expanded = " ".join(axes)
    arr = rearrange(arr, f"{pattern} -> {expanded}", **lengths)
    return arr, _named_axes(arr.ndim, expanded)


def _prepare(arr, pattern, time_axis, lengths):
    """Delegate rearrangement grammar and shape validation to einops."""
    if pattern is None:
        if lengths:
            raise ValueError("Axis lengths require a pattern")
        return arr, _default_axes(arr.ndim, time_axis)
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a string")
    if re.search(r"\bfps\b", pattern) or "fps" in lengths:
        raise ValueError("fps is reserved for playback speed")
    if "->" in pattern:
        return _arrow(arr, pattern, time_axis, lengths)
    if time_axis is not None:
        raise ValueError(TIME_AXIS_TWICE)
    return _shorthand(arr, pattern, lengths)


def _named_axes(ndim, pattern):
    """Axis names from a space-separated pattern, expanding one ellipsis to batch axes."""
    axes = pattern.split()
    if axes.count("...") > 1:
        raise ValueError("pattern may contain only one ellipsis")
    if "..." in axes:
        count = ndim - len(axes) + 1
        if count < 0:
            raise ValueError("pattern has more axes than the array")
        index = axes.index("...")
        prefix = "eschbatch"
        while any(axis.startswith(prefix) for axis in axes):
            prefix += "x"
        axes[index : index + 1] = [f"{prefix}{i}" for i in range(count)]
    if len(axes) != ndim or len(set(axes)) != len(axes):
        raise ValueError("pattern must name each array axis exactly once")
    if any(re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", axis) is None or axis.endswith("_") for axis in axes):
        raise ValueError("Use axis names and optional ...; apply composite einops rearrangements before draw")
    return axes


def _default_axes(ndim, time_axis):
    """Batch axes, then rows and columns last, with an optional time axis."""
    axes = [f"batch{i}" for i in range(ndim)]
    spatial = list(range(ndim))
    if time_axis is not None:
        if isinstance(time_axis, bool) or not isinstance(time_axis, Integral) or not -ndim <= time_axis < ndim:
            raise ValueError("time_axis must index an existing array axis")
        time_axis %= ndim
        axes[time_axis] = "t"
        spatial.remove(time_axis)
    # The last two spatial axes (or fewer) become columns and rows.
    for index, name in zip(reversed(spatial), ("w", "h"), strict=False):
        axes[index] = name
    return axes


def _pattern_argument(arr, data, pattern):
    """Accept draw(array, pattern), draw(pattern, array) or draw(array, pattern=...)."""
    if isinstance(arr, str):
        if pattern is not None:
            raise ValueError("pattern was supplied twice")
        return data, arr
    if data is not None:
        if not isinstance(data, str) or pattern is not None:
            raise ValueError("Use draw(array, pattern) or draw(array, pattern=pattern)")
        return arr, data
    return arr, pattern


def _voronoi_only(**used):
    """Reject Voronoi options for square and circle glyphs, naming the first one used."""
    for name, value in used.items():
        if value:
            label = "positions in draw" if name == "positions" else name
            raise ValueError(f"{label} is only supported for shape='voronoi'")


def _shape(shape, *, dot, **voronoi_options):
    """The glyph shape, rejecting options that only the other kind of plot uses."""
    if shape == "voronoi":
        if dot:
            raise ValueError("dot is only supported for square and circle glyphs")
        return shape
    shape = shape_name(shape)
    _voronoi_only(**voronoi_options)
    return shape


def _lengths(pattern, col, axes_lengths, axis_sizes):
    """einops axis lengths from axes_lengths and keywords; an integer col is a length."""
    lengths = dict(axes_lengths or {})
    # Preserve col='navy' as a color alias, while col=2 supplies an axis length.
    if isinstance(col, Integral) and isinstance(pattern, str) and re.search(r"\bcol\b", pattern):
        if "col" in lengths:
            raise ValueError("An axis length was supplied twice")
        lengths["col"], col = col, "black"
    if lengths.keys() & axis_sizes.keys():
        raise ValueError("An axis length was supplied twice")
    lengths.update(axis_sizes)
    if any(isinstance(size, bool) or not isinstance(size, Integral) or size < 1 for size in lengths.values()):
        raise ValueError("Axis lengths must be positive integers")
    return lengths, col


def _labels(pattern, axes, ndim):
    """Pattern names of single-name output axes, used for split filenames."""
    if pattern is None:
        return {}
    if "->" in pattern:
        return {axes[index]: group[0] for index, group in _output_groups(pattern, ndim) if len(group) == 1}
    names = ParsedExpression(pattern).identifiers
    return {axis: axis for axis in axes if axis in names}


def _row_col(batch, sizes, columns):
    """Panel axes named row and col fix the layout; columns must agree."""
    if set(batch) - {"row", "col"}:
        raise ValueError("Combine all panel dimensions into row and col axes")
    expected = sizes.get("col", 1)
    if columns is not None and columns != expected:
        raise ValueError("columns conflicts with the pattern's row/col layout")
    return [axis for axis in ("row", "col") if axis in batch], expected


def _panels(axes, sizes, columns):
    """Panel axes in layout order, panel columns (None: choose later), and legacy n/c layout."""
    batch = [axis for axis in axes if axis not in ("h", "w", "t")]
    if set(batch) & {"row", "col"}:
        batch, columns = _row_col(batch, sizes, columns)
    # Preserve the existing n/c row/column convention, regardless of input order.
    legacy = bool(batch) and set(batch) <= {"n", "c"}
    if legacy:
        batch = [axis for axis in ("n", "c") if axis in batch]
    return batch, columns, legacy


def draw(
    arr,
    data=None,
    debug=False,
    *,
    pattern=None,
    time_axis=None,
    shape="square",
    positions=None,
    dots=False,
    monochrome=False,
    edges=None,
    edge=None,
    edge_pos=None,
    hide_outer=False,
    dot=False,
    fps=1,
    columns=None,
    gap=5,
    col="black",
    pos=None,
    neg=None,
    color_by_magnitude=None,
    normalize=True,
    axes_lengths=None,
    **axis_sizes,
):
    """Render any-rank real array; trailing spatial axes form a grid.

    Leading axes become panels. Scalars and vectors get missing singleton axes.
    Use pattern='... h w t' or time_axis=-1 to animate. Named axes other than
    h (row), w (column), and t (time) become panels. draw('h w', arr) also works.
    Arrow patterns rearrange first, then use output order for spatial layout;
    an output axis containing t is time. Pass einops axis lengths as keywords
    or via axes_lengths for names that collide with drawing options.
    dot=True draws zero as an outline-sized square or circle instead of nothing.
    """
    arr, pattern = _pattern_argument(arr, data, pattern)
    arr = values_array(arr, normalize=normalize)
    shape = _shape(
        shape,
        dot=dot,
        positions=positions is not None,
        dots=dots,
        monochrome=monochrome,
        edges=edges is not None,
        edge=edge is not None,
        edge_pos=edge_pos is not None,
        hide_outer=hide_outer,
    )
    if color_by_magnitude is None:
        color_by_magnitude = shape == "voronoi"
    rate(fps)
    lengths, col = _lengths(pattern, col, axes_lengths, axis_sizes)
    try:
        arr, axes = _prepare(arr, pattern, time_axis, lengths)
    except EinopsError as error:
        raise ValueError(str(error)) from error
    sizes = dict(zip(axes, arr.shape, strict=True))
    labels = _labels(pattern, axes, arr.ndim)
    batch, layout_columns, legacy = _panels(axes, sizes, columns)
    target = " ".join([f"({' '.join(batch)})" if batch else "1", *[axis if axis in axes else "1" for axis in ("h", "w", "t")]])
    array = rearrange(arr, f"{' '.join(axes)} -> {target}")
    panels, h, w, _ = array.shape
    if layout_columns is None:
        layout_columns = sizes.get("c", 1) if legacy else ceil(sqrt(panels))
    if isinstance(layout_columns, bool) or not isinstance(layout_columns, Integral) or layout_columns < 1:
        raise ValueError("columns must be a positive integer")
    layout_columns = min(layout_columns, panels)
    radius = 0.5 if shape == "voronoi" else sqrt(float(np.max(np.abs(array)))) / (2 if shape == "square" else 2.1)
    drawing = Drawing(
        h, w, ceil(panels / layout_columns), layout_columns, pad=max(1, radius + 0.1), debug=debug, panels=panels, gap=gap
    )
    drawing._panel_axes = tuple(batch)
    drawing._panel_shape = tuple(sizes[axis] for axis in batch)
    drawing._filename_axes = {labels[axis]: i for i, axis in enumerate(batch) if axis in labels}
    drawing._save_columns = columns
    options = {"fps": fps, "col": col, "pos": pos, "neg": neg, "color_by_magnitude": color_by_magnitude, "normalize": False}
    if shape != "voronoi":
        return grid_fn(drawing, array, shape=shape, dot=dot, **options)
    from .voronoi import voronoi_grid

    return voronoi_grid(
        drawing,
        array,
        positions,
        dots=dots,
        monochrome=monochrome,
        edges=edges,
        edge=edge,
        edge_pos=edge_pos,
        hide_outer=hide_outer,
        **options,
    )
