"""Static and moving-site Voronoi layers clipped using Shapely/GEOS."""

from dataclasses import dataclass
from functools import cache
from itertools import pairwise

import numpy as np
from shapely import MultiPoint, Polygon, box, voronoi_polygons
from shapely.geometry.polygon import orient

from .animation import animate, number
from .color import color
from .glyph import STROKE, glyph
from .motion import point_timelines
from .values import rate, values_array

# Cells extend past the panel by MARGIN so that their own strokes draw only
# interior edges; clipping hides the margin. It exceeds half of STROKE.
MARGIN = 0.05
# Largest distance, in grid units, between a drawn outline and its sampled cell.
TOLERANCE = 0.02
# Geometry samples per second, and halvings of a sample interval where a
# linear morph misses the true cell.
RATE = 15
REFINE = 4


def _cells(points, bounds):
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Voronoi sites must be distinct within each panel, including motion samples")
    cells = (
        [bounds] if len(points) == 1 else voronoi_polygons(MultiPoint(points[:, ::-1]), extend_to=bounds, ordered=True).geoms
    )
    result = []
    for cell in cells:
        polygon = cell.intersection(bounds)
        if polygon.is_empty or polygon.geom_type != "Polygon":
            raise ValueError("Could not construct a nonempty Voronoi cell for each site")
        # Consistent orientation lets outlines be matched vertex by vertex.
        result.append(np.asarray(orient(polygon).exterior.coords)[:-1])
    return result


def _on_frame(start, end, bounds):
    """Whether a segment lies along the artificial clipping rectangle."""
    xmin, ymin, xmax, ymax = bounds
    return any(start[axis] == end[axis] == limit for axis, limit in ((0, xmin), (0, xmax), (1, ymin), (1, ymax)))


def _segment_key(start, end):
    """Adjacent clipped polygons can differ at machine precision; canonicalize
    below SVG output precision so both cells share one segment."""
    return tuple(sorted(tuple(round(value, 12) for value in point) for point in (start, end)))


def _interior_edges(cells, bounds):
    """Unique cell edges and the cells owning them, excluding the clipping rectangle."""
    segments = {}
    for index, cell in enumerate(cells):
        cell = [tuple(point) for point in cell.tolist()]
        for start, end in zip(cell, cell[1:] + cell[:1], strict=True):
            if start != end and not _on_frame(start, end, bounds.bounds):
                segments.setdefault(_segment_key(start, end), set()).add(index)
    return dict(sorted(segments.items()))


def _touches_boundary(cell, bounds):
    """Whether a convex cell reaches the panel edge (its margin lies beyond it)."""
    xmin, ymin, xmax, ymax = bounds
    tolerance = 1e-10 * max(1, xmax - xmin, ymax - ymin)
    (left, top), (right, bottom) = cell.min(axis=0), cell.max(axis=0)
    return left <= xmin + tolerance or top <= ymin + tolerance or right >= xmax - tolerance or bottom >= ymax - tolerance


def _align(reference, cell):
    """Spread a cell's vertices over the reference's slots, cyclically and in order.

    Every vertex fills at least one slot; repeated vertices add only zero-length
    edges. Minimizing squared displacement lets linear morphs follow each vertex,
    including vertices that appear or vanish as the topology changes.
    """
    slots, count = len(reference), len(cell)
    rotations = (np.arange(count)[:, None] + np.arange(count)) % count
    distance = np.sum((reference[None, :, None] - cell[rotations][:, None]) ** 2, axis=-1)
    cost = np.full((count, count), np.inf)
    cost[:, 0] = distance[:, 0, 0]
    advanced = np.zeros((slots, count, count), dtype=bool)
    for slot in range(1, slots):
        moved = np.full_like(cost, np.inf)
        moved[:, 1:] = cost[:, :-1]
        advanced[slot] = moved < cost
        cost = np.minimum(cost, moved) + distance[:, slot]
    rotation, vertex, path = int(np.argmin(cost[:, -1])), count - 1, []
    for slot in range(slots - 1, -1, -1):
        path.append(rotations[rotation, vertex])
        vertex -= advanced[slot, rotation, vertex]
    return cell[path[::-1]]


def _pad(vertices, slots):
    """Repeat the last vertex; earlier frames follow a later slot split this way."""
    return np.concatenate([vertices, np.repeat(vertices[-1:], slots - len(vertices), axis=0)])


def _track(cell, times):
    """One cell's outlines at loop times, as vertex lists of one length.

    cell(t) gives the true outline. Where the linear morph between neighbouring
    times misses the true cell halfway by more than TOLERANCE, the interval is
    halved, up to REFINE times, so every cell stays continuous and close to its
    neighbours.
    """
    track, stamps = [cell(times[0])], [times[0]]

    def extend(start, end, depth):
        target = cell(end)
        slots = max(len(track[-1]), len(target))
        target = _align(_pad(track[-1], slots), target)
        if depth:
            middle = (start + end) / 2
            morph = Polygon((_pad(track[-1], slots) + target) / 2)
            if morph.hausdorff_distance(Polygon(cell(middle))) > TOLERANCE:
                extend(start, middle, depth - 1)
                extend(middle, end, depth - 1)
                return
        track.append(target)
        stamps.append(end)

    for start, end in pairwise(times):
        extend(start, end, REFINE)
    slots = max(map(len, track))
    return np.stack([_pad(vertices, slots) for vertices in track]), np.array(stamps)


def _outline(vertices):
    return " ".join(f"{number(x)},{number(y)}" for x, y in vertices)


def _visibility(e, node, visible, duration):
    states = ["visible" if value else "hidden" for value in visible]
    node.update({"visibility": states[0]})
    animate(e, node, "visibility", [*states, states[0]], np.linspace(0, 1, len(states) + 1), duration, mode="discrete")


@dataclass(frozen=True)
class Style:
    """Validated Voronoi options shared by every panel."""

    fps: float
    col: object
    pos: object
    neg: object
    color_by_magnitude: bool
    maximum: float
    dots: bool
    monochrome: bool
    edges: bool
    edge: str
    edge_pos: str | None
    hide_outer: bool


def _style(*, dots, monochrome, hide_outer, edges, edge, edge_pos, fps, **colors):
    """Check option types; default edge colors; drop an edge_pos equal to edge."""
    for name, flag in (("dots", dots), ("monochrome", monochrome), ("hide_outer", hide_outer)):
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
    edges = True if edges is None else edges
    if not isinstance(edges, (bool, np.bool_)):
        raise TypeError("edges must be a boolean or None")
    edge = color(edge) if edge is not None else ("black" if monochrome else "white")
    edge_pos = color(edge_pos) if edge_pos is not None and color(edge_pos) != edge else None
    rate(fps)
    return Style(
        fps, **colors, dots=dots, monochrome=monochrome, edges=edges, edge=edge, edge_pos=edge_pos, hide_outer=hide_outer
    )


def _sampler(points, bounds):
    """All cells at a loop fraction t, as sites move linearly between frames; cached."""
    frames = points.shape[1]

    @cache
    def cells_at(t):
        frame, fraction = divmod(t * frames, 1)
        start, end = points[:, int(frame) % frames], points[:, (int(frame) + 1) % frames]
        return _cells((1 - fraction) * start + fraction * end, bounds)

    return cells_at


def _clipped_layer(e):
    """A panel layer whose regions group clips each cell's margin.

    Each layer carries its clip so copied panels stay self-contained. Equal ids
    always define the same panel rectangle. (A nested <svg> viewport would
    start its own SMIL timeline, which pausing misses.)
    """
    layer = e.dwg.g()
    clip = layer.add(e.dwg.clipPath(id=f"esch-panel-{number(e.w)}x{number(e.h)}"))
    clip.add(e.dwg.rect(insert=(-0.5, -0.5), size=(e.w, e.h)))
    return layer, layer.add(e.dwg.g(clip_path=f"url(#{clip['id']})"))


def _region(e, regions, values, track, times, *, stroke, visible, duration, style):
    """One cell: its fill timeline, stroke, visibility and outline morph."""
    if style.monochrome:
        node = regions.add(e.dwg.polygon(fill="none", stroke_linejoin="round"))
    else:
        node = glyph(
            e,
            regions,
            values,
            0,
            0,
            "voronoi",
            style.fps,
            style.col,
            pos=style.pos,
            neg=style.neg,
            color_by_magnitude=style.color_by_magnitude,
            color_max=style.maximum,
            vertices=[],
        )
    node.points = [(number(x), number(y)) for x, y in track[0]]
    node.update({"stroke": stroke, "stroke-width": STROKE})
    if style.hide_outer:
        _visibility(e, node, visible, duration)
    animate(e, node, "points", list(map(_outline, track)), times, duration, track=track, tolerance=TOLERANCE)


def _regions(e, regions, panel, cells_at, times, visible, *, stroked, duration, style):
    """Every cell's region. Cells stroke their own edges, positive cells first so
    that an edge between a positive and a negative cell takes the later color."""
    positive = panel[:, 0] > 0
    order = np.argsort(~positive, kind="stable") if stroked and style.edge_pos is not None else range(len(panel))
    for index in order:
        track, track_times = _track(lambda t, index=index: cells_at(t)[index], times)
        stroke = (style.edge_pos if style.edge_pos is not None and positive[index] else style.edge) if stroked else "none"
        _region(
            e,
            regions,
            panel[index],
            track,
            track_times,
            stroke=stroke,
            visible=visible[:, index],
            duration=duration,
            style=style,
        )


def _edge_colors(edges, visible, live, style):
    """Color of each interior edge with a visible owner; edges between two live
    (positive) cells take edge_pos."""
    return {
        segment: style.edge_pos if len(owners) == 2 and all(live[i] for i in owners) else style.edge
        for segment, owners in edges.items()
        if any(visible[owner] for owner in owners)
    }


def _edge_paths(e, regions, samples, panel, visible, *, steps, times, duration, bounds, style):
    """Interior edges as discrete paths, for edge colors that follow changing signs."""
    frames = panel.shape[1]
    frame, step = np.divmod(np.arange(len(samples)), steps)
    fraction = (step / steps)[:, None]
    live = ((1 - fraction) * panel[:, frame].T + fraction * panel[:, (frame + 1) % frames].T) > 0
    colored = [
        _edge_colors(_interior_edges(sample, bounds), visible[index], live[index], style)
        for index, sample in enumerate(samples)
    ]
    for color_ in (style.edge, style.edge_pos):
        segments = [[segment for segment, shade in edges.items() if shade == color_] for edges in colored]
        if max(map(len, segments)):
            regions.add(_edge_path(e, segments, color_, times, duration))


def _edge_path(e, segments, color_, times, duration):
    """One color's edges per sample as a discretely animated path."""
    count = max(map(len, segments))
    # Keep the command list stable when edge topology changes.
    padded = [frame + [((0.0, 0.0), (0.0, 0.0))] * (count - len(frame)) for frame in segments]
    paths = [" ".join(f"M{number(a)},{number(b)}L{number(c)},{number(d)}" for (a, b), (c, d) in frame) for frame in padded]
    border = e.dwg.path(d=paths[0], fill="none", stroke=color_, stroke_width=STROKE, stroke_linecap="butt")
    animate(e, border, "d", [*paths, paths[0]], times, duration, mode="discrete")
    return border


def _dots(e, layer, points, panel, visible, *, duration, style):
    """Site markers above all regions, on the same timeline as meshes."""
    for index, (trajectory, value) in enumerate(zip(points, panel, strict=True)):
        dot = glyph(
            e,
            layer,
            np.ones_like(value) if style.monochrome else value,
            trajectory[:, 1],
            trajectory[:, 0],
            "circle",
            style.fps,
            style.col,
            pos="black" if style.monochrome else style.pos,
            neg=None if style.monochrome else style.neg,
            marker=True,
        )
        if style.monochrome:
            dot.update({"fill": "black"})
        if style.hide_outer:
            _visibility(e, dot, visible[:, index], duration)


def _timeline(frames, fps, *, sampled):
    """Geometry samples per frame and their loop times, including the closing time 1.

    Sample geometry densely; keyframes keep only what interpolation misses.
    """
    steps = max(1, int(np.ceil(RATE / fps))) if sampled else 1
    return steps, np.linspace(0, 1, (frames if sampled else 1) * steps + 1)


def _layer(e, points, panel, style):
    """One panel's regions, edges and dots; points are (site, frame, 2)."""
    inner = (-0.5, -0.5, e.w - 0.5, e.h - 0.5)
    bounds = box(*inner).buffer(MARGIN, join_style="mitre")
    frames = points.shape[1]
    duration = frames / style.fps
    moving = np.any(points != points[:, :1])
    # Edge colors that follow changing signs need separately animated edges.
    changing = np.any((panel > 0) != (panel[:, :1] > 0))
    stroked = style.edges and (style.edge_pos is None or not changing)
    steps, times = _timeline(frames, style.fps, sampled=moving or (style.edges and not stroked))
    cells_at = _sampler(points, bounds)
    samples = [cells_at(t) for t in times[:-1]]
    visible = np.array([[not style.hide_outer or not _touches_boundary(cell, inner) for cell in sample] for sample in samples])
    layer, regions = _clipped_layer(e)
    if stroked or not style.monochrome:
        keys = times if moving else times[[0, -1]]
        _regions(e, regions, panel, cells_at, keys, visible, stroked=stroked, duration=duration, style=style)
    if style.edges and not stroked:
        _edge_paths(
            e, regions, samples, panel, visible, steps=steps, times=times, duration=duration, bounds=bounds, style=style
        )
    if style.dots:
        _dots(e, layer, points, panel, visible, duration=duration, style=style)
    return layer


def voronoi_fn(
    e,
    positions,
    arr,
    *,
    fps=1,
    col="black",
    pos=None,
    neg=None,
    color_by_magnitude=True,
    normalize=True,
    dots=False,
    monochrome=False,
    edges=None,
    edge=None,
    edge_pos=None,
    hide_outer=False,
):
    """Color nearest-site regions; positions are (panel, point, [row, column]).

    Values have shape (panel, point[, time]); positions may also have a time
    axis before the coordinate pair. Motion loops linearly between samples.
    """
    values = values_array(arr, normalize=normalize)
    positions, values = point_timelines(e, positions, values)
    style = _style(
        dots=dots,
        monochrome=monochrome,
        hide_outer=hide_outer,
        edges=edges,
        edge=edge,
        edge_pos=edge_pos,
        fps=fps,
        col=col,
        pos=pos,
        neg=neg,
        color_by_magnitude=color_by_magnitude,
        maximum=float(np.max(np.abs(values))),
    )
    if np.any(positions < -0.5) or np.any(positions > np.array([e.h, e.w]) - 0.5):
        raise ValueError("Voronoi sites must lie within the panel's cell bounds")
    # Build every layer before changing the existing drawing, so errors leave it intact.
    layers = [_layer(e, points, panel, style) for points, panel in zip(positions, values, strict=True)]
    for group, layer in zip(e.gs, layers, strict=True):
        group.elements.extend(layer.elements)
    return e


def voronoi_grid(e, arr, positions=None, **options):
    """Adapt canonical (panel, h, w, t) grid data to Voronoi sites."""
    panels, h, w, frames = arr.shape
    if positions is None:
        positions = np.moveaxis(np.indices((h, w)), 0, -1)
    positions = values_array(positions)
    if positions.shape == (h, w, 2):
        positions = np.broadcast_to(positions, (panels, h, w, 2))
    elif positions.shape == (h, w, frames, 2) and positions.shape != (panels, h, w, 2):
        positions = np.broadcast_to(positions, (panels, h, w, frames, 2))
    if positions.shape == (panels, h, w, 2):
        positions = positions.reshape(panels, h * w, 2)
    elif positions.shape == (panels, h, w, frames, 2):
        positions = positions.reshape(panels, h * w, frames, 2)
    else:
        raise ValueError("Voronoi grid positions must have shape [panels,] h, w, [time,] 2 matching values")
    return voronoi_fn(e, positions, arr.reshape(panels, h * w, frames), **options)
