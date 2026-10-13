"""Experimental variable-length Voronoi clips with shared moving geometry.

This repository-only exporter uses kinetic Delaunay events to retain
neighbours, then switches a bounded set of clip slots using native SMIL. It is
not an exact-arithmetic certificate implementation or a public esch API.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from shapely import MultiPoint, delaunay_triangles

import esch

from .kinetic import kinetic_intervals


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float
    neighbors: tuple[int, ...]


def _clip(cell, normal, limit):
    result = []
    for start, end in zip(cell, np.roll(cell, -1, axis=0), strict=True):
        a, b = start @ normal - limit, end @ normal - limit
        if a <= 0:
            result.append(start)
        if (a <= 0) != (b <= 0):
            result.append(start + a / (a - b) * (end - start))
    return np.asarray(result)


def _extreme(a, b, c, maximum):
    t = np.clip(-b / (2 * np.where(a != 0, a, 1)), 0, 1)
    values = np.stack([c, a + b + c, a * t * t + b * t + c])
    return values.max(axis=0) if maximum else values.min(axis=0)


def interval_neighbors(start, end, bounds, *, owners=None):
    """Conservatively bound required neighbours for the entire linear interval.

    f_j(x,t) <= 0 means x is closer to i than j. f is affine in x and
    quadratic in normalized t. Relaxed midpoint inequalities enclose all cells
    allowed by an initial neighbour set. An omitted constraint is discarded
    only when it holds throughout that enclosure for every t in [0,1].
    """
    xmin, ymin, xmax, ymax = bounds
    corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    midpoint = (start + end) / 2
    ids = {tuple(point): i for i, point in enumerate(midpoint)}
    neighbors = [set() for _ in start]
    for line in delaunay_triangles(MultiPoint(midpoint), only_edges=True).geoms:
        i, j = [ids[tuple(point)] for point in line.coords]
        neighbors[i].add(j)
        neighbors[j].add(i)
    velocity = end - start
    tolerance = 1e-10 * max(1, xmax - xmin, ymax - ymin) ** 2
    result = []
    for i in range(len(start)) if owners is None else owners:
        point = start[i]
        a = velocity[i] @ velocity[i] - np.sum(velocity**2, axis=1)
        bn = 2 * (velocity - velocity[i])
        bc = 2 * (point @ velocity[i] - np.sum(start * velocity, axis=1))
        cn = 2 * (start - point)
        cc = point @ point - np.sum(start**2, axis=1)
        envelope = corners.copy()
        for _ in range(3):
            for j in sorted(neighbors[i]):
                b, c = envelope @ bn[j] + bc[j], envelope @ cn[j] + cc[j]
                relaxation = np.max(0.25 * a[j] + 0.5 * b + c - _extreme(a[j], b, c, False)) + tolerance
                envelope = _clip(envelope, cn[j] + 0.5 * bn[j], relaxation - (cc[j] + 0.5 * bc[j] + 0.25 * a[j]))
                if len(envelope) < 3 or not np.isfinite(envelope).all():
                    break
            else:
                continue
            break
        if len(envelope) < 3 or not np.isfinite(envelope).all():
            result.append(tuple(j for j in range(len(start)) if j != i))
            continue
        b = bn @ envelope.T + bc[:, None]
        c = cn @ envelope.T + cc[:, None]
        maximum = _extreme(a[:, None], b, c, True).max(axis=1)
        retained = set(np.flatnonzero(maximum > -tolerance)) | neighbors[i]
        retained.discard(i)
        result.append(tuple(sorted(retained)))
    return result


def plan_chunks(points, bounds):
    """Merge per-cell schedules from kinetic events, in input-frame units."""
    points = np.asarray(points, dtype=float)
    return [list(chunks) for chunks in _cached_plans(points.shape, points.tobytes(), tuple(bounds))]


@lru_cache(maxsize=4)
def _cached_plans(shape, data, bounds):
    # Gallery recipes intentionally repeat the same calculation. Cache only
    # immutable geometry plans, never drawings or serialized test results.
    points = np.frombuffer(data, dtype=float).reshape(shape)
    _, frames, _ = points.shape
    plans = [[] for _ in points]
    for frame in range(frames):
        for a, b, neighbors in kinetic_intervals(points[:, frame], points[:, (frame + 1) % frames], bounds, interval_neighbors):
            for i, row in enumerate(neighbors):
                plans[i].append(Chunk(frame + a, frame + b, tuple(row)))
    # Merge consecutive equal neighbour sets, including across input frames.
    for i, chunks in enumerate(plans):
        merged = []
        for chunk in chunks:
            if merged and merged[-1].neighbors == chunk.neighbors:
                merged[-1] = Chunk(merged[-1].start, chunk.end, chunk.neighbors)
            else:
                merged.append(chunk)
        plans[i] = merged
    return tuple(tuple(chunks) for chunks in plans)


def _separations(points):
    """Minimum pair separation in each segment, including the closing segment."""
    result = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = points[j] - points[i]
            v = np.roll(d, -1, axis=0) - d
            denominator = np.sum(v**2, axis=-1)
            t = np.clip(-np.sum(d * v, axis=-1) / np.where(denominator, denominator, 1), 0, 1)
            result[i, j] = np.linalg.norm(d + t[:, None] * v, axis=-1)
    return result


def _slots(chunks):
    """Keep surviving neighbours in their slots to avoid needless switches."""
    slots = [None] * max(len(chunk.neighbors) for chunk in chunks)
    states = []
    for chunk in chunks:
        wanted = set(chunk.neighbors)
        slots = [j if j in wanted else None for j in slots]
        newcomers = iter(sorted(wanted - set(slots)))
        slots = [next(newcomers, None) if j is None else j for j in slots]
        states.append((chunk.start, tuple(slots)))
    return states


def adaptive_example(positions, colors, *, h, w, fps=1):
    """Single panel, fixed colors, white internal edges; no embedded scripts."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 2 or not all(positions.shape):
        raise ValueError("Expected nonempty (site, time, 2) positions")
    if not np.isfinite(positions).all() or np.any(positions < -0.5) or np.any(positions > np.array([h, w]) - 0.5):
        raise ValueError("Sites must be finite and inside the panel")
    if len(colors) != len(positions) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Provide one color per site and positive finite fps")
    points = positions[..., ::-1]
    bounds = (-0.5, -0.5, w - 0.5, h - 0.5)
    diagonal = np.hypot(h, w)
    separations = _separations(points)
    if any(np.min(values) < 1e-10 * max(1, diagonal) for values in separations.values()):
        raise ValueError("Sites must never collide, including between frames")
    plans = plan_chunks(points, bounds)
    drawing = esch.Drawing(h, w)
    svg = drawing.dwg
    frames = points.shape[1]
    duration = f"{frames / fps:g}s"
    used = {}
    for i, chunks in enumerate(plans):
        for chunk in chunks:
            for j in chunk.neighbors:
                used.setdefault((i, j), set()).update(range(int(chunk.start), int(np.ceil(chunk.end))))
    for (i, j), active in sorted(used.items()):
        indices = sorted({0, frames} | active | {frame + 1 for frame in active})
        scale = 2 * diagonal / separations[min(i, j), max(i, j)][sorted(active)].min()
        p, q = points[i, np.array(indices) % frames], points[j, np.array(indices) % frames]
        difference, midpoint = q - p, (p + q) / 2
        tangent = scale * np.stack([-difference[:, 1], difference[:, 0]], axis=-1)
        poses = np.stack([midpoint + tangent, midpoint - tangent, midpoint - scale * difference], axis=1)
        values = [" ".join(f"{x:.12g},{y:.12g}" for x, y in pose) for pose in poses]
        polygon = svg.polygon(points=poses[0].tolist(), id=f"plane-{i}-{j}", fill="none")
        if frames > 1:
            polygon.add(
                svg.animate(
                    attributeName="points",
                    values=";".join(values),
                    keyTimes=";".join(f"{t / frames:.12g}" for t in indices),
                    dur=duration,
                    repeatCount="indefinite",
                    calcMode="linear",
                )
            )
        svg.defs.add(polygon)
        clip = svg.clipPath(id=f"planeclip-{i}-{j}", clipPathUnits="userSpaceOnUse")
        clip.add(svg.use(href=f"#plane-{i}-{j}"))
        svg.defs.add(clip)
    panel_clip = svg.clipPath(id="panel", clipPathUnits="userSpaceOnUse")
    panel_clip.add(svg.rect(insert=(-0.5, -0.5), size=(w, h), fill="none"))
    svg.defs.add(panel_clip)
    panel = svg.g(clip_path="url(#panel)")
    drawing.gs[0].add(panel)
    for i, (chunks, color) in enumerate(zip(plans, colors, strict=True)):
        states = _slots(chunks)
        group = panel
        for slot in range(len(states[0][1])):
            timeline = []
            for time, row in states:
                href = "none" if row[slot] is None else f"url(#planeclip-{i}-{row[slot]})"
                if not timeline or href != timeline[-1][1]:
                    timeline.append((time, href))
            child = svg.g(clip_path=timeline[0][1])
            if len(timeline) > 1:
                timeline.append((frames, timeline[0][1]))
                child.add(
                    svg.animate(
                        attributeName="clip-path",
                        values=";".join(href for _, href in timeline),
                        keyTimes=";".join(f"{t / frames:.12g}" for t, _ in timeline),
                        dur=duration,
                        repeatCount="indefinite",
                        calcMode="discrete",
                    )
                )
            group.add(child)
            group = child
        group.add(svg.rect(insert=(-0.5, -0.5), size=(w, h), fill=color))
        for j in sorted(set().union(*(set(chunk.neighbors) for chunk in chunks))):
            timeline = []
            for chunk in chunks:
                opacity = "1" if j in chunk.neighbors else "0"
                if not timeline or opacity != timeline[-1][1]:
                    timeline.append((chunk.start, opacity))
            stroke = svg.use(href=f"#plane-{i}-{j}", fill="none", stroke="white", stroke_width=0.04, opacity=timeline[0][1])
            if len(timeline) > 1:
                timeline.append((frames, timeline[0][1]))
                stroke.add(
                    svg.animate(
                        attributeName="opacity",
                        values=";".join(value for _, value in timeline),
                        keyTimes=";".join(f"{t / frames:.12g}" for t, _ in timeline),
                        dur=duration,
                        repeatCount="indefinite",
                        calcMode="discrete",
                    )
                )
            group.add(stroke)
    return drawing
