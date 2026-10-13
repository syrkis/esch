"""Compact reusable clip-slot experiment with byte-cost chunk selection."""

import xml.etree.ElementTree as ET
from functools import lru_cache
from itertools import pairwise

import numpy as np
from shapely import MultiPoint, Polygon, box, voronoi_polygons

import esch

from .adaptive_clips import Chunk, _separations, _slots, plan_chunks


def number(value, digits):
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def pose_text(pose, digits):
    return " ".join(f"{number(x, digits)},{number(y, digits)}" for x, y in pose)


def simplify_timeline(timeline, tolerance):
    """RDP simplification between jumps; retain both sides of every switch."""
    times = np.array([time for time, _ in timeline])
    poses = np.array([np.fromstring(value.replace(",", " "), sep=" ").reshape(3, 2) for _, value in timeline])
    jumps = np.flatnonzero(np.diff(times) == 0)
    starts, ends = [0, *(jumps + 1)], [*jumps, len(times) - 1]
    retained = set(starts) | set(ends)
    pending = list(zip(starts, ends, strict=True))
    while pending:
        a, b = pending.pop()
        if b <= a + 1:
            continue
        fractions = (times[a + 1 : b] - times[a]) / (times[b] - times[a])
        interpolated = poses[a] + fractions[:, None, None] * (poses[b] - poses[a])
        errors = np.linalg.norm(poses[a + 1 : b] - interpolated, axis=-1).max(axis=-1)
        offset = int(errors.argmax())
        if errors[offset] > tolerance:
            middle = a + 1 + offset
            retained.add(middle)
            pending.extend([(a, middle), (middle, b)])
    return [timeline[index] for index in sorted(retained)]


def sites_at(points, times):
    times = np.asarray(times)
    frame = times.astype(int) % points.shape[1]
    fraction = times % 1
    return points[:, frame] * (1 - fraction)[None, :, None] + points[:, (frame + 1) % points.shape[1]] * fraction[None, :, None]


def plane_poses(points, i, j, times, diagonal):
    sites = sites_at(points[[i, j]], times)
    difference = sites[1] - sites[0]
    velocity = np.diff(difference, axis=0)
    denominator = np.sum(velocity**2, axis=1)
    fraction = np.clip(-np.sum(difference[:-1] * velocity, axis=1) / np.where(denominator, denominator, 1), 0, 1)
    separation = np.linalg.norm(difference[:-1] + fraction[:, None] * velocity, axis=1).min()
    scale = 2 * diagonal / separation
    midpoint = sites.mean(axis=0)
    tangent = scale * np.stack([-difference[:, 1], difference[:, 0]], axis=-1)
    return np.stack([midpoint + tangent, midpoint - tangent, midpoint - scale * difference], axis=1)


@lru_cache(maxsize=128)
def _costs(shape, data, owner, chunks, diagonal, digits):
    """Reuse geometry/byte measurements across runtime-weight candidates."""
    points = np.frombuffer(data, dtype=float).reshape(shape)
    frames = points.shape[1]
    boundaries = np.linspace(0, frames, 4 * frames + 1)
    rows = []
    for a, b in pairwise(boundaries):
        rows.append(set().union(*(set(chunk.neighbors) for chunk in chunks if chunk.start < b and chunk.end > a)))
    result = []
    for end in range(1, len(boundaries)):
        neighbors = set()
        for start in range(end - 1, max(-1, end - 17), -1):
            neighbors.update(rows[start])
            a, b = boundaries[start], boundaries[end]
            times = sorted({a, b} | set(range(int(np.ceil(a)), int(np.floor(b)) + 1)))
            # This is the actual coordinate/time payload before slots are
            # stitched together, plus a fixed transition overhead per clip.
            size = 0
            for j in sorted(neighbors):
                poses = plane_poses(points, owner, j, times, diagonal)
                size += (
                    sum(
                        len(pose_text(pose, digits)) + len(number(time / frames, 10)) + 2
                        for time, pose in zip(times, poses, strict=True)
                    )
                    + 8
                )
            result.append((start, end, size, Chunk(a, b, tuple(sorted(neighbors)))))
    return tuple(result)


def partition(points, owner, chunks, diagonal, digits, clip_penalty):
    """Shortest path over quarter-frame boundaries, scored by encoded bytes.

    Candidates span up to four input frames. The runtime penalty accounts for
    active clips integrated over time; it is a proxy, not a browser FPS prediction.
    """
    count = 4 * points.shape[1]
    costs = [0.0] + [np.inf] * count
    choices = {}
    for start, end, size, chunk in _costs(points.shape, points.tobytes(), owner, tuple(chunks), diagonal, digits):
        cost = costs[start] + size + clip_penalty * (chunk.end - chunk.start) * len(chunk.neighbors)
        if cost < costs[end]:
            costs[end] = cost
            choices[end] = (start, chunk)
    result = []
    end = count
    while end:
        end, chunk = choices[end]
        result.append(chunk)
    return list(reversed(result))


def compact_example(positions, colors, *, h, w, fps=1, digits=3, clip_penalty=32, plans=None):
    """Return a Drawing and size/work report; pure vector SMIL, no scripts.

    `plans` is an internal benchmark hook for an already computed kinetic plan.
    Coordinate rounding must be checked against the desired geometry tolerance.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 2 or not all(positions.shape):
        raise ValueError("Expected nonempty (site, time, 2) positions")
    if not np.isfinite([h, w, fps, clip_penalty]).all() or min(h, w, fps) <= 0 or clip_penalty < 0:
        raise ValueError("Dimensions and fps must be positive and finite; clip_penalty must be nonnegative")
    if not isinstance(digits, int) or not 1 <= digits <= 12:
        raise ValueError("digits must be an integer from 1 to 12")
    if (
        len(colors) != len(positions)
        or not np.isfinite(positions).all()
        or np.any(positions < -0.5)
        or np.any(positions > [h - 0.5, w - 0.5])
    ):
        raise ValueError("Provide one color per site and finite positions inside the panel")
    points = positions[..., ::-1]
    diagonal = np.hypot(h, w)
    if any(np.min(values) < 1e-10 * max(1, diagonal) for values in _separations(points).values()):
        raise ValueError("Sites must never collide, including between frames")
    if plans is None:
        plans = plan_chunks(points, (-0.5, -0.5, w - 0.5, h - 0.5))
    drawing = esch.Drawing(h, w)
    svg = drawing.dwg
    frames = points.shape[1]
    duration = f"{frames / fps:g}s"
    clip = svg.clipPath(id="compact-panel")
    clip.add(svg.rect(insert=(-0.5, -0.5), size=(w, h), fill="none"))
    svg.defs.add(clip)
    panel = svg.g(clip_path="url(#compact-panel)")
    drawing.gs[0].add(panel)
    empty = np.array([[-2 * diagonal, -2 * diagonal], [6 * diagonal, -2 * diagonal], [-2 * diagonal, 6 * diagonal]])
    clip_counts, chunk_counts, keyframes, original_keyframes = [], [], 0, 0
    for owner, original in enumerate(plans):
        chunks = partition(points, owner, original, diagonal, digits, clip_penalty)
        states = _slots(chunks)
        slots = len(states[0][1])
        clip_counts.append(slots)
        chunk_counts.append(len(chunks))
        group = panel
        references = []
        for slot in range(slots):
            timeline = []
            for chunk, (_, row) in zip(chunks, states, strict=True):
                times = sorted({chunk.start, chunk.end} | set(range(int(np.ceil(chunk.start)), int(np.floor(chunk.end)) + 1)))
                poses = (
                    np.repeat(empty[None], len(times), axis=0)
                    if row[slot] is None
                    else plane_poses(points, owner, row[slot], times, diagonal)
                )
                for time, pose in zip(times, poses, strict=True):
                    value = pose_text(pose, digits)
                    if timeline and time == timeline[-1][0] and value == timeline[-1][1]:
                        continue
                    timeline.append((time, value))
            original_keyframes += len(timeline)
            timeline = simplify_timeline(timeline, 10.0**-digits / 2)
            identifier = f"slot-{owner}-{slot}"
            polygon = svg.polygon(
                points=[tuple(pair.split(",")) for pair in timeline[0][1].split()], id=identifier, fill="none"
            )
            if frames > 1:
                polygon.add(
                    svg.animate(
                        attributeName="points",
                        values=";".join(value for _, value in timeline),
                        keyTimes=";".join(number(time / frames, 10) for time, _ in timeline),
                        dur=duration,
                        repeatCount="indefinite",
                        calcMode="linear",
                    )
                )
            svg.defs.add(polygon)
            clip = svg.clipPath(id=f"clip-{identifier}", clipPathUnits="userSpaceOnUse")
            clip.add(svg.use(href=f"#{identifier}"))
            svg.defs.add(clip)
            child = svg.g(clip_path=f"url(#clip-{identifier})")
            group.add(child)
            group = child
            references.append(identifier)
            keyframes += len(timeline)
        group.add(svg.rect(insert=(-0.5, -0.5), size=(w, h), fill=colors[owner], id=f"cell-{owner}"))
        for identifier in references:
            group.add(svg.use(href=f"#{identifier}", fill="none", stroke="white", stroke_width=0.04))
    return drawing, {
        "bytes": len(drawing.tostring().encode()),
        "digits": digits,
        "clip_penalty": clip_penalty,
        "slots": sum(clip_counts),
        "max_slots_per_cell": max(clip_counts),
        "chunks": sum(chunk_counts),
        "keyframes": keyframes,
        "original_keyframes": original_keyframes,
    }


def measure_error(drawing, positions, h, w):
    """Check serialized triangles on a fixed and offset temporal validation grid.

    This measures sampled Hausdorff error in grid units. It is deliberately
    separate from the optimizer's cost calculations, but is not an all-times
    proof of numerical stability or a substitute for browser visual review.
    """
    points = np.asarray(positions, dtype=float)[..., ::-1]
    frames = points.shape[1]
    root = ET.fromstring(drawing.tostring())
    ns = {"s": "http://www.w3.org/2000/svg"}
    tracks = [[] for _ in points]
    for polygon in root.findall(".//s:polygon", ns):
        owner = int(polygon.get("id").split("-")[1])
        animation = polygon.find("s:animate", ns)
        if animation is None:
            times, values = [0, 1], [polygon.get("points")] * 2
        else:
            times = list(map(float, animation.get("keyTimes").split(";")))
            values = animation.get("values").split(";")
        poses = np.array([np.fromstring(value.replace(",", " "), sep=" ").reshape(3, 2) for value in values])
        tracks[owner].append((np.asarray(times) * frames, poses))
    times = sorted(
        set(np.arange(0, frames, 0.125))
        | {frame + offset for frame in range(frames) for offset in (0.173, 0.419, 0.731)}
        | {frames - 1e-8}
    )
    bounds = box(-0.5, -0.5, w - 0.5, h - 0.5)
    maximum, count = 0.0, 0
    for time in times:
        sites = sites_at(points, [time])[:, 0]
        reference = [bounds] if len(sites) == 1 else voronoi_polygons(MultiPoint(sites), extend_to=bounds, ordered=True).geoms
        for owner, expected in enumerate(reference):
            actual = bounds
            for keys, poses in tracks[owner]:
                index = min(np.searchsorted(keys, time, side="right") - 1, len(keys) - 2)
                fraction = (time - keys[index]) / (keys[index + 1] - keys[index])
                pose = poses[index] + fraction * (poses[index + 1] - poses[index])
                actual = actual.intersection(Polygon(pose))
            error = actual.hausdorff_distance(expected.intersection(bounds))
            if not np.isfinite(error):
                return np.inf, count
            maximum = max(maximum, error)
            count += 1
    return maximum, count


def optimized_example(positions, colors, *, h, w, fps=1, tolerance=0.005, runtime_weight=1024):
    """Choose precision and chunk costs for this input; return Drawing/report.

    `tolerance` is a sampled geometric error budget, in grid units. Selection
    minimizes actual SVG bytes plus `runtime_weight * clip_slots` among four
    cost weights. Runtime weight is a workload proxy, not a promised FPS.
    """
    if not np.isfinite([tolerance, runtime_weight]).all() or tolerance <= 0 or runtime_weight < 0:
        raise ValueError("tolerance must be positive and finite; runtime_weight must be finite and nonnegative")
    candidates = []
    for penalty in (0, 32, 128, 512):
        for digits in range(max(1, int(np.ceil(-np.log10(tolerance)))), 13):
            drawing, report = compact_example(positions, colors, h=h, w=w, fps=fps, digits=digits, clip_penalty=penalty)
            error, checks = measure_error(drawing, positions, h, w)
            if error <= tolerance:
                report.update(checked_max_error=error, checks=checks)
                candidates.append((drawing, report))
                break
        else:
            raise ValueError("Unable to meet the sampled geometry tolerance at 12 decimal places")
    drawing, selected = min(candidates, key=lambda candidate: candidate[1]["bytes"] + runtime_weight * candidate[1]["slots"])
    report = dict(selected)
    report.update(tolerance=tolerance, runtime_weight=runtime_weight, candidates=[entry for _, entry in candidates])
    return drawing, report
