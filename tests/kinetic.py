"""Floating-point kinetic Delaunay certificates for linear site trajectories.

An event heap schedules incircle and hull-orientation roots. GEOS rebuilds
the triangulation after an event; unchanged certificates retain their cached
roots. Tiny intervals around numerical roots use conservative enclosures.
This is not an exact-arithmetic kinetic triangulation implementation.
"""

import heapq
from itertools import combinations

import numpy as np
from numpy.polynomial import polynomial as poly
from shapely import MultiPoint, delaunay_triangles


def _cross(a, b):
    return poly.polysub(poly.polymul(a[0], b[1]), poly.polymul(a[1], b[0]))


def certificate(start, velocity, ids):
    """Orientation (three sites) or incircle (four sites) polynomial."""
    origin = ids[-1]
    rows = [np.stack([start[i] - start[origin], velocity[i] - velocity[origin]], axis=1) for i in ids[:-1]]
    if len(rows) == 2:
        return _cross(*rows)
    norms = [poly.polyadd(poly.polymul(row[0], row[0]), poly.polymul(row[1], row[1])) for row in rows]
    return poly.polyadd(
        poly.polysub(poly.polymul(norms[0], _cross(rows[1], rows[2])), poly.polymul(norms[1], _cross(rows[0], rows[2]))),
        poly.polymul(norms[2], _cross(rows[0], rows[1])),
    )


def topology(sites):
    ids = {tuple(point): i for i, point in enumerate(sites)}
    edges = {}
    for triangle in delaunay_triangles(MultiPoint(sites)).geoms:
        vertices = tuple(ids[tuple(point)] for point in list(triangle.exterior.coords)[:3])
        for a, b in combinations(vertices, 2):
            edges.setdefault(tuple(sorted((a, b))), set()).update(vertices)
    neighbors = [set() for _ in sites]
    keys = set()
    for (a, b), vertices in edges.items():
        neighbors[a].add(b)
        neighbors[b].add(a)
        if len(vertices) == 4:
            keys.add(tuple(sorted(vertices)))
        else:
            # Any site crossing a hull edge can change the triangulation.
            keys.update(tuple(sorted((a, b, i))) for i in range(len(sites)) if i not in (a, b))
    if not edges:
        # Singleton, pair, or entirely collinear sites. All pairs are safe.
        neighbors = [set(range(len(sites))) - {i} for i in range(len(sites))]
        keys.update(combinations(range(len(sites)), 3))
    return tuple(tuple(sorted(row)) for row in neighbors), keys


def kinetic_intervals(start, end, bounds, enclosure):
    """Yield (start, end, neighbour sets) in normalized segment time."""
    velocity = end - start
    if not np.any(velocity):
        yield 0.0, 1.0, topology(start)[0]
        return
    cache, queue, active = {}, [], set()
    guard = 1e-7
    now = 0.0

    def schedule(keys, after):
        nonlocal active
        for key in keys - active:
            if key not in cache:
                coefficients = certificate(start, velocity, key)
                scale = np.max(np.abs(coefficients))
                roots = poly.polyroots(coefficients / scale) if scale else []
                cache[key] = sorted(float(root.real) for root in roots if abs(root.imag) < 1e-8 and 0 < root.real < 1)
            for root in cache[key]:
                if root > after:
                    heapq.heappush(queue, (root, key))
        active = keys

    # Guard the initial instant too: input frames can be exactly cocircular.
    initial = min(guard, 1.0)
    yield 0.0, initial, enclosure(start, start + initial * velocity, bounds)
    now = initial
    while now < 1:
        neighbors, keys = topology(start + now * velocity)
        schedule(keys, now)
        while queue and (queue[0][1] not in active or queue[0][0] <= now):
            heapq.heappop(queue)
        event = queue[0][0] if queue else 1.0
        before, after = max(now, event - guard), min(1.0, event + guard)
        if before > now:
            yield now, before, neighbors
        if after > before:
            yield before, after, enclosure(start + before * velocity, start + after * velocity, bounds)
        now = after
