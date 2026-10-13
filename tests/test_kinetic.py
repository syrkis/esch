"""Kinetic schedules checked against independently constructed Voronoi cells."""

import unittest

import numpy as np
from numpy.polynomial.polynomial import polyval
from shapely import Polygon, box

from .adaptive_clips import _clip, interval_neighbors
from .kinetic import certificate, kinetic_intervals


class KineticTests(unittest.TestCase):
    def test_certificate_polynomials_match_determinants(self):
        rng = np.random.default_rng(431)
        start, velocity = rng.normal(size=(2, 4, 2))
        for size in (3, 4):
            coefficients = certificate(start, velocity, tuple(range(size)))
            for time in np.linspace(0, 1, 17):
                sites = start[:size] + time * velocity[:size]
                relative = sites[:-1] - sites[-1]
                matrix = relative if size == 3 else np.column_stack([relative, np.sum(relative**2, axis=1)])
                self.assertAlmostEqual(polyval(time, coefficients), np.linalg.det(matrix), places=10)

    def test_events_cover_flips_hull_changes_and_frame_degeneracies(self):
        rng = np.random.default_rng(922)
        scenes = [(rng.uniform(0, 3, (12, 2)), rng.uniform(0, 3, (12, 2))) for _ in range(4)]
        square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        scenes.append((square, square + [[0, 0], [0, 0], [-0.3, 0.2], [0, 0]]))
        line = np.array([[0, 1], [1, 1], [2, 1], [3, 1]], dtype=float)
        scenes.append((line, line + [[0, 0], [0, -0.5], [0, 0.5], [0, 0]]))
        bounds = box(-0.5, -0.5, 3.5, 3.5)
        total = 0
        for start, end in scenes:
            intervals = list(kinetic_intervals(start, end, bounds.bounds, interval_neighbors))
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], 1)
            previous = 0
            for a, b, neighbors in intervals:
                self.assertEqual(a, previous)
                self.assertGreater(b, a)
                previous = b
                # Both sides of every event, plus the middle of each interval.
                for time in (a, (a + b) / 2, b):
                    sites = start + time * (end - start)
                    for i in range(len(sites)):
                        # All-pairs construction is stable even at collinear
                        # instants where GEOS Voronoi may return invalid cells.
                        reference = np.array(bounds.exterior.coords)[:-1]
                        for j in range(len(sites)):
                            if i != j:
                                reference = _clip(
                                    reference, 2 * (sites[j] - sites[i]), sites[j] @ sites[j] - sites[i] @ sites[i]
                                )
                        cell = np.array(bounds.exterior.coords)[:-1]
                        for j in neighbors[i]:
                            cell = _clip(cell, 2 * (sites[j] - sites[i]), sites[j] @ sites[j] - sites[i] @ sites[i])
                        self.assertLess(Polygon(cell).symmetric_difference(Polygon(reference)).area, 1e-7, (a, b, time, i))
                total += 1
        self.assertGreater(total, 100)
