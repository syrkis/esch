"""Geometry, discontinuous slot changes, and optimization regression checks."""

import unittest
import xml.etree.ElementTree as ET
from itertools import pairwise

import numpy as np
from shapely import MultiPoint, Polygon, box, voronoi_polygons

from .adaptive_clips import plan_chunks
from .compact_clips import compact_example, optimized_example, simplify_timeline
from .test_adaptive_clips import NS, value_at


class CompactClipTests(unittest.TestCase):
    def test_simplifier_preserves_discontinuous_switches(self):
        timeline = [(0, "0,0 1,0 0,1"), (1, "1,0 2,0 1,1"), (2, "2,0 3,0 2,1"), (2, "8,0 9,0 8,1"), (3, "9,0 10,0 9,1")]
        self.assertEqual(simplify_timeline(timeline, 1e-9), [timeline[index] for index in (0, 2, 3, 4)])

    def test_serialized_cells_at_switches_and_between_input_frames(self):
        positions = np.random.default_rng(612).uniform(0, 2, (9, 5, 2))
        drawing, report = compact_example(positions, ["red"] * 4 + ["blue"] * 5, h=3, w=3, clip_penalty=1024)
        root = ET.fromstring(drawing.tostring())
        ids = {node.get("id"): node for node in root.iter() if node.get("id")}
        parents = {child: parent for parent in root.iter() for child in parent}
        times = set(np.random.default_rng(122).uniform(0, 5, 40)) | {0, 1, 4.999999}
        discontinuities = 0
        for animation in root.findall(".//s:animate", NS):
            keys = list(map(float, animation.get("keyTimes").split(";")))
            for a, b in pairwise(keys):
                if a == b:
                    discontinuities += 1
                    times.update((a * 5 - 1e-7, a * 5, a * 5 + 1e-7))
        self.assertGreater(discontinuities, 0)
        panel = box(-0.5, -0.5, 2.5, 2.5)
        for time in sorted(t for t in times if 0 <= t < 5):
            sites = ((1 - time % 1) * positions[:, int(time)] + (time % 1) * positions[:, (int(time) + 1) % 5])[:, ::-1]
            reference = voronoi_polygons(MultiPoint(sites), extend_to=panel, ordered=True)
            for owner, expected in enumerate(reference.geoms):
                actual = panel
                node = ids[f"cell-{owner}"]
                while node in parents:
                    node = parents[node]
                    clip = node.get("clip-path", "")
                    if not clip.startswith("url(#clip-slot"):
                        continue
                    polygon = ids[clip[10:-1]]
                    actual = actual.intersection(Polygon(value_at(polygon, "points", time / 5)))
                self.assertLess(actual.hausdorff_distance(expected.intersection(panel)), 0.005, (time, owner))
        self.assertEqual(report["bytes"], len(drawing.tostring().encode()))
        self.assertFalse(root.findall(".//s:script", NS))
        self.assertFalse(root.findall(".//s:filter", NS))
        self.assertEqual({a.get("attributeName") for a in root.findall(".//s:animate", NS)}, {"points"})

    def test_cached_plans_cannot_be_mutated_by_callers(self):
        points = np.array([[[0, 0]], [[1, 0]]], dtype=float)
        first = plan_chunks(points, (-0.5, -0.5, 1.5, 0.5))
        first[0].clear()
        self.assertTrue(plan_chunks(points, (-0.5, -0.5, 1.5, 0.5))[0])

    def test_optimizer_measures_error_and_selects_actual_byte_cost(self):
        positions = np.array([[[0, 0], [0.1, 0.2]], [[1, 1], [0.8, 1]], [[0, 1], [0.1, 0.8]]])
        drawing, report = optimized_example(positions, ["red", "blue", "red"], h=2, w=2, tolerance=0.01, runtime_weight=0)
        self.assertEqual(report["bytes"], len(drawing.tostring().encode()))
        self.assertEqual(report["bytes"], min(candidate["bytes"] for candidate in report["candidates"]))
        self.assertTrue(all(candidate["checked_max_error"] <= 0.01 for candidate in report["candidates"]))
        self.assertGreater(report["checks"], 0)
        with self.assertRaises(ValueError):
            optimized_example(positions, ["red"] * 3, h=2, w=2, tolerance=0)

    def test_static_singleton_and_invalid_inputs(self):
        drawing, report = compact_example([[[0, 0]]], ["red"], h=1, w=1)
        self.assertEqual(report["slots"], 0)
        self.assertNotIn("<animate", drawing.tostring())
        for digits in (0, 13, 1.5):
            with self.assertRaises(ValueError):
                compact_example([[[0, 0]]], ["red"], h=1, w=1, digits=digits)
        with self.assertRaisesRegex(ValueError, "collide"):
            compact_example([[[0, 0], [1, 1]], [[1, 1], [0, 0]]], ["red", "blue"], h=2, w=2)
