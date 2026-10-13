"""Check the clipping prototype against independently computed Voronoi cells."""

import unittest
import xml.etree.ElementTree as ET

import numpy as np
from shapely import MultiPoint, Polygon, box, voronoi_polygons

from .halfplanes import clipping_example, halfplanes


class HalfplaneTests(unittest.TestCase):
    def test_interpolated_clips_match_voronoi_including_loop(self):
        points = np.random.default_rng(94).uniform(0, 4, (9, 4, 2))
        bounds = box(-0.5, -0.5, 4.5, 4.5)
        clips = list(halfplanes(points, bounds.bounds))
        edge_counts = set()
        for time in np.arange(0, 4, 0.137):
            frame, fraction = int(time), time % 1
            sites = (1 - fraction) * points[:, frame] + fraction * points[:, (frame + 1) % 4]
            expected = voronoi_polygons(MultiPoint(sites), extend_to=bounds, ordered=True)
            actual = [bounds for _ in points]
            for i, _, poses in clips:
                # Exercise the actual serialized coordinate precision too.
                poses = np.array([float(f"{v:.12g}") for v in poses.flat]).reshape(poses.shape)
                triangle = (1 - fraction) * poses[frame] + fraction * poses[(frame + 1) % 4]
                actual[i] = actual[i].intersection(Polygon(triangle))
            edge_counts.add(tuple(len(cell.exterior.coords) for cell in actual))
            for cell, reference in zip(actual, expected.geoms, strict=True):
                self.assertLess(cell.symmetric_difference(reference.intersection(bounds)).area, 2e-4)
        self.assertGreater(len(edge_counts), 1)

    def test_collisions_between_keyframes_are_rejected(self):
        positions = np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]])
        with self.assertRaisesRegex(ValueError, "collide"):
            clipping_example(positions, ["red", "blue"], h=2, w=2)

    def test_native_animation_and_shared_geometry(self):
        positions = np.array([[[0, 0], [0, 1]], [[1, 1], [1, 0]]])
        drawing = clipping_example(positions, ["red", "blue"], h=2, w=2, fps=2)
        root = ET.fromstring(drawing.tostring())
        ns = {"s": "http://www.w3.org/2000/svg"}
        self.assertFalse(root.findall(".//s:script", ns))
        animations = root.findall(".//s:animate", ns)
        self.assertEqual(len(animations), 2)
        for animation in animations:
            self.assertEqual(animation.get("calcMode"), "linear")
            self.assertEqual(animation.get("dur"), "1s")
            values = animation.get("values").split(";")
            self.assertEqual(len(values), 3)
            self.assertEqual(values[0], values[-1])
