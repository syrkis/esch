"""Independent geometry and serialization checks for adaptive SVG clipping."""

import unittest
import xml.etree.ElementTree as ET
from itertools import pairwise

import numpy as np
from shapely import MultiPoint, Polygon, box, voronoi_polygons

from .adaptive_clips import _clip, adaptive_example, interval_neighbors, plan_chunks
from .visual import inline_svg

NS = {"s": "http://www.w3.org/2000/svg"}


def value_at(node, attribute, phase):
    animation = node.find(f"s:animate[@attributeName='{attribute}']", NS)
    if animation is None:
        return node.get(attribute)
    values = animation.get("values").split(";")
    times = np.array(list(map(float, animation.get("keyTimes").split(";"))))
    index = min(np.searchsorted(times, phase, side="right") - 1, len(values) - 2)
    if animation.get("calcMode") == "discrete":
        return values[index]
    start, end = [np.array([list(map(float, pair.split(","))) for pair in values[k].split()]) for k in (index, index + 1)]
    fraction = (phase - times[index]) / (times[index + 1] - times[index])
    return (1 - fraction) * start + fraction * end


class AdaptiveClipTests(unittest.TestCase):
    def test_interval_enclosures_match_independent_voronoi(self):
        rng = np.random.default_rng(638)
        bounds = box(-0.5, -0.5, 6.5, 10.5)
        for _ in range(6):
            start = rng.uniform([0, 0], [6, 10], (16, 2))
            end = np.clip(start + rng.normal(0, 0.15, (16, 2)), [0, 0], [6, 10])
            neighbors = interval_neighbors(start, end, bounds.bounds)
            for fraction in (0, 0.037, 0.293, 0.618, 0.981, 1):
                sites = (1 - fraction) * start + fraction * end
                expected = voronoi_polygons(MultiPoint(sites), extend_to=bounds, ordered=True)
                for i, reference in enumerate(expected.geoms):
                    cell = np.array(bounds.exterior.coords)[:-1]
                    for j in neighbors[i]:
                        cell = _clip(cell, 2 * (sites[j] - sites[i]), sites[j] @ sites[j] - sites[i] @ sites[i])
                    self.assertLess(Polygon(cell).symmetric_difference(reference.intersection(bounds)).area, 1e-8)

    def test_plans_cover_time_and_merge_quiet_cells(self):
        points = np.moveaxis(np.indices((3, 3)), 0, -1).reshape(9, 1, 2).astype(float)
        plans = plan_chunks(np.repeat(points, 4, axis=1), (-0.5, -0.5, 2.5, 2.5))
        self.assertTrue(all(len(chunks) == 1 for chunks in plans))
        self.assertTrue(all(chunks[0].start == 0 and chunks[0].end == 4 for chunks in plans))
        moving = np.random.default_rng(51).uniform(0, 2, (9, 3, 2))
        plans = plan_chunks(moving, (-0.5, -0.5, 2.5, 2.5))
        lengths = set()
        for i, chunks in enumerate(plans):
            self.assertEqual(chunks[0].start, 0)
            self.assertEqual(chunks[-1].end, 3)
            for previous, following in pairwise(chunks):
                self.assertEqual(previous.end, following.start)
            for chunk in chunks:
                self.assertNotIn(i, chunk.neighbors)
                lengths.add(chunk.end - chunk.start)
        self.assertGreater(len(lengths), 1)

    def test_exported_clip_switches_preserve_cells_between_keyframes(self):
        positions = np.random.default_rng(73).uniform(0, 2, (9, 3, 2))
        drawing = adaptive_example(positions, ["red"] * 4 + ["blue"] * 5, h=3, w=3, fps=2)
        svg = drawing.tostring()
        root = ET.fromstring(svg)
        ids = {node.get("id"): node for node in root.iter() if node.get("id")}
        parents = {child: parent for parent in root.iter() for child in parent}
        cells = [node for node in root.findall(".//s:rect", NS) if node.get("fill") in ("red", "blue")]
        self.assertEqual(len(cells), 9)
        self.assertTrue(root.findall('.//s:animate[@attributeName="clip-path"]', NS))
        self.assertFalse(root.findall('.//s:animate[@attributeName="xlink:href"]', NS))
        self.assertFalse(root.findall(".//s:script", NS))
        self.assertFalse(root.findall(".//s:filter", NS))
        bounds = box(-0.5, -0.5, 2.5, 2.5)
        for time in (0, 0.137, 0.5, 0.999, 1, 1.875, 2.999):
            frame, fraction = int(time), time % 1
            sites = ((1 - fraction) * positions[:, frame] + fraction * positions[:, (frame + 1) % 3])[:, ::-1]
            expected = voronoi_polygons(MultiPoint(sites), extend_to=bounds, ordered=True)
            for node, reference in zip(cells, expected.geoms, strict=True):
                region = bounds
                while node in parents:
                    node = parents[node]
                    clip = value_at(node, "clip-path", time / 3)
                    if clip in (None, "none", "url(#panel)"):
                        continue
                    use = ids[clip[5:-1]].find("s:use", NS)
                    polygon = ids[use.get("{http://www.w3.org/1999/xlink}href")[1:]]
                    region = region.intersection(Polygon(value_at(polygon, "points", time / 3)))
                self.assertLess(region.symmetric_difference(reference.intersection(bounds)).area, 2e-6)
        scoped = inline_svg(svg, "adaptive")
        self.assertNotIn("url(#planeclip-", scoped)
        self.assertIn("url(#adaptive-planeclip-", scoped)

    def test_single_site_and_invalid_inputs(self):
        root = ET.fromstring(adaptive_example([[[0, 0]]], ["red"], h=1, w=1).tostring())
        self.assertFalse(root.findall(".//s:animate", NS))
        with self.assertRaisesRegex(ValueError, "collide"):
            adaptive_example([[[0, 0], [1, 1]], [[1, 1], [0, 0]]], ["red", "blue"], h=2, w=2)
        for positions in ([], [[[np.nan, 0]]], [[[5, 5]]]):
            with self.assertRaises(ValueError):
                adaptive_example(positions, ["red"], h=1, w=1)
        for fps in (0, np.nan, -1):
            with self.assertRaises(ValueError):
                adaptive_example([[[0, 0]]], ["red"], h=1, w=1, fps=fps)
