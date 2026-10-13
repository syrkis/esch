"""Individual-mask filter graph and native motion regression coverage."""

import unittest
import xml.etree.ElementTree as ET

import numpy as np

from .cell_fields import cell_example
from .visual import inline_svg

NS = {"s": "http://www.w3.org/2000/svg"}


class CellFieldTests(unittest.TestCase):
    def test_competitors_exclude_the_current_site(self):
        positions = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        root = ET.fromstring(cell_example(positions, [1, 1, -1, -1], h=2, w=2).tostring())
        filter_ = root.find(".//s:filter", NS)
        dependencies = {}
        for node in filter_:
            if node.tag.endswith("feImage"):
                dependencies[node.get("result")] = {int(node.get("result").split("-")[-1])}
            elif node.tag.endswith("feBlend"):
                dependencies[node.get("result")] = dependencies[node.get("in")] | dependencies[node.get("in2")]
            elif node.get("result", "").startswith("other-blue-"):
                site = int(node.get("result").split("-")[-1])
                dependencies[node.get("result")] = dependencies[node.get("in")]
                self.assertEqual(dependencies[node.get("in")], set(range(4)) - {site})
        self.assertEqual(len(root.findall(".//s:feMorphology", NS)), 4)
        self.assertFalse(root.findall(".//s:animate", NS))

    def test_alignment_animation_and_internal_references(self):
        positions = np.array([[[0, 0], [0.2, 0.7]], [[1, 1], [0.8, 0.4]]])
        svg = cell_example(positions, [1, -1], h=2, w=2, fps=2).tostring()
        root = ET.fromstring(svg)
        gradients = root.findall(".//s:radialGradient", NS)
        images = root.findall(".//s:feImage", NS)
        self.assertEqual(len(gradients), 2)
        self.assertFalse(root.findall(".//s:script", NS))
        for image, gradient, trajectory in zip(images, gradients, positions, strict=True):
            self.assertTrue(image.get("{http://www.w3.org/1999/xlink}href").startswith("#field-"))
            for attribute, axis, offset in (("cx", 1, "x"), ("cy", 0, "y")):
                animation = gradient.find(f"s:animate[@attributeName='{attribute}']", NS)
                coordinates = np.array(list(map(float, animation.get("values").split(";")))) + float(image.get(offset))
                np.testing.assert_allclose(coordinates, np.r_[trajectory[:, axis], trajectory[0, axis]])
                self.assertEqual(animation.get("calcMode"), "linear")
                self.assertEqual(animation.get("dur"), "1s")
        scoped = ET.fromstring(inline_svg(svg, "example"))
        identifiers = {node.get("id") for node in scoped.iter() if node.get("id")}
        self.assertTrue(all(identifier.startswith("example-") for identifier in identifiers))
        for node in scoped.iter():
            for key, value in node.attrib.items():
                if key.endswith("href"):
                    self.assertIn(value[1:], identifiers)
                elif value.startswith("url(#"):
                    self.assertIn(value[5:-1], identifiers)

    def test_invalid_inputs(self):
        positions = np.array([[[0, 0]], [[1, 1]]])
        for teams in ([1], [0, -1], [np.nan, 1]):
            with self.assertRaises(ValueError):
                cell_example(positions, teams, h=2, w=2)
        for fps in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                cell_example(positions, [1, -1], h=2, w=2, fps=fps)
        for points in (positions[:1], positions + 4, positions + np.nan):
            with self.assertRaises(ValueError):
                cell_example(points, [1, -1], h=2, w=2)
