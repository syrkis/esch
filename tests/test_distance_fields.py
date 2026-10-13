"""Regression coverage for the repository-only SVG distance-field experiment."""

import unittest
import xml.etree.ElementTree as ET

import numpy as np

from .distance_fields import distance_example

NS = {"s": "http://www.w3.org/2000/svg"}


class DistanceFieldTests(unittest.TestCase):
    def test_animation_tracks_sites_and_closes_loop(self):
        positions = np.array([[[0, 0], [0.2, 0.7]], [[1, 1], [0.8, 0.4]]])
        root = ET.fromstring(distance_example(positions, [1, -1], h=2, w=2, fps=2).tostring())
        gradients = root.findall(".//s:radialGradient", NS)
        self.assertEqual(len(gradients), 2)
        self.assertFalse(root.findall(".//s:script", NS))
        for gradient, trajectory, color in zip(gradients, positions, ["red", "blue"], strict=True):
            self.assertEqual(gradient.get("gradientUnits"), "userSpaceOnUse")
            self.assertEqual(gradient.find("s:stop", NS).get("stop-color"), color)
            for attribute, axis in (("cx", 1), ("cy", 0)):
                animation = gradient.find(f"s:animate[@attributeName='{attribute}']", NS)
                values = np.array(list(map(float, animation.get("values").split(";"))))
                np.testing.assert_allclose(values, np.r_[trajectory[:, axis], trajectory[0, axis]])
                self.assertEqual(animation.get("dur"), "1s")
                self.assertEqual(animation.get("calcMode"), "linear")

    def test_filter_comparison_selects_nearest_team(self):
        positions = np.array([[[0, 0]], [[1, 1]]])
        root = ET.fromstring(distance_example(positions, [1, -1], h=2, w=2).tostring())
        self.assertFalse(root.findall(".//s:animate", NS))
        filter_ = root.find(".//s:filter", NS)
        self.assertEqual(filter_.get("color-interpolation-filters"), "sRGB")
        matrix = np.array(list(map(float, filter_.find("s:feColorMatrix", NS).get("values").split()))).reshape(4, 5)
        transfer = filter_.find("s:feComponentTransfer", NS)
        self.assertEqual(transfer.get("in"), "smooth")
        self.assertEqual(filter_.find("s:feGaussianBlur", NS).get("in"), "difference")
        tables = np.array(
            [list(map(float, transfer.find(f"s:feFunc{channel}", NS).get("tableValues").split())) for channel in "RGB"]
        )
        radius = float(root.find(".//s:radialGradient", NS).get("r"))
        for point, expected in (([0.1, 0.1], [229, 57, 53]), ([0.9, 0.9], [30, 136, 229])):
            strengths = 1 - np.linalg.norm(positions[:, 0] - point, axis=-1) / radius
            result = np.clip(matrix @ np.array([strengths[0], 0, strengths[1], 1, 1]), 0, 1)
            colors = tables[np.arange(3), (result[:3] >= 0.5).astype(int)]
            np.testing.assert_allclose(colors * 255, expected)

    def test_optional_boundaries_use_expanded_fields_and_final_clip(self):
        positions = np.array([[[0, 0]], [[1, 1]]])
        for boundaries in (False, True):
            root = ET.fromstring(distance_example(positions, [1, -1], h=2, w=2, boundaries=boundaries).tostring())
            self.assertEqual(len(root.findall(".//s:feMorphology", NS)), 2 if boundaries else 0)
            clip = root.find(".//s:clipPath/s:rect", NS)
            self.assertEqual(float(clip.get("width")), 2)
            self.assertGreater(float(root.find(".//s:filter", NS).get("width")), 2)
            for field in root.findall(".//s:g/s:rect", NS):
                self.assertEqual(field.get("style"), "mix-blend-mode:lighten")
                self.assertGreater(float(field.get("width")), 2)

    def test_invalid_inputs(self):
        positions = np.array([[[0, 0]], [[1, 1]]])
        for teams in ([1, 1], [1], [1, 0]):
            with self.assertRaises(ValueError):
                distance_example(positions, teams, h=2, w=2)
        for fps in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                distance_example(positions, [1, -1], h=2, w=2, fps=fps)
        with self.assertRaises(ValueError):
            distance_example(positions + np.nan, [1, -1], h=2, w=2)
        with self.assertRaises(ValueError):
            distance_example(positions + 4, [1, -1], h=2, w=2)
        with self.assertRaises(TypeError):
            distance_example(positions, [1, -1], h=2, w=2, boundaries="yes")
