"""Filename axis selection preserves the already-rendered panels."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import esch

NS = {"s": "http://www.w3.org/2000/svg"}


class SaveTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix=".esch-save-", dir=".")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def read(self, name):
        return ET.parse(self.directory / name).getroot()

    def test_user_example_exports_each_inner_array(self):
        values = np.arange(1, 25).reshape(3, 2, 4)
        drawing = esch.draw(values, "n h w")
        drawing.save(self.directory / "test_{n}.svg")
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()), ["test_0.svg", "test_1.svg", "test_2.svg"])
        for index in range(3):
            actual = self.read(f"test_{index}.svg")
            expected = ET.fromstring(esch.draw(values[index] / 24, normalize=False).tostring())
            self.assertEqual(ET.tostring(actual), ET.tostring(expected))

    def test_multiple_fields_and_partial_selection(self):
        drawing = esch.draw(np.ones((2, 3, 2, 4)), "row col h w", gap=4)
        drawing.save(self.directory / "r{row}_c{col:02d}.svg")
        self.assertEqual(len(list(self.directory.glob("r*.svg"))), 6)
        for row in range(2):
            for col in range(3):
                root = self.read(f"r{row}_c{col:02d}.svg")
                self.assertEqual(len(root.findall("s:g", NS)), 1)
                self.assertEqual(len(root.findall(".//s:rect", NS)), 8)
        drawing.save(self.directory / "column_{col}.svg")
        root = self.read("column_1.svg")
        self.assertEqual(len(root.findall("s:g", NS)), 2)
        self.assertEqual(root.get("viewBox"), "0,0,5.0,5.5")

    def test_animation_colors_layers_and_original_are_preserved(self):
        values = np.linspace(-10, 20, 48).reshape(2, 2, 4, 3)
        drawing = esch.draw(values, "n h w t", pos="red", neg="blue", color_by_magnitude=True, fps=7)
        esch.grid_fn(drawing, np.ones((2, 2, 4)) * 0.1, shape="circle", normalize=False)
        original = drawing.tostring()
        drawing.save(self.directory / "animated_{n}.svg")
        source = ET.fromstring(original).findall("s:g", NS)
        for index in range(2):
            group = self.read(f"animated_{index}.svg").find("s:g", NS)
            self.assertEqual([ET.tostring(node) for node in group], [ET.tostring(node) for node in source[index]])
            self.assertTrue(group.findall(".//s:animate", NS))
        self.assertEqual(drawing.tostring(), original)

    def test_arrow_output_axis_names_and_packed_shorthand(self):
        values = np.ones((6, 2, 4))
        for index, pattern in enumerate(("(n c) h w -> n c h w", "(n c) h w")):
            drawing = esch.draw(values, pattern, n=2)
            drawing.save(self.directory / f"case{index}_{{n}}.svg")
            self.assertEqual(len(self.read(f"case{index}_1.svg").findall("s:g", NS)), 3)

    def test_aliases_repeated_fields_and_literal_braces(self):
        drawing = esch.draw(np.ones((2, 2, 4)), "n h w")
        drawing.saveas(self.directory / "repeat_{n}_{n:02d}.svg")
        esch.save(drawing, self.directory / "alias_{n}.svg")
        drawing.save(self.directory / "literal_{{braces}}.svg")
        self.assertTrue((self.directory / "repeat_1_01.svg").exists())
        self.assertTrue((self.directory / "alias_1.svg").exists())
        self.assertTrue((self.directory / "literal_{braces}.svg").exists())

    def test_invalid_templates_fail_before_writing(self):
        drawing = esch.draw(np.ones((12, 2, 4)), "n h w")
        for filename in (
            "bad_{missing}.svg",
            "bad_{h}.svg",
            "bad_{}.svg",
            "bad_{n!r}.svg",
            "bad_{n.real}.svg",
            "bad_{n:{n}}.svg",
            "bad_{n:.0e}.svg",
            "bad_{n.svg",
        ):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                drawing.save(self.directory / filename)
            self.assertEqual(list(self.directory.iterdir()), [])
        with self.assertRaisesRegex(ValueError, "panel axis"):
            esch.Drawing(2, 4).save(self.directory / "manual_{n}.svg")

    def test_positional_pattern_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            esch.draw(np.ones((2, 3)), "h w", pattern="h w")
