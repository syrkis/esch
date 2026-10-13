"""Lines joining points in order, static and moving."""

import unittest
import xml.etree.ElementTree as ET

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import esch

from .smil import sample

NS = {"s": "http://www.w3.org/2000/svg"}
PATH = [[[0, 0], [1, 2], [2, 3]]]


def lines(drawing):
    root = ET.fromstring(drawing.tostring())
    return root.findall(".//s:polyline", NS) + root.findall(".//s:polygon", NS)


def vertices(text):
    return np.array([[float(value) for value in point.split(",")] for point in text.split()])


class LineTests(unittest.TestCase):
    def test_static_line_joins_points_in_order(self):
        (node,) = lines(esch.line_fn(esch.Drawing(3, 4), PATH, col=[0, 116, 217]))
        self.assertEqual(node.tag.rsplit("}", 1)[-1], "polyline")
        # (row, column) positions become (x, y) = (column, row) points.
        self.assertEqual(node.get("points"), "0,0 2,1 3,2")
        self.assertEqual((node.get("fill"), node.get("stroke"), node.get("stroke-width")), ("none", "#0074d9", "0.04"))
        self.assertIsNone(node.find("s:animate", NS))
        (closed,) = lines(esch.line_fn(esch.Drawing(3, 4), PATH, closed=True))
        self.assertEqual(closed.tag.rsplit("}", 1)[-1], "polygon")

    def test_rejects_bad_input_before_drawing(self):
        drawing = esch.Drawing(2, 2)
        original = drawing.tostring()
        for positions in ([[0, 0]], [[[0, 0, 0]]], np.zeros((2, 2, 2))):
            with self.subTest(positions=np.shape(positions)), self.assertRaisesRegex(ValueError, "positions"):
                esch.line_fn(drawing, positions)
        with self.assertRaisesRegex(ValueError, "fps"):
            esch.line_fn(drawing, [[[0, 0], [1, 1]]], fps=0)
        self.assertEqual(drawing.tostring(), original)

    @given(
        arrays(
            np.float64,
            st.tuples(st.integers(2, 5), st.integers(1, 4)).map(lambda shape: (*shape, 2)),
            elements=st.floats(0, 4),
        )
    )
    def test_vertices_pass_through_every_frame_and_loop(self, positions):
        (node,) = lines(esch.line_fn(esch.Drawing(5, 5), positions[None], fps=2))
        animation = node.find("s:animate", NS)
        frames = positions.shape[1]

        def shown(t):
            return vertices(node.get("points")) if animation is None else sample(animation, t).reshape(-1, 2)

        for frame in range(frames):
            np.testing.assert_allclose(shown(frame / frames), positions[:, frame, ::-1], atol=1.5e-3)
        np.testing.assert_allclose(shown(1), shown(0), atol=1e-9)
