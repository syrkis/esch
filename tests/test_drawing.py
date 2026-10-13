"""Behavioral regression tests; run with python -m unittest discover."""

import itertools
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np
from einops import rearrange
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import esch

from .smil import sample

NS = {"s": "http://www.w3.org/2000/svg"}


def xml(drawing):
    return ET.fromstring(drawing.tostring())


def nodes(drawing, kind):
    return xml(drawing).findall(f".//s:{kind}", NS)


class DrawingTests(unittest.TestCase):
    def test_default_gap_is_five(self):
        values = np.ones((2, 3, 4, 8))
        self.assertEqual(esch.draw(values).tostring(), esch.draw(values, gap=5).tostring())
        drawing = esch.Drawing(4, 8, row=2, col=3)
        self.assertAlmostEqual(drawing.sub_width - 8, 8 / 5)
        self.assertAlmostEqual(drawing.sub_height - 4, 4 / 5)
        legacy = esch.draw(np.ones((5, 2, 7)), columns=3, gap=None)
        self.assertEqual(xml(legacy).get("viewBox"), "0,0,24,6")

    def test_gap_divides_panel_dimensions(self):
        for gap in (2, 4, 8, 2.5):
            with self.subTest(gap=gap):
                drawing = esch.draw(np.ones((2, 3, 4, 8)), pattern="row col h w", gap=gap)
                self.assertAlmostEqual(drawing.sub_width - 8, 8 / gap)
                self.assertAlmostEqual(drawing.sub_height - 4, 4 / gap)
                self.assertAlmostEqual(drawing.total_width, 9 + 2 * (8 + 8 / gap))
                self.assertAlmostEqual(drawing.total_height, 5 + 4 + 4 / gap)
                self.assertEqual(nodes(drawing, "g")[1].get("transform"), f"translate({1 + 8 + 8 / gap},1.0)")

    def test_gap_single_panel_and_partial_row(self):
        self.assertEqual(esch.draw(np.ones((4, 8)), gap=4).total_width, 9)
        drawing = esch.draw(np.ones((5, 4, 8)), columns=3, gap=4)
        self.assertEqual(len(nodes(drawing, "g")), 5)
        self.assertEqual(xml(drawing).get("viewBox"), "0,0,29.0,10.0")
        self.assertEqual(nodes(drawing, "g")[4].get("transform"), "translate(11.0,6.0)")

    def test_gap_direct_drawing_and_validation(self):
        drawing = esch.Drawing(4, 8, row=2, col=3, gap=4)
        self.assertEqual((drawing.sub_width, drawing.sub_height), (10, 5))
        for gap in (0, -1, np.nan, np.inf, True, "4", [4]):
            with self.subTest(gap=gap), self.assertRaisesRegex(ValueError, "gap"):
                esch.draw(np.ones((2, 4, 8)), gap=gap)

    def test_gap_preserves_animation_and_inner_geometry(self):
        values = np.linspace(-1, 1, 48).reshape(2, 3, 4, 2)
        plain = esch.draw(values, time_axis=-1)
        spaced = esch.draw(values, time_axis=-1, gap=4)
        self.assertEqual([ET.tostring(n) for n in nodes(plain, "rect")], [ET.tostring(n) for n in nodes(spaced, "rect")])

    def test_rgb_colors_match_hex_across_renderers(self):
        positive = np.array([0, 116, 217])
        negative = np.array([255, 128, 0])
        for shape in ("square", "circle"):
            for magnitude in (False, True):
                for animated in (False, True):
                    for renderer in ("draw", "grid", "mesh"):
                        with self.subTest(shape=shape, magnitude=magnitude, animated=animated, renderer=renderer):
                            values = np.array([[-1, 0.25], [0.5, -1]]) if animated else np.array([-1, 0.5])
                            results = []
                            for pos, neg in ((positive, negative), ("#0074d9", "#ff8000")):
                                options = {"shape": shape, "pos": pos, "neg": neg, "color_by_magnitude": magnitude}
                                if renderer == "draw":
                                    drawing = esch.draw(values, pattern="w t" if animated else "w", **options)
                                else:
                                    drawing = esch.Drawing(1, 2)
                                    if renderer == "grid":
                                        esch.grid_fn(drawing, values[None, None], **options)
                                    else:
                                        esch.mesh_fn(drawing, [[[0, 0], [0, 1]]], values[None], **options)
                                results.append(drawing.tostring())
                            self.assertEqual(*results)
        np.testing.assert_array_equal(positive, [0, 116, 217])
        np.testing.assert_array_equal(negative, [255, 128, 0])

    def test_rgb_sequences_and_float_ranges(self):
        for color, expected in (
            ([0, 116, 217], "#0074d9"),
            ((0, 116, 217), "#0074d9"),
            (np.array([0, 116, 217], dtype=np.uint8), "#0074d9"),
            ([0.0, 116.0, 217.0], "#0074d9"),
            ([0.0, 0.5, 1.0], "#0080ff"),
            ([0, 1, 0], "#000100"),
        ):
            with self.subTest(color=color):
                self.assertEqual(nodes(esch.draw([1], pos=color), "rect")[0].get("fill"), expected)
                self.assertEqual(nodes(esch.draw([-1], neg=color), "rect")[0].get("stroke"), expected)
        self.assertEqual(nodes(esch.draw([1], col=[0, 116, 217]), "rect")[0].get("fill"), "#0074d9")

    def test_invalid_rgb_colors(self):
        for color in (
            [0, 1],
            [0, 1, 2, 3],
            [[0, 1, 2]],
            [-1, 0, 0],
            [256, 0, 0],
            [np.nan, 0, 0],
            [np.inf, 0, 0],
            [1j, 0, 0],
            ["red", "green", "blue"],
        ):
            for argument in ("pos", "neg"):
                with self.subTest(color=color, argument=argument), self.assertRaises(ValueError):
                    esch.draw([-1, 1], **{argument: color})

    def test_normalization_preserves_relative_areas_and_signs(self):
        values = np.array([-100.0, -25.0, 0, 25.0, 100.0])
        original = values.copy()
        for shape, kind, dimension in (("square", "rect", "width"), ("circle", "circle", "r")):
            drawing = esch.draw(values, shape=shape)
            sizes = np.array([float(g.get(dimension)) for g in nodes(drawing, kind)])
            # SVG radii are rounded to five decimals; squaring ratios amplifies that error.
            np.testing.assert_allclose((sizes / sizes[-1]) ** 2, [1, 0.25, 0, 0.25, 1], atol=2e-5)
            self.assertEqual(drawing.tostring(), esch.draw(values / 100, shape=shape, normalize=False).tostring())
        np.testing.assert_array_equal(values, original)

    def test_normalization_is_global_across_panels_and_frames(self):
        values = np.array([[[[1.0, 4.0]]], [[[16.0, -64.0]]]])
        for shape in ("square", "circle"):
            drawing = esch.draw(values, time_axis=-1, shape=shape)
            expected = esch.draw(values / 64, time_axis=-1, shape=shape, normalize=False)
            self.assertEqual(drawing.tostring(), expected.tostring())

    def test_normalization_zero_and_opt_out(self):
        for shape, kind, dimension in (("square", "rect", "width"), ("circle", "circle", "r")):
            drawing = esch.draw(np.zeros((2, 3)), shape=shape)
            self.assertTrue(all(float(g.get(dimension)) == 0 for g in nodes(drawing, kind)))
        self.assertEqual(float(nodes(esch.draw([4], normalize=False), "rect")[0].get("width")), 2)
        self.assertEqual(float(nodes(esch.draw([4]), "rect")[0].get("width")), 1)

    def test_normalization_in_layer_apis(self):
        values = np.array([[1.0, -4.0]])
        for renderer in ("grid", "mesh"):
            for normalize in (True, False):
                a, b = esch.Drawing(1, 2), esch.Drawing(1, 2)
                expected = values / 4 if normalize else values
                if renderer == "grid":
                    esch.grid_fn(a, values[:, None], normalize=normalize)
                    esch.grid_fn(b, expected[:, None], normalize=False)
                else:
                    positions = [[[0, 0], [0, 1]]]
                    esch.mesh_fn(a, positions, values, normalize=normalize)
                    esch.mesh_fn(b, positions, expected, normalize=False)
                self.assertEqual(a.tostring(), b.tostring())

    def test_named_panel_rows_and_columns(self):
        source = np.linspace(0.1, 1, 120).reshape(2, 3, 4, 5)
        expected = esch.draw(source, pattern="n c h w").tostring()
        for pattern, arr in (
            ("row col h w", source),
            ("row col h w -> col w row h", source),
            ("col row h w -> row col h w", source.transpose(1, 0, 2, 3)),
        ):
            with self.subTest(pattern=pattern):
                drawing = esch.draw(arr, pattern=pattern)
                self.assertEqual((drawing.row, drawing.col), (2, 3))
                self.assertEqual(drawing.tostring(), expected)

    def test_interleaved_arrays_in_named_columns(self):
        source = np.linspace(-1, 1, 72).reshape(8, 3, 3)
        expected = esch.draw(source.reshape(4, 2, 3, 3), pattern="t c h w", fps=4).tostring()
        for pattern in ("(t col) h w -> col h w t", "(t col) ... -> t col ...", "(t col) h w"):
            with self.subTest(pattern=pattern):
                drawing = esch.draw(source, pattern=pattern, col=2, fps=4)
                self.assertEqual((drawing.row, drawing.col, drawing.h, drawing.w), (1, 2, 3, 3))
                self.assertEqual(drawing.tostring(), expected)

    def test_named_rows_and_legacy_color(self):
        drawing = esch.draw(np.ones((3, 2, 4)), pattern="row h w -> row h w", col="navy")
        self.assertEqual((drawing.row, drawing.col), (3, 1))
        self.assertTrue(all(g.get("fill") == "navy" for g in nodes(drawing, "rect")))

    def test_named_layout_rejects_conflicts(self):
        for options in (
            {"pattern": "col h w", "columns": 3},
            {"pattern": "col extra w"},
            {"pattern": "col h w -> (col h) w"},
            {"pattern": "col h w -> col h w", "time_axis": 0},
            {"pattern": "col h w", "col": 2, "axes_lengths": {"col": 2}},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                esch.draw(np.ones((2, 3, 4)), **options)

    def test_rearrangement_patterns(self):
        source = np.arange(1, 121).reshape(2, 3, 4, 5) / 120
        patterns = (
            "b c h w -> (b c) h w",
            "b c h w -> (b h) (c w)",
            "b c h w -> b c w h",
            "... h w -> ... w h",
            "... h w -> (...) h w",
            "b c h w -> 1 b c h w",
        )
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                expected = esch.draw(rearrange(source, pattern)).tostring()
                self.assertEqual(esch.draw(source, pattern=pattern).tostring(), expected)
                self.assertEqual(esch.draw(pattern, source).tostring(), expected)

    def test_split_axes_and_keyword_lengths(self):
        source = np.arange(24).reshape(6, 4) / 24
        pattern = "(batch h) w -> batch h w"
        expected = esch.draw(rearrange(source, pattern, h=2)).tostring()
        self.assertEqual(esch.draw(source, pattern=pattern, h=2).tostring(), expected)
        self.assertEqual(esch.draw(source, pattern="(batch h) w", h=2).tostring(), expected)
        collision = "(shape h) w -> shape h w"
        self.assertEqual(esch.draw(source, pattern=collision, axes_lengths={"shape": 3}).tostring(), expected)

    def test_rearranged_time_axes(self):
        source = np.linspace(-1, 1, 120).reshape(2, 3, 4, 5)
        for pattern, axis in (("b h w t -> t b h w", 0), ("... h w t -> ... t w h", 1), ("b h w t -> h w (b t)", 2)):
            with self.subTest(pattern=pattern):
                expected = esch.draw(rearrange(source, pattern), time_axis=axis, fps=5).tostring()
                self.assertEqual(esch.draw(source, pattern=pattern, fps=5).tostring(), expected)
        pattern = "b h w frames -> frames b h w"
        self.assertEqual(
            esch.draw(source, pattern=pattern, time_axis=0).tostring(),
            esch.draw(rearrange(source, pattern), time_axis=0).tostring(),
        )

    def test_rearrange_scalar_singleton_and_empty_ellipsis(self):
        for source, pattern in (
            (np.array(0.5), " -> 1 1"),
            (np.ones((1, 2)), "1 w -> w"),
            (np.ones((2, 3)), "... h w -> ... w h"),
        ):
            with self.subTest(pattern=pattern):
                self.assertEqual(
                    esch.draw(source, pattern=pattern).tostring(), esch.draw(rearrange(source, pattern)).tostring()
                )

    def test_invalid_rearrangements(self):
        with self.assertRaisesRegex(TypeError, "pattern must be a string"):
            esch.draw(np.ones((2, 3)), pattern=42)
        for options in (
            {"pattern": "h w -> h"},
            {"pattern": "h w -> h h"},
            {"pattern": "(a h) w -> a h w"},
            {"pattern": "(a h) w -> a h w", "a": 4},
            {"pattern": "h w -> h w", "unknown": 2},
            {"h": 2},
            {"pattern": "fps w -> fps w"},
            {"pattern": "fps w"},
            {"pattern": "h t -> t h", "time_axis": 0},
            {"pattern": "h w -> h w", "h": 2, "axes_lengths": {"h": 2}},
            {"pattern": "h w -> h w", "h": 0.5},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                esch.draw(np.ones((2, 3)), **options)

    def test_arbitrary_rank(self):
        for shape in ((), (5,), (2, 5), (3, 2, 5), (2, 3, 2, 5), (2, 1, 3, 2, 5), (2, 1, 3, 2, 2, 5)):
            with self.subTest(shape=shape):
                drawing = esch.draw(np.ones(shape))
                self.assertEqual(len(nodes(drawing, "rect")), np.prod(shape))
                self.assertEqual(
                    (drawing.h, drawing.w), ((1, 1) if not shape else (1, shape[0]) if len(shape) == 1 else shape[-2:])
                )
                self.assertFalse(nodes(drawing, "animate"))
                self.assertGreater(drawing.total_width, 0)
                self.assertGreater(drawing.total_height, 0)

    def test_no_time_guessing(self):
        self.assertFalse(nodes(esch.draw(np.ones((2, 100))), "animate"))

    def test_axis_permutations_preserve_values(self):
        source = np.arange(1, 25).reshape(2, 3, 4) / 24
        expected = esch.draw(source, pattern="batch h w").tostring()
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                pattern = " ".join(["batch", "h", "w"][i] for i in order)
                self.assertEqual(esch.draw(source.transpose(order), pattern=pattern).tostring(), expected)

    def test_time_axis_in_every_position(self):
        arr = np.linspace(-1, 1, 120).reshape(2, 3, 4, 5)
        expected = esch.draw(arr, pattern="batch h w t").tostring()
        for axis in range(4):
            moved = np.moveaxis(arr, -1, axis)
            self.assertEqual(esch.draw(moved, time_axis=axis).tostring(), expected)
            self.assertEqual(esch.draw(moved, time_axis=axis - 4).tostring(), expected)

    def test_ellipsis_and_legacy_call(self):
        arr = np.ones((2, 3, 4, 5, 6))
        self.assertEqual(esch.draw(arr, pattern="... h w t").tostring(), esch.draw(arr, time_axis=-1).tostring())
        self.assertEqual(esch.draw("n c h w t", arr).tostring(), esch.draw(arr, pattern="n c h w t").tostring())
        self.assertEqual(esch.draw(np.ones((2, 3)), pattern="... h w").tostring(), esch.draw(np.ones((2, 3))).tostring())

    def test_legacy_panel_order(self):
        arr = np.arange(1, 7).reshape(2, 3, 1, 1) / 6
        drawing = esch.draw("n c h w", arr)
        self.assertEqual((drawing.row, drawing.col), (2, 3))
        self.assertEqual(drawing.tostring(), esch.draw("c n h w", arr.transpose(1, 0, 2, 3)).tostring())
        self.assertEqual(len(nodes(drawing, "g")), 6)

    def test_missing_dimensions(self):
        for pattern, shape, expected in (
            ("t", (3,), (1, 1)),
            ("h t", (4, 3), (4, 1)),
            ("w t", (4, 3), (1, 4)),
            ("", (), (1, 1)),
        ):
            with self.subTest(pattern=pattern):
                drawing = esch.draw(np.ones(shape), pattern=pattern)
                self.assertEqual((drawing.h, drawing.w), expected)

    def test_single_frame_matches_static(self):
        arr = np.array([[-1.0, 0, 0.25, 1]])
        for shape in ("square", "circle"):
            self.assertEqual(
                esch.draw(arr, shape=shape).tostring(), esch.draw(arr[..., None], time_axis=-1, shape=shape).tostring()
            )

    def test_animation_first_frame_loop_and_timing(self):
        for shape, kind, dimension in (("square", "rect", "width"), ("circle", "circle", "r")):
            with self.subTest(shape=shape):
                drawing = esch.draw(np.array([-0.25, 1, 0]), pattern="t", shape=shape, fps=2)
                node = nodes(drawing, kind)[0]
                static = nodes(esch.draw(-0.25, shape=shape, normalize=False), kind)[0]
                self.assertEqual(node.attrib, static.attrib)
                animations = {a.get("attributeName"): a for a in nodes(drawing, "animate")}
                size = animations[dimension]
                self.assertAlmostEqual(sample(size, 0)[0], float(node.get(dimension)))
                self.assertAlmostEqual(sample(size, 1)[0], float(node.get(dimension)))
                peak = nodes(esch.draw(1, shape=shape), kind)[0]
                self.assertAlmostEqual(sample(size, 1 / 3)[0], float(peak.get(dimension)), places=4)
                self.assertTrue(all(a.get("dur") == "1.5s" for a in animations.values()))
                # -0.25 -> 1 crosses zero at 1/15; 0 at 2/3 continues toward -0.25.
                fills = [sample(animations["fill"], t) for t in (0, 0.05, 0.1, 0.5, 0.7, 0.99)]
                self.assertEqual(fills, ["white", "white", "black", "black", "white", "white"])

    def test_signed_geometry_and_color(self):
        for shape, kind, size in (("square", "rect", "width"), ("circle", "circle", "r")):
            glyphs = nodes(esch.draw([-1, 0, 1], shape=shape, col="navy"), kind)
            self.assertEqual([g.get("fill") for g in glyphs], ["white", "navy", "navy"])
            self.assertTrue(all(float(g.get(size)) >= 0 for g in glyphs))
            self.assertEqual(glyphs[0].get(size), glyphs[2].get(size))

    def test_dot_draws_zero_as_an_outline(self):
        for shape, kind, size, floor in (("square", "rect", "width", 0.04), ("circle", "circle", "r", 0.02)):
            with self.subTest(shape=shape):
                zero, small = nodes(esch.draw([0, 0.25], shape=shape, dot=True, normalize=False), kind)
                plain = nodes(esch.draw([0, 0.25], shape=shape, normalize=False), kind)
                # The outline (0.04 wide) covers the whole glyph: no fill shows.
                self.assertEqual((float(zero.get(size)), float(zero.get("stroke-width"))), (floor, 0.04))
                self.assertEqual(float(plain[0].get(size)), 0)
                self.assertEqual(small.attrib, plain[1].attrib)
                moving = esch.draw([1, -1], pattern="t", shape=shape, dot=True)
                animations = {a.get("attributeName"): a for a in nodes(moving, "animate")}
                self.assertAlmostEqual(sample(animations[size], 0.25)[0], floor)
        with self.assertRaisesRegex(ValueError, "dot"):
            esch.draw([0, 1], shape="voronoi", dot=True)

    @given(arrays(np.float64, st.integers(1, 6), elements=st.floats(-10, 10)))
    def test_dot_floors_glyph_size_at_the_outline(self, values):
        for shape, kind, size, floor in (("square", "rect", "width", 0.04), ("circle", "circle", "r", 0.02)):
            plain = [float(node.get(size)) for node in nodes(esch.draw(values, shape=shape), kind)]
            dotted = [float(node.get(size)) for node in nodes(esch.draw(values, shape=shape, dot=True), kind)]
            self.assertEqual(dotted, np.maximum(plain, floor).tolist())

    def test_opposite_values_shrink_through_zero_and_loop(self):
        for shape, dimension in (("square", "width"), ("circle", "r")):
            for magnitude in (False, True):
                with self.subTest(shape=shape, magnitude=magnitude):
                    drawing = esch.draw(
                        [1, -1],
                        pattern="t",
                        shape=shape,
                        fps=2,
                        pos="red",
                        neg="blue",
                        color_by_magnitude=magnitude,
                    )
                    animations = {a.get("attributeName"): a for a in nodes(drawing, "animate")}
                    times = (0, 0.125, 0.25, 0.375, 0.5, 0.75, 1)
                    sizes = np.array([sample(animations[dimension], t)[0] for t in times])
                    self.assertGreater(sizes[0], 0)
                    np.testing.assert_allclose(sizes, sizes[0] * np.array([1, 0.5, 0, 0.5, 1, 0, 1]))
                    self.assertTrue(all(a.get("dur") == "1s" for a in animations.values()))
                    for attribute in ("fill", "stroke"):
                        if magnitude:
                            colors = [sample(animations[attribute], t) for t in (0, 0.25, 0.5, 0.75)]
                            np.testing.assert_allclose(colors, [[255, 0, 0], [255, 255, 255], [0, 0, 255], [255, 255, 255]])
                        else:
                            colors = [sample(animations[attribute], t) for t in (0.1, 0.3, 0.6, 0.8)]
                            self.assertEqual(colors, ["red", "blue", "blue", "red"])
                    if shape == "square":
                        for coordinate in ("x", "y"):
                            positions = np.array([sample(animations[coordinate], t)[0] for t in times])
                            np.testing.assert_allclose(positions + sizes / 2, 0, atol=1e-3)

    def test_unequal_sign_crossings_keep_original_frame_times(self):
        drawing = esch.draw([1, -0.25], pattern="t", pos="red", neg="blue")
        animations = {a.get("attributeName"): a for a in nodes(drawing, "animate")}
        widths = [sample(animations["width"], t)[0] for t in (0, 0.2, 0.4, 0.5, 0.6, 0.8, 1)]
        np.testing.assert_allclose(widths, [1, 0.5, 0, 0.5, 0, 0.5, 1])
        colors = [sample(animations["fill"], t) for t in (0.3, 0.45, 0.55, 0.65)]
        self.assertEqual(colors, ["red", "blue", "blue", "red"])
        self.assertEqual(animations["width"].get("dur"), "2s")

    def test_same_sign_animation_needs_no_zero_crossing(self):
        drawing = esch.draw([0.25, 1], pattern="t")
        width = next(a for a in nodes(drawing, "animate") if a.get("attributeName") == "width")
        widths = [sample(width, t)[0] for t in (0, 0.25, 0.5, 0.75, 1)]
        np.testing.assert_allclose(widths, [0.5, 0.75, 1, 0.75, 0.5])

    def test_custom_sign_colors_across_renderers(self):
        colors = {"pos": "#155e75", "neg": "#c2410c"}
        for shape, kind in (("square", "rect"), ("circle", "circle")):
            for animated in (False, True):
                for renderer in ("draw", "grid", "mesh"):
                    with self.subTest(shape=shape, animated=animated, renderer=renderer):
                        values = np.array([[-1, 1], [1, -1]]) if animated else np.array([-1, 1])
                        if renderer == "draw":
                            drawing = esch.draw(values, pattern="w t" if animated else "w", shape=shape, **colors)
                        else:
                            drawing = esch.Drawing(1, 2)
                            if renderer == "grid":
                                esch.grid_fn(drawing, values[None, None], shape=shape, **colors)
                            else:
                                esch.mesh_fn(drawing, [[[0, 0], [0, 1]]], values[None], shape=shape, **colors)
                        glyphs = nodes(drawing, kind)
                        self.assertEqual([g.get("fill") for g in glyphs], ["#c2410c", "#155e75"])
                        self.assertEqual([g.get("stroke") for g in glyphs], ["#c2410c", "#155e75"])
                        fills = [a for a in nodes(drawing, "animate") if a.get("attributeName") == "fill"]
                        self.assertEqual(len(fills), 2 if animated else 0)
                        if animated:
                            shown = [sample(fills[0], t) for t in (0, 0.2, 0.3, 0.7, 0.8)]
                            self.assertEqual(shown, ["#c2410c", "#c2410c", "#155e75", "#155e75", "#c2410c"])
                            self.assertEqual(fills[0].get("calcMode"), "discrete")
                            strokes = [a for a in nodes(drawing, "animate") if a.get("attributeName") == "stroke"]
                            self.assertEqual([a.get("values") for a in strokes], [a.get("values") for a in fills])
                            self.assertTrue(all(a.get("calcMode") == "discrete" for a in strokes))

    def test_color_defaults_and_precedence(self):
        drawing = esch.draw([-1, 1], col="navy", neg="orange")
        self.assertEqual([g.get("fill") for g in nodes(drawing, "rect")], ["orange", "navy"])
        drawing = esch.draw([-1, 1], col="navy", pos="green")
        self.assertEqual([g.get("fill") for g in nodes(drawing, "rect")], ["white", "green"])
        self.assertTrue(all(g.get("stroke") == "green" for g in nodes(drawing, "rect")))
        explicit_white = esch.draw([-1, 1], pos="green", neg="white")
        self.assertEqual([g.get("stroke") for g in nodes(explicit_white, "rect")], ["white", "green"])
        drawing = esch.draw([-1, 1], pattern="t", pos="navy", neg="navy")
        self.assertFalse([a for a in nodes(drawing, "animate") if a.get("attributeName") == "fill"])

    def test_magnitude_colors_share_scale_across_panels_and_frames(self):
        options = {"pos": "red", "neg": "#00f", "color_by_magnitude": True}
        values = np.array([[-2, -1, 0], [0.5, 1, 2]])
        expected = ["#0000ff", "#8080ff", "#ffffff", "#ffbfbf", "#ff8080", "#ff0000"]
        for shape, kind in (("square", "rect"), ("circle", "circle")):
            drawing = esch.draw(values, pattern="batch w", shape=shape, **options)
            self.assertEqual([g.get("fill") for g in nodes(drawing, kind)], expected)
            self.assertEqual([g.get("stroke") for g in nodes(drawing, kind)], expected)
            plain = esch.draw(values, pattern="batch w", shape=shape)
            geometry = ("width", "height", "x", "y") if kind == "rect" else ("r", "cx", "cy")
            for colored, original in zip(nodes(drawing, kind), nodes(plain, kind), strict=True):
                self.assertEqual([colored.get(key) for key in geometry], [original.get(key) for key in geometry])
            animated = esch.draw(values, pattern="batch t", shape=shape, **options)
            fills = [a for a in nodes(animated, "animate") if a.get("attributeName") == "fill"]
            self.assertEqual(fills[0].get("values"), ";".join(expected[:3] + expected[:1]))
            self.assertEqual(fills[1].get("values"), ";".join(expected[3:] + expected[3:4]))
            self.assertTrue(all(a.get("calcMode") == "linear" for a in fills))
            strokes = [a for a in nodes(animated, "animate") if a.get("attributeName") == "stroke"]
            self.assertEqual([a.get("values") for a in strokes], [a.get("values") for a in fills])
            self.assertTrue(all(a.get("calcMode") == "linear" for a in strokes))

    def test_magnitude_mesh_uses_unscaled_value_ratios(self):
        drawing = esch.Drawing(1, 2)
        esch.mesh_fn(drawing, [[[0, 0], [0, 1]]], [[0.5, 1]], pos="red", color_by_magnitude=True)
        self.assertEqual([g.get("fill") for g in nodes(drawing, "circle")], ["#ff8080", "#ff0000"])

    def test_magnitude_zero_and_single_frame(self):
        drawing = esch.draw([0, 0], color_by_magnitude=True)
        self.assertTrue(all(g.get("fill") == g.get("stroke") == "#ffffff" for g in nodes(drawing, "rect")))
        options = {"pos": "navy", "neg": "orange", "color_by_magnitude": True}
        values = np.array([[-1, 0.25]])
        self.assertEqual(
            esch.draw(values, **options).tostring(), esch.draw(values[..., None], time_axis=-1, **options).tostring()
        )
        with self.assertRaisesRegex(ValueError, "Magnitude colors"):
            esch.draw([1], pos="not-a-color", color_by_magnitude=True)

    def test_rectangular_layout_and_partial_last_row(self):
        drawing = esch.draw(np.ones((5, 2, 7)), columns=3)
        self.assertEqual((drawing.row, drawing.col), (2, 3))
        self.assertEqual(len(nodes(drawing, "g")), 5)
        self.assertEqual(xml(drawing).get("viewBox"), "0,0,24.8,5.4")
        self.assertEqual(nodes(drawing, "g")[4].get("transform"), "translate(9.4,3.4)")

    def test_large_values_fit_viewbox(self):
        for shape, kind in (("square", "rect"), ("circle", "circle")):
            drawing = esch.draw(100, shape=shape, normalize=False)
            node = nodes(drawing, kind)[0]
            radius = float(node.get("r")) if kind == "circle" else float(node.get("width")) / 2
            self.assertGreater(drawing.pad, radius)

    def test_input_unchanged(self):
        arr = np.arange(24).reshape(2, 3, 4)
        original = arr.copy()
        esch.draw(arr, time_axis=-1)
        np.testing.assert_array_equal(arr, original)

    def test_glyph_size_and_array_coordinates(self):
        square = nodes(esch.draw(np.full((2, 3), 0.25), normalize=False), "rect")[-1]
        self.assertEqual(float(square.get("width")), 0.5)
        self.assertEqual(float(square.get("x")) + 0.25, 2)
        self.assertEqual(float(square.get("y")) + 0.25, 1)
        circle = nodes(esch.draw(np.full((2, 3), 0.25), shape="circle", normalize=False), "circle")[-1]
        self.assertAlmostEqual(float(circle.get("r")), 0.5 / 2.1, places=3)
        self.assertEqual((float(circle.get("cx")), float(circle.get("cy"))), (2, 1))

    def test_shape_aliases(self):
        for alias, shape in (("sphere", "circle"), ("cube", "square")):
            self.assertEqual(esch.draw([-1, 1], shape=alias).tostring(), esch.draw([-1, 1], shape=shape).tostring())

    def test_invalid_arrays(self):
        for arr in ([], np.empty((2, 0)), [np.nan], [np.inf], [1j], ["hello"], None, [[1], [2, 3]]):
            with self.subTest(arr=repr(arr)), self.assertRaises(ValueError):
                esch.draw(arr)

    def test_invalid_options(self):
        options = [
            {"pattern": "h h"},
            {"pattern": "h"},
            {"pattern": "h w t"},
            {"pattern": "... ..."},
            {"pattern": "(h w)"},
            {"pattern": "h w", "time_axis": 0},
            {"time_axis": 2},
            {"time_axis": -3},
            {"time_axis": True},
            {"time_axis": 0.5},
            {"fps": 0},
            {"fps": -1},
            {"fps": np.nan},
            {"columns": 0},
            {"columns": 1.5},
            {"columns": True},
            {"shape": "triangle"},
        ]
        for kwargs in options:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                esch.draw(np.ones((2, 3)), **kwargs)

    def test_debug_does_not_duplicate_panels(self):
        drawing = esch.draw(np.ones((2, 3, 4)), debug=True)
        self.assertEqual(len(nodes(drawing, "g")), 2)
        self.assertEqual(len(nodes(drawing, "rect")), 24 + 2 + 1)

    def test_save_and_notebook_svg(self):
        drawing = esch.draw([0.25, 1], pattern="t")
        self.assertEqual(drawing._repr_svg_(), drawing.tostring())
        with patch.object(drawing.dwg, "saveas") as saveas:
            drawing.save(Path("test.svg"))
            saveas.assert_called_once_with("test.svg")
        with patch.object(drawing.dwg, "saveas") as saveas:
            esch.save(drawing, Path("other.svg"))
            saveas.assert_called_once_with("other.svg")

    def test_show_preserves_native_animation(self):
        from IPython.display import SVG

        drawing = esch.draw([0.25, 1], pattern="t")
        with patch("IPython.display.display") as display:
            esch.show(drawing)
            svg = display.call_args.args[0]
            self.assertIsInstance(svg, SVG)
            root = ET.fromstring(svg.data)
            self.assertTrue(root.findall(".//s:animate", NS))


class LayerTests(unittest.TestCase):
    def test_mesh_coordinates_and_short_color_keywords(self):
        drawing = esch.Drawing(1, 2)
        esch.mesh_fn(drawing, positions=[[[0, 0], [0, 1]]], arr=[[-1, 1]], pos="teal", neg="orange")
        glyphs = nodes(drawing, "circle")
        self.assertEqual([g.get("fill") for g in glyphs], ["orange", "teal"])
        self.assertEqual([g.get("stroke") for g in glyphs], ["orange", "teal"])

    def test_grid_layers_and_rectangular_ticks(self):
        drawing = esch.Drawing(h=2, w=5)
        self.assertIs(esch.grid_fn(drawing, np.ones((1, 2, 5)), shape="square", ticks=True), drawing)
        esch.grid_fn(drawing, np.full((1, 2, 5), 0.1), shape="circle", col="red")
        self.assertEqual(len(nodes(drawing, "g")), 1)
        self.assertEqual(len(nodes(drawing, "rect")), 10)
        self.assertEqual(len(nodes(drawing, "circle")), 10)
        self.assertEqual(len(nodes(drawing, "line")), 14)

    @given(arrays(np.float64, st.integers(1, 6), elements=st.floats(-4, 4)))
    def test_unscaled_mesh_sizes_glyphs_like_draw(self, values):
        positions = np.stack([np.zeros(len(values)), np.arange(len(values))], axis=-1)[None]
        for shape, kind, size in (("square", "rect", "width"), ("circle", "circle", "r")):
            drawing = esch.Drawing(1, len(values))
            esch.mesh_fn(drawing, positions, values[None], shape=shape, normalize=False, scaled=False)
            grid = esch.draw(values, shape=shape, normalize=False)
            self.assertEqual([n.get(size) for n in nodes(drawing, kind)], [n.get(size) for n in nodes(grid, kind)])

    def test_mesh_shapes_static_and_animated(self):
        for shape, kind in (("square", "rect"), ("circle", "circle")):
            for animated in (False, True):
                with self.subTest(shape=shape, animated=animated):
                    drawing = esch.Drawing(4, 5)
                    arr = np.array([[[0.5, -1], [1, 0.5]]]) if animated else np.array([[0.5, 1]])
                    self.assertIs(esch.mesh_fn(drawing, [[[1, 2], [2, 3]]], arr, shape=shape), drawing)
                    self.assertEqual(len(nodes(drawing, kind)), 2)
                    self.assertEqual(bool(nodes(drawing, "animate")), animated)
                    first = nodes(drawing, kind)[0]
                    if kind == "circle":
                        self.assertEqual((float(first.get("cx")), float(first.get("cy"))), (2, 1))

    def test_rejects_shape_mismatches_before_mutating(self):
        drawing = esch.Drawing(2, 3)
        original = drawing.tostring()
        for arr in (np.ones((1, 2, 2)), np.ones((2, 2, 3)), np.ones((2, 3))):
            with self.assertRaises(ValueError):
                esch.grid_fn(drawing, arr)
        with self.assertRaises(ValueError):
            esch.mesh_fn(drawing, [[[0, 0]]], [[1, 2]])
        self.assertEqual(drawing.tostring(), original)


if __name__ == "__main__":
    unittest.main()
