"""Geometry, animation, and compatibility checks for fixed Voronoi sites."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from shapely import Point, Polygon, box, union_all

import esch
from esch.voronoi import RATE, TOLERANCE, _align

from .smil import sample

NS = {"s": "http://www.w3.org/2000/svg"}


def polygons(drawing):
    root = ET.fromstring(drawing.tostring())
    return root.findall(".//s:polygon", NS)


def panel(drawing):
    """The clip rectangle that hides each cell's margin beyond the panel."""
    rect = ET.fromstring(drawing.tostring()).find(".//s:clipPath/s:rect", NS)
    x, y, w, h = (float(rect.get(key)) for key in ("x", "y", "width", "height"))
    return box(x, y, x + w, y + h)


def outline(points):
    return np.array([list(map(float, point.split(","))) for point in points.split()])


def geometry(node, clip=None, t=None):
    """A cell as drawn: its outline at loop fraction t, clipped to the panel."""
    animation = node.find("s:animate[@attributeName='points']", NS)
    static = t is None or animation is None
    polygon = Polygon(outline(node.get("points")) if static else sample(animation, t).reshape(-1, 2))
    return polygon if clip is None else polygon.intersection(clip)


class VoronoiTests(unittest.TestCase):
    def test_hide_outer_keeps_only_interior_cell_and_dot(self):
        drawing = esch.draw(np.ones((3, 3)), shape="voronoi", hide_outer=True, dots=True)
        root = ET.fromstring(drawing.tostring())
        visible = root.findall('.//s:polygon[@visibility="visible"]', NS)
        self.assertEqual(len(visible), 1)
        self.assertAlmostEqual(geometry(visible[0]).area, 1)
        self.assertEqual(len(root.findall('.//s:circle[@visibility="visible"]', NS)), 1)
        # The interior cell draws all four of its edges.
        self.assertEqual(visible[0].get("stroke"), "white")
        for value in (1, [1, 1]):
            root = ET.fromstring(esch.draw(value, shape="voronoi", hide_outer=True, dots=True).tostring())
            self.assertFalse(root.findall('.//*[@visibility="visible"]', NS))
            self.assertFalse(root.findall(".//s:path", NS))
        with self.assertRaisesRegex(TypeError, "hide_outer"):
            esch.draw([1], shape="voronoi", hide_outer="yes")
        with self.assertRaisesRegex(ValueError, "hide_outer"):
            esch.draw([1], hide_outer=True)

    def test_hide_outer_tracks_motion_and_loops(self):
        sites = np.moveaxis(np.indices((3, 3)), 0, -1).reshape(1, 9, 2).astype(float)
        end = sites.copy()
        end[0, 4] = [-0.4, 1.2]
        positions = np.stack([sites, end], axis=-2)
        drawing = esch.voronoi_fn(esch.Drawing(3, 3), positions, np.ones((1, 9)), hide_outer=True, dots=True, fps=1)
        root = ET.fromstring(drawing.tostring())
        dots = root.findall(".//s:circle", NS)
        changed = False
        for node, dot in zip(polygons(drawing), dots, strict=True):
            visibility = node.find("s:animate[@attributeName='visibility']", NS)
            if visibility is not None:
                changed = True
                self.assertEqual(visibility.attrib, dot.find("s:animate[@attributeName='visibility']", NS).attrib)
                self.assertEqual(sample(visibility, 0), sample(visibility, 1))
            # Visibility follows the geometry samples: check at those samples.
            for t in np.arange(2 * RATE) / (2 * RATE):
                state = node.get("visibility") if visibility is None else sample(visibility, t)
                cell = geometry(node, t=t)
                gap = min(cell.bounds[0] + 0.5, cell.bounds[1] + 0.5, 2.5 - cell.bounds[2], 2.5 - cell.bounds[3])
                if abs(gap) > 2 * TOLERANCE:
                    self.assertEqual(state, "hidden" if gap < 0 else "visible")
        self.assertTrue(changed)

    def test_positive_neighbors_get_distinct_edges(self):
        for moving in (False, True):
            sites = np.array([[[0, 0], [0, 1], [0, 2], [0, 3]]])
            if moving:
                sites = np.stack([sites, sites + [0, 0.1]], axis=-2)
            drawing = esch.voronoi_fn(
                esch.Drawing(1, 4),
                sites,
                [[1, 1, -1, -1]],
                pos="black",
                neg="white",
                edge="black",
                edge_pos="white",
                color_by_magnitude=False,
            )
            # Cells stroke their own edges; negative cells paint last, so only
            # edges between two positive cells keep the positive edge color.
            cells = polygons(drawing)
            self.assertEqual([p.get("fill") for p in cells], ["black", "black", "white", "white"])
            self.assertEqual([p.get("stroke") for p in cells], ["white", "white", "black", "black"])
            self.assertFalse(ET.fromstring(drawing.tostring()).findall(".//s:path", NS))
            if moving:
                for node in cells:
                    self.assertTrue(geometry(node, t=0).equals(geometry(node, t=1)))
                    self.assertFalse(geometry(node, t=0).equals(geometry(node, t=0.5)))

    def test_positive_edges_follow_changing_signs_without_motion(self):
        values = np.array([[1, -1], [1, -1], [-1, 1], [-1, 1]])[None]
        drawing = esch.draw(values, "h w t", shape="voronoi", edge="black", edge_pos=[255, 255, 255], fps=1)
        # Changing signs need separately animated edges; cells then draw none.
        self.assertTrue(all(p.get("stroke") == "none" for p in polygons(drawing)))
        paths = {p.get("stroke"): p for p in ET.fromstring(drawing.tostring()).findall(".//s:path", NS)}
        animation = paths["#ffffff"].find("s:animate", NS)
        self.assertEqual(sample(animation, 0), "M0.5,-0.55L0.5,0.55")
        self.assertEqual(sample(animation, 0.5), "M2.5,-0.55L2.5,0.55")
        self.assertEqual(sample(animation, 0.99), sample(animation, 0))
        with self.assertRaisesRegex(ValueError, "edge_pos"):
            esch.draw([1], edge_pos="white")

    def test_monochrome_cells_and_optional_black_dots(self):
        positions = np.array([[[[0, 0], [0.2, 0.3]], [[1, 1], [0.8, 0.7]]]])
        values = np.array([[[1, -1], [0, 0.5]]])
        for animated in (False, True):
            for dots in (False, True):
                with self.subTest(animated=animated, dots=dots):
                    drawing = esch.voronoi_fn(
                        esch.Drawing(2, 2),
                        positions if animated else positions[:, :, 0],
                        values if animated else values[..., 0],
                        monochrome=True,
                        dots=dots,
                        pos="red",
                        neg="blue",
                        color_by_magnitude=True,
                        fps=2,
                    )
                    root = ET.fromstring(drawing.tostring())
                    cells = root.findall(".//s:polygon", NS)
                    self.assertEqual(len(cells), 2)
                    self.assertTrue(all(p.get("fill") == "none" and p.get("stroke") == "black" for p in cells))
                    markers = root.findall(".//s:circle", NS)
                    self.assertEqual(len(markers), 2 if dots else 0)
                    for marker in markers:
                        self.assertEqual(marker.get("fill"), "black")
                        self.assertEqual(marker.get("stroke"), "black")
                        self.assertAlmostEqual(float(marker.get("r")), 0.04)
                    animations = root.findall(".//s:animate", NS)
                    self.assertEqual(bool(animations), animated)
                    self.assertTrue(all(a.get("attributeName") in ("points", "cx", "cy") for a in animations))
                    self.assertTrue(all(a.get("dur") == "1s" for a in animations))
        grid = esch.draw(values.reshape(1, 2, 2), "h w t", shape="voronoi", monochrome=True, dots=True)
        self.assertTrue(all(p.get("fill") == "none" for p in polygons(grid)))
        self.assertFalse(ET.fromstring(grid.tostring()).findall(".//s:animate", NS))
        with self.assertRaisesRegex(TypeError, "monochrome"):
            esch.draw([1], shape="voronoi", monochrome="yes")
        with self.assertRaisesRegex(ValueError, "monochrome"):
            esch.draw([1], monochrome=True)

    def test_optional_cell_edges(self):
        custom = esch.draw([1, -1], shape="voronoi", pos="black", neg="white", edge=[0, 0, 0], color_by_magnitude=False)
        self.assertEqual([p.get("fill") for p in polygons(custom)], ["black", "white"])
        self.assertEqual([p.get("stroke") for p in polygons(custom)], ["#000000"] * 2)
        with self.assertRaisesRegex(ValueError, "edge"):
            esch.draw([1], edge="black")
        for monochrome in (False, True):
            for edges in (None, False, True):
                drawing = esch.draw([1, -1], shape="voronoi", monochrome=monochrome, edges=edges, dots=True)
                enabled = True if edges is None else edges
                stroke = ("black" if monochrome else "white") if enabled else "none"
                self.assertTrue(all(p.get("stroke") == stroke for p in polygons(drawing)))
                self.assertEqual(len(polygons(drawing)), 0 if monochrome and not enabled else 2)
                self.assertFalse(ET.fromstring(drawing.tostring()).findall(".//s:path", NS))
                self.assertEqual(len(ET.fromstring(drawing.tostring()).findall(".//s:circle", NS)), 2)
        with self.assertRaisesRegex(TypeError, "edges"):
            esch.draw([1], shape="voronoi", edges="yes")
        with self.assertRaisesRegex(ValueError, "edges"):
            esch.draw([1], edges=True)

    def test_topology_changes_keep_stable_svg_point_lists(self):
        centers = np.moveaxis(np.indices((3, 4)), 0, -1)
        phase = np.arange(12).reshape(3, 4, 1) * 0.7 + np.arange(8) * 2 * np.pi / 8
        moving = centers[..., None, :] + 0.35 * np.stack([np.sin(phase), np.cos(phase)], axis=-1)
        drawing = esch.draw(np.sin(phase), "h w t", shape="voronoi", positions=moving, fps=8)
        clip = panel(drawing)
        changed_topology = False
        for node in polygons(drawing):
            animation = node.find("s:animate[@attributeName='points']", NS)
            self.assertIsNone(animation.get("calcMode"))  # linear morphs
            self.assertEqual(node.get("stroke-linejoin"), "round")
            outlines = [outline(v) for v in animation.get("values").split(";")]
            self.assertEqual({len(p) for p in outlines}, {len(node.get("points").split())})
            changed_topology |= len({len(np.unique(p, axis=0)) for p in outlines}) > 1
        self.assertTrue(changed_topology, "Regression fixture must exercise changing vertex counts")
        # Check between frames, where cells interpolate within about TOLERANCE.
        sites = moving.reshape(12, 8, 2)
        for index in range(64):
            cells = [geometry(node, clip, index / 64) for node in polygons(drawing)]
            self.assertLess(union_all(cells).symmetric_difference(clip).area, 0.3)
            self.assertAlmostEqual(sum(p.area for p in cells), 12, delta=0.3)
            left, fraction = divmod(index / 8, 1)
            xy = ((1 - fraction) * sites[:, int(left) % 8] + fraction * sites[:, (int(left) + 1) % 8])[:, ::-1]
            for site, cell in zip(xy, cells, strict=True):
                self.assertTrue(cell.buffer(TOLERANCE).covers(Point(site)))
                for vertex in np.asarray(cell.exterior.coords):
                    self.assertTrue(np.all(np.sum((vertex - site) ** 2) <= np.sum((xy - vertex) ** 2, axis=1) + 0.1))

    def test_outlines_follow_vertices_so_smooth_motion_needs_few_keyframes(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        # A rotated start and a spare slot: vertices keep their nearest slots.
        reference = square[[0, 1, 1, 2, 3]]
        np.testing.assert_allclose(_align(reference, square[[2, 3, 0, 1]] + 0.01), reference + 0.01)
        # A vertex appearing takes over a repeated slot beside its neighbours.
        born = np.array([[0, 0], [1, 0], [1, 1], [0.5, 1.2], [0, 1]], dtype=float)
        np.testing.assert_allclose(_align(reference, born), born)
        sites = np.array([[[0, 0], [0, 2], [2, 1]]], dtype=float)
        moving = np.stack([sites, sites + [0.3, 0.2]], axis=-2)
        drawing = esch.voronoi_fn(esch.Drawing(3, 3), moving, np.ones((1, 3)), fps=1)
        for node in polygons(drawing):
            frames = node.find("s:animate[@attributeName='points']", NS).get("values").split(";")
            self.assertLess(len(frames), RATE, "Smooth motion should need few of its geometry samples")

    def test_site_dots_are_optional_outlined_and_follow_motion(self):
        sites = [[[[0, 0], [0.2, 0.4]], [[1, 1], [1.2, 1.4]]]]
        for animated in (False, True):
            positions = sites if animated else np.asarray(sites)[:, :, 0]
            values = [[[1, -1], [-1, 1]]] if animated else [[1, -1]]
            drawing = esch.voronoi_fn(esch.Drawing(2, 2), positions, values, dots=True, pos=[0, 116, 217], neg="orange", fps=2)
            root = ET.fromstring(drawing.tostring())
            dots = root.findall(".//s:circle", NS)
            self.assertEqual(len(dots), 2)
            self.assertEqual([p.get("stroke") for p in dots], ["#0074d9", "orange"])
            for dot in dots:
                self.assertEqual(dot.get("fill"), "white")
                self.assertEqual(float(dot.get("r")), 0.04)
                border = root.find(".//s:polygon", NS)
                self.assertEqual(2 * float(dot.get("r")), 2 * float(border.get("stroke-width")))
                self.assertIsNone(dot.find("s:animate[@attributeName='r']", NS))
                self.assertIsNone(dot.find("s:animate[@attributeName='fill']", NS))
            if animated:
                motion = dots[0].find("s:animate[@attributeName='cx']", NS)
                self.assertEqual(motion.get("dur"), "1s")
                np.testing.assert_allclose([sample(motion, t)[0] for t in (0, 0.25, 0.5, 0.75, 1)], [0, 0.2, 0.4, 0.2, 0])
            # Clipped regions first, then markers above every region.
            layer = root.find("s:g", NS)
            self.assertEqual([n.tag.rsplit("}", 1)[-1] for n in layer], ["clipPath", "g", "circle", "circle"])
            self.assertEqual([n.tag.rsplit("}", 1)[-1] for n in layer.find("s:g", NS)], ["polygon", "polygon"])
        self.assertFalse(ET.fromstring(esch.draw([1, -1], shape="voronoi").tostring()).findall(".//s:circle", NS))
        with self.assertRaisesRegex(ValueError, "dots"):
            esch.draw([1], dots=True)
        with self.assertRaisesRegex(TypeError, "dots"):
            esch.draw([1], shape="voronoi", dots="yes")

    def test_moving_cells_tile_bounds_and_loop(self):
        sites = np.array([[[[0, 0], [0.4, 0.2]], [[0, 2], [1, 2]], [[2, 1], [2, 0.5]]]])
        drawing = esch.voronoi_fn(esch.Drawing(3, 3), sites, [[1, -1, 0.5]], fps=2)
        clip = panel(drawing)
        animations = [p.find("s:animate[@attributeName='points']", NS) for p in polygons(drawing)]
        self.assertTrue(all(a.get("dur") == "1s" for a in animations))
        for node in polygons(drawing):
            self.assertTrue(geometry(node, t=0).equals(geometry(node, t=1)))
            self.assertFalse(geometry(node, t=0).equals(geometry(node, t=0.5)))
        for t in np.arange(30) / 30:
            cells = [geometry(node, clip, t) for node in polygons(drawing)]
            self.assertAlmostEqual(union_all(cells).area, 9, delta=0.05)
            self.assertAlmostEqual(sum(c.area for c in cells), 9, delta=0.05)

    def test_motion_grid_and_time_validation(self):
        # Preserve static per-panel coordinates when dimensions coincide.
        fixed = np.stack([np.moveaxis(np.indices((2, 2)), 0, -1)] * 2)
        static = esch.draw(np.ones((2, 2, 2, 2)), "col h w t", shape="voronoi", positions=fixed)
        self.assertTrue(all(p.find("s:animate[@attributeName='points']", NS) is None for p in polygons(static)))
        sites = np.moveaxis(np.indices((2, 3)), 0, -1)
        moving = np.stack([sites, sites + 0.2], axis=-2)
        values = np.ones((2, 2, 3, 2))
        a = esch.draw(values, "col h w t", shape="voronoi", positions=moving)
        b = esch.draw(values, "col h w t", shape="voronoi", positions=np.stack([moving, moving]))
        self.assertEqual(a.tostring(), b.tostring())
        for renderer in (esch.mesh_fn, esch.voronoi_fn):
            with self.assertRaisesRegex(ValueError, "time lengths"):
                renderer(esch.Drawing(2, 3), moving.reshape(1, 6, 2, 2), np.ones((1, 6, 3)))

    def test_mesh_motion_matches_size_zero_crossings(self):
        for shape, attr in [("circle", "cx"), ("square", "x")]:
            drawing = esch.mesh_fn(esch.Drawing(3, 3), [[[[0, 0], [2, 2]]]], [[[1, -1]]], shape=shape, fps=2)
            root = ET.fromstring(drawing.tostring())
            motion = root.find(f".//s:animate[@attributeName='{attr}']", NS)
            self.assertEqual(motion.get("dur"), "1s")
            offset = 0 if shape == "circle" else sample(root.find(".//s:animate[@attributeName='width']", NS), 0)[0] / 2
            coordinates = [sample(motion, t)[0] for t in (0, 0.25, 0.75, 1)]
            # Zero crossings coincide with the halfway position.
            np.testing.assert_allclose(coordinates, [-offset, 1, 1, -offset], atol=1e-3)

    def test_cells_cover_panel_and_belong_to_nearest_site(self):
        sites = np.array([[[0, 0], [0.7, 3], [2, 1], [3, 4], [1.5, 2.1]]])
        drawing = esch.voronoi_fn(esch.Drawing(4, 5), sites, [[1, -1, 0.5, 0.25, -0.25]], pos="teal", neg="orange")
        clip = panel(drawing)
        self.assertTrue(clip.equals(box(-0.5, -0.5, 4.5, 3.5)))
        cells = [geometry(node, clip) for node in polygons(drawing)]
        self.assertEqual(len(cells), 5)
        # Coordinates are written to 0.001 grid units.
        self.assertAlmostEqual(union_all(cells).symmetric_difference(clip).area, 0, places=2)
        self.assertAlmostEqual(sum(cell.area for cell in cells), 20, places=2)
        xy = sites[0, :, ::-1]
        for site, cell in zip(xy, cells, strict=True):
            self.assertTrue(cell.covers(Point(site)))
            for vertex in np.asarray(cell.exterior.coords):
                own = np.sum((vertex - site) ** 2)
                self.assertTrue(np.all(own <= np.sum((xy - vertex) ** 2, axis=1) + 1e-2))

    def test_single_two_and_collinear_sites(self):
        drawing = esch.draw([1, 1], shape="voronoi", monochrome=True)
        clip = panel(drawing)
        cells = [geometry(node, clip) for node in polygons(drawing)]
        self.assertTrue(cells[0].equals(box(-0.5, -0.5, 0.5, 0.5)) and cells[1].equals(box(0.5, -0.5, 1.5, 0.5)))
        singleton = esch.draw(1, shape="voronoi", monochrome=True)
        self.assertTrue(geometry(polygons(singleton)[0], panel(singleton)).equals(box(-0.5, -0.5, 0.5, 0.5)))
        for count in (1, 2, 4):
            with self.subTest(count=count):
                sites = np.array([[[0, index] for index in range(count)]])
                drawing = esch.voronoi_fn(esch.Drawing(1, count), sites, np.ones((1, count)))
                self.assertEqual(len(polygons(drawing)), count)
                self.assertAlmostEqual(sum(geometry(node, panel(drawing)).area for node in polygons(drawing)), count)
        self.assertEqual(len(polygons(esch.draw(0.5, shape="voronoi"))), 1)

    def test_animation_reuses_color_timing_and_never_moves_vertices(self):
        options = {"pos": np.array([0, 116, 217]), "neg": "orange", "color_by_magnitude": True}
        values = np.array([[[[0.25, -1]]]])
        voronoi = esch.draw(values, "n h w t", shape="voronoi", fps=4, **options)
        square = esch.draw(values, "n h w t", fps=4, **options)
        region = polygons(voronoi)[0]
        animations = region.findall("s:animate", NS)
        self.assertEqual([a.get("attributeName") for a in animations], ["fill"])
        expected = ET.fromstring(square.tostring()).find(".//s:animate[@attributeName='fill']", NS)
        self.assertEqual(animations[0].attrib, expected.attrib)
        self.assertEqual(animations[0].get("dur"), "0.5s")
        self.assertEqual(region.get("stroke"), "white")
        static = esch.draw(values[..., :1], "n h w t", shape="voronoi", **options)
        self.assertFalse(polygons(static)[0].findall("s:animate", NS))

    def test_grid_positions_broadcast_and_rearrangement(self):
        values = np.linspace(-1, 1, 24).reshape(2, 3, 4)
        sites = np.moveaxis(np.indices((3, 4)), 0, -1) + 0.1
        shared = esch.draw(values, "col h w", shape="voronoi", positions=sites)
        per_panel = esch.draw(values, "col h w", shape="voronoi", positions=np.stack([sites, sites]))
        self.assertEqual(shared.tostring(), per_panel.tostring())
        rearranged = esch.draw(values.transpose(1, 2, 0), "h w col -> col h w", shape="voronoi", positions=sites)
        self.assertEqual(shared.tostring(), rearranged.tostring())
        self.assertEqual((shared.row, shared.col), (1, 2))

    def test_split_exports_keep_polygon_geometry_and_colors(self):
        drawing = esch.draw(np.arange(1, 13).reshape(2, 2, 3), "n h w", shape="voronoi", dots=True)
        original = ET.fromstring(drawing.tostring()).findall("s:g", NS)
        with TemporaryDirectory(prefix=".esch-voronoi-", dir=".") as directory:
            drawing.save(Path(directory) / "cell_{n}.svg")
            for index in range(2):
                exported = ET.parse(Path(directory) / f"cell_{index}.svg").getroot().find("s:g", NS)
                self.assertEqual([ET.tostring(n) for n in original[index]], [ET.tostring(n) for n in exported])

    def test_invalid_sites_leave_drawing_unchanged(self):
        for sites in ([[[0, 0], [0, 0]]], [[[0, 0], [0, 5]]], [[[0, 0], [np.nan, 1]]], [[[0, 0]]]):
            drawing = esch.Drawing(2, 2)
            original = drawing.tostring()
            with self.subTest(sites=sites), self.assertRaises(ValueError):
                esch.voronoi_fn(drawing, sites, [[1, 2]])
            self.assertEqual(drawing.tostring(), original)
        with self.assertRaises(ValueError):
            esch.draw(np.ones((2, 3)), shape="voronoi", positions=np.ones((2, 3, 4, 2)))

    def test_magnitude_default_does_not_change_existing_shapes(self):
        for shape in ("square", "circle"):
            self.assertEqual(
                esch.draw([0.25, 1], shape=shape).tostring(),
                esch.draw([0.25, 1], shape=shape, color_by_magnitude=False).tostring(),
            )
        drawing = esch.draw([0.25, 1], shape="voronoi", pos="red")
        self.assertEqual([node.get("fill") for node in polygons(drawing)], ["#ffbfbf", "#ff0000"])
        zero = esch.draw([0, 0], shape="voronoi")
        self.assertTrue(all(node.get("fill") == "#ffffff" for node in polygons(zero)))
