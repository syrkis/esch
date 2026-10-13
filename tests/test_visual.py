"""Always regenerate every visual example when running the test suite."""

import unittest
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

from .recipes import recipe_for
from .visual import build_gallery


class VisualTests(unittest.TestCase):
    def test_gallery(self):
        fixtures, path = build_gallery()
        page = path.read_text()
        experimental = (path.parent / "experimental.html").read_text()
        self.assertIn('href="experimental.html"', page)
        self.assertIn('href="index.html"', experimental)
        self.assertEqual(len(fixtures), len({case.name for case in fixtures}))
        for case in fixtures:
            with self.subTest(case=case.name):
                link = f'href="{case.name}.svg"'
                separate = "experiment" in case.name or (
                    case.name.startswith(("voronoi-", "territorial-control"))
                    and (path.parent / f"{case.name}.svg").stat().st_size > 1_000_000
                )
                selected_page, other_page = (experimental, page) if separate else (page, experimental)
                self.assertIn(link, selected_page)
                self.assertNotIn(link, other_page)
                root = ET.parse(path.parent / f"{case.name}.svg").getroot()
                clips = set(root.iterfind(".//{http://www.w3.org/2000/svg}clipPath/*"))
                glyphs = [
                    node
                    for node in root.iter()
                    if node.tag.rsplit("}", 1)[-1] in ("circle", "rect", "polygon")
                    and node.get("fill") != "none"
                    and node not in clips
                ]
                animations = root.findall(".//{http://www.w3.org/2000/svg}animate")
                self.assertEqual(len(glyphs), case.glyphs)
                self.assertEqual(bool(animations), case.animated)
                self.assertFalse(root.findall(".//{http://www.w3.org/2000/svg}script"))
                if case.name.startswith("territorial-control"):
                    ns = {"s": "http://www.w3.org/2000/svg"}
                    bw = case.name.endswith("black-white")
                    self.assertFalse(root.findall(".//s:circle", ns))
                    self.assertEqual(
                        [p.get("fill") for p in root.findall(".//s:polygon", ns)],
                        ["black" if bw else "#e53935"] * 32 + ["white" if bw else "#1e88e5"] * 32,
                    )
                    # Cells stroke their own edges; black cells' shared edges are white.
                    self.assertEqual(
                        [p.get("stroke") for p in root.findall(".//s:polygon", ns)],
                        ["white"] * 32 + ["black" if bw else "white"] * 32,
                    )
                    self.assertTrue(all(a.get("attributeName") in ("points", "d", "visibility") for a in animations))
                    if "inner" in case.name:
                        self.assertTrue(root.findall('.//s:polygon[@visibility="hidden"]', ns))
                        self.assertTrue(root.findall('.//s:animate[@attributeName="visibility"]', ns))
                    self.assertTrue(all(a.get("dur") == "16s" for a in animations))
                recipe = recipe_for(case.name)
                self.assertIn(escape(recipe.source(f"{case.name}.svg")), selected_page)
                # Execute exactly the displayed standalone recipe, including
                # saving, and compare it with the independent visual fixture.
                with TemporaryDirectory(prefix=".esch-recipe-", dir=".") as directory:
                    target = Path(directory) / f"{case.name}.svg"
                    namespace = {}
                    exec(compile(recipe.source(target), f"<recipe {case.name}>", "exec"), namespace)  # noqa: S102 — trusted repository recipes
                    self.assertEqual(ET.tostring(ET.parse(target).getroot()), ET.tostring(root))
                if case.split is not None:
                    axis, count = case.split
                    self.assertIn(escape(recipe.source(f"{case.name}_{{{axis}}}.svg")), selected_page)
                    with TemporaryDirectory(prefix=".esch-recipe-split-", dir=".") as directory:
                        target = Path(directory) / f"{case.name}_{{{axis}}}.svg"
                        exec(compile(recipe.source(target), "<split recipe>", "exec"), {})  # noqa: S102 — trusted repository recipes
                        for index in range(count):
                            actual = ET.parse(Path(directory) / f"{case.name}_{index}.svg").getroot()
                            expected = ET.parse(path.parent / f"{case.name}_{index}.svg").getroot()
                            self.assertEqual(ET.tostring(actual), ET.tostring(expected))
                    for index in range(count):
                        exported = ET.parse(path.parent / f"{case.name}_{index}.svg").getroot()
                        self.assertEqual(len(exported.findall("{http://www.w3.org/2000/svg}g")), 1)
        print(f"\nManual visual review: {path.resolve()} ({len(fixtures)} examples)")
