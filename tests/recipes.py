"""Small standalone recipes for the visual fixtures, checked against their SVGs."""

from dataclasses import dataclass

SIGNED = "signed = np.array([[-1, -0.5, 0, 0.5, 1], [0.2, 0.4, 0.6, 0.8, 1]])"
BLOCKS = "blocks = np.linspace(-1, 1, 120).reshape(2, 3, 4, 5)"
MOVING = """time = np.arange(32) * 2 * np.pi / 32
centers = np.moveaxis(np.indices((3, 4)), 0, -1)
phase = np.arange(12).reshape(3, 4, 1) * 0.7 + time
moving = centers[..., None, :] + 0.35 * np.stack([np.sin(phase), np.cos(phase)], axis=-1)"""
SITES = """rng = np.random.default_rng(2026)
sites = np.moveaxis(np.indices((6, 9)), 0, -1) + rng.uniform(-0.35, 0.35, (6, 9, 2))
time = np.arange(32) * 2 * np.pi / 32
wave = np.sin(sites[..., 0, None] * 0.9 + sites[..., 1, None] * 0.65 - time)"""
WAVE = """time = np.arange(24) * 2 * np.pi / 24
row, column = np.mgrid[:4, :8]
wave = np.sin(row[..., None] * 0.8 + column[..., None] * 0.5 - time)"""
MORPH = """ease = (1 - np.cos(np.linspace(0, np.pi, 12))) / 2  # slow out, slow in
progress = np.concatenate([np.zeros(6), ease, np.ones(6), ease[::-1]])  # 0 ring, 1 path
angle = np.arange(16) * 2 * np.pi / 16
ring = 1.5 + 1.5 * np.stack([-np.cos(angle), np.sin(angle)], axis=-1)
snake = np.array([(row, column if row % 2 == 0 else 3 - column) for row in range(4) for column in range(4)], float)
points = (ring[:, None] + progress[None, :, None] * (snake - ring)[:, None])[None]  # (panel, point, time, 2)"""
BATCH = WAVE + "\nbatch = np.stack([wave * factor for factor in (1, 0.7, 0.4, 0.2, 0.9, 0.5)]).reshape(2, 3, 4, 8, 24)"


@dataclass(frozen=True)
class Recipe:
    setup: str
    plot: str

    def source(self, filename):
        return f"import numpy as np\nimport esch\n\n{self.setup}\n\n{self.plot}\ndrawing.save({str(filename)!r})\n"


def recipe_for(name):
    """Return data setup and a concrete plotting call, without fixture variables."""
    if name == "voronoi-compact-experiment":
        return Recipe(
            "from tests.compact_clips import optimized_example  # experimental, repository checkout required\n"
            + recipe_for("territorial-control").setup,
            'drawing, report = optimized_example(positions[0], ["#e53935"] * 32 + ["#1e88e5"] * 32, h=7, w=11)',
        )
    if name == "voronoi-adaptive-experiment":
        return Recipe(
            "from tests.adaptive_clips import adaptive_example  # experimental, repository checkout required\n"
            + recipe_for("territorial-control").setup,
            'drawing = adaptive_example(positions[0], ["#e53935"] * 32 + ["#1e88e5"] * 32, h=7, w=11)',
        )
    if name == "voronoi-cell-experiment":
        return Recipe(
            "from tests.cell_fields import cell_example  # experimental, repository checkout required\n"
            + recipe_for("territorial-control").setup,
            "drawing = cell_example(positions[0], teams[0], h=7, w=11)",
        )
    if name.startswith("voronoi-distance-experiment"):
        return Recipe(
            "from tests.distance_fields import distance_example  # experimental, repository checkout required\n"
            + recipe_for("territorial-control").setup,
            f"drawing = distance_example(positions[0], teams[0], h=7, w=11, boundaries={name.endswith('-boundaries')})",
        )
    if name == "voronoi-clipping-experiment":
        return Recipe(
            "from tests.halfplanes import clipping_example  # experimental, repository checkout required\n"
            + recipe_for("territorial-control").setup,
            'drawing = clipping_example(positions[0], ["#e53935"] * 32 + ["#1e88e5"] * 32, h=7, w=11)',
        )
    if name in (
        "territorial-control",
        "territorial-control-black-white",
        "territorial-control-inner",
        "territorial-control-inner-black-white",
    ):
        colors = (
            'pos="black", neg="white", edge="black", edge_pos="white"'
            if name.endswith("black-white")
            else 'pos="#e53935", neg="#1e88e5", edge="white"'
        )
        return Recipe(
            """rng = np.random.default_rng(64)
waypoints = rng.uniform([0, 0], [6, 10], size=(64, 16, 2))
# Smooth each agent's independent random route, including the loop seam.
positions = (np.roll(waypoints, 1, axis=1) + 2 * waypoints + np.roll(waypoints, -1, axis=1))[None] / 4
# positions: (panel, agent, time, 2), coordinates in (row, column) order
teams = np.array([[1] * 32 + [-1] * 32])  # 64 units, fixed team membership""",
            f"""drawing = esch.Drawing(7, 11)
esch.voronoi_fn(
    drawing, positions, teams, fps=1,
    {colors},
    color_by_magnitude=False, dots=False, edges=True,
    hide_outer={"inner" in name},
)""",
        )
    animated = name.endswith("animated") or "moving" in name
    pattern = "h w t" if animated else "h w"
    if name.startswith(("voronoi-moving", "voronoi-dots-", "voronoi-monochrome-", "voronoi-colored-")):
        values = "np.sin(phase)" if animated or name == "voronoi-colored-no-edges" else "np.sin(phase[..., 0])"
        animated = animated or name == "voronoi-colored-no-edges"
        pattern = "h w t" if animated else "h w"
        positions = "moving" if animated else "moving[..., 0, :]"
        options = ["monochrome=True"] if "monochrome" in name else ['pos="teal"', 'neg="orange"']
        if "dots" in name or "moving-edges-" in name:
            options.append("dots=True")
        if "no-edges" in name or name.endswith("edges-False"):
            options.append("edges=False")
        elif name.endswith("edges-True"):
            options.append("edges=True")
        return Recipe(
            MOVING,
            f'drawing = esch.draw(\n    {values}, {pattern!r}, shape="voronoi",\n    positions={positions}, fps=16, {", ".join(options)},\n)',
        )
    if name in ("voronoi-static", "voronoi-animated", "voronoi-panels"):
        values = "np.stack([wave, -wave])" if name.endswith("panels") else ("wave" if animated else "wave[..., 0]")
        pattern = "col h w t" if name.endswith("panels") else pattern
        return Recipe(
            SITES,
            f'drawing = esch.draw(\n    {values}, {pattern!r}, shape="voronoi", positions=sites,\n    fps=16, pos="#0074d9", neg="#ff8000",\n)',
        )
    if name in ("voronoi-singleton", "voronoi-collinear"):
        values = "np.array(1)" if name.endswith("singleton") else "np.array([-1, -0.3, 0.4, 1])"
        return Recipe(f"values = {values}", 'drawing = esch.draw(values, shape="voronoi", pos="teal", neg="orange")')
    if name.startswith("mesh-moving-"):
        shape = name.removeprefix("mesh-moving-")
        return Recipe(
            MOVING,
            f"""drawing = esch.Drawing(3, 4)
esch.mesh_fn(
    drawing, moving.reshape(1, 12, 32, 2), np.sin(phase).reshape(1, 12, 32),
    shape={shape!r}, fps=16, pos="teal", neg="orange", color_by_magnitude=True,
)""",
        )
    if name.startswith("mesh-"):
        shape = name.split("-")[1]
        setup = """time = np.arange(24) * 2 * np.pi / 24
angle = np.arange(12) * 2 * np.pi / 12
positions = np.stack([2 + 1.5 * np.sin(angle), 3 + 2.5 * np.cos(angle)], axis=-1)[None]"""
        setup += "\nvalues = " + ("np.cos(angle[:, None] - time)[None]" if animated else "np.cos(angle)[None]")
        return Recipe(
            setup,
            f'drawing = esch.Drawing(h=5, w=7)\nesch.mesh_fn(drawing, positions, values, shape={shape!r}, pos="#155e75", fps=12)',
        )
    if name.startswith(("rgb-array-colors-", "magnitude-colors-", "custom-colors-")):
        shape = "circle" if name.startswith("rgb") else name.split("-")[2]
        values = "signed"
        if animated:
            values = (
                "signed[..., None] * np.cos(np.arange(24) * 2 * np.pi / 24)"
                if name.startswith("magnitude")
                else "np.stack([signed, -signed], axis=-1)"
            )
        options = (
            "pos=np.array([0, 116, 217]), neg=np.array([255, 128, 0]), color_by_magnitude=True"
            if name.startswith("rgb")
            else 'pos="#155e75", neg="#c2410c"'
        )
        if name.startswith("magnitude"):
            options += ", fps=12, color_by_magnitude=True"
        return Recipe(
            SIGNED + f"\nvalues = {values}",
            f"drawing = esch.draw(\n    values, {pattern!r}, shape={shape!r},\n    {options},\n)",
        )
    if name.startswith("normalized-areas-"):
        shape = name.removeprefix("normalized-areas-")
        return Recipe(
            "values = [-100, -25, 0, 25, 100]",
            f'drawing = esch.draw(values, shape={shape!r}, pos="teal", neg="orange", color_by_magnitude=True)',
        )
    if name.startswith("panel-gap-"):
        gap = int(name.removeprefix("panel-gap-"))
        return Recipe(BLOCKS, f'drawing = esch.draw(blocks, "row col h w", gap={gap}, debug=True)')
    if name.startswith("signed-"):
        return Recipe(SIGNED, f"drawing = esch.draw(signed, shape={name.removeprefix('signed-')!r})")
    if name.startswith("rank-"):
        rank = int(name.removeprefix("rank-"))
        dims = (2,) * max(0, rank - 2) + (3, 5)[-min(rank, 2) :] if rank else ()
        return Recipe(
            f"dims = {dims!r}\narr = np.linspace(0.15, 1, int(np.prod(dims))).reshape(dims)", "drawing = esch.draw(arr)"
        )
    if name == "line-morph":
        return Recipe(
            MORPH,
            """drawing = esch.Drawing(4, 4)
esch.line_fn(drawing, points, fps=12)
esch.mesh_fn(drawing, points, np.full((1, 16), 0.1), shape="circle", fps=12, normalize=False, scaled=False)""",
        )
    if name.startswith("zero-dot-"):
        shape = name.removeprefix("zero-dot-")
        return Recipe(SIGNED, f'drawing = esch.draw(signed, shape={shape!r}, pos="teal", neg="orange", dot=True)')
    if name == "wave-dot":
        return Recipe(WAVE, "drawing = esch.draw(wave, time_axis=-1, fps=12, dot=True)")
    if name.startswith("wave-"):
        return Recipe(WAVE, f"drawing = esch.draw(wave, time_axis=-1, shape={name.removeprefix('wave-')!r}, fps=12)")
    recipes = {
        "voronoi-hide-outer-static": Recipe(
            "values = np.ones((3, 3))", 'drawing = esch.draw(values, shape="voronoi", hide_outer=True, dots=True, pos="teal")'
        ),
        "voronoi-sign-edges": Recipe(
            "values = np.array([[[1, -1], [1, -1], [-1, 1], [-1, 1]]])",
            'drawing = esch.draw(values, "h w t", shape="voronoi", pos="black", neg="white",\n                    edge="black", edge_pos="white", color_by_magnitude=False)',
        ),
        "split-panels": Recipe(
            "arr = np.linspace(-1, 1, 24).reshape(3, 2, 4)", 'drawing = esch.draw(arr, "n h w", pos="teal", neg="orange")'
        ),
        "named-row-col": Recipe(BLOCKS, 'drawing = esch.draw(blocks, "row col h w")'),
        "interleaved-columns": Recipe(
            "interleaved = np.linspace(-1, 1, 72).reshape(8, 3, 3)",
            'drawing = esch.draw(interleaved, "(t col) ... -> t col ...", col=2, fps=4)',
        ),
        "rearrange-tiles": Recipe(BLOCKS, 'drawing = esch.draw(blocks, "b c h w -> (b h) (c w)")'),
        "rearrange-split": Recipe(
            "arr = np.linspace(-1, 1, 24).reshape(6, 4)", 'drawing = esch.draw(arr, "(batch h) w -> batch h w", h=2)'
        ),
        "rearrange-time": Recipe(BLOCKS, 'drawing = esch.draw(blocks, "b h w t -> t b w h", fps=5)'),
        "zero": Recipe("arr = np.zeros((2, 4))", "drawing = esch.draw(arr)"),
        "singleton": Recipe("arr = np.ones((1, 1))", "drawing = esch.draw(arr)"),
        "large-value": Recipe("arr = np.array(4)", "drawing = esch.draw(arr)"),
        "time-first": Recipe(WAVE, 'drawing = esch.draw(np.moveaxis(wave, -1, 0), "t h w", fps=12)'),
        "t": Recipe("time = np.arange(24) * 2 * np.pi / 24", 'drawing = esch.draw(np.sin(time), "t", fps=12)'),
        "h-t": Recipe(WAVE, 'drawing = esch.draw(wave[:, 0], "h t", fps=12)'),
        "w-t": Recipe(WAVE, 'drawing = esch.draw(wave[0], "w t", fps=12)'),
        "single-frame": Recipe(SIGNED, 'drawing = esch.draw(signed[..., None], "h w t")'),
        "named-panels": Recipe(BATCH, 'drawing = esch.draw("n c h w t", batch, fps=12)'),
        "ellipsis": Recipe(BATCH, 'drawing = esch.draw(batch, "... h w t", fps=12)'),
        "partial-row": Recipe(SIGNED, "drawing = esch.draw(np.stack([signed] * 5), columns=3, debug=True)"),
        "layers-and-ticks": Recipe(
            SIGNED,
            """drawing = esch.Drawing(h=2, w=5)
esch.grid_fn(drawing, np.ones((1, 2, 5)), shape="square", pos="#e2e8f0", ticks=True)
esch.grid_fn(drawing, np.abs(signed)[None] * 0.4, shape="circle", pos="#155e75", normalize=False)""",
        ),
    }
    return recipes[name]
