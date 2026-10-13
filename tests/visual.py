"""Deterministic visual fixtures and a self-contained manual review gallery."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np

import esch

from . import recipes
from .adaptive_clips import adaptive_example
from .cell_fields import cell_example
from .compact_clips import optimized_example
from .distance_fields import distance_example
from .halfplanes import clipping_example
from .recipes import recipe_for


@dataclass
class Case:
    name: str
    description: str
    drawing: esch.Drawing
    glyphs: int
    animated: bool = False
    split: tuple[str, int] | None = None
    paused: bool = False


def cases():
    rng = np.random.default_rng(64)
    waypoints = rng.uniform([0, 0], [6, 10], size=(64, 16, 2))
    positions = (np.roll(waypoints, 1, axis=1) + 2 * waypoints + np.roll(waypoints, -1, axis=1))[None] / 4
    teams = np.array([[1] * 32 + [-1] * 32])
    yield Case(
        "voronoi-adaptive-experiment",
        "Experimental kinetic vector clipping: shared moving half-planes, event-driven variable-length cell chunks, "
        "and white edges between every cell. No SVG scripts or raster filters. Starts paused; inspect chunk transitions.",
        adaptive_example(positions[0], ["#e53935"] * 32 + ["#1e88e5"] * 32, h=7, w=11),
        64,
        True,
        paused=True,
    )
    yield Case(
        "voronoi-compact-experiment",
        "Experimental optimized clip slots: automatic variable chunks and coordinate precision, "
        "with a sampled 0.005-grid-unit error budget. Smooth native SVG animation and white internal edges. "
        "Starts paused; inspect transitions and the loop.",
        optimized_example(positions[0], ["#e53935"] * 32 + ["#1e88e5"] * 32, h=7, w=11)[0],
        64,
        True,
        paused=True,
    )
    yield Case(
        "voronoi-cell-experiment",
        "Experimental individual distance-field masks: white edges between all cells, including teammates. "
        "Approximate rasterized boundaries, no outer frame or embedded scripts. Slow at large sizes; starts paused.",
        cell_example(positions[0], teams[0], h=7, w=11),
        65,
        True,
        paused=True,
    )
    for boundaries in (False, True):
        yield Case(
            "voronoi-distance-experiment" + ("-boundaries" if boundaries else ""),
            "Experimental rasterized SVG filters: 64 agents, animated gradient centers, no SVG scripts. "
            "Team territories merge; individual teammate cells are not shown. Borders are approximate. "
            + ("White front lines separate teams, without an outer frame." if boundaries else "No boundary strokes."),
            distance_example(positions[0], teams[0], h=7, w=11, boundaries=boundaries),
            64,
            True,
        )
    yield Case(
        "voronoi-clipping-experiment",
        "Experimental: 64 agents, native linear animation of half-plane clips. No intermediate cell frames or SVG scripts. "
        "Compare against territorial-control; check seams, topology changes and playback speed. Not a public esch API.",
        clipping_example(positions[0], ["#e53935"] * 32 + ["#1e88e5"] * 32, h=7, w=11),
        64,
        True,
    )
    for name, positive, negative, edge in (
        ("territorial-control", "#e53935", "#1e88e5", "white"),
        ("territorial-control-black-white", "black", "white", "black"),
        ("territorial-control-inner", "#e53935", "#1e88e5", "white"),
        ("territorial-control-inner-black-white", "black", "white", "black"),
    ):
        yield Case(
            name,
            f"64 units, 32 per team, follow independent smoothed random waypoints. 1 fps, 16-second loop; {edge} internal edges, no dots."
            + (" White separators distinguish neighboring black cells." if name.endswith("black-white") else "")
            + (" Boundary-touching cells are hidden; their territory stays blank." if "inner" in name else ""),
            esch.voronoi_fn(
                esch.Drawing(7, 11),
                positions,
                teams,
                fps=1,
                pos=positive,
                neg=negative,
                edge=edge,
                edge_pos="white" if name.endswith("black-white") else None,
                color_by_magnitude=False,
                dots=False,
                edges=True,
                hide_outer="inner" in name,
            ),
            64,
            True,
        )
    rng = np.random.default_rng(2026)
    yield Case(
        "voronoi-hide-outer-static",
        "Only the central cell and its dot remain; the eight boundary cells are hidden without reallocating territory.",
        esch.draw(np.ones((3, 3)), shape="voronoi", hide_outer=True, dots=True, pos="teal"),
        18,
    )
    yield Case(
        "voronoi-sign-edges",
        "White edges separate two positive black cells; other edges stay black. Fixed sites, changing signs.",
        esch.draw(
            np.array([[[1, -1], [1, -1], [-1, 1], [-1, 1]]]),
            "h w t",
            shape="voronoi",
            pos="black",
            neg="white",
            edge="black",
            edge_pos="white",
            color_by_magnitude=False,
        ),
        4,
        True,
    )
    sites = np.moveaxis(np.indices((6, 9)), 0, -1) + rng.uniform(-0.35, 0.35, (6, 9, 2))
    time = np.arange(32) * 2 * np.pi / 32
    wave = np.sin(sites[..., 0, None] * 0.9 + sites[..., 1, None] * 0.65 - time)
    centers = np.moveaxis(np.indices((3, 4)), 0, -1)
    phase = np.arange(12).reshape(3, 4, 1) * 0.7 + time
    moving = centers[..., None, :] + 0.35 * np.stack([np.sin(phase), np.cos(phase)], axis=-1)
    yield Case(
        "voronoi-moving",
        "Moving sites change region areas and edges; colors vary independently. The loop closes smoothly.",
        esch.draw(np.sin(phase), "h w t", shape="voronoi", positions=moving, fps=16, pos="teal", neg="orange"),
        12,
        True,
    )
    for animated in (False, True):
        yield Case(
            f"voronoi-dots-{'moving' if animated else 'static'}",
            "White site dots with sign-colored outlines. Check cell boundaries for spikes and dots for continuous motion.",
            esch.draw(
                np.sin(phase) if animated else np.sin(phase[..., 0]),
                "h w t" if animated else "h w",
                shape="voronoi",
                positions=moving if animated else moving[..., 0, :],
                dots=True,
                fps=16,
                pos="teal",
                neg="orange",
            ),
            24,
            animated,
        )
    for animated in (False, True):
        for dots in (False, True):
            yield Case(
                f"voronoi-monochrome-{'moving' if animated else 'static'}{'-dots' if dots else ''}",
                "Black internal boundaries with transparent cells and no outer frame. Optional dots have diameter twice the edge width, excluding outlines.",
                esch.draw(
                    np.sin(phase) if animated else np.sin(phase[..., 0]),
                    "h w t" if animated else "h w",
                    shape="voronoi",
                    positions=moving if animated else moving[..., 0, :],
                    monochrome=True,
                    dots=dots,
                    fps=16,
                ),
                12 if dots else 0,
                animated,
            )
    for edges in (False, True):
        yield Case(
            f"voronoi-monochrome-moving-edges-{edges}",
            "Compare optional black cell boundaries; dots are enabled independently.",
            esch.draw(
                np.sin(phase), "h w t", shape="voronoi", positions=moving, monochrome=True, dots=True, edges=edges, fps=16
            ),
            12,
            True,
        )
    yield Case(
        "voronoi-colored-no-edges",
        "Colored moving cells with all boundary strokes disabled.",
        esch.draw(np.sin(phase), "h w t", shape="voronoi", positions=moving, edges=False, pos="teal", neg="orange", fps=16),
        12,
        True,
    )
    for shape in ("circle", "square"):
        yield Case(
            f"mesh-moving-{shape}",
            "The same moving sites drawn as points, with signed size and color animation.",
            esch.mesh_fn(
                esch.Drawing(3, 4),
                moving.reshape(1, 12, 32, 2),
                np.sin(phase).reshape(1, 12, 32),
                shape=shape,
                fps=16,
                pos="teal",
                neg="orange",
                color_by_magnitude=True,
            ),
            12,
            True,
        )
    for animated in (False, True):
        yield Case(
            f"voronoi-{'animated' if animated else 'static'}",
            "Fixed irregular sites. Regions tile the panel; values fade blue/ orange through white. Geometry stays fixed.",
            esch.draw(
                wave if animated else wave[..., 0],
                pattern="h w t" if animated else "h w",
                shape="voronoi",
                positions=sites,
                fps=16,
                pos="#0074d9",
                neg="#ff8000",
            ),
            54,
            animated,
        )
    yield Case(
        "voronoi-panels",
        "Two synchronized Voronoi panels with shared positions and global color scaling.",
        esch.draw(np.stack([wave, -wave]), "col h w t", shape="voronoi", positions=sites, fps=16, pos="#0074d9", neg="#ff8000"),
        108,
        True,
    )
    for name, values in (("singleton", np.array(1)), ("collinear", np.array([-1, -0.3, 0.4, 1]))):
        yield Case(
            f"voronoi-{name}",
            "Degenerate site layout: bounded, nonoverlapping regions still cover the panel.",
            esch.draw(values, shape="voronoi", pos="teal", neg="orange"),
            values.size,
        )
    yield Case(
        "split-panels",
        "Saving split-panels_{n}.svg exports each of these three panels, without rescaling.",
        esch.draw(np.linspace(-1, 1, 24).reshape(3, 2, 4), "n h w", pos="teal", neg="orange"),
        24,
        split=("n", 3),
    )
    signed = np.array([[-1, -0.5, 0, 0.5, 1], [0.2, 0.4, 0.6, 0.8, 1]])
    for animated in (False, True):
        yield Case(
            f"rgb-array-colors-{'animated' if animated else 'static'}",
            "RGB arrays: positive [0, 116, 217], negative [255, 128, 0]. Matching outlines; magnitude fades toward white.",
            esch.draw(
                np.stack([signed, -signed], axis=-1) if animated else signed,
                pattern="h w t" if animated else "h w",
                shape="circle",
                pos=np.array([0, 116, 217]),
                neg=np.array([255, 128, 0]),
                color_by_magnitude=True,
            ),
            10,
            animated,
        )
    for shape in ("square", "circle"):
        yield Case(
            f"normalized-areas-{shape}",
            "Values -100, -25, 0, 25, 100. Small glyphs have one quarter the area; colors also encode magnitude.",
            esch.draw([-100, -25, 0, 25, 100], shape=shape, pos="teal", neg="orange", color_by_magnitude=True),
            5,
        )
    blocks = np.linspace(-1, 1, 120).reshape(2, 3, 4, 5)
    for gap in (2, 4, 5, 8):
        yield Case(
            f"panel-gap-{gap}",
            f"Two rows × three columns of 4 × 5 arrays. Horizontal gap 5/{gap}, vertical gap 4/{gap}; blue panel bounds.",
            esch.draw(blocks, pattern="row col h w", gap=gap, debug=True),
            120,
        )
    yield Case(
        "named-row-col",
        "row col h w: exactly two panel rows and three panel columns, each 4 × 5.",
        esch.draw(blocks, pattern="row col h w"),
        120,
    )
    interleaved = np.linspace(-1, 1, 72).reshape(8, 3, 3)
    yield Case(
        "interleaved-columns",
        "(t col) ... -> t col ..., col=2: two 3 × 3 arrays side by side, four shared frames.",
        esch.draw(interleaved, pattern="(t col) ... -> t col ...", col=2, fps=4),
        18,
        True,
    )
    yield Case(
        "rearrange-tiles",
        "b c h w -> (b h) (c w): combine panels into a single 8 × 15 grid.",
        esch.draw(blocks, pattern="b c h w -> (b h) (c w)"),
        120,
    )
    yield Case(
        "rearrange-split",
        "(batch h) w -> batch h w, h=2: split a packed axis into three panels.",
        esch.draw(np.linspace(-1, 1, 24).reshape(6, 4), pattern="(batch h) w -> batch h w", h=2),
        24,
    )
    yield Case(
        "rearrange-time",
        "b h w t -> t b w h: time first, spatial axes transposed, two animated panels.",
        esch.draw(blocks, pattern="b h w t -> t b w h", fps=5),
        24,
        True,
    )
    for shape in ("square", "circle"):
        for animated in (False, True):
            values = signed[..., None] * np.cos(np.arange(24) * 2 * np.pi / 24) if animated else signed
            yield Case(
                f"magnitude-colors-{shape}-{'animated' if animated else 'static'}",
                "Magnitude blends white toward teal/orange using one scale across the full array. Outlines fade too.",
                esch.draw(
                    values,
                    pattern="h w t" if animated else "h w",
                    shape=shape,
                    fps=12,
                    pos="#155e75",
                    neg="#c2410c",
                    color_by_magnitude=True,
                ),
                10,
                animated,
            )
    for shape in ("square", "circle"):
        for animated in (False, True):
            values = np.stack([signed, -signed], axis=-1) if animated else signed
            yield Case(
                f"custom-colors-{shape}-{'animated' if animated else 'static'}",
                "Positive teal, negative orange, with matching outlines. Equal-magnitude sign flips shrink to zero, then regrow.",
                esch.draw(
                    values,
                    pattern="h w t" if animated else "h w",
                    shape=shape,
                    pos="#155e75",
                    neg="#c2410c",
                ),
                10,
                animated,
            )
    for shape in ("square", "circle"):
        yield Case(
            f"signed-{shape}",
            "2 × 5. Negative values white, positive black; zero invisible. Compare equal magnitudes.",
            esch.draw(signed, shape=shape),
            10,
        )
    for rank in range(7):
        dims = (2,) * max(0, rank - 2) + (3, 5)[-min(rank, 2) :] if rank else ()
        arr = np.linspace(0.15, 1, int(np.prod(dims))).reshape(dims)
        yield Case(
            f"rank-{rank}",
            f"Shape {dims}. Last two axes are the grid; leading axes become panels in array order.",
            esch.draw(arr),
            arr.size,
        )
    for name, arr in (("zero", np.zeros((2, 4))), ("singleton", np.ones((1, 1))), ("large-value", np.array(4))):
        yield Case(
            name, f"Shape {arr.shape}. Check nonzero canvas dimensions and unclipped boundaries.", esch.draw(arr), arr.size
        )

    time = np.arange(24) * 2 * np.pi / 24
    row, column = np.mgrid[:4, :8]
    wave = np.sin(row[..., None] * 0.8 + column[..., None] * 0.5 - time)
    for shape in ("square", "circle"):
        yield Case(
            f"wave-{shape}",
            "4 × 8 × 24, time last, 12 fps. Check smooth size changes, sign flips, and the two-second loop.",
            esch.draw(wave, time_axis=-1, shape=shape, fps=12),
            32,
            True,
        )
    for shape in ("square", "circle"):
        yield Case(
            f"zero-dot-{shape}",
            "dot=True: the zero in the middle of the top row is an outline-sized glyph whose fill cannot show.",
            esch.draw(signed, shape=shape, pos="teal", neg="orange", dot=True),
            10,
        )
    namespace = {"np": np}
    exec(recipes.MORPH, namespace)  # noqa: S102 — the recipe's own setup, so both stay identical
    morph = esch.Drawing(4, 4)
    esch.line_fn(morph, namespace["points"], fps=12)
    esch.mesh_fn(morph, namespace["points"], np.full((1, 16), 0.1), shape="circle", fps=12, normalize=False, scaled=False)
    yield Case(
        "line-morph",
        "16 points glide from a ring onto a serpentine path and back. The line keeps their order; dots use draw sizing.",
        morph,
        16,
        True,
    )
    yield Case(
        "wave-dot",
        "Square wave with dot=True: glyphs shrink to an outline-sized square at each sign flip instead of vanishing.",
        esch.draw(wave, time_axis=-1, fps=12, dot=True),
        32,
        True,
    )
    yield Case(
        "time-first",
        "Same square wave with time first. Should match wave-square at every playback position.",
        esch.draw(np.moveaxis(wave, -1, 0), pattern="t h w", fps=12),
        32,
        True,
    )
    for pattern, arr in (("t", np.sin(time)), ("h t", wave[:, 0]), ("w t", wave[0])):
        yield Case(
            pattern.replace(" ", "-"),
            f"Pattern '{pattern}'. Missing spatial dimensions become singleton axes.",
            esch.draw(arr, pattern=pattern, fps=12),
            arr.size // 24,
            True,
        )
    yield Case(
        "single-frame",
        "One time frame: identical to signed-square, with no animation.",
        esch.draw(signed[..., None], pattern="h w t"),
        10,
    )
    batch = np.stack([wave * factor for factor in (1, 0.7, 0.4, 0.2, 0.9, 0.5)]).reshape(2, 3, 4, 8, 24)
    yield Case(
        "named-panels",
        "Pattern 'n c h w t': two rows, three columns. No duplicated panels.",
        esch.draw("n c h w t", batch, fps=12),
        192,
        True,
    )
    yield Case(
        "ellipsis",
        "Pattern '... h w t': same six panels as named-panels, in the same order.",
        esch.draw(batch, pattern="... h w t", fps=12),
        192,
        True,
    )
    yield Case(
        "partial-row",
        "Five rectangular panels in three columns. Last slot empty; red canvas, blue panel boundaries.",
        esch.draw(np.stack([signed] * 5), columns=3, debug=True),
        50,
    )

    drawing = esch.Drawing(h=2, w=5)
    esch.grid_fn(drawing, np.ones((1, 2, 5)), shape="square", col="#e2e8f0", ticks=True)
    esch.grid_fn(drawing, np.abs(signed)[None] * 0.4, shape="circle", col="#155e75", normalize=False)
    yield Case(
        "layers-and-ticks", "Gray squares with blue circles. Ticks on all four sides of a rectangular grid.", drawing, 20
    )

    angle = np.arange(12) * 2 * np.pi / 12
    positions = np.stack([2 + 1.5 * np.sin(angle), 3 + 2.5 * np.cos(angle)], axis=-1)[None]
    for shape in ("square", "circle"):
        for animated in (False, True):
            drawing = esch.Drawing(h=5, w=7)
            values = np.cos(angle[:, None] - time)[None] if animated else np.cos(angle)[None]
            esch.mesh_fn(drawing, positions, values, shape=shape, col="#155e75", fps=12)
            yield Case(
                f"mesh-{shape}-{'animated' if animated else 'static'}",
                "Fixed elliptical point positions in (row, column) order. Check size, color, and clipping.",
                drawing,
                12,
                animated,
            )


def recipe_html(name, filename):
    recipe = recipe_for(name)
    plot = f"{recipe.plot}\ndrawing.save({filename!r})"
    setup = f"import numpy as np\nimport esch\n\n{recipe.setup}"
    return (
        '<section class="recipe" aria-label="Python example">'
        f"<h3>Plot code</h3><pre><code>{escape(plot)}</code></pre>"
        f"<details><summary>Imports and example data</summary><pre><code>{escape(setup)}</code></pre></details>"
        '<button type="button" data-action="copy">Copy complete example</button>'
        '<span class="copy-status" role="status"></span>'
        f'<textarea class="copy-source" aria-label="Complete Python example" readonly hidden>{escape(recipe.source(filename))}</textarea>'
        "</section>"
    )


def inline_svg(svg, prefix):
    """Scope definitions so inline gallery examples cannot share filter IDs."""
    identifiers = {node.get("id"): f"{prefix}-{node.get('id')}" for node in ET.fromstring(svg).iter() if node.get("id")}
    return re.sub(
        r'(?P<reference>\bid="|href="#|url\(#)(?P<identifier>[^"\)]+)',
        lambda match: match["reference"] + identifiers.get(match["identifier"], match["identifier"]),
        svg,
    )


def build_gallery(output="visual"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cards = []
    experimental_cards = []
    fixtures = list(cases())
    for case in fixtures:
        case.drawing.save(output / f"{case.name}.svg")
        svg = case.drawing.tostring()
        separate = "experiment" in case.name or (
            case.name.startswith(("voronoi-", "territorial-control")) and len(svg.encode()) > 1_000_000
        )
        destination = experimental_cards if separate else cards
        root = ET.fromstring(svg)
        animation = root.find(".//{http://www.w3.org/2000/svg}animate")
        duration = animation.get("dur")[:-1] if animation is not None else "0"
        controls = (
            (
                f'<div class="controls"><button type="button" data-action="pause">Pause</button>'
                f'<button type="button" data-action="restart">Restart</button>'
                f'<label>Time <input type="range" min="0" max="{duration}" step="0.01" value="0" '
                f'aria-label="Playback time for {escape(case.name)}"></label></div>'
            )
            if case.animated
            else '<p class="static">Static</p>'
        )
        destination.append(
            f'<article data-paused="{str(case.paused).lower()}"><h2><a href="{case.name}.svg">{escape(case.name)}</a></h2>'
            f'<p>{escape(case.description)}</p><div class="plot">{inline_svg(svg, case.name)}</div>{controls}'
            f"{recipe_html(case.name, f'{case.name}.svg')}</article>"
        )
        if case.split is not None:
            axis, count = case.split
            case.drawing.save(output / f"{case.name}_{{{axis}}}.svg")
            for index in range(count):
                name = f"{case.name}_{index}.svg"
                exported = (output / name).read_text().split("?>", 1)[-1].lstrip()
                destination.append(
                    f'<article><h2><a href="{name}">{name}</a></h2>'
                    f"<p>Exported panel {axis}={index}; compare its size and color with the source above.</p>"
                    f'<div class="plot">{inline_svg(exported, name)}</div><p class="static">Static</p>'
                    f"{recipe_html(case.name, f'{case.name}_{{{axis}}}.svg')}</article>"
                )
    page = (
        """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>esch · visual regression gallery</title>
<style>
body { font: 16px/1.5 system-ui, sans-serif; margin: 2rem auto; padding: 0 1.5rem; max-width: 1400px; color: #17212b; background: #f5f5f3; }
h1 { margin-bottom: .25rem; } h2 { font-size: 1rem; margin: 0; } a { color: #155e75; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr)); gap: 1rem; }
article { background: white; padding: 1.25rem; border: 1px solid #ddd; border-radius: 8px; }
article p { min-height: 4.5em; font-size: .85rem; color: #4b5563; }
.plot { height: 260px; background: white; } .plot svg { width: 100%; height: 100%; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin-top: 1rem; }
button { padding: .35rem .7rem; cursor: pointer; } input { max-width: 140px; vertical-align: middle; }
article .static { min-height: 0; } header { margin-bottom: 2rem; max-width: 850px; }
.recipe { margin-top: 1rem; border-top: 1px solid #ddd; padding-top: .5rem; }
.recipe h3 { font-size: .9rem; margin: .5rem 0; }
pre { font-size: .78rem; line-height: 1.5; padding: .75rem; background: #f5f5f3; overflow-x: auto; }
article { min-width: 0; } summary { cursor: pointer; font-size: .85rem; }
.recipe button { margin-top: .75rem; } .copy-status { font-size: .8rem; margin-left: .5rem; }
.copy-source { box-sizing: border-box; width: 100%; height: 14rem; margin-top: .5rem; }
</style><header><h1>esch visual regression gallery</h1>
__GALLERY_NAVIGATION__
<p>Deterministic examples regenerated by the test suite. Automated checks verify SVG structure;
visual approval is manual. Inspect proportions, spacing, clipping, sign colors, layers, and animation loops.
Open an SVG using its title. Pause and scrub animations to compare frames.</p>
<p>Each plot includes its Python call. Expand the data setup or copy a complete runnable example.</p>
<p>When changing rendering, compare this gallery with a previously saved copy and record your approval in the review.</p>
</header><main class="grid">"""
        + "__GALLERY_CARDS__"
        + """</main>
<script>
document.querySelectorAll('[data-action="copy"]').forEach(button => {
  button.addEventListener('click', async () => {
    const recipe = button.closest('.recipe');
    const source = recipe.querySelector('.copy-source');
    const status = recipe.querySelector('.copy-status');
    try {
      await navigator.clipboard.writeText(source.value);
      status.textContent = 'Copied';
    } catch {
      source.hidden = false;
      source.focus(); source.select();
      status.textContent = 'Selected — press Ctrl+C or ⌘C to copy';
    }
  });
});
document.querySelectorAll('article').forEach(card => {
  const svg = card.querySelector('svg');
  const pause = card.querySelector('[data-action="pause"]');
  if (!pause) return;
  const slider = card.querySelector('input');
  if (card.dataset.paused === 'true') {
    svg.pauseAnimations(); svg.setCurrentTime(0); pause.textContent = 'Play';
  }
  pause.addEventListener('click', () => {
    if (svg.animationsPaused()) { svg.unpauseAnimations(); pause.textContent = 'Pause'; }
    else { svg.pauseAnimations(); pause.textContent = 'Play'; }
  });
  card.querySelector('[data-action="restart"]').addEventListener('click', () => {
    svg.setCurrentTime(0); svg.unpauseAnimations(); pause.textContent = 'Pause'; slider.value = 0;
  });
  slider.addEventListener('input', () => {
    svg.pauseAnimations(); svg.setCurrentTime(Number(slider.value)); pause.textContent = 'Play';
  });
});
</script></html>"""
    )
    (output / "index.html").write_text(
        page.replace("__GALLERY_CARDS__", "\n".join(cards)).replace(
            "__GALLERY_NAVIGATION__", '<p><a href="experimental.html">Large and experimental Voronoi examples →</a></p>'
        )
    )
    (output / "experimental.html").write_text(
        page.replace("esch visual regression gallery", "esch · large and experimental Voronoi")
        .replace("esch · visual regression gallery", "esch · large and experimental Voronoi")
        .replace("__GALLERY_CARDS__", "\n".join(experimental_cards))
        .replace(
            "__GALLERY_NAVIGATION__", '<p><a href="index.html">← Main gallery</a>. Large files and rendering experiments.</p>'
        )
    )
    return fixtures, output / "index.html"
