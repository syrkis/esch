# esch

Turn arrays of any rank into clean, optionally animated SVGs.

For copyable examples of every plot, open [the visual index](visual/index.html).
Each card shows its plotting call, expandable data setup, and a complete
example you can copy and run. Tests check that the recipes reproduce the SVGs.

```python
import numpy as np
import esch

x = np.linspace(-1, 1, 60).reshape(3, 4, 5)
plot = esch.draw(x)
plot.save("array.svg")
plot  # Native SVG display in a notebook.
```

Without a pattern, the last two axes are grid rows and columns; all leading
axes become panels in array order. A vector is a single row, and a scalar is a
single glyph. Panels wrap into a roughly square grid; use `columns=4` to choose
the number of panel columns.

Time is always explicit. Array sizes never determine whether a plot animates:

```python
time = np.linspace(0, 2 * np.pi, 48, endpoint=False)
movie = x[..., None] * np.cos(time)
esch.draw(movie, time_axis=-1, fps=24).save("movie.svg")
# Equivalent:
esch.draw(movie, pattern="... h w t", fps=24).save("movie.svg")
```

Named patterns follow einops conventions:

| Axis | Meaning |
| --- | --- |
| `h` | Grid rows |
| `w` | Grid columns |
| `t` | Animation frames |
| `row`, `col` | Panel rows and columns |
| Any other name | Panel axis |
| `...` | Zero or more panel axes |

Missing spatial or time axes are inserted as singleton dimensions. Axis order
is arbitrary: `"t layer w h"` works. Names must be unique. Use either a pattern
or `time_axis`; supplying both is an error. An explicit time axis of length one
renders exactly like a static plot.

For compatibility, `n` and `c`, when they are the only panel axes, map to panel
rows and columns. Other panel axes flatten in their order in the pattern.
`esch.draw("n c h w t", movie)` remains supported.

Name panel axes `row` and `col` to specify layout directly, without `columns`:

```python
# Input shape (2*T, N, N), alternating between two arrays at each timestep:
esch.draw(scores, pattern="(t col) h w -> col h w t", col=2, fps=4)
# Equivalently, leave the inner array dimensions unnamed:
esch.draw(scores, pattern="(t col) ... -> t col ...", col=2, fps=4)
```

`row` and `col` set panel placement; `h` and `w` set the inner grid. Missing
`row` or `col` means one panel row or column. In arrow patterns using these
panel names, named `h`/`w` keep their roles regardless of output order;
otherwise, unnamed trailing axes supply the inner grid. Combine additional
panel dimensions into `row` or `col`; conflicting `columns` is rejected.
Numeric `col=2` supplies the einops axis length when the pattern names `col`.
The legacy string `col="navy"` still sets color; prefer `pos`.

For everyday use, keep the pattern as the second argument and reuse a small
dictionary of style options:

```python
style = dict(shape="circle", pos="teal", neg="orange", color_by_magnitude=True)
esch.draw(scores, "(t col) h w -> col h w t", col=2, fps=4, **style).save("scores.svg")
```

Use `row`/`col` for panel layout, `pos`/`neg` for colors, and `dots`/`edges`
for Voronoi details. An axis named in a save path, such as `"scores_{col}.svg"`,
exports separate files without another drawing call.

Full einops rearrangements work directly in `pattern`, including grouping,
splitting, permutations, singleton axes, and ellipses:

```python
weights = np.ones((2, 3, 4, 5))
esch.draw(weights, pattern="model head h w -> (model head) h w")
esch.draw(weights, pattern="model head h w -> (model h) (head w)")

# Supply lengths to split a packed input axis, just like einops.rearrange:
packed = np.ones((6, 5))
esch.draw(packed, pattern="(batch h) w -> batch h w", h=2)
esch.draw("(batch h) w -> batch h w", packed, h=2)  # Pattern-first also works.

# Shorthand retains named axis roles and can now split packed axes too:
esch.draw(weights, pattern="model head h w")
esch.draw(packed, pattern="(batch h) w", h=2)
```

With an arrow and no explicit `row`/`col`, the **output order** determines layout: the last two non-time
axes are rows and columns; earlier axes are panels. Thus `h w -> w h`
actually transposes the plot. The shorthand's special `n`/`c` layout does not
apply to arrow patterns.

An output axis containing `t` is time, including a group such as `(step t)`.
Alternatively, use `time_axis` to identify an output axis when no output axis
contains `t`. For example, `b h w t -> t b w h` transposes the grid while
moving time first. `fps` is reserved for playback speed and cannot be an axis
name. Axis names colliding with other drawing options can be supplied through
`axes_lengths={"shape": 2}`; ordinary lengths can be passed as keywords.
Duplicate length arguments and inconsistent shapes raise errors.

## Appearance and animation

`draw(array, *, pattern=None, time_axis=None, shape="square", positions=None, fps=1,
columns=None, gap=5, col="black", pos=None, neg=None,
color_by_magnitude=None, normalize=True, debug=False, axes_lengths=None, **axis_sizes)` returns a `Drawing`.

`color_by_magnitude=None` retains solid colors for squares/circles and enables
magnitude coloring for Voronoi regions. Explicit `True` or `False` overrides it.

```python
esch.draw(x, pos="teal", neg="orange")
esch.draw(movie, time_axis=-1, pos="#155e75", neg="#c2410c")
esch.draw(x, pos="teal", neg="orange", color_by_magnitude=True)
esch.draw(x, pos=np.array([0, 116, 217]), neg=[255, 128, 0])
```

Both color options also work with `grid_fn` and `mesh_fn`, for static and
animated squares and circles. For keyword-based mesh calls, pass coordinates
as `positions=...` so `pos` can specify the color.
`pos` overrides the legacy `col`
option; when omitted, `col` controls the positive color. An explicit
`neg` controls both negative fills and outlines. When omitted
(`None`), negative values retain white fills with positive-colored outlines.

Colors accept strings or one RGB triplet as a list, tuple, or NumPy array.
Integer channels use 0–255. Floating-point triplets entirely within 0–1
are scaled to 0–255; otherwise they use the 0–255 range and are rounded.
Arrays must contain exactly three finite, nonnegative channels at most 255.
RGB inputs work with animation and magnitude coloring without modifying them.

With `color_by_magnitude=True`, fills blend from white to the sign color in
proportion to `abs(value) / max(abs(array))`. The scale is shared across all
panels and frames in that call (each layer call gets its own scale). All-zero
arrays stay white. Outlines fade toward white too; glyph sizes are unchanged.
Use RGB triplets, CSS3 color names, or `#RGB`/`#RRGGBB` hex colors for this option. Colors
interpolate smoothly between animation frames when enabled; the default
mode keeps discrete sign-color changes. Colors remain opaque, so fading is
toward white even on a nonwhite background.

- By default, input is divided by its maximum absolute value, yielding
  `[-1, 1]` without shifting zero or changing signs. One scale covers every
  panel and animation frame; all-zero inputs stay zero. `grid_fn` and
  `mesh_fn` normalize each supplied layer independently as well.
- Squares have side length `sqrt(abs(value))`; circles have radius
  `sqrt(abs(value)) / 2.1`. Choose `shape="circle"` for circles.
  Thus area is proportional to magnitude within either shape: a quarter of
  the magnitude gives half the side/radius and a quarter of the area.
  This changes square sizing from the previous linear side-length mapping.
- Positive values use `pos` (or `col`); negative values use
  `neg` for both fill and outline when supplied. Without it,
  negative values are white with positive-colored outlines. Zero has zero
  size. The background is transparent.
- `dot=True` (in `draw`, `grid_fn` and `mesh_fn`) keeps zero visible: squares
  never shrink below a side of 0.04 grid units, and circles below a diameter
  of 0.04, the outline width. The outline then covers the whole glyph, so a
  zero shows as a small dot whose fill cannot be seen. Larger values are
  unchanged; animated glyphs shrink to the dot at sign flips.
- Use `normalize=False` for already scaled data or consistent scales across
  separate calls. Square-root sizing still applies. Unnormalized magnitudes
  above one can overlap neighboring cells; outer padding accommodates the
  largest glyph in `draw`.
- Animated geometry interpolates between frames, including the last-to-first
  transition. Sign changes shrink through zero before growing in the other
  color, even for equal opposite values. Without magnitude coloring, fill and
  outline switch at the zero crossing; with it, they fade through white.
  Circle radii and square sides interpolate linearly between these keyframes.
  A loop lasts
  `number_of_frames / fps` seconds and starts at the first input frame.
- To keep files small, coordinates are written to 0.001 grid units, keyframes
  that interpolation reproduces are dropped, and `keyTimes` are omitted when
  they match SMIL's default spacing.
- `fps` must be positive and finite. Empty arrays, complex values, NaN, and
  infinity are rejected. Input arrays are never modified.
- `debug=True` shows panel and canvas boundaries.

SVG animations play in browsers and native notebook SVG output. Static
rasterizers and PDF converters generally do not preserve animation.

## Layers and meshes

Use `gap` to set spacing between panels as a divisor of their dimensions:
`esch.draw(x, gap=4)` leaves one quarter of a panel's width between columns
and one quarter of its height between rows. `gap=2` gives more space;
`gap=8` gives less. The default is `gap=5` (one fifth of each panel dimension)
for both `draw` and `Drawing`. Numeric gaps must be positive and finite;
`gap=None` retains the old padding-based spacing.
Panel dimensions are cell counts (`w` and `h`), excluding outer padding.
Outer margins stay unchanged. With `normalize=False`, oversized glyphs or
ticks may extend into a small requested gap.

Use `Drawing`, `grid_fn`, and `mesh_fn` when you want to compose layers.
All rendering paths use the same glyphs and return the drawing for chaining.

```python
e = esch.Drawing(h=4, w=5, row=1, col=1)
esch.grid_fn(e, np.ones((1, 4, 5)), shape="square", col="lightgray", ticks=True)
esch.mesh_fn(e, [[[1.2, 2.4], [2.5, 3.0]]], [[0.8, -0.5]], col="navy")
e.save("layers.svg")
```

Grid values have shape `(panel, row, column[, time])`. Mesh positions have
shape `(panel, point, 2)`, with coordinates in **(row, column)** order;
mesh values have shape `(panel, point[, time])`. Moving positions have shape
`(panel, point, time, 2)` and interpolate linearly, including the loop back
to the first frame. Static values or positions broadcast over time; otherwise
their time lengths must match. Mesh values retain the historical scaling by
`1 / (sqrt(point_count) * 2.1)` before glyph sizing; pass `scaled=False` to
size mesh glyphs as `draw` does (square side `sqrt(|value|)` grid units).

`esch.line_fn(e, positions, col="black", fps=1, closed=False)` joins points in
order with a thin line. Positions have the mesh shape, `(panel, point[, time], 2)`,
and vertices move exactly like mesh glyphs, so a line and a mesh on the same
positions stay together. `closed=True` also joins the last point to the first.
Draw the line before the mesh to keep glyphs on top:

```python
e = esch.Drawing(h=4, w=4)
esch.line_fn(e, points, fps=12)  # points: (1, 16, time, 2)
esch.mesh_fn(e, points, np.full((1, 16), 0.1), shape="circle", fps=12, normalize=False, scaled=False)
```

`Drawing.h` and `Drawing.w` count grid cells; `row` and `col` count panels.
`pad` is the margin between an outer cell center and the panel edge (default
1, minimum 0.55). When composing layers directly, choose padding large enough
for your values and keep mesh positions within the panel.

`Drawing.save(path)`, `Drawing.saveas(path)`, and `esch.save(drawing, path)`
write SVG. `drawing.dwg` exposes the underlying svgwrite drawing.
`esch.show(drawing)` displays native SVG in a notebook.

## Voronoi regions

`shape="voronoi"` assigns each value a nearest-site region clipped to its
panel. Sites can move while values animate the colors; region area is
determined by site positions, not value magnitude. This mode uses Shapely/GEOS
for the geometry and shares the existing normalization, RGB colors, animation
timing, panel layout, and split-file saving. Regions have thin white borders.

```python
rng = np.random.default_rng(0)
sites = np.moveaxis(np.indices((6, 9)), 0, -1)
sites = sites + rng.uniform(-0.35, 0.35, sites.shape)
time = np.arange(32) * 2 * np.pi / 32
values = np.sin(sites[..., 0, None] + sites[..., 1, None] - time)

esch.draw(
    values, "h w t", shape="voronoi", positions=sites,
    fps=16, pos=[0, 116, 217], neg=[255, 128, 0],
).save("voronoi.svg")
```

Without `positions`, sites are the regular cell centers. Supply `(h, w, 2)`
positions to share one layout across all panels, or `(panels, h, w, 2)` for
different layouts. Coordinates are **(row, column)** and match the final grid
after rearrangement; panels follow the rendered panel order. Sites must be
distinct within a panel and inside `[-0.5, h-0.5] × [-0.5, w-0.5]`.
For moving sites, supply `(h, w, time, 2)` or `(panels, h, w, time, 2)`
matching the values' explicit time axis. For example:

If a shared moving shape also matches `(panels, h, w, 2)`, the existing
static per-panel interpretation takes precedence. Include the panel axis
explicitly to disambiguate moving coordinates in that case.

```python
phase = np.arange(54).reshape(6, 9, 1) + time
centers = np.moveaxis(np.indices((6, 9)), 0, -1)
moving = centers[..., None, :] + 0.3 * np.stack([np.sin(phase), np.cos(phase)], axis=-1)
esch.draw(values, "h w t", shape="voronoi", positions=moving, fps=16,
          pos="teal", neg="orange").save("moving.svg")
```

Sites travel linearly between frames and back to the first frame. Geometry
is recomputed at least 15 times per second (or at the input fps if higher).
Cell outlines then morph linearly between keyframes that stay within 0.02 grid
units of the true cells at those samples. Where a morph misses the true cell
halfway between samples, as when cells gain or lose edges, the interval is
halved up to four times, so neighboring cells stay continuous. Fast, chaotic
motion still needs many keyframes. Avoid coincident sites; duplicates at input
or sampled positions raise `ValueError`.

Add `dots=True` to `draw(..., shape="voronoi")` or `voronoi_fn(...)` to show
the generating site in each cell. Dots have a white center, fixed radius of
0.04 grid units (diameter is twice the 0.04 cell-edge width, excluding the
dot outline), and an outline colored by `pos` or `neg` (falling back to
the positive color when `neg` is omitted). They move with the sites and keep
their full outline color regardless of magnitude. Dots are off by default.

Use `monochrome=True` for unfilled cells with black internal boundaries,
independent of values and color settings. The rectangular outer border is omitted.
Combine it with `dots=True` for solid black site
dots; otherwise no dots are drawn. Both options work with static or moving
positions and with `voronoi_fn`.

`edges` controls cell boundaries independently of dots. By default boundaries
are black in monochrome mode and white in colored mode; only boundaries
between cells are drawn. `edges=False` hides them. Monochrome without edges
or dots is entirely white.

Use `edge="black"` (or an RGB triplet) to override the boundary color without
changing cell fills. For two-team territory in black and white, use
`pos="black", neg="white", edge="black", color_by_magnitude=False` with
fixed +1/-1 team values and moving positions. The gallery includes this
alongside a red/blue version with white boundaries.

Set `edge_pos="white"` to give boundaries shared by two positive cells their
own color. In the black/white territory example, positive cells are black:
this reveals individual units within black territory while other boundaries
stay black. It works with moving sites. When value signs change over time,
boundaries are drawn as separate paths that follow the interpolated signs at
the geometry sampling rate, which makes files larger.

Set `hide_outer=True` to hide cells that touch the panel boundary, including
their site dots and edges shared only by hidden cells. Remaining cells keep
their original areas; omitted regions are transparent. During motion this
is recomputed at every geometry sample, so cells can appear or disappear
when they start or stop touching the boundary. The default is `False`.
This works in both `draw(..., shape="voronoi")` and `voronoi_fn(...)`.
The gallery includes both territory color schemes with outer cells hidden.

```python
esch.draw(values, "h w t", shape="voronoi", positions=moving,
          monochrome=True, dots=True, fps=16).save("voronoi-black-white.svg")
```

Animated polygon point lists retain a constant length as cells gain or lose
edges, using repeated vertices that preserve the exact geometry. Each cell
strokes its own boundary with round joins. Cells extend slightly past the
panel, and a clip path hides that margin, so the rectangular outer frame is
never drawn. Positive cells are drawn first, so an edge between a positive
and a negative cell takes the negative cell's (`edge`) color.

For arbitrary point counts rather than gridded values, use the layer API:

```python
e = esch.Drawing(h=4, w=5)
esch.voronoi_fn(e, positions=[[[0, 0], [1, 3], [3, 1]]],
               arr=[[1, -0.5, 0.25]], pos="teal", neg="orange")
e.save("regions.svg")
```

Layer values have shape `(panel, point[, time])`; positions have shape
`(panel, point, 2)` or `(panel, point, time, 2)`. Static values broadcast over
moving positions. Magnitude coloring and normalization default to `True`.
Single-site, two-site, and collinear layouts are supported. Geometry uses
Shapely 2.1's ordered Voronoi cells, requiring GEOS 3.12+ (included in current
Shapely binary wheels). Existing square/circle rendering is unchanged.

The repository's [experimental gallery](visual/experimental.html) includes a
separate kinetic clipping prototype (`tests.adaptive_clips.adaptive_example`).
It shares animated half-planes and switches neighbours at predicted Delaunay
events, using native SVG animation with white internal edges and no scripts.
A heap schedules incircle and hull-orientation polynomial roots; GEOS rebuilds
the triangulation after events. Small intervals around roots retain conservative
neighbour sets. This uses floating-point arithmetic, not exact certificates,
and remains experimental: export cost, SVG size, and browser performance can
still be substantial. It does not change the default renderer or public API.

The same gallery now includes an automatically optimized **compact clip-slot**
variant. It reuses a fixed set of clip polygons, chooses variable-length chunks
by dynamic programming, rounds coordinates to a checked precision, and removes
redundant linear keyframes while preserving instantaneous neighbour switches.

```python
from tests.compact_clips import optimized_example  # repository experiment

drawing, report = optimized_example(
    moving_sites,  # (site, time, 2), row/column coordinates
    ["#e53935"] * 32 + ["#1e88e5"] * 32,
    h=7, w=11, fps=1,
    tolerance=0.005,
    runtime_weight=1024,
)
drawing.save("compact.svg")
print(report)
```

`tolerance` is measured in grid units: `0.005` is half a pixel when one grid
unit occupies 100 pixels. Precision increases automatically if sampled boundary
error exceeds this budget. These checks do not prove an error bound at every
instant. `runtime_weight=0` selects the smallest measured SVG among candidates;
larger values favour fewer animated clip slots. This is a browser-work estimate,
not a frame-rate guarantee. The report lists actual byte sizes, slot counts,
keyframes, selected precision, and sampled errors for all four candidates.
The optimizer uses NumPy/GEOS and adds no dependencies or embedded scripts.

## Separate files by panel axis

Use a named panel axis in a filename field to export one file per index:

```python
esch.draw(arr, "n h w").save("test_{n}.svg")
# test_0.svg, test_1.svg, ..., each containing one h × w array.

esch.draw(arr, "n c h w t", fps=4).save("test_{n}_{c:02d}.svg")
# One animated file per (n, c) combination; indices start at zero.
```

Both array-first `draw(arr, pattern)` and pattern-first `draw(pattern, arr)`
calls work. Without filename fields, saving still writes the complete drawing.
Selecting only some panel axes leaves the other panels together in each file.
Files retain the original normalization, colors, animations, and panel layers;
the selected panels are repositioned into their own canvas with the same gap.

Filename fields refer to explicit named **panel** axes. For arrow patterns,
use ungrouped output names, e.g. `b h w -> b h w` with `test_{b}.svg`.
Grouped output factors and anonymous ellipsis axes cannot be selected by name;
keep a factor as a separate output axis to export it. Inner grid axes and time
are retained, rather than split into files. Manually constructed `Drawing`
objects have no pattern names and support ordinary filenames only.

Integer formatting such as `{n:03d}` and repeated fields are supported.
Unknown fields, invalid formats, and duplicate generated paths are rejected
before any files are written. Use `{{` and `}}` for literal filename braces.
Destination directories must already exist. Root-level custom SVG additions
are not copied into split files; panel content and SVG definitions are.

## Development and visual review

Python 3.11 and uv:

```sh
uv sync
just test    # pytest, in parallel
just check   # ruff and vulture
ruff format --check esch tests main.py
```

Every full test run regenerates **`visual/index.html`** and standalone SVGs.
Open the gallery in a browser for manual review. It includes scalars through
rank-six arrays, signed squares and circles, zero/singleton/large values,
axis arrangements, one-frame and looping animations, panel layouts, layers,
ticks, and static/animated meshes. Animations have pause, restart, and scrub
controls. Fixtures are deterministic.

Automated tests check data placement, SVG geometry, validation failures,
layering, timing, and gallery completeness. Passing tests do **not** imply
visual approval. Inspect the gallery after rendering changes; add an example
for every new rendering behavior. To generate it independently:

```sh
uv run python main.py
uv run python main.py --output visual-before
```

Save a gallery to a different directory before future rendering changes for
manual before/after comparison. Each HTML file embeds its plots and can be
kept as a self-contained review artifact. No pixel baselines have been
approved yet.

Install the optional notebook tools with `uv sync --extra notebook`.

## Changes from the earlier API

The README's former `init`/`tile` examples did not match the implementation;
use `draw` instead. The pattern-first `draw` call, exported layer functions,
`Drawing`, `show`, and `save` remain available. `sphere` and `cube` remain
aliases for circle and square.

Layout now consistently treats height as rows and width as columns.
Singleton grids have a nonzero canvas, and debug rendering no longer
duplicates panels. Static negative squares render valid positive geometry.
Circle sizing now uses the former animated radius formula for both static
and animated plots, so existing static circle/mesh plots can change size.
Outlines are consistently 0.04 grid units. Animation starts at the first
frame and no longer adds an extra frame to the loop duration.

Notebook display now keeps the SVG rather than converting through PDF and
Matplotlib; it preserves animation and uses the SVG's original colors.
The historical `show(dpi=...)` and `save(scale=...)` parameters remain accepted
but have no effect.

## Typst

`lib.typ` is a separate wrapper for placing a plot with axis labels, a title,
and an optional caption. It does not render arrays or preserve SVG animation
in static documents.
