# Compact Voronoi experiment

The compact exporter reduces the 64-unit, 16-second territorial-control example
from 6.59 MB to **2.71 MB**, with white edges between every cell, continuous native
SVG interpolation, and no scripts or raster filters. It remains a repository
experiment, separate from the public renderer.

Open [the experimental gallery](visual/experimental.html) and find
`voronoi-compact-experiment`. It starts paused. The regular gallery remains at
[visual/index.html](visual/index.html).

## Measured results

Measurements use the existing deterministic 64-unit fixture: 32 red and 32 blue
sites, 16 input frames, 1 input fps, a 7-by-11 panel, and independently smoothed
random trajectories. MB means decimal megabytes.

| Encoding | SVG size | Reserved clip slots | Chromium playback |
| --- | ---: | ---: | ---: |
| Previous kinetic shared planes | 6.59 MB | 662 | about 35 fps |
| Compact, smallest candidate | 2.57 MB | 1,538 | not remeasured after final keyframe removal |
| Compact, automatically selected | **2.71 MB** | **1,261** | **about 43 fps** |
| Compact, stronger clip penalty | 3.47 MB | 973 | not measured |

The selected export retains 42,709 triangle poses instead of 50,239 before
simplification. Its 1,437 cell chunks have variable lengths. Three coordinate
decimal places suffice for the requested 0.005-grid-unit error budget. Maximum
observed boundary error is **0.002645 grid units**, roughly **0.265 pixels at
100 pixels per grid unit**, across 11,328 cell/time comparisons.

An additional Chromium screenshot check found no incorrectly coloured interior
pixels among 98,972 samples at eight times, including immediately before, at,
and after an input-frame boundary. White edges and anti-aliased pixels are
excluded from that colour-classification count. This is not an exhaustive
image-equivalence test or manual visual approval.

Playback was measured in isolated headless Chromium at an 1100-by-800 viewport
using a three-second requestAnimationFrame window. These are local observations,
not cross-browser performance guarantees. The selected SVG compresses to about
0.81 MB with gzip; that is a compressed size, not the standalone `.svg` size.
The isolated gallery card at a 500-by-700 viewport measured approximately
60 fps, and its initial paused state and play control both worked.

## What the optimizer does

1. Reuses the kinetic Delaunay planner's conservative neighbour schedules.
2. Uses dynamic programming to choose independent variable-length chunks for
   each cell. Candidate boundaries are quarter input frames; candidate spans
   extend up to four input frames. The cost measures encoded coordinate/time
   payload and penalizes clips active over time.
3. Reuses a bounded collection of animated clip polygons for each cell.
   Duplicate SMIL key times change a slot's geometry instantly when its assigned
   neighbour changes. White boundary strokes reference those same polygons.
   This removes thousands of separate pair definitions and visibility tracks.
4. Removes redundant linear keyframes using Ramer–Douglas–Peucker simplification,
   retaining both sides of every instantaneous switch.
5. Tests four clip-cost weights and selects by **actual serialized SVG bytes +
   runtime_weight × reserved clip slots**. This runtime term is a workload proxy.
6. Chooses decimal precision and checks the serialized geometry against fresh
   Voronoi cells at regular and offset times. Failed precision checks cause a
   higher-precision retry. The error budget is sampled, not proven for all time.

The dynamic-programming payload cost omits some whole-document interactions,
so this is a bounded search with measured final candidates, not a claim of a
globally optimal SVG encoding. Immutable geometry plans and cost tables are
cached within the Python process; drawings and validation results are not cached.

## Usage

```python
from tests.compact_clips import optimized_example

drawing, report = optimized_example(
    positions,  # (site, time, 2), row/column coordinates
    ["#e53935"] * 32 + ["#1e88e5"] * 32,
    h=7, w=11, fps=1,
    tolerance=0.005,
    runtime_weight=1024,
)
drawing.save("compact.svg")
print(report)
```

Use `runtime_weight=0` to select the smallest measured candidate. Larger weights
prefer fewer clips. `tolerance` controls the sampled coordinate-error budget,
not time sampling or input fps. The report exposes every candidate's size,
precision, slots, keyframes, and checked error.

## Limits and rejected approach

Export still takes minutes on this fixture because the kinetic event analysis
is expensive. Encoding and checking all compact candidates took about 76 seconds
with the kinetic plan already available. Repeated exports reuse the analysis.

A direct polygon-morph prototype was also investigated. It needed excessive
subdivision around changing edge correspondence and near-degenerate cells; a
small regression scene grew to 455 KB, versus 45 KB with reusable clip slots.
That prototype was not retained in the repository.

No JAX dependency was added. The substantial size reduction came from changing
the SVG representation. Accelerating geometry preprocessing would reduce export
time, but would not remove SVG markup or browser clipping work by itself.

This is useful progress, not a tiny-file solution: a dense moving 64-unit scene
still needs several megabytes without scripts. The default renderer and public
API are unchanged. Regression tests cover interpolation, both sides of clip
switches, loop-adjacent times, precision selection, byte-cost selection, static
inputs, collisions, and cache isolation.

Final validation: **100 unit tests passed**, regenerating 78 visual examples and
executing their displayed recipes. All 27 Python files pass the formatting
check. Ruff reports four pre-existing issues in `esch/__init__.py`, `esch/draw.py`,
and `tests/test_drawing.py`; the changed Python files have no lint failures.
