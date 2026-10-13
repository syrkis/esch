"""Experimental native-SVG Voronoi clipping; deliberately not a public API.

Each cell intersects one animated half-plane per other site. All triangle
vertices are affine in the moving site positions, so linear SMIL interpolation
preserves the bisectors even when Voronoi topology changes. This trades stored
geometry frames for quadratic browser work. No scripting is embedded in SVG.
"""

import numpy as np

import esch


def halfplanes(points, bounds):
    """Yield (owner, other, triangle poses), in SVG (x, y) coordinates."""
    xmin, ymin, xmax, ymax = bounds
    diagonal = np.hypot(xmax - xmin, ymax - ymin)
    for i in range(len(points)):
        for j in range(len(points)):
            if i == j:
                continue
            difference = points[j] - points[i]
            velocity = np.roll(difference, -1, axis=0) - difference
            denominator = np.sum(velocity**2, axis=-1)
            fraction = np.clip(-np.sum(difference * velocity, axis=-1) / np.where(denominator, denominator, 1), 0, 1)
            separation = np.linalg.norm(difference + fraction[:, None] * velocity, axis=-1).min()
            if separation < 1e-10 * max(1, diagonal):
                raise ValueError("The clipping experiment requires sites that never collide, including between frames")
            # The midpoint remains inside the bounds. Every point in the bounds
            # is within one diagonal of it. A triangle extending two diagonals
            # in the tangent/normal directions therefore covers the half-plane
            # within the bounds, at every interpolated time (including closure).
            scale = 2 * diagonal / separation
            midpoint = (points[j] + points[i]) / 2
            tangent = scale * np.stack([-difference[:, 1], difference[:, 0]], axis=-1)
            yield i, j, np.stack([midpoint + tangent, midpoint - tangent, midpoint - scale * difference], axis=1)


def clipping_example(positions, colors, *, h, w, fps=1):
    """One panel, fixed colors, white internal edges; positions (site, time, 2).

    Input coordinates use esch's (row, column) convention. This proof of concept
    does not implement dots, changing colors, hide_outer, or the pattern API.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 2 or not all(positions.shape):
        raise ValueError("Expected nonempty (site, time, 2) positions")
    if not np.isfinite(positions).all() or np.any(positions < -0.5) or np.any(positions > np.array([h, w]) - 0.5):
        raise ValueError("Sites must be finite and inside the panel")
    if len(colors) != len(positions) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Provide one color per site and positive finite fps")
    drawing = esch.Drawing(h, w)
    svg = drawing.dwg
    frames = positions.shape[1]
    for i, j, poses in halfplanes(positions[..., ::-1], (-0.5, -0.5, w - 0.5, h - 0.5)):
        values = [" ".join(f"{x:.12g},{y:.12g}" for x, y in pose) for pose in poses]
        polygon = svg.polygon(points=poses[0].tolist(), id=f"half-{i}-{j}", fill="none")
        if frames > 1:
            polygon.add(
                svg.animate(
                    attributeName="points",
                    values=";".join([*values, values[0]]),
                    dur=f"{frames / fps:g}s",
                    repeatCount="indefinite",
                    calcMode="linear",
                )
            )
        svg.defs.add(polygon)
        clip = svg.clipPath(id=f"clip-{i}-{j}", clipPathUnits="userSpaceOnUse")
        clip.add(svg.use(href=f"#half-{i}-{j}"))
        svg.defs.add(clip)
    # Clip the remote ends of the finite triangles to the panel, without
    # stroking the panel rectangle. Only bisectors can appear as visible edges.
    panel_clip = svg.clipPath(id="panel", clipPathUnits="userSpaceOnUse")
    panel_clip.add(svg.rect(insert=(-0.5, -0.5), size=(w, h), fill="none"))
    svg.defs.add(panel_clip)
    panel = svg.g(clip_path="url(#panel)")
    drawing.gs[0].add(panel)
    for i, color in enumerate(colors):
        group = panel
        for j in range(len(positions)):
            if i != j:
                child = svg.g(clip_path=f"url(#clip-{i}-{j})")
                group.add(child)
                group = child
        group.add(svg.rect(insert=(-0.5, -0.5), size=(w, h), fill=color))
        for j in range(len(positions)):
            if i != j:
                group.add(svg.use(href=f"#half-{i}-{j}", fill="none", stroke="white", stroke_width=0.04))
    return drawing
