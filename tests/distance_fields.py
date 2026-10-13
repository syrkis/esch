"""Experimental two-team territory using animated SVG distance gradients.

Filters rasterize internally. This is an approximate, repository-only prototype,
not the vector Voronoi renderer or a public plotting API.
"""

import numpy as np

import esch


def distance_example(positions, teams, *, h, w, fps=1, boundaries=False):
    """Animate fixed +/- teams; optional white boundaries separate teams only."""
    positions = np.asarray(positions, dtype=float)
    teams = np.asarray(teams)
    if positions.ndim != 3 or positions.shape[-1] != 2 or not all(positions.shape):
        raise ValueError("Expected nonempty (site, time, 2) positions")
    if not np.isfinite(positions).all() or np.any(positions < -0.5) or np.any(positions > np.array([h, w]) - 0.5):
        raise ValueError("Sites must be finite and inside the panel")
    if teams.shape != (len(positions),) or not np.isin(teams, [-1, 1]).all() or len(np.unique(teams)) != 2:
        raise ValueError("Provide both teams, with one +1 or -1 label per site")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not isinstance(boundaries, (bool, np.bool_)):
        raise TypeError("boundaries must be a boolean")
    drawing = esch.Drawing(h, w)
    svg = drawing.dwg
    # Compute beyond the viewport so blur and morphology do not create an
    # artificial frame around the diagram. Clip only the finished output.
    margin = 0.1
    radius = np.hypot(h + 2 * margin, w + 2 * margin)
    filter_ = svg.filter(
        id="territory",
        filterUnits="userSpaceOnUse",
        x=-0.5 - margin,
        y=-0.5 - margin,
        width=w + 2 * margin,
        height=h + 2 * margin,
        **{"color-interpolation-filters": "sRGB"},
    )
    # Amplify before smoothing so the comparison is not quantized back to a
    # single byte around .5. A small blur reduces gradient dithering at borders.
    filter_.feColorMatrix(type="matrix", values="32 0 -32 0 0.5 32 0 -32 0 0.5 32 0 -32 0 0.5 0 0 0 1 0", result="difference")
    filter_.feGaussianBlur(in_="difference", stdDeviation=0.015, result="smooth")
    transfer = filter_.feComponentTransfer(in_="smooth", result="colored")
    transfer.feFuncR("discrete", tableValues=f"{30 / 255} {229 / 255}")
    transfer.feFuncG("discrete", tableValues=f"{136 / 255} {57 / 255}")
    transfer.feFuncB("discrete", tableValues=f"{229 / 255} {53 / 255}")
    if boundaries:
        filter_.feColorMatrix(in_="smooth", type="matrix", values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0", result="alpha")
        threshold = filter_.feComponentTransfer(in_="alpha", result="mask")
        threshold.feFuncA("discrete", tableValues="0 1")
        filter_.feMorphology(in_="mask", operator="dilate", radius=0.02, result="outer")
        filter_.feMorphology(in_="mask", operator="erode", radius=0.02, result="inner")
        filter_.feComposite(in_="outer", in2="inner", operator="out", result="front")
        filter_.feFlood(flood_color="white", result="white")
        filter_.feComposite(in_="white", in2="front", operator="in", result="stroke")
        filter_.feMerge(["colored", "stroke"])
    svg.defs.add(filter_)
    clip = svg.clipPath(id="panel", clipPathUnits="userSpaceOnUse")
    clip.add(svg.rect(insert=(-0.5, -0.5), size=(w, h), fill="none"))
    svg.defs.add(clip)
    panel = svg.g(clip_path="url(#panel)")
    filtered = svg.g(filter="url(#territory)")
    fields = svg.g(style="isolation:isolate")
    filtered.add(fields)
    panel.add(filtered)
    drawing.gs[0].add(panel)
    for index, (trajectory, team) in enumerate(zip(positions, teams, strict=True)):
        row, column = trajectory[0]
        gradient = svg.radialGradient(
            id=f"distance-{index}",
            center=(column, row),
            r=radius,
            gradientUnits="userSpaceOnUse",
            **{"color-interpolation": "sRGB"},
        )
        gradient.add_stop_color(0, "red" if team > 0 else "blue")
        gradient.add_stop_color(1, "black")
        if len(trajectory) > 1:
            for attribute, coordinates in (("cx", trajectory[:, 1]), ("cy", trajectory[:, 0])):
                values = [f"{value:.10g}" for value in coordinates]
                gradient.add(
                    svg.animate(
                        attributeName=attribute,
                        values=";".join([*values, values[0]]),
                        dur=f"{len(trajectory) / fps:g}s",
                        repeatCount="indefinite",
                        calcMode="linear",
                    )
                )
        svg.defs.add(gradient)
        fields.add(
            svg.rect(
                insert=(-0.5 - margin, -0.5 - margin),
                size=(w + 2 * margin, h + 2 * margin),
                fill=f"url(#distance-{index})",
                style="mix-blend-mode:lighten",
            )
        )
    return drawing
