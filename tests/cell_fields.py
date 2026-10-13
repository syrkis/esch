"""Repository-only experiment: one SVG distance-field mask per Voronoi cell."""

import numpy as np

import esch


def cell_example(positions, teams, *, h, w, fps=1):
    """Approximate individual cells, fixed team colors, white internal edges."""
    positions = np.asarray(positions, dtype=float)
    teams = np.asarray(teams)
    if positions.ndim != 3 or positions.shape[-1] != 2 or not all(positions.shape):
        raise ValueError("Expected nonempty (site, time, 2) positions")
    if len(positions) < 2:
        raise ValueError("The individual-cell experiment requires at least two sites")
    if not np.isfinite(positions).all() or np.any(positions < -0.5) or np.any(positions > np.array([h, w]) - 0.5):
        raise ValueError("Sites must be finite and inside the panel")
    if teams.shape != (len(positions),) or not np.isin(teams, [-1, 1]).all():
        raise ValueError("Provide one +1 or -1 team label per site")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    drawing = esch.Drawing(h, w)
    svg = drawing.dwg
    margin = 0.1
    origin, size = (-0.5 - margin, -0.5 - margin), (w + 2 * margin, h + 2 * margin)
    radius = np.hypot(*size)
    filter_ = svg.filter(
        id="cells",
        filterUnits="userSpaceOnUse",
        x=origin[0],
        y=origin[1],
        width=size[0],
        height=size[1],
        **{"color-interpolation-filters": "sRGB"},
    )
    for index, trajectory in enumerate(positions):
        row, column = trajectory[0]
        gradient = svg.radialGradient(
            id=f"distance-{index}",
            center=(column - origin[0], row - origin[1]),
            r=radius,
            gradientUnits="userSpaceOnUse",
            **{"color-interpolation": "sRGB"},
        )
        gradient.add_stop_color(0, "red")
        gradient.add_stop_color(1, "black")
        if len(trajectory) > 1:
            for attribute, coordinates in (("cx", trajectory[:, 1] - origin[0]), ("cy", trajectory[:, 0] - origin[1])):
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
        field = svg.rect(id=f"field-{index}", insert=(0, 0), size=size, fill=f"url(#distance-{index})")
        svg.defs.add(field)
        filter_.feImage(href=f"#field-{index}", start=origin, size=size, result=f"site-{index}")
    maximum = "site-0"
    prefixes = [maximum]
    for index in range(1, len(positions)):
        filter_.feBlend(in_=maximum, in2=f"site-{index}", mode="lighten", result=f"max-{index}")
        maximum = f"max-{index}"
        prefixes.append(maximum)
    suffixes = [f"site-{len(positions) - 1}"]
    for index in range(len(positions) - 2, -1, -1):
        filter_.feBlend(in_=suffixes[-1], in2=f"site-{index}", mode="lighten", result=f"suffix-{index}")
        suffixes.append(f"suffix-{index}")
    suffixes.reverse()
    layers = []
    filter_.feFlood(flood_color="white", result="background")
    layers.append("background")
    for index, team in enumerate(teams):
        # Exclude the site itself: signed comparisons can be smoothed across
        # bisectors, unlike equality with a maximum that includes this site.
        if index == 0:
            other = suffixes[1]
        elif index == len(positions) - 1:
            other = prefixes[index - 1]
        else:
            other = f"other-{index}"
            filter_.feBlend(in_=prefixes[index - 1], in2=suffixes[index + 1], mode="lighten", result=other)
        filter_.feColorMatrix(
            in_=other, type="matrix", values="0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0", result=f"other-blue-{index}"
        )
        filter_.feBlend(in_=f"site-{index}", in2=f"other-blue-{index}", mode="lighten", result=f"difference-{index}")
        filter_.feColorMatrix(
            in_=f"difference-{index}",
            type="matrix",
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 32 0 -32 0 0.5",
            result=f"alpha-{index}",
        )
        filter_.feGaussianBlur(in_=f"alpha-{index}", stdDeviation=0.015, result=f"smooth-{index}")
        threshold = filter_.feComponentTransfer(in_=f"smooth-{index}", result=f"mask-{index}")
        threshold.feFuncA("discrete", tableValues="0 1")
        filter_.feMorphology(in_=f"mask-{index}", operator="erode", radius=0.02, result=f"inner-{index}")
        filter_.feFlood(flood_color="#e53935" if team > 0 else "#1e88e5", result=f"color-{index}")
        filter_.feComposite(in_=f"color-{index}", in2=f"inner-{index}", operator="in", result=f"cell-{index}")
        layers.append(f"cell-{index}")
    filter_.feMerge(layers)
    svg.defs.add(filter_)
    clip = svg.clipPath(id="panel", clipPathUnits="userSpaceOnUse")
    clip.add(svg.rect(insert=(-0.5, -0.5), size=(w, h), fill="none"))
    svg.defs.add(clip)
    group = svg.g(clip_path="url(#panel)")
    group.add(svg.rect(insert=origin, size=size, fill="white", filter="url(#cells)"))
    drawing.gs[0].add(group)
    return drawing
