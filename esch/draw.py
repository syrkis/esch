# draw.py
# SVG drawing functions for hinton plots
# by: Noah Syrkis

import svgwrite
from jax import Array
import jax.numpy as jnp
from tqdm import tqdm
from typing import Optional, Tuple


def make(x: Array, size: int = 10) -> svgwrite.Drawing:
    """Create optimized SVG drawing."""
    width, height = x.shape
    # Initialize with explicit pixel dimensions
    dwg = svgwrite.Drawing(size=(f"{width * size}px", f"{height * size}px"))
    dwg.viewbox(0, 0, width * size, height * size)
    dwg["width"] = "100%"
    dwg["height"] = "100%"
    dwg["preserveAspectRatio"] = "xMidYMid meet"
    group = dwg.g()

    non_zero = jnp.nonzero(x)
    for i, j in zip(non_zero[0], non_zero[1]):
        value = x[i, j]
        rect_size = jnp.abs(value)
        rect_width = rect_size * size * 0.8

        # Set fill color based on sign
        fill_color = "white" if value < 0 else "black"

        offset_x = (size - rect_width) / 2
        offset_y = (size - rect_width) / 2

        pos_x = j * size + offset_x
        pos_y = i * size + offset_y

        group.add(
            dwg.rect(
                insert=(f"{pos_x:.1f}", f"{pos_y:.1f}"),
                size=(f"{rect_width:.1f}", f"{rect_width:.1f}"),
                fill=fill_color,
                stroke="black" if value < 0 else "none",  # Add border for white rectangles
                stroke_width="1" if value < 0 else "0",
            )
        )

    dwg.add(group)
    return dwg


def play(frames: Array, rate: int = 20) -> svgwrite.Drawing:
    """Create single SVG with animated rectangles."""
    if len(frames) == 0:
        raise ValueError("No frames provided")

    width, height = frames[0].shape
    size = 10
    # Initialize with explicit pixel dimensions
    dwg = svgwrite.Drawing(size=(f"{width * size}px", f"{height * size}px"))
    dwg.viewbox(0, 0, height * size, width * size)
    dwg["width"] = "100%"
    dwg["height"] = "100%"
    dwg["preserveAspectRatio"] = "xMidYMid meet"

    base_group = dwg.g()
    non_zero = jnp.nonzero(frames[0])
    # Set progress bar length to number of non-zero elements

    total_elements = len(non_zero[0])
    for i, j in tqdm(zip(non_zero[0], non_zero[1]), total=total_elements):
        values = [frame[i, j] for frame in frames]
        sizes = [jnp.abs(v) for v in values]
        rect_widths = [s * size * 0.8 for s in sizes]

        # Set fill color based on sign of first frame
        # Note: If you want color to change during animation, this needs to be handled differently
        fill_color = "white" if values[0] < 0 else "black"

        offsets = [(size - w) / 2 for w in rect_widths]

        cell_x = j * size
        cell_y = i * size

        initial_offset = offsets[0]
        pos_x = cell_x + initial_offset
        pos_y = cell_y + initial_offset

        size_str = ";".join(f"{w:.1f}" for w in rect_widths)
        x_position_str = ";".join(f"{cell_x + offset:.1f}" for offset in offsets)
        y_position_str = ";".join(f"{cell_y + offset:.1f}" for offset in offsets)

        rect = dwg.rect(
            insert=(f"{pos_x:.1f}", f"{pos_y:.1f}"),
            width="0",
            height="0",
            fill=fill_color,
            stroke="black" if values[0] < 0 else "none",  # Add border for white rectangles
            stroke_width="1" if values[0] < 0 else "0",
        )
        rect.add(
            dwg.animate(attributeName="width", values=size_str, dur=f"{len(frames)/rate}s", repeatCount="indefinite")
        )
        rect.add(
            dwg.animate(attributeName="height", values=size_str, dur=f"{len(frames)/rate}s", repeatCount="indefinite")
        )
        rect.add(
            dwg.animate(attributeName="x", values=x_position_str, dur=f"{len(frames)/rate}s", repeatCount="indefinite")
        )
        rect.add(
            dwg.animate(attributeName="y", values=y_position_str, dur=f"{len(frames)/rate}s", repeatCount="indefinite")
        )
        base_group.add(rect)

    dwg.add(base_group)
    return dwg
