# draw.py
import svgwrite
from tqdm import tqdm
from typing import Optional, Tuple, List, Any
from numpy import ndarray
import numpy as np


def setup_drawing(width: int, height: int, size: int, padding: int = 0) -> svgwrite.Drawing:
    """Initialize and setup SVG drawing with common properties."""
    total_width = width * size + (padding * 2)
    total_height = height * size + (padding * 2)

    dwg = svgwrite.Drawing(size=(f"{total_width}px", f"{total_height}px"))
    dwg.viewbox(-padding, -padding, total_width, total_height)
    dwg["width"] = "100%"
    dwg["height"] = "100%"
    dwg["preserveAspectRatio"] = "xMidYMid meet"

    # Add style element for FiraCode font
    style = dwg.style("""
        @import url(https://cdn.jsdelivr.net/npm/firacode@6.2.0/distr/fira_code.css);
        text { font-family: 'Fira Code', monospace; }
    """)
    dwg.defs.add(style)

    return dwg


def get_rect_properties(value: float, size: int) -> dict:
    """Calculate common rectangle properties based on value."""
    rect_size = np.abs(value)
    rect_width = rect_size * size * 0.8
    fill_color = "white" if value < 0 else "black"
    stroke = {"stroke": "black", "stroke_width": "0.5"} if value < 0 else {"stroke": "none", "stroke_width": "0"}
    offset = (size - rect_width) / 2
    return {"rect_width": rect_width, "fill_color": fill_color, "offset": offset, **stroke}


def calculate_position(i: int, j: int, size: int, offset: float) -> Tuple[float, float]:
    """Calculate position for rectangle."""
    return j * size + offset, i * size + offset


def make(
    x: ndarray,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xticks: Optional[List] = None,
    yticks: Optional[List] = None,
    size: int = 10,
) -> svgwrite.Drawing:
    """Create optimized SVG drawing."""
    x = x.T
    width, height = x.shape
    # Add padding for ticks and labels
    padding = size * 2
    dwg = setup_drawing(width, height, size, padding)

    # Create group for rectangles
    group = dwg.g()

    non_zero = np.nonzero(x)
    for i, j in zip(non_zero[0], non_zero[1]):
        value = x[i, j]
        props = get_rect_properties(value, size)  # type: ignore
        pos_y, pos_x = calculate_position(i, j, size, props["offset"])

        group.add(
            dwg.rect(
                insert=(f"{pos_x:.1f}", f"{pos_y:.1f}"),
                size=(f"{props['rect_width']:.1f}", f"{props['rect_width']:.1f}"),
                fill=props["fill_color"],
                stroke=props["stroke"],
                stroke_width=props["stroke_width"],
            )
        )

    dwg.add(group)

    # Add ticks and labels
    add_ticks_and_labels(dwg, size, width, height, xlabel, ylabel, xticks, yticks)

    return dwg


def add_ticks_and_labels(
    dwg: svgwrite.Drawing,
    size: int,
    width: int,
    height: int,
    xlabel: Optional[str],
    ylabel: Optional[str],
    xticks: Optional[List],
    yticks: Optional[List],
) -> None:
    """Add axis ticks and labels to the drawing."""
    tick_length = size * 0.3
    text_offset = size * 0.6
    tick_offset = size * 0.4

    # Create group for ticks and labels
    tick_group = dwg.g()

    # Generate default ticks if none provided
    # Add x-axis ticks and labels
    if xticks is not None:
        for pos, label in xticks:
            x = pos * size + size / 2
            # Draw tick mark, moved down by tick_offset
            tick_group.add(
                dwg.line(
                    start=(x, height * size + tick_offset),
                    end=(x, height * size + tick_offset + tick_length),
                    stroke="black",
                    stroke_width=0.5,
                )
            )
            tick_group.add(
                dwg.text(
                    label,
                    insert=(x, height * size + tick_offset + tick_length + text_offset),  # Adjusted spacing
                    text_anchor="middle",
                    dominant_baseline="hanging",  # Align text from top
                    font_size=f"{size * 0.6}px",
                )
            )

    if yticks is not None:
        for pos, label in yticks:
            y = pos * size + size / 2
            # Draw tick mark, moved left by tick_offset
            tick_group.add(
                dwg.line(start=(-tick_offset, y), end=(-tick_offset - tick_length, y), stroke="black", stroke_width=0.5)
            )
            # Add tick label, adjusted position
            tick_group.add(
                dwg.text(
                    label,
                    insert=(-tick_offset - tick_length - text_offset, y),
                    text_anchor="end",
                    dominant_baseline="middle",
                    font_size=f"{size * 0.6}px",
                )
            )

    # X-axis label
    if xlabel is not None:
        tick_group.add(
            dwg.text(
                xlabel,
                insert=(width * size / 2, height * size + (size) + (tick_offset / 4)),
                text_anchor="middle",
                font_size=f"{size * 0.6}px",
            )
        )

    # Y-axis label
    if ylabel is not None:
        tick_group.add(
            dwg.text(
                ylabel,
                insert=(-size - tick_offset, height * size / 2),
                text_anchor="middle",
                font_size=f"{size * 0.6}px",
                transform=f"rotate(-90, {- (size/ 2) - (tick_offset)}, {height * size / 2})",
            )
        )

    dwg.add(tick_group)


def create_animation_values(frames: List[ndarray], i: int, j: int, size: int) -> dict:
    """Create animation values for a specific position."""
    values = [frame[i, j] for frame in frames]
    sizes = [float(np.abs(v)) for v in values]  # Convert to float
    rect_widths = [s * size * 0.8 for s in sizes]
    offsets = [(size - w) / 2 for w in rect_widths]

    start_pos_x, start_pos_y = calculate_position(j, i, size, float(offsets[0]))  # Swap i,j and convert offset to float

    return {
        "size_str": ";".join(f"{w:.1f}" for w in rect_widths),
        "initial_pos": (start_pos_x, start_pos_y),
        "x_position_str": ";".join(f"{i * size + offset:.1f}" for offset in offsets),  # Use j for x
        "y_position_str": ";".join(f"{j * size + offset:.1f}" for offset in offsets),  # Use i for y
        "values": values,
    }


def play(
    frames,
    xlabel: str | None,
    ylabel: str | None,
    xticks: Optional[List] = None,
    yticks: Optional[List] = None,
    size: int = 10,
    rate: int = 20,
) -> svgwrite.Drawing:
    """Create single SVG with animated rectangles."""
    if len(frames) == 0:
        raise ValueError("No frames provided")

    frames = [frame.T for frame in frames]
    width, height = frames[0].shape
    padding = size * 2
    dwg = setup_drawing(width, height, size, padding)
    base_group = dwg.g()

    non_zero = np.nonzero(frames[0])
    total_elements = len(non_zero[0])

    for i, j in tqdm(zip(non_zero[0], non_zero[1]), total=total_elements):
        anim_values = create_animation_values(frames, i, j, size)
        props = get_rect_properties(anim_values["values"][0], size)

        rect = dwg.rect(
            insert=(f"{anim_values['initial_pos'][0]:.1f}", f"{anim_values['initial_pos'][1]:.1f}"),
            width=f"{props['rect_width']:.1f}",
            height=f"{props['rect_width']:.1f}",
            fill=props["fill_color"],
            stroke=props["stroke"],
            stroke_width=props["stroke_width"],
        )

        duration = f"{len(frames)/rate}s"
        for attr, values in [
            ("width", anim_values["size_str"]),
            ("height", anim_values["size_str"]),
            ("x", anim_values["x_position_str"]),
            ("y", anim_values["y_position_str"]),
        ]:
            rect.add(dwg.animate(attributeName=attr, values=values, dur=duration, repeatCount="indefinite"))

        base_group.add(rect)

    dwg.add(base_group)
    add_ticks_and_labels(dwg, size, width, height, xlabel, ylabel, xticks, yticks)
    return dwg
