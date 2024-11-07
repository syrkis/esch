# draw.py
import svgwrite
from tqdm import tqdm
from typing import Optional, Tuple, List, Any
from numpy import ndarray
import numpy as np
from functools import lru_cache


def setup_drawing(
    width: int,  # width of single plot
    height: int,  # height of single plot
    size: int,
    n_plots: int = 1,  # number of plots
    padding: int = 0,
) -> svgwrite.Drawing:
    """Initialize and setup SVG drawing with common properties.

    Layout direction is determined by the aspect ratio of individual plots:
    - If width > height: arrange plots in a column
    - If width <= height: arrange plots in a row
    """
    plot_width = width * size
    plot_height = height * size
    plot_spacing = padding  # Using padding as plot spacing for now

    # Determine layout based on aspect ratio
    is_column_layout = plot_width > plot_height

    if is_column_layout:
        # Column layout
        total_width = plot_width + (padding * 2)
        total_height = (plot_height * n_plots) + (plot_spacing * (n_plots - 1)) + (padding * 2)
    else:
        # Row layout
        total_width = (plot_width * n_plots) + (plot_spacing * (n_plots - 1)) + (padding * 2)
        total_height = plot_height + (padding * 2)

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


def get_plot_offset(plot_index: int, width: int, height: int, size: int, padding: int = 0) -> tuple[float, float]:
    """Calculate offset for a plot.

    Args:
        plot_index: Index of the plot (0-based)
        width: Width of a single plot in units
        height: Height of a single plot in units
        size: Size of each unit
        padding: Padding between plots

    Returns:
        tuple[float, float]: (x, y) coordinate offsets for the plot
    """
    plot_width = width * size
    plot_height = height * size
    is_column_layout = plot_width > plot_height

    if is_column_layout:
        x_offset = 0
        y_offset = plot_index * (plot_height + padding)
    else:
        x_offset = plot_index * (plot_width + padding)
        y_offset = 0

    return x_offset, y_offset


@lru_cache
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
    """Create SVG drawing for single or multiple plots.

    Args:
        x: 2D array (height × width) or 3D array (frames × height × width)
        ...
    """
    # Handle both 2D and 3D arrays
    if x.ndim == 2:
        x = x[np.newaxis, ...]  # Add frame dimension

    # x = x.transpose(0, 2, 1)  # Transpose each frame
    n_plots, height, width = x.shape

    # Setup drawing with correct number of plots
    padding = size * 2
    dwg = setup_drawing(width, height, size, n_plots, padding)

    # Create plots one by one
    for plot_idx in range(n_plots):
        frame = x[plot_idx]
        x_offset, y_offset = get_plot_offset(plot_idx, width, height, size, padding)

        # Create group for this plot
        plot_group = dwg.g()

        # Draw rectangles for this frame
        non_zero = np.nonzero(frame)
        for i, j in zip(non_zero[0], non_zero[1]):
            value = frame[i, j]
            props = get_rect_properties(value, size)
            pos_x, pos_y = calculate_position(i, j, size, props["offset"])

            # Adjust positions by offset
            pos_x += x_offset
            pos_y += y_offset

            plot_group.add(
                dwg.rect(
                    insert=(f"{pos_x:.1f}", f"{pos_y:.1f}"),
                    size=(f"{props['rect_width']:.1f}", f"{props['rect_width']:.1f}"),
                    fill=props["fill_color"],
                    stroke=props["stroke"],
                    stroke_width=props["stroke_width"],
                )
            )

        dwg.add(plot_group)

        # Add ticks and labels for this plot
        add_ticks_and_labels(
            dwg,
            size,
            width,
            height,
            xlabel if plot_idx == n_plots - 1 else None,  # Only add xlabel to last plot for column layout
            ylabel if plot_idx == 0 else None,  # Only add ylabel to first plot
            xticks,
            yticks,
            x_offset,
            y_offset,
        )

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
    x_offset: float = 0,
    y_offset: float = 0,
) -> None:
    """Add axis ticks and labels to the drawing with offset support."""
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
    dwg = setup_drawing(width, height, size, 1, padding)
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
