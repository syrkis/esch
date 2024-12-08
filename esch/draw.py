# draw.py
import svgwrite
from tqdm import tqdm
from typing import Tuple, List
from numpy import ndarray
import numpy as np
from functools import lru_cache
from .edge import EdgeConfigs, add_ticks_and_labels


def setup_drawing(
    width: int,  # width of single plot
    height: int,  # height of single plot
    size: int,
    n_plots: int = 1,  # number of plots
    padding: int = 10,
) -> svgwrite.Drawing:
    """Initialize and setup SVG drawing with common properties.

    Layout direction is determined by the aspect ratio of individual plots:
    - If width > height: arrange plots in a column
    - If width <= height: arrange plots in a row
    """
    # padding = size * 3  # Original padding might have been size * 2
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
        text {
            font-family: 'STIX Two', 'STIX Two Math', 'STIXMath', 'DejaVu Math TeX Glyph',
                        'DejaVu Serif', 'Cambria Math', 'Latin Modern Math', 'FiraCode',
                        'serif';
        }
    """)
    dwg.defs.add(style)
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
    rect_size = np.sqrt(np.abs(value))
    rect_width = rect_size * size * 0.95
    fill_color = "white" if value < 0 else "black"
    stroke = {"stroke": "black", "stroke_width": "0.5"} if value < 0 else {"stroke": "none", "stroke_width": "0"}
    offset = (size - rect_width) / 2
    return {"rect_width": rect_width, "fill_color": fill_color, "offset": offset, **stroke}


def calculate_position(i: int, j: int, size: int, offset: float) -> Tuple[float, float]:
    """Calculate position for rectangle."""
    return j * size + offset, i * size + offset


def make(
    x: ndarray,
    edge: EdgeConfigs,
    size: int = 10,
    font_size: float = 0.9,
) -> svgwrite.Drawing:
    """Create SVG drawing for single or multiple plots.

    Args:
        x: 2D array (height × width) or 3D array (frames × height × width)
        ...
    """
    # Handle both 2D and 3D arrays
    if x.ndim == 2:
        x = x[np.newaxis, ...]  # Add frame dimension

    n_plots, height, width = x.shape

    # Setup drawing with correct number of plots
    padding = size * 4
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
        add_ticks_and_labels(dwg, size, width, height, edge, plot_idx, n_plots, x_offset, y_offset, font_size)

    return dwg


def create_animation_values(frames: List[ndarray], i: int, j: int, size: int) -> dict:
    """Create animation values for a specific position."""
    values = [frame[i, j] for frame in frames]
    sizes = [float(np.abs(v)) for v in values]  # Convert to float
    rect_widths = [s * size * 0.95 for s in sizes]
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
    edge: EdgeConfigs,
    size: int = 10,
    rate: int = 20,
    font_size: float = 0.9,
) -> svgwrite.Drawing:
    """Create single SVG with animated rectangles."""
    if len(frames) == 0:
        raise ValueError("No frames provided")

    if frames.ndim == 3:
        # frames = [frame[np.newaxis, ...] for frame in frames]
        frames = frames[np.newaxis, ...]
    frames = np.concat((frames[:, -1][:, np.newaxis, ...], frames), axis=1)
    frames = frames.transpose(0, 1, 3, 2)
    # frames = np.array([frame.transpose(0, 2, 1) for frame in frames])
    # width, height = frames[0].shape
    n_plots, n_frames, width, height = frames.shape

    padding = size * 2
    dwg = setup_drawing(width, height, size, n_plots, padding)
    for plot_idx in range(n_plots):
        frame_sequence = frames[plot_idx]
        x_offset, y_offset = get_plot_offset(plot_idx, width, height, size, padding)

        plot_group = dwg.g()

        non_zero = np.nonzero(frame_sequence[0])
        total_elements = len(non_zero[0])

        for i, j in tqdm(zip(non_zero[0], non_zero[1]), total=total_elements):
            anim_values = create_animation_values(frame_sequence, i, j, size)
            props = get_rect_properties(anim_values["values"][0], size)

            rect = dwg.rect(
                insert=(
                    f"{anim_values['initial_pos'][0] + x_offset:.1f}",
                    f"{anim_values['initial_pos'][1] + y_offset:.1f}",
                ),
                width=f"{props['rect_width']:.1f}",
                height=f"{props['rect_width']:.1f}",
                fill=props["fill_color"],
                stroke=props["stroke"],
                stroke_width=props["stroke_width"],
            )

            duration = f"{n_frames / rate}s"
            for attr, values in [
                ("width", anim_values["size_str"]),
                ("height", anim_values["size_str"]),
                ("x", ";".join(f"{float(pos_x) + x_offset:.1f}" for pos_x in anim_values["x_position_str"].split(";"))),
                ("y", ";".join(f"{float(pos_y) + y_offset:.1f}" for pos_y in anim_values["y_position_str"].split(";"))),
            ]:
                rect.add(dwg.animate(attributeName=attr, values=values, dur=duration, repeatCount="indefinite"))

            plot_group.add(rect)
            add_ticks_and_labels(dwg, size, width, height, edge, plot_idx, n_plots, x_offset, y_offset, font_size)
        dwg.add(plot_group)
    return dwg
