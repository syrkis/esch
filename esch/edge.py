from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import svgwrite


@dataclass
class EdgeConfig:
    ticks: Optional[List[Tuple[float, str]]] = None
    label: Optional[Union[str, List[str]]] = None
    show_on: Union[str, List[int], None] = None


@dataclass
class EdgeConfigs:
    top: Optional[EdgeConfig] = None
    right: Optional[EdgeConfig] = None
    bottom: Optional[EdgeConfig] = None
    left: Optional[EdgeConfig] = None


def should_show_edge(config: EdgeConfig, plot_idx: int, n_plots: int) -> bool:
    """Determine if a particular edge should be shown on a plot."""
    if config is None:
        return False
    if config.show_on == "all":
        return True
    if config.show_on == "first" and plot_idx == 0:
        return True
    if config.show_on == "last" and plot_idx == n_plots - 1:
        return True
    if isinstance(config.show_on, list) and plot_idx in config.show_on:
        return True
    return False


def draw_tick_line(dwg, x1, y1, x2, y2):
    """Draw a tick line."""
    return dwg.line(start=(x1, y1), end=(x2, y2), stroke="black", stroke_width=0.5)


def add_tick_label(dwg, text, x, y, anchor, rotation=None, font_size=0.9):
    """Add tick label text to the drawing."""
    text_params = {
        "insert": (x, y),
        "text_anchor": anchor,
        "dominant_baseline": "middle",
        "font_size": f"{font_size}px",
    }
    if rotation:
        text_params["transform"] = rotation
    return dwg.text(text, **text_params)


def add_axis_label(dwg, label, x, y, anchor, rotation=None, font_size=0.9):
    """Add an axis label to the drawing."""
    text_params = {"insert": (x, y), "text_anchor": anchor, "font_size": f"{font_size}px"}
    if rotation:
        text_params["transform"] = rotation
    return dwg.text(label, **text_params)


def add_ticks_and_labels(
    dwg: svgwrite.Drawing,
    size: int,
    width: int,
    height: int,
    edge_configs: EdgeConfigs,
    plot_idx: int,
    n_plots: int,
    x_offset: float = 0,
    y_offset: float = 0,
    font_size: float = 0.9,
) -> None:
    """Add axis ticks and labels to the drawing with offset support."""
    tick_length = size * 0.6
    text_offset = size * 1.2
    tick_offset = size * 0.6

    tick_group = dwg.g()

    for edge_name, config in edge_configs.__dict__.items():
        if should_show_edge(config, plot_idx, n_plots):
            ticks = config.ticks
            label = config.label

            if edge_name == "bottom":
                y_start = height * size + tick_offset + y_offset
                if ticks:
                    for pos, tick_label in ticks:
                        x = pos * size + size / 2 + x_offset
                        tick_group.add(draw_tick_line(dwg, x, y_start, x, y_start + tick_length))
                        tick_group.add(
                            add_tick_label(
                                dwg, tick_label, x, y_start + tick_length + text_offset, "middle", font_size=font_size
                            )
                        )

                # Use plot-specific label if label is a list
                if isinstance(label, list) and plot_idx < len(label):
                    axis_label = label[plot_idx]
                else:
                    axis_label = label

                if axis_label:
                    tick_group.add(
                        add_axis_label(
                            dwg,
                            axis_label,
                            width * size / 2 + x_offset,
                            y_start + tick_length + size,
                            "middle",
                            font_size=font_size,
                        )
                    )

            elif edge_name == "left":
                x_start = -tick_offset + x_offset
                if ticks:
                    for pos, tick_label in ticks:
                        y = pos * size + size / 2 + y_offset
                        tick_group.add(draw_tick_line(dwg, x_start, y, x_start - tick_length, y))
                        tick_group.add(
                            add_tick_label(
                                dwg, tick_label, x_start - tick_length - text_offset, y, "end", font_size=font_size
                            )
                        )

                # Use plot-specific label if label is a list
                if isinstance(label, list) and plot_idx < len(label):
                    axis_label = label[plot_idx]
                else:
                    axis_label = label

                if axis_label:
                    tick_group.add(
                        add_axis_label(
                            dwg,
                            axis_label,
                            x_start - tick_length - size,
                            height * size / 2 + y_offset,
                            "middle",
                            f"rotate(-90, {x_start - tick_length - size}, {height * size / 2 + y_offset})",
                            font_size=font_size,
                        )
                    )

            # Extend logic to 'top' and 'right' edges if needed
            elif edge_name == "top":
                y = y_offset - tick_offset  # Correct positioning taking into account y_offset
                if ticks:
                    for pos, tick_label in ticks:
                        x = pos * size + size / 2 + x_offset
                        tick_group.add(draw_tick_line(dwg, x, y, x, y - tick_length))
                        tick_group.add(
                            add_tick_label(
                                dwg, tick_label, x, y - tick_length - text_offset, "middle", font_size=font_size
                            )
                        )

                if isinstance(label, list) and plot_idx < len(label):
                    axis_label = label[plot_idx]
                else:
                    axis_label = label

                if axis_label:
                    tick_group.add(
                        add_axis_label(
                            dwg,
                            axis_label,
                            width * size / 2 + x_offset,
                            y - tick_length - size,
                            "middle",
                            font_size=font_size,
                        )
                    )

            elif edge_name == "right":  # Correct approach for the right edge
                x = width * size + tick_offset + x_offset
                if ticks:
                    for pos, tick_label in ticks:
                        y = pos * size + size / 2 + y_offset
                        tick_group.add(draw_tick_line(dwg, x, y, x + tick_length, y))
                        tick_group.add(
                            add_tick_label(
                                dwg, tick_label, x + tick_length + text_offset, y, "start", font_size=font_size
                            )
                        )

                axis_label = (
                    label if not isinstance(label, list) else (label[plot_idx] if plot_idx < len(label) else None)
                )
                if axis_label:
                    tick_group.add(
                        add_axis_label(
                            dwg,
                            axis_label,
                            x + tick_length + size,
                            height * size / 2 + y_offset,
                            "middle",
                            f"rotate(90, {x + tick_length + size}, {height * size / 2 + y_offset})",
                            font_size=font_size,
                        )
                    )

    dwg.add(tick_group)
