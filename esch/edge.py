from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import svgwrite


@dataclass
class EdgeConfig:
    ticks: Optional[List[Tuple[float, str]]] = None
    label: Optional[str] = None
    show_on: Union[str, List[int], None] = None


@dataclass
class EdgeConfigs:
    top: Optional[EdgeConfig] = None
    right: Optional[EdgeConfig] = None
    bottom: Optional[EdgeConfig] = None
    left: Optional[EdgeConfig] = None


def should_show_edge(config: EdgeConfig, plot_idx: int, n_plots: int) -> bool:
    """Determine if a particular edge should be shown on a plot."""
    if config is None or config.show_on is None:
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
) -> None:
    """Add axis ticks and labels to the drawing with offset support."""
    tick_length = size * 0.3
    text_offset = size * 0.6
    tick_offset = size * 0.4

    y_start, x_start = 0, 0

    # Create group for ticks and labels
    tick_group = dwg.g()

    for edge_name, config in edge_configs.__dict__.items():
        if should_show_edge(config, plot_idx, n_plots):
            ticks = config.ticks
            label = config.label

            if edge_name == "bottom" and ticks:
                for pos, tick_label in ticks:
                    x = pos * size + size / 2 + x_offset
                    y_start = height * size + tick_offset + y_offset

                    # Draw tick mark at the bottom
                    tick_group.add(
                        dwg.line(
                            start=(x, y_start),
                            end=(x, y_start + tick_length),
                            stroke="black",
                            stroke_width=0.5,
                        )
                    )
                    tick_group.add(
                        dwg.text(
                            tick_label,
                            insert=(x, y_start + tick_length + text_offset),
                            text_anchor="middle",
                            dominant_baseline="hanging",
                            font_size=f"{size * 0.6}px",
                        )
                    )
                # Add X-axis label if provided
                if label:
                    tick_group.add(
                        dwg.text(
                            label,
                            insert=(width * size / 2 + x_offset, y_start + tick_length + size),
                            text_anchor="middle",
                            font_size=f"{size * 0.6}px",
                        )
                    )

            elif edge_name == "left" and ticks:
                for pos, tick_label in ticks:
                    y = pos * size + size / 2 + y_offset
                    x_start = -tick_offset + x_offset

                    # Draw tick mark on the left
                    tick_group.add(
                        dwg.line(start=(x_start, y), end=(x_start - tick_length, y), stroke="black", stroke_width=0.5)
                    )
                    tick_group.add(
                        dwg.text(
                            tick_label,
                            insert=(x_start - tick_length - text_offset, y),
                            text_anchor="end",
                            dominant_baseline="middle",
                            font_size=f"{size * 0.6}px",
                        )
                    )
                # Add Y-axis label if provided
                if label:
                    tick_group.add(
                        dwg.text(
                            label,
                            insert=(x_start - tick_length - size, height * size / 2 + y_offset),
                            text_anchor="middle",
                            font_size=f"{size * 0.6}px",
                            transform=f"rotate(-90, {x_start - tick_length - size}, {height * size / 2 + y_offset})",
                        )
                    )
            # Extend logic to 'top' and 'right' edges if needed

    dwg.add(tick_group)
