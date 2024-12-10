# base.py
# main hinton plot interface
# by: Noah Syrkis

from typing import Optional
from . import draw, data, edge
from einops import rearrange

# im
from numpy import ndarray
import numpy as np


class Plot:
    """Hinton plot visualization."""

    def __init__(
        self,
        array: ndarray,
        fps: int = 20,
        size: int = 10,
        edge: edge.EdgeConfigs = edge.EdgeConfigs(),
        font_size: float = 12,
    ):
        self.data = data.prep(array)
        self.rate = fps
        self.size = size
        self.edge = edge
        self._dwg = None
        self.png: Optional[bytes] = None
        self.font_size = font_size

    def static(self) -> None:
        """Create static plot."""
        self._dwg = draw.make(self.data, self.edge, self.size, self.font_size)

    def animate(self) -> None:
        """Create animated plot with optimized SVG."""
        self._dwg = draw.play(self.data, self.edge, self.size, self.rate, self.font_size)

    def save(self, path: str) -> None:
        """Save plot to file."""
        self._dwg.saveas(path)  # type: ignore


def mesh(
    array: np.ndarray,
    fps: int = 20,
    size: int = 10,
    path: Optional[str] = None,
    edge: edge.EdgeConfigs = edge.EdgeConfigs(),
    font_size: float = 0.9,
) -> Optional[Plot]:
    match array.ndim, array.shape:
        case 1, _:
            array = array[np.newaxis, ...]
            animated = False
        case 2, d if (min(d) / max(d)) < 0.05:
            animated = True
            array = rearrange(array, "t s -> 1 t 1 s")
        case 2, d if (min(d) / max(d)) >= 0.05:
            animated = False
        case 3, d if d[0] > 10:  # time or small multiples
            array = array[np.newaxis, ...]
            animated = True
        case 3, d if d[0] <= 10:  # animation with singles
            animated = False
        case 4, _:  # animation with multiples
            animated = True
        case _, _:
            animated = False

    if animated:
        step_size = int(np.floor(array.shape[0] / 1001) + 1)
        fps = int(fps / step_size)
        array = array[::step_size]

    p = Plot(array, fps, size, edge, font_size)
    p.animate() if animated else p.static()
    p._dwg.saveas(path) if path else None  # type: ignore
    return p  # type: ignore
