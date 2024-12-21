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
        act: ndarray,
        pos: ndarray,
        fps: int = 20,
        size: int = 10,
        edge: edge.EdgeConfigs = edge.EdgeConfigs(),
        font_size: float = 12,
    ):
        self.data = data.prep(act)
        self.pos = pos
        self.rate = fps
        self.size = size
        self.edge = edge
        self._dwg = None
        self.png: Optional[bytes] = None
        self.font_size = font_size

    def static(self) -> None:
        """Create static plot."""
        self._dwg = draw.make(self.data, self.pos, self.edge, self.size, self.font_size)

    def animate(self) -> None:
        """Create animated plot with optimized SVG."""
        self._dwg = draw.play(self.data, self.pos, self.edge, self.size, self.rate, self.font_size)

    def save(self, path: str) -> None:
        """Save plot to file."""
        self._dwg.saveas(path)  # type: ignore


def mesh(
    act: np.ndarray,
    pos: np.ndarray = np.array([]),
    size: int = 10,
    fps: int = 1,
    path: Optional[str] = None,
    edge: edge.EdgeConfigs = edge.EdgeConfigs(),
    font_size: float = 0.9,
) -> Optional[Plot]:
    match act.ndim, act.shape:
        case 1, _:
            act = act[np.newaxis, np.newaxis, ...]
            animated = False
        case 2, d if (min(d) / max(d)) < 0.05:
            animated = True
            act = rearrange(act, "t s -> 1 t 1 s")
        case 2, d if (min(d) / max(d)) >= 0.05:
            animated = False
            act = act[np.newaxis, ...]
        case 3, d if d[0] > 10:  # time or small multiples
            act = act[np.newaxis, ...]
            animated = True
        case 3, d if d[0] <= 10:  # animation with singles
            animated = False
        case 4, _:  # animation with multiples
            animated = True
        case _, _:
            animated = False

    if animated:
        step_size = int(np.floor(act.shape[0] / 1001) + 1)
        fps = int(fps / step_size)
        act = act[::step_size]

    p = Plot(act, pos, fps, size, edge, font_size)
    p.animate() if animated else p.static()
    p._dwg.saveas(path) if path else None  # type: ignore
    return p  # type: ignore
