# base.py
# main hinton plot interface
# by: Noah Syrkis

from typing import Optional
from . import draw, data, edge

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
        # if len(self.data.shape) < 3:
        # raise ValueError("Data must be 3D for animation")

        # Pass entire data tensor at once
        # self.data = np.concat((self.data[-1][np.newaxis, ...], self.data), axis=0)
        self._dwg = draw.play(self.data, self.edge, self.size, self.rate, self.font_size)

    def save(self, path: str) -> None:
        """Save plot to file."""
        # if self._dwg is None:
        # if len(self.data.shape) > 2:
        # self.animate()
        # else:
        # self.static()
        self._dwg.saveas(path)  # type: ignore


def mesh(
    array,
    animated: bool = False,
    fps: int = 20,
    size: int = 10,
    path: Optional[str] = None,
    edge: edge.EdgeConfigs = edge.EdgeConfigs(),
    font_size: float = 0.9,
) -> Optional[Plot]:
    array = np.array(array)
    if animated:
        step_size = int(np.floor(array.shape[0] / 1001) + 1)
        fps = int(fps / step_size)
        array = array[::step_size]

    p = Plot(array, fps, size, edge, font_size)

    if animated:
        p.animate()
    else:
        p.static()

    # Pass the drawing object directly
    # util.display_fn(p._dwg)

    if path:
        p.save(path)
        return p
    return p
