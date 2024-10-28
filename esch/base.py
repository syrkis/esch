# base.py
# main hinton plot interface
# by: Noah Syrkis

from typing import Optional, Union
from jax import Array
import jax.numpy as jnp
from . import draw, data
from tqdm import tqdm


class Plot:
    """Hinton plot visualization."""

    def __init__(
        self,
        array: Array,
        rate: int = 20,
        size: int = 10,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        """Initialize plot with data and parameters."""
        self.data = data.prep(array)  # preprocess data
        self.rate = rate
        self.size = size
        self.xlabel = xlabel
        self.ylabel = ylabel
        self._dwg = None

    def static(self) -> None:
        """Create static plot."""
        if len(self.data.shape) > 2:
            self.data = self.data[0]  # take first frame if animated
        self._dwg = draw.make(self.data, self.size)

    def animate(self) -> None:
        """Create animated plot with optimized SVG."""
        if len(self.data.shape) < 3:
            raise ValueError("Data must be 3D for animation")

        # Pass entire data tensor at once
        self._dwg = draw.play(self.data, self.rate)

    def save(self, path: str) -> None:
        """Save plot to file."""
        if self._dwg is None:
            if len(self.data.shape) > 2:
                self.animate()
            else:
                self.static()
        self._dwg.saveas(path)


def plot(
    array: Array,
    animated: bool = False,
    rate: int = 20,
    size: int = 10,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    path: Optional[str] = None,
) -> Optional[Plot]:
    """Create and optionally save a Hinton plot."""
    p = Plot(array, rate, size, xlabel, ylabel)

    if animated:
        p.animate()
    else:
        p.static()

    if path:
        p.save(path)
        return None
    return p
