# base.py
# main hinton plot interface
# by: Noah Syrkis

from typing import Optional, Union
from . import draw, data
from tqdm import tqdm
from typing import List
from numpy import ndarray
import numpy as np


class Plot:
    """Hinton plot visualization."""

    def __init__(
        self,
        array: ndarray,
        rate: int = 20,
        size: int = 10,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xticks: Optional[List] = None,  # Add these parameters
        yticks: Optional[List] = None,
    ):
        self.data = data.prep(array)
        self.rate = rate
        self.size = size
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xticks = xticks
        self.yticks = yticks
        self._dwg = None

    def static(self) -> None:
        """Create static plot."""
        if len(self.data.shape) > 2:
            self.data = self.data[0]  # take first frame if animated
        self._dwg = draw.make(self.data, self.xlabel, self.ylabel, self.xticks, self.yticks, self.size)

    def animate(self) -> None:
        """Create animated plot with optimized SVG."""
        if len(self.data.shape) < 3:
            raise ValueError("Data must be 3D for animation")

        # Pass entire data tensor at once
        self._dwg = draw.play(self.data, self.xlabel, self.ylabel, self.xticks, self.yticks, self.size, self.rate)

    def save(self, path: str) -> None:
        """Save plot to file."""
        if self._dwg is None:
            if len(self.data.shape) > 2:
                self.animate()
            else:
                self.static()
        self._dwg.saveas(path)  # type: ignore


def plot(
    array,
    animated: bool = False,
    rate: int = 20,
    size: int = 10,  # not sure this is needed
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xticks: Optional[List] = None,
    yticks: Optional[List] = None,
    path: Optional[str] = None,
) -> Optional[Plot]:  # todo dynamically chagne rate and step size, to keep it small
    array = np.array(array)
    # max frames around 1000
    if animated:
        step_size = int(np.floor(array.shape[0] / 1000) + 1)
        rate = int(rate / step_size)
        array = array[::step_size]
    """Create and optionally save a Hinton plot."""
    # if yticks is not None:
    # y_ticks = [((array.shape[-2] - pos) % array.shape[-2], label) for pos, label in yticks]
    p = Plot(array, rate, size, xlabel, ylabel, xticks, yticks)

    if animated:
        p.animate()
    else:
        p.static()

    if path:
        p.save(path)
        return None
    return p
