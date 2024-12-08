# base.py
# main hinton plot interface
# by: Noah Syrkis

from typing import Optional
from . import draw, data, edge

# im
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cairosvg import svg2png
import darkdetect
import io


class Plot:
    """Hinton plot visualization."""

    def __init__(
        self,
        array: ndarray,
        fps: int = 20,
        size: int = 10,
        edge: edge.EdgeConfigs = edge.EdgeConfigs(),
        font_size: float = 0.9,
    ):
        self.data = data.prep(array)
        self.rate = fps
        self.size = size
        self.edge = edge
        # self.low_label = xlabel
        # self.left_label = ylabel
        # self.low_ticks = xticks
        # self.left_ticks = yticks
        self._dwg = None
        self.png: Optional[bytes] = None
        self.font_size = font_size

    def static(self) -> None:
        """Create static plot."""
        # if len(self.data.shape) > 2:
        # self.data = self.data[0]  # take first frame if animated
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


def tile(
    array,
    animated: bool = False,
    fps: int = 20,
    size: int = 10,  # not sure this is needed
    # xlabel: Optional[str] = None,
    # ylabel: Optional[str] = None,
    # xticks: Optional[List] = None,
    # yticks: Optional[List] = None,
    path: Optional[str] = None,
    edge: edge.EdgeConfigs = edge.EdgeConfigs(),
    font_size: float = 0.9,
) -> Optional[Plot]:  # todo dynamically chagne rate and step size, to keep it small
    array = np.array(array)
    # max frames around 1000
    if animated:
        step_size = int(np.floor(array.shape[0] / 1001) + 1)
        fps = int(fps / step_size)
        array = array[::step_size]
    """Create and optionally save a Hinton plot."""
    p = Plot(array, fps, size, edge, font_size)

    if animated:
        p.animate()
    else:
        p.static()

    # p.figure = esch_nb(p)
    p.png = esch_nb(p)  # type: ignore
    if path:
        p.save(path)
        return p
    return p


def esch_nb(p):
    # Convert SVG to PNG with high scale
    is_dark = darkdetect.isDark()
    if is_dark:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.axis("off")
    svg_png = svg2png(
        p._dwg.tostring(),
        scale=4,  # Increase scale factor
    )
    # Display the image with higher quality
    bytes = io.BytesIO(svg_png)  # type: ignore
    img = mpimg.imread(bytes, format="png")  # type: ignore
    # flip color if dark
    # set ax facecolor to black
    # img = img / img.max()
    if is_dark:
        img = (img.min(axis=(-1)) - img.max(axis=(-1))) < 0
    ax.imshow(
        img,
        interpolation="hermite",
        cmap="gray",
    )
    plt.tight_layout()
    # plt.show()
    return bytes
