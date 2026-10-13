"""Turn arrays of any rank into static or animated SVG drawings."""

from .draw import draw
from .drawing import Drawing, save, show
from .line import line_fn
from .plot import grid_fn, mesh_fn
from .voronoi import voronoi_fn

__all__ = [
    "Drawing",
    "draw",
    "grid_fn",
    "line_fn",
    "mesh_fn",
    "save",
    "show",
    "voronoi_fn",
]
