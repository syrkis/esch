# base.py
#     main hinton plot interface
# by: Noah Syrkis

from typing import Optional
import numpy as np
from einops import rearrange
from . import data, draw, edge

from esch.util import Array


def mesh(
    act: Array,
    pos: Array = np.array([]),
    shp: str = "rect",  # circle or rect (for now)
    size: int = 10,  # maybe delete this
    fps: int = 1,
    path: Optional[str] = None,
    edge: edge.EdgeConfigs = edge.EdgeConfigs(),
    font_size: float = 0.9,
):
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

    act = data.prep(act)
    if animated:
        step_size = int(np.floor(act.shape[0] / 1001) + 1)
        fps = int(fps / step_size)
        act = act[::step_size]

    if animated:
        dwg = draw.play(act, pos, edge, size, fps, font_size)
    else:
        dwg = draw.make(act, pos, edge, size, font_size)
    if path:
        dwg.saveas(path)
    return dwg
