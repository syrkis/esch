# combining
import numpy as np
from esch.plot import draw
from esch.util import setup_drawing, display_fn


def sims(geo, pos, **kwargs):
    non_zero_x, non_zero_y = np.where(geo != 0)
    geo = np.array([non_zero_x, non_zero_y]).T
    ink = np.ones(geo.shape[0])[None, ..., None]
    dwg = setup_drawing(ink, geo[None, ...])
    dwg = draw(dwg, ink, geo)
    ink = np.ones(pos.shape[1])[None, ..., None]
    dwg = draw(dwg, ink, pos, shp="circle")
    path = kwargs.get("path", None)
    dwg.saveas(path) if path else display_fn(dwg)
    return
