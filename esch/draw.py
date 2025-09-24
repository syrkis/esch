# from dataclasses import dataclass
import svgwrite
import numpy as np
from itertools import product
from esch.atom import square_fn, circle_fn
from einops import rearrange


def draw(pat: str, arr, debug=False):
    arr: np.ndarray = np.array(arr)
    assert arr.ndim == len(pat.split())

    tar = f"({'n' if 'n' in pat else 1} {'c' if 'c' in pat else 1}) {'h' if 'h' in pat else 1} {'w' if 'w' in pat else 1} {'t' if 't' in pat else 1}"

    n: int = 1 if "n" not in pat else arr.shape[pat.split().index("n")]  # columns
    c: int = 1 if "c" not in pat else arr.shape[pat.split().index("c")]  # rows
    h: int = 1 if "h" not in pat else arr.shape[pat.split().index("h")]  # outer columns
    w: int = 1 if "w" not in pat else arr.shape[pat.split().index("w")]  # outer rows
    t: int = 1 if "t" not in pat else arr.shape[pat.split().index("t")]  # time

    e = Esching(n, c, h, w, t, pad=1, debug=debug)

    # Add empty dimensions if missing for any pattern element
    arr = rearrange(arr, f"{pat} -> {tar}")
    assert arr.shape[0] == e.gs.__len__()

    for idx, (sub, g) in enumerate(zip(arr, e.gs)):
        print(idx)
        for x in range(sub.shape[0]):
            for y in range(sub.shape[1]):
                square_fn(sub[x, y], x, y, e, g, e.fps)
        e.dwg.add(g)

    return e


# @dataclass
class Esching:
    def __init__(self, n: int, c: int, h: int, w: int, t: int, pad: int = 1, debug=False):
        # cfg
        self.pad = pad
        self.fps = 1

        # dimensions
        self.n = n  # outer rows
        self.c = c  # outer cols
        self.h = h - 1  # inner rows
        self.w = w - 1  # inner cols
        self.t = t  # time

        # derivatives
        self.sub_width = self.w + self.pad * 2
        self.sub_height = self.h + self.pad * 2
        self.tot_width = self.sub_width * self.c
        self.tot_height = self.sub_height * self.n

        # setup drawing
        self.dwg = svgwrite.Drawing(size=None, preserveAspectRatio="xMidYMid meet")
        self.dwg.viewbox(minx=0, miny=0, width=self.tot_width, height=self.tot_height)  # type: ignore
        self.dwg.defs.add(self.dwg.style("""* { stroke-width: 0.1; }"""))

        idxs = list(product(range(self.n), range(self.c)))
        self.gs = [self.dwg.g() for i, j in idxs]
        [g.translate(pad + self.sub_width * j, pad + self.sub_height * i) for g, (i, j) in zip(self.gs, idxs)]
        self.debug() if debug else None

    def save(self, filename):
        self.dwg.saveas(filename)

    def debug(self):
        print(self.gs.__len__())
        print("tot width  : \t\t", self.tot_width)
        print("tot height : \t\t", self.tot_height)
        print("sub width  : \t\t", self.sub_width)
        print("sub height : \t\t", self.sub_height)

        self.dwg.add(self.dwg.rect(insert=(0, 0), size=(self.tot_width, self.tot_height), stroke="red", fill="none"))
        for g in self.gs:
            g.add(
                self.dwg.rect(
                    insert=(0, 0),
                    size=(self.sub_width - self.pad * 2, self.sub_height - self.pad * 2),
                    stroke="blue",
                    fill="none",
                )
            )
            self.dwg.add(g)
