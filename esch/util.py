# util.py
#   esch functions
# by: Noah Syrkis

# imports
import tempfile
import darkdetect
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from pdf2image import convert_from_path
from reportlab.graphics import renderPDF
from svglib import svglib
from dataclasses import dataclass


# Types
@dataclass
class Drawing:
    h: int  # rows
    w: int  # rows
    row: int  # for small multiples
    col: int  # for small multipels
    pad: int = 1
    tick: float = 0.3
    debug: bool = False

    def __post_init__(self: "Drawing"):
        # constants
        # self.pad = float(min(self.w, self.h) / 8)
        pad, w, h, row, col = self.pad, self.w, self.h, self.row, self.col

        # subplots dims
        self.sub_width = pad + w + pad  #  w and pad on both sides
        self.sub_height = pad + h + pad  # h and pad on both sides

        # total dims
        self.total_height = row * self.sub_width  # could add pad again around that, but fuck it for now
        self.total_width = col * self.sub_height  # could add pad around that, but fuck it for now

        # setup dwg
        self.dwg = svgwrite.Drawing(size=None, preserveAspectRatio="xMidYMid meet")
        self.dwg.viewbox(minx=0, miny=0, width=self.total_width, height=self.total_height)  # type: ignore
        # set default stroke color to black and stroke width to 0.02
        self.dwg.defs.add(
            self.dwg.style("""
            * {
                stroke: black;
                stroke-width: 0.1;
            }
        """)
        )

        # make groups or group
        idxs = [(i, j) for i in range(row) for j in range(col)]
        self.gs = [self.dwg.g() for i, j in idxs]
        [g.translate(pad + self.sub_height * j, pad + self.sub_width * i) for g, (i, j) in zip(self.gs, idxs)]

        # debug?
        self._debug() if self.debug else None

    def _debug(self):
        # # add red box around viewbox of dwg
        # self.dwg.add(
        #     self.dwg.rect(
        #         insert=(0, 0), size=(self.total_width, self.total_height), stroke="red", stroke_width=1, fill="none"
        #     )
        # )

        # add blue rectangles to each group
        idxs = [(i, j) for i in range(self.row) for j in range(self.col)]
        for (i, j), g in zip(idxs, self.gs):
            # insert = (self.sub_plot_height * i, self.sub_plot_width * j)
            g.add(
                self.dwg.rect(
                    insert=(-self.pad, -self.pad),
                    size=(self.h + 2 * self.pad, self.w + 2 * self.pad),
                    stroke="blue",
                    stroke_width=0.1,
                    fill="none",
                )
            )
            g.add(
                self.dwg.rect(
                    insert=(-0, -0),
                    size=(self.h, self.w),
                    stroke="red",
                    stroke_width=0.1,
                    fill="none",
                )
            )
        # add separator lines between subplots
        for i in range(self.row + 1):
            y = i * self.sub_width
            self.dwg.add(self.dwg.line(start=(0, y), end=(self.total_width, y), stroke="green", stroke_width=0.1))

        for j in range(self.col + 1):
            x = j * self.sub_height
            self.dwg.add(self.dwg.line(start=(x, 0), end=(x, self.total_height), stroke="green", stroke_width=0.1))


def save(dwg, filename, scale=80):
    dwg.saveas(filename)


def show(img, dpi=300):
    with tempfile.NamedTemporaryFile(suffix=".svg") as svg_file, tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file:
        img.saveas(svg_file.name)
        img = svglib.svg2rlg(svg_file.name)
        assert img is not None
        renderPDF.drawToFile(img, pdf_file.name)
        img = np.array(convert_from_path(pdf_file.name, dpi=2000)[0])
        img = img / 255.0

        # Create figure with calculated size
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(1 - img if darkdetect.isDark() else img)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(pad=0)
        fig.set_facecolor("black" if darkdetect.isDark() else "white")
        ax.set_facecolor("black" if darkdetect.isDark() else "white")
        for spine in ax.spines.values():
            spine.set_visible(False)
