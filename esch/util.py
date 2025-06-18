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
    w: int
    h: int
    row: int = 1  # for small multiples
    col: int = 1  # for small multipels
    debug: bool = True

    def __post_init__(self: "Drawing"):
        # constants
        self.pad = min(self.w, self.h) / 10
        pad, w, h, row, col = self.pad, self.w, self.h, self.row, self.col

        # subplots dims
        self.sub_plot_width = pad + w + pad
        self.sub_plot_height = pad + h + pad

        # total dims
        self.total_width = row * self.sub_plot_width
        self.total_height = col * self.sub_plot_height

        # setup dwg
        self.dwg = svgwrite.Drawing(size=None, preserveAspectRatio="xMidYMid meet")
        self.dwg.viewbox(minx=0, miny=0, width=self.total_width, height=self.total_height)  # type: ignore

        # make groups or group
        gs = [(self.dwg.g(), i, j) for i in range(row) for j in range(col)]
        [g.translate(2 * pad + self.sub_plot_width * j, 2 * pad + self.sub_plot_height * i) for g, i, j in gs]
        self.gs = [g for g, _, _ in gs]

        # debug?
        self._debug() if self.debug else None

    def _debug(self):
        # add red box around viewbox of dwg
        self.dwg.add(
            self.dwg.rect(
                insert=(0, 0), size=(self.total_width, self.total_height), stroke="red", stroke_width=2, fill="none"
            )
        )


#


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
