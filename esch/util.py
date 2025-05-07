# util.py
#   esch functions
# by: Noah Syrkis

# imports
import tempfile
from typing import Union

import darkdetect
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from pdf2image import convert_from_path
from reportlab.graphics import renderPDF
from svglib import svglib
from dataclasses import dataclass

# Types
Array = Union[np.ndarray, jnp.ndarray]

# Constants
BASE_SIZE: int = 1
PAD = 0.1
# BASE_FONT_SIZE: int = 10
# PAD: int = 10


@dataclass
class Esch:
    num: int
    row: int
    col: int

    @property
    def offset(self):  # dwg, ink, pos):
        return (np.array([(0, self.sub_h + PAD) if self.ratio > 1 else (self.sub_w + PAD, 0)]))[..., None]

    @property
    def step(self):
        return max(self.row, self.col)

    @property
    def ratio(self):
        return self.col / self.row

    @property
    def sub_w(self):
        return self.ratio if self.ratio < 1 else 1

    @property
    def sub_h(self):
        return 1 if self.ratio <= 1 else (1 / self.ratio)

    @property
    def fig_w(self):
        return (
            (self.sub_w if self.sub_w > self.sub_h else self.sub_w * self.num)
            + (self.num * PAD * (self.ratio > 1))
            + PAD
        )

    @property
    def fig_h(self):
        return (
            (self.sub_h if self.sub_h > self.sub_w else self.sub_h * self.num)
            + (self.num * PAD * (self.ratio < 1))
            + PAD
        )

    def __post_init__(self):
        self.dwg = svgwrite.Drawing(size=(f"{self.fig_w}pt", f"{self.fig_h}pt"))
        self.dwg["width"], self.dwg["height"], self.dwg["preserveAspectRatio"] = "100%", "100%", "xMidYMid meet"
        self.dwg.viewbox(0, 0, self.fig_w, self.fig_h)  # type: ignore


def init(*shape):
    return Esch(*shape)


def save(e: Esch, path):
    e.dwg.saveas(path)
    return e


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
