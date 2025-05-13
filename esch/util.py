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


def init(x_range, y_range, rows=1, cols=1):
    dwg = svgwrite.Drawing(size=None, preserveAspectRatio="xMidYMid meet")
    dwg.viewbox(-1, -1, (x_range + 1) * rows, (y_range + 1) * cols)
    return dwg


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
