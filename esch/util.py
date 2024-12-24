# util.py
#   esch util functions
# by: Noah Syrkis

# imports
import tempfile

import darkdetect
import matplotlib.pyplot as plt
import numpy as np
from pdf2image import convert_from_path
from reportlab.graphics import renderPDF
from svglib import svglib
from typing import Union
import jax.numpy as jnp


# Types
Array = Union[np.ndarray, jnp.ndarray]

# Constants
BASE_SIZE: int = 1
PADDING = 0.1
# BASE_FONT_SIZE: int = 10
# PADDING: int = 10


def display_fn(img):
    with (
        tempfile.NamedTemporaryFile(suffix=".svg") as svg_file,
        tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file,
        tempfile.NamedTemporaryFile(suffix=".png") as png_file,
    ):
        img.saveas(svg_file.name)
        img = svglib.svg2rlg(svg_file.name)
        assert img is not None
        renderPDF.drawToFile(img, pdf_file.name)
        images = convert_from_path(pdf_file.name)
        images[0].save(png_file.name, "PNG")

        img = np.array(images[0]) / 255.0

        # Create figure with calculated size
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(1 - img if darkdetect.isDark() else img)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(pad=0)
        fig.set_facecolor("black" if darkdetect.isDark() else "white")
        ax.set_facecolor("black" if darkdetect.isDark() else "white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # plt.show()
