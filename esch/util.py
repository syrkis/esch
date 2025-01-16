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
import svgwrite


# Types
Array = Union[np.ndarray, jnp.ndarray]

# Constants
BASE_SIZE: int = 1
PADDING = 0.1
# BASE_FONT_SIZE: int = 10
# PADDING: int = 10


def display_fn(img, dpi=300):
    with (
        tempfile.NamedTemporaryFile(suffix=".svg") as svg_file,
        tempfile.NamedTemporaryFile(suffix=".pdf") as pdf_file,
        # tempfile.NamedTemporaryFile(suffix=".png") as png_file,
    ):
        img.saveas(svg_file.name)
        img = svglib.svg2rlg(svg_file.name)
        assert img is not None
        renderPDF.drawToFile(img, pdf_file.name)
        img = np.array(convert_from_path(pdf_file.name, dpi=2000)[0])
        img = img / 255.0
        # images.save(png_file.name, "PNG")

        # Create figure with calculated size
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
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


# %% utils stuff that should probably be in a separate file #######################
def setup_drawing(ink: Array, pos: Array):
    # pos = pos[0, 0, :, :]
    # print(pos.shape, ink.shape)
    # exit()
    n_plots, n_points, n_steps = ink.shape
    print(f"n_plots: {n_plots}, n_points: {n_points}, n_steps: {n_steps}")
    if pos.ndim == 2:
        print(f"width: {pos[:, 0].max()}, height: {pos[:, 1].max()}")
        sub_plot_width, sub_plot_height = pos[:, 0].max().item() + PADDING, pos[:, 1].max().item() + PADDING
    else:
        print(f"width: {pos[:, :, :, 0].max()}, height: {pos[:, :, 1].max()}")
        sub_plot_width, sub_plot_height = pos[:, :, :, 0].max().item() + PADDING, pos[:, :, :, 1].max().item() + PADDING
    ratio = sub_plot_width / sub_plot_height
    # exit()
    if ratio > 1:
        total_width = sub_plot_width + PADDING
        total_height = (sub_plot_height) * ink.shape[0] + PADDING
        offset = np.array([0, sub_plot_height])
    else:
        total_width = (sub_plot_width) * ink.shape[0] + PADDING
        total_height = sub_plot_height + PADDING
        offset = np.array([sub_plot_width, 0])

    offset = offset if pos.ndim == 2 else offset[None, None, :, None]

    dwg = svgwrite.Drawing(size=(f"{total_width}pt", f"{total_height}pt"))
    dwg["width"], dwg["height"] = "100%", "100%"
    dwg["preserveAspectRatio"] = "xMidYMid meet"
    # print(width, height, total_height, total_width)
    dwg.viewbox(0, 0, total_width, total_height)  # TODO: add padding
    dwg.defs.add(dwg.style("text {font-family: 'Computer Modern', 'serif';}"))
    # print(total_width, total_height)
    # exit()
    return dwg, offset
