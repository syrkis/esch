# util.py
#   esch util functions
# by: Noah Syrkis

# imports
import tempfile

import os
import darkdetect
import matplotlib.pyplot as plt
import numpy as np
from pdf2image import convert_from_path
from reportlab.graphics import renderPDF
from svglib import svglib


def display_fn(img):
    with tempfile.NamedTemporaryFile(suffix=".svg") as svg_file, tempfile.NamedTemporaryFile(
        suffix=".pdf"
    ) as pdf_file, tempfile.NamedTemporaryFile(suffix=".png") as png_file:
        img.save(svg_file.name)
        img = svglib.svg2rlg(svg_file.name)
        assert img is not None
        renderPDF.drawToFile(img, pdf_file.name)
        images = convert_from_path(pdf_file.name)
        images[0].save(png_file.name, "PNG")

        img = np.array(images[0]) / 255.0

        # Set default maximum figure size (in inches)
        max_width = 12
        max_height = 8

        # Calculate figure size maintaining aspect ratio
        img_height, img_width = img.shape[:2]
        aspect_ratio = img_width / img_height

        if img_width / max_width > img_height / max_height:
            fig_width = max_width
            fig_height = fig_width / aspect_ratio
        else:
            fig_height = max_height
            fig_width = fig_height * aspect_ratio

        # Create figure with calculated size
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
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
        plt.show()
