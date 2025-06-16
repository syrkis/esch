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


# Types


def init(x_range, y_range, rows=1, cols=1, pad=1, border=False):
    dwg = svgwrite.Drawing(size=None, preserveAspectRatio="xMidYMid meet")
    dwg.viewbox(-pad, -pad, (x_range + pad) * rows, (y_range + pad) * cols)
    dwg.add(dwg.rect(insert=(-pad, -pad), size=((x_range + pad) * rows, (y_range + pad) * cols), fill="white"))  # bg

    rows, cols = cols, rows

    if border:
        if rows > 1 or cols > 1:
            # Border around each subplot
            for row in range(rows):
                for col in range(cols):
                    x_offset = col * (x_range + pad)
                    y_offset = row * (y_range + pad)
                    dwg.add(
                        dwg.rect(
                            insert=(x_offset - pad / 2, y_offset - pad / 2),
                            size=(x_range, y_range),
                            fill="none",
                            stroke="black",
                            stroke_width=0.1,
                        )
                    )
        else:
            # Border around the entire plot
            dwg.add(
                dwg.rect(
                    insert=(-pad / 2, -pad / 2), size=(x_range, y_range), fill="none", stroke="black", stroke_width=0.1
                )
            )
    # if (rows > 1 or cols > 1) and line:
    # Vertical lines between columns
    # if cols > 1:
    # for i in range(1, cols):
    # x = i * (x_range + 1) - 1
    # dwg.add(dwg.line(start=(x, +1), end=(x, (y_range - 1) * rows - 1), stroke="black", stroke_width=0.1))

    # Horizontal lines between rows
    # if rows > 1:
    # for i in range(1, rows):
    # y = i * (y_range + 1) - 1
    # dwg.add(dwg.line(start=(+1, y), end=((x_range - 1) * cols - 1, y), stroke="black", stroke_width=0.1))
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
