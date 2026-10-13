"""Drawing: a grid of panels in one SVG, with saving and notebook display."""

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from math import ceil, sqrt
from numbers import Integral, Real
from string import Formatter

import numpy as np
import svgwrite


def _positive_integer(value):
    return not isinstance(value, bool) and isinstance(value, Integral) and value >= 1


@dataclass
class Drawing:
    """A grid of panels; h and w count cells, row and col count panels."""

    h: int
    w: int
    row: int = 1
    col: int = 1
    pad: float = 1
    tick: float = 0.3
    debug: bool = False
    panels: int | None = None
    gap: float | None = 5

    def __post_init__(self):
        count = self._count()
        self._layout()
        self._canvas(count)

    def _count(self):
        """Validate the fields; return the number of panels drawn."""
        for name in ("h", "w", "row", "col"):
            if not _positive_integer(getattr(self, name)):
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.pad) or self.pad < 0.55:
            raise ValueError("pad must be finite and at least 0.55")
        if not np.isfinite(self.tick) or not 0 < self.tick <= 0.5:
            raise ValueError("tick must be in (0, 0.5]")
        count = self.row * self.col if self.panels is None else self.panels
        if not _positive_integer(count) or count > self.row * self.col:
            raise ValueError("panels must fit within the panel grid")
        return count

    def _layout(self):
        """Panel spacing and canvas size."""
        if self.gap is None:
            self.sub_width = self.w - 1 + 2 * self.pad
            self.sub_height = self.h - 1 + 2 * self.pad
            self.total_width = self.col * self.sub_width
            self.total_height = self.row * self.sub_height
            return
        if isinstance(self.gap, bool) or not isinstance(self.gap, Real) or not np.isfinite(self.gap) or self.gap <= 0:
            raise ValueError("gap must be a positive finite divisor")
        # Panel bounds span w × h cells. Padding belongs to the outer canvas;
        # the requested gap is the space between adjacent panel bounds.
        self.sub_width = self.w + self.w / self.gap
        self.sub_height = self.h + self.h / self.gap
        self.total_width = self.w - 1 + 2 * self.pad + (self.col - 1) * self.sub_width
        self.total_height = self.h - 1 + 2 * self.pad + (self.row - 1) * self.sub_height
        if not np.isfinite([self.sub_width, self.sub_height, self.total_width, self.total_height]).all():
            raise ValueError("gap is too small for finite layout dimensions")

    def _canvas(self, count):
        """The SVG document with one translated group per panel."""
        self.dwg = svgwrite.Drawing(size=("100%", "100%"), preserveAspectRatio="xMidYMid meet")
        self.dwg.viewbox(0, 0, self.total_width, self.total_height)
        self.gs = []
        for index in range(count):
            row, col = divmod(index, self.col)
            group = self.dwg.g()
            group.translate(self.pad + col * self.sub_width, self.pad + row * self.sub_height)
            self.gs.append(group)
            self.dwg.add(group)
            if self.debug:
                group.add(
                    self.dwg.rect(insert=(-0.5, -0.5), size=(self.w, self.h), fill="none", stroke="blue", stroke_width=0.02)
                )
        if self.debug:
            self.dwg.add(
                self.dwg.rect(size=(self.total_width, self.total_height), fill="none", stroke="red", stroke_width=0.02)
            )

    def save(self, filename):
        """Save SVG; named panel fields such as {n} export separate drawings."""
        filename = str(filename)
        fields = self._fields(filename)
        if not fields:
            self.dwg.saveas(filename.format())
            return
        axis_indices = [self._filename_axes[field] for field in fields]
        combinations = list(product(*(range(self._panel_shape[index]) for index in axis_indices)))
        filenames = [filename.format(**dict(zip(fields, values, strict=True))) for values in combinations]
        if len(set(filenames)) != len(filenames):
            raise ValueError("Filename formatting must give every slice a unique path")
        for values, destination in zip(combinations, filenames, strict=True):
            selected = dict(zip(axis_indices, values, strict=True))
            self._panel_slice(selected).dwg.saveas(destination)

    def _fields(self, filename):
        """Panel axes named in a filename, in order of first use."""
        fields = []
        for _, field, spec, conversion in Formatter().parse(filename):
            if field is None:
                continue
            if field not in getattr(self, "_filename_axes", {}):
                raise ValueError(f"Unknown filename panel axis {field!r}; use an explicit, ungrouped panel axis")
            if conversion is not None or "{" in spec or "}" in spec:
                raise ValueError("Filename fields support integer formatting, without conversions or nested fields")
            if field not in fields:
                fields.append(field)
        return fields

    def _panel_slice(self, selected):
        """Copy rendered panels so overlays, animation, and global scales survive."""
        indices = np.arange(len(self.gs)).reshape(self._panel_shape)
        indices = indices[tuple(selected.get(i, slice(None)) for i in range(len(self._panel_shape)))].reshape(-1)
        remaining = {
            name: size
            for i, (name, size) in enumerate(zip(self._panel_axes, self._panel_shape, strict=True))
            if i not in selected
        }
        if set(self._panel_axes) <= {"row", "col"}:
            columns = remaining.get("col", 1)
        elif self._save_columns is not None:
            columns = min(self._save_columns, len(indices))
        elif set(self._panel_axes) <= {"n", "c"}:
            columns = remaining.get("c", 1)
        else:
            columns = ceil(sqrt(len(indices)))
        result = Drawing(
            self.h,
            self.w,
            ceil(len(indices) / columns),
            columns,
            pad=self.pad,
            tick=self.tick,
            debug=self.debug,
            panels=len(indices),
            gap=self.gap,
        )
        result.dwg.defs.elements = deepcopy(self.dwg.defs.elements)
        for group, index in zip(result.gs, indices, strict=True):
            original = self.gs[index]
            group.elements = deepcopy(original.elements)
            group.attribs.update({key: deepcopy(value) for key, value in original.attribs.items() if key != "transform"})
        return result

    saveas = save

    def tostring(self):
        return self.dwg.tostring()

    def _repr_svg_(self):
        return self.tostring()


def save(dwg, filename, scale=80):
    """Save a Drawing or svgwrite.Drawing. scale is retained for compatibility."""
    dwg.saveas(str(filename))


def show(img, dpi=300):
    """Display native SVG in a notebook, preserving animation; dpi is unused."""
    from IPython.display import SVG, display

    del dpi  # Accepted for compatibility; SVG has no resolution.
    return display(SVG(data=img.tostring()))
