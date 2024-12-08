import platform

if platform.system() == "Darwin":
    from ctypes.macholib import dyld  # type: ignore

    dyld.DEFAULT_LIBRARY_FALLBACK.append("/opt/homebrew/lib")

from .tile import tile  # noqa
from .data import prep  # noqa
from .edge import EdgeConfig, EdgeConfigs  # noqa
from .ring import ring
from .axis import axis


__all__ = ["tile", "prep", "EdgeConfig", "EdgeConfigs", "ring", "axis"]
