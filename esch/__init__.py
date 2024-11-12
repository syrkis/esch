import platform

if platform.system() == "Darwin":
    from ctypes.macholib import dyld  # type: ignore

    dyld.DEFAULT_LIBRARY_FALLBACK.append("/opt/homebrew/lib")
from .plot import plot  # noqa
from .data import prep  # noqa
from .edge import EdgeConfig, EdgeConfigs  # noqa


__all__ = ["plot", "prep", "EdgeConfig", "EdgeConfigs"]
