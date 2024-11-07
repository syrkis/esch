from ctypes.macholib import dyld  # type: ignore

dyld.DEFAULT_LIBRARY_FALLBACK.append("/opt/homebrew/lib")
from .plot import plot
from .data import prep


__all__ = ["plot", "prep"]
