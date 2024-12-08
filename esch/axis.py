from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class axis:
    side: Optional[str]
    ticks: Optional[List[Tuple[int, str]]] = None
    label: Optional[str] = None
    plots: Optional[List[int]] = None  # for small multiples


@dataclass
class Axes:
    l: Optional[axis] = None  # noqa
    r: Optional[axis] = None
    t: Optional[axis] = None
    b: Optional[axis] = None
