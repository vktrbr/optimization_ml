from __future__ import annotations
from typing import List, TypedDict, AnyStr, Tuple
from numbers import Real, Integral


class Point(TypedDict):
    """
    Class with an output optimization point
    """
    point: Real
    f_value: Real


class HistoryGDFS(TypedDict):
    """
    Class with an optimization history of GDFS
    (gradient descent with fractal step)
    """
    iteration: List[Integral]
    f_value: List[Real]
    x: List[Real]