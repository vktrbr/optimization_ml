from __future__ import annotations
from typing import List, TypedDict
from numbers import Real


class History(TypedDict):
    """
    Class with an optimization history
    """
    iteration: List[int]
    middle_point: List[Real]
    f_value: List[Real]
    left_point: List[Real]
    right_point: List[Real]


class Point(TypedDict):
    """
    Class with an output optimization point
    """
    point: Real
    f_value: Real
