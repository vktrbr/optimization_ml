from __future__ import annotations
from typing import List, TypedDict
from numbers import Real


class HistoryGSS(TypedDict):
    """
    Class with an optimization history of GSS
    """
    iteration: List[int]
    middle_point: List[Real]
    f_value: List[Real]
    left_point: List[Real]
    right_point: List[Real]


class HistorySPI(TypedDict):
    """
    Class with an optimization history of SPI
    """
    iteration: List[int]
    f_value: List[Real]
    x_left: List[Real]
    x_right: List[Real]
    x_middle: List[Real]


class Point(TypedDict):
    """
    Class with an output optimization point
    """
    point: Real
    f_value: Real
