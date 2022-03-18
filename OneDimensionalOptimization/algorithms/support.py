from __future__ import annotations
from typing import List, TypedDict, AnyStr
from numbers import Real, Integral


class HistoryGSS(TypedDict):
    """
    Class with an optimization history of GSS
    """
    iteration: List[Integral]
    middle_point: List[Real]
    f_value: List[Real]
    left_point: List[Real]
    right_point: List[Real]


class HistorySPI(TypedDict):
    """
    Class with an optimization history of SPI
    """
    iteration: List[Integral]
    f_value: List[Real]
    x0: List[Real]
    x1: List[Real]
    x2: List[Real]


class HistoryBrant(TypedDict):
    """
    Class with an optimization history of Brant's algorithm
    """
    iteration: List[Integral]

    f_least: List[Real]
    f_middle: List[Real]
    f_largest: List[Real]

    x_least: List[Real]
    x_middle: List[Real]
    x_largest: List[Real]

    left_bound: List[Real]
    right_bound: List[Real]

    type_step: List[AnyStr]


class Point(TypedDict):
    """
    Class with an output optimization point
    """
    point: Real
    f_value: Real
