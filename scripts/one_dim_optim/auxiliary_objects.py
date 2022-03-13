from __future__ import annotations

from typing import List
import sys
from numbers import Real

if sys.version_info >= (3, 8):
    from typing import TypedDict 
else:
    from typing_extensions import TypedDict


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
