from __future__ import annotations
from typing import List, TypedDict, AnyStr, Literal
from numbers import Real, Integral


class Point(TypedDict):
    """
    Class with an output optimization point
    """
    point: Real
    f_value: Real


class HistoryGradDescent(TypedDict):
    """
    Class with an optimization history of gradient descent methods
    """
    iteration: List[Integral]
    f_value: List[Real]
    f_grad_norm: List[Real]
    x: List[Real]
    message: AnyStr


def update_history_grad_descent(history: HistoryGradDescent, values: List) -> HistoryGradDescent:
    """
    Update HistoryGradDescent with values, which contains iteration, f_value, f_grad_norm, x as a list
    :param history: object of HistoryGradDescent
    :param values: new values that need to append in history in order iteration, f_value, f_grad_norm, x
    :return: updated history
    """
    key: Literal['iteration', 'f_value', 'f_grad_norm', 'x']
    for i, key in enumerate(['iteration', 'f_value', 'f_grad_norm', 'x']):
        history[key] = values[i]
    return history
