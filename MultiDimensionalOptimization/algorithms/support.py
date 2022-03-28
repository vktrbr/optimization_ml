from __future__ import annotations
from typing import List, TypedDict, AnyStr, Literal, Callable, Sequence
from numbers import Real, Integral
import numpy as np
import os
import sys


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
    x: List[Sequence]
    message: AnyStr


class HiddenPrints:
    """
    Object hides print. Working with context manager "with"::

        >>> with HiddenPrints():
        >>>     print("It won't be printed")
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def update_history_grad_descent(history: HistoryGradDescent, values: List) -> HistoryGradDescent:
    """
    Update HistoryGradDescent with values, which contains iteration, f_value, f_grad_norm, x as a list
    :param history: object of HistoryGradDescent
    :param values: new values that need to append in history in order iteration, f_value, f_grad_norm, x
    :return: updated history
    """
    key: Literal['iteration', 'f_value', 'f_grad_norm', 'x']
    for i, key in enumerate(['iteration', 'f_value', 'f_grad_norm', 'x']):
        history[key].append(values[i])
    return history


def gradient(function: Callable,
             x0: np.ndarray,
             delta_x=1e-8) -> np.ndarray:
    """
    Calculate and return a gradient using a two-side difference
    :param function: callable that depends on the first positional argument
    :param x0: the point at which we calculate the gradient
    :param delta_x: precision of differentiation
    :return: vector np.ndarray with the gradient at the point
    """

    grad = []
    x0 = np.array(x0, dtype=float)

    for i in range(len(x0)):
        delta = np.zeros_like(x0)
        delta[i] += delta_x
        delta_x_vec_plus = x0 + delta
        delta_x_vec_minus = x0 - delta
        grad_i = (function(delta_x_vec_plus) - function(delta_x_vec_minus)) / (2 * delta_x)
        grad.append(grad_i)

    grad = np.array(grad)
    return grad
