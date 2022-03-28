from __future__ import annotations
import numpy as np

from typing import Tuple, Callable
from MultiDimensionalOptimization.algorithms.support import *

def gradient_descent_constant_step(function: Callable[[np.ndarray], Real],
                                   x0: np.ndarray,
                                   epsilon: Real = 1e-5,
                                   gamma: float = 0.1,
                                   max_iter: int = 500,
                                   verbose: bool = False,
                                   keep_history: bool = False) -> Tuple[Point, HistoryGradDescent]:



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
    for i in range(len(x0)):
        delta_x_vec_plus = x0.copy()
        delta_x_vec_minus = x0.copy()
        delta_x_vec_plus[i] += delta_x
        delta_x_vec_minus[i] -= delta_x
        grad_i = (function(delta_x_vec_plus) - function(delta_x_vec_minus)) / (2 * delta_x)
        grad.append(grad_i)

    grad = np.array(grad)
    return grad
