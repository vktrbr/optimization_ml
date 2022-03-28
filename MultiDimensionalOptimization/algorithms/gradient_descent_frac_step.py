from __future__ import annotations
import numpy as np

from typing import Tuple, Callable, Any, Literal
from MultiDimensionalOptimization.algorithms.support import *


def gradient_descent_frac_step(function: Callable[[np.ndarray], Real],
                               x0: np.ndarray,
                               epsilon: Real = 1e-5,
                               gamma: float = 0.1,
                               delta: float = 0.1,
                               lambda0: float = 0.1,
                               max_iter: int = 500,
                               verbose: bool = False,
                               keep_history: bool = False,
                               **kwargs) -> Tuple[Point, HistoryGDFS]:
    """
    Algorithm with descent step
    Requirements: 0 < 𝜆 < 1, 0 < 𝛿 < 1

    Code example::

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param x0: numpy ndarray which is initial approximation
    :param epsilon: optimization accuracy
    :param gamma: gradient step
    :param delta: value of the crushing parameter
    :param lambda0: initial step
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """

    if keep_history:
        history: HistoryGDFS = {'iteration': [0],
                               'f_value': [function(x0, **kwargs)],
                                'x': List[Real]}
    else:
        history: HistoryGDFS = {'iteration': [], 'f_value': [], 'x': []}

    if verbose:
        print(f'Iteration: {0} \t|\t point = {x0} '
              f'\t|\t f(point) = {function(x0, **kwargs): 0.3f}')
    x_k = x0
    for i in range( max_iter):
        t = x_k - gamma * gradient(function, x_k)
        if function(t) - function(x_k) <= -gamma * delta * (sum(map(lambda x: x ** 2, gradient(function, x_k)))):
            x_k -= gamma * gradient(function, x_k)
            if sum(map(lambda x: x ** 2, gradient(function, x_k))) ** 0.5 < epsilon:
                break
        else:
            gamma = gamma * lambda0

        return {'point': x_k 'f_value': function(x_k)}, history


    def gradient(function: Callable,
                 x0: np.ndarray,
                 delta_x=1e-8,
                 **kwargs) -> np.ndarray:
        """
        Calculate gradient
        :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
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
            grad_i = (function(delta_x_vec_plus, **kwargs) - function(delta_x_vec_minus, **kwargs)) / (2 * delta_x)
            grad.append(grad_i)

        grad = np.array(grad)
        return grad


