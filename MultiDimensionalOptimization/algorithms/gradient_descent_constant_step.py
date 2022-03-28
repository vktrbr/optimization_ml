from __future__ import annotations
import numpy as np

from typing import Tuple, Callable, Sequence
from numbers import Integral, Real
from MultiDimensionalOptimization.algorithms.support import Point, HistoryGradDescent, update_history_grad_descent


def gradient_descent_constant_step(function: Callable[[np.ndarray], Real],
                                   x0: Sequence[Real],
                                   epsilon: Real = 1e-5,
                                   gamma: Real = 0.1,
                                   max_iter: Integral = 500,
                                   verbose: bool = False,
                                   keep_history: bool = False) -> Tuple[Point, HistoryGradDescent]:
    """
    Algorithm with constant step. Documentation: paragraph 2.2.2, page 3.
    The gradient of the function shows us the direction of increasing the function.
    The idea is to move in the opposite direction to x𝑘+1 where 𝑓(x𝑘+1) < 𝑓(x𝑘).
    But, if we add a gradient to x𝑘 without changes, our method will often diverge.
    So we need to add a gradient with some weight 𝛾.

    Code example::
        >>> def func(x): return x[0] ** 2 + x[1] ** 2
        >>> x_0 = [1, 2]
        >>> solution = gradient_descent_constant_step(func, x_0)
        >>> print(solution[0])
        {'point': array([1.91561942e-06, 3.83123887e-06]), 'f_value': 1.834798903191018e-11}

    :param function: callable that depends on the first positional argument
    :param x0: numpy ndarray which is initial approximation
    :param epsilon: optimization accuracy
    :param gamma: gradient step
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """
    x0 = np.array(x0)
    func_k = function(x0)
    if keep_history:
        grad_f0 = gradient(function, x0)
        history: HistoryGradDescent = {'iteration': [0],
                                       'f_value': [func_k],
                                       'f_grad_norm': [grad_f0],
                                       'x': [x0]}
    else:
        history: HistoryGradDescent = {'iteration': [], 'f_value': [], 'x': [], 'f_grad_norm': []}

    if verbose:
        print(f'Iteration: {0} \t|\t point = {x0} '
              f'\t|\t f(point) = {func_k: 0.3f}')

    x_k = x0
    try:
        for i in range(max_iter):
            grad_k = gradient(function, x_k)
            func_k = function(x_k)
            if np.sum(grad_k ** 2) ** 0.5 < epsilon:
                history['message'] = 'Optimization terminated successfully. code 0'
                break
            else:
                x_k = x_k - gamma * grad_k
            if keep_history:
                history = update_history_grad_descent(history, values=[i + 1, func_k, grad_k, x_k])
            if verbose:
                round_precision = -int(np.log10(epsilon))
                print(f'Iteration: {i+1} \t|\t point = {np.round(x_k, round_precision)} '
                      f'\t|\t f(point) = {np.round(func_k, round_precision)}')

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'
    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return {'point': x_k, 'f_value': function(x_k)}, history


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


if __name__ == '__main__':
    def paraboloid(x): return x[0] ** 2 + x[1] ** 2
    start_point = [1, 2]
    output = gradient_descent_constant_step(paraboloid, start_point, keep_history=True)
    print(output[1], output[0])
