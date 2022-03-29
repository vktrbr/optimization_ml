from __future__ import annotations

from typing import Tuple, Sequence
from MultiDimensionalOptimization.algorithms.support import *
from OneDimensionalOptimization.algorithms.brent import brent


def gradient_descent_optimal_step(function: Callable[[np.ndarray], Real],
                                  x0: Sequence[Real],
                                  epsilon: Real = 1e-5,
                                  max_iter: Integral = 500,
                                  verbose: bool = False,
                                  keep_history: bool = False) -> Tuple[Point, HistoryMDO]:
    """
    Algorithm with optimal step. Documentation: paragraph 2.2.4, page 5
    The idea is to choose a gamma that minimizes the function in the direction f'(x_k)

    Code example::

        >>> def func(x): return -np.e ** (- x[0] ** 2 - x[1] ** 2)
        >>> x_0 = [1, 2]
        >>> solution = gradient_descent_optimal_step(func,x_0)
        >>> print(solution[0])
        {'point': array([9.21321369e-08, 1.84015366e-07]), 'f_value': -0.9999999999999577}


    :param function: callable that depends on the first positional argument
    :param x0: numpy ndarray which is initial approximation
    :param epsilon: optimization accuracy
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.
    """

    x_k = np.array(x0, dtype=float)
    func_k = function(x0)
    grad_k = gradient(function, x_k)
    round_precision = -int(np.log10(epsilon))

    if keep_history:
        grad_f0 = gradient(function, x_k)
        history: HistoryMDO = {'iteration': [0],
                               'f_value': [func_k],
                               'f_grad_norm': [sum(grad_f0 ** 2) ** 0.5],
                               'x': [x_k]}
    else:
        history: HistoryMDO = {'iteration': [], 'f_value': [], 'x': [], 'f_grad_norm': []}

    if verbose:
        print(f'Iteration: {0} \t|\t point = {np.round(x_k, round_precision)} '
              f'\t|\t f(point) = {round(func_k, round_precision)}')

    try:
        for i in range(max_iter - 1):
            with HiddenPrints():
                gamma = brent(lambda gam: function(x_k - gam * grad_k), (0, 1))[0]['point']
            x_k = x_k - gamma * grad_k
            grad_k = gradient(function, x_k)

            if keep_history:
                func_k = function(x_k)
                history = update_history_grad_descent(history, values=[i + 1, func_k, sum(grad_k ** 2) ** 0.5, x_k])

            if verbose:
                func_k = function(x_k)
                print(f'Iteration: {i + 1} \t|\t point = {np.round(x_k, round_precision)} '
                      f'\t|\t f(point) = {round(func_k, round_precision)}')

            if sum(grad_k ** 2) ** 0.5 < epsilon:
                history['message'] = 'Optimization terminated successfully. code 0'
                break

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return {'point': x_k, 'f_value': function(x_k)}, history


if __name__ == '__main__':
    import numpy as np


    def expon(x): return -np.e ** (- x[0] ** 2 - x[1] ** 2)


    start_point = [1, 2]
    output = gradient_descent_optimal_step(expon, start_point, keep_history=True, verbose=True)

    print(output[1], output[0])
