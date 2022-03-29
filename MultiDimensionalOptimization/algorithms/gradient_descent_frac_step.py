from __future__ import annotations

from typing import Tuple
from MultiDimensionalOptimization.algorithms.support import *


def gradient_descent_frac_step(function: Callable[[np.ndarray], Real],
                               x0: Sequence[Real],
                               epsilon: Real = 1e-5,
                               gamma: Real = 0.1,
                               delta: Real = 0.1,
                               lambda0: Real = 0.1,
                               max_iter: Integral = 500,
                               verbose: bool = False,
                               keep_history: bool = False) -> Tuple[Point, HistoryMDO]:
    """
    Algorithm with fractional step. Documentation: paragraph 2.2.3, page 4
    Requirements: 0 < lambda0 < 1 is the step multiplier, 0 < delta < 1.

    Code example::

        >>> def func(x): return x[0] ** 2 + x[1] ** 2
        >>> x_0 = [1, 2]
        >>> solution = gradient_descent_frac_step(func, x_0)
        >>> print(solution[0])
        {'point': array([1.91561942e-06, 3.83123887e-06]), 'f_value': 1.834798903191018e-11}


    :param function: callable that depends on the first positional argument
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

            t = x_k - gamma * grad_k
            func_t = function(t)

            while not func_t - func_k <= - gamma * delta * sum(grad_k ** 2):
                gamma = gamma * lambda0
                t = x_k - gamma * grad_k
                func_t = function(t)

            x_k = t
            func_k = func_t
            grad_k = gradient(function, x_k)

            if keep_history:
                history = update_history_grad_descent(history, values=[i + 1, func_k, sum(grad_k ** 2) ** 0.5, x_k])

            if verbose:
                print(f'Iteration: {i + 1} \t|\t point = {np.round(x_k, round_precision)} '
                      f'\t|\t f(point) = {round(func_k, round_precision)}')

            if np.sum(grad_k ** 2) ** 0.5 < epsilon:
                history['message'] = 'Optimization terminated successfully. code 0'
                break
        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return {'point': x_k, 'f_value': function(x_k)}, history


if __name__ == '__main__':
    def paraboloid(x): return x[0] ** 2 + x[1] ** 2


    start_point = [1, 2]
    output = gradient_descent_frac_step(paraboloid, start_point, keep_history=True, verbose=True)
    print(output[1], output[0])
