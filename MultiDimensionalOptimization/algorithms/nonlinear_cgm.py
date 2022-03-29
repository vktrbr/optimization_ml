from __future__ import annotations

from typing import Tuple
from MultiDimensionalOptimization.algorithms.support import *
from scipy.optimize import line_search
from OneDimensionalOptimization.algorithms.brent import brent
import warnings


def nonlinear_cgm(function: Callable[[np.ndarray], Real],
                  x0: Sequence[Real],
                  epsilon: Real = 1e-5,
                  max_iter: Integral = 500,
                  verbose: bool = False,
                  keep_history: bool = False) -> Tuple[Point, HistoryMDO]:
    """
    Paragraph 2.4.1 page 6
    Algorithm works when the function is approximately quadratic near the minimum, which is the case when the
    function is twice differentiable at the minimum and the second derivative is non-singular there.


    Code example::
        >>> def func(x): return 10 * x[0] ** 2 + x[1] ** 2 / 5
        >>> x_0 = [1, 2]
        >>> solution = nonlinear_cgm(func, x_0)
        >>> print(solution[0])
        {'point': array([-1.70693616e-07,  2.90227591e-06]), 'f_value': 1.9760041961386155e-12}

    :param function: callable that depends on the first positional argument
    :param x0: numpy ndarray which is initial approximation
    :param epsilon: optimization accuracy
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """
    x_k = np.array(x0, dtype=float)
    grad_k = gradient(function, x_k)
    p_k = grad_k
    func_k = function(x_k)
    round_precision = -int(np.log10(epsilon))
    if keep_history:
        history: HistoryMDO = {'iteration': [0],
                               'f_value': [func_k],
                               'f_grad_norm': [np.sum(grad_k ** 2) ** 0.5],
                               'x': [x_k]}
    else:
        history: HistoryMDO = {'iteration': [], 'f_value': [], 'x': [], 'f_grad_norm': []}

    if verbose:
        print(f'Iteration: {0} '
              f'\t|\t f(point) = {np.round(func_k, round_precision)}'
              f'\t|\t point = {np.round(x_k, round_precision)} ')

    try:
        for i in range(max_iter - 1):

            if np.sum(grad_k ** 2) ** 0.5 < epsilon:
                history['message'] = 'Optimization terminated successfully. code 0'
                break
            else:
                with HiddenPrints():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gamma = line_search(function,
                                            lambda x: gradient(function, x),
                                            x_k,
                                            p_k)[0]
                    if gamma is None:
                        gamma = brent(lambda gam: function(x_k - gam * p_k), (0, 1))[0]['point']

                x_k = x_k - gamma * p_k
                grad_k_new = gradient(function, x_k)
                beta_fr = (grad_k_new @ grad_k_new.reshape(-1, 1)) / (grad_k @ grad_k.reshape(-1, 1))
                p_k = grad_k_new + beta_fr * p_k
                grad_k = grad_k_new
                func_k = function(x_k)

            if keep_history:
                history = update_history_grad_descent(history, values=[i + 1, func_k, np.sum(grad_k ** 2) ** 0.5, x_k])

            if verbose:
                print(f'Iteration: {i + 1} '
                      f'\t|\t f(point) = {np.round(func_k, round_precision)}'
                      f'\t|\t point = {np.round(x_k, round_precision)}')

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return {'point': x_k, 'f_value': function(x_k)}, history


if __name__ == '__main__':
    def paraboloid(x): return 10 * x[0] ** 2 + x[1] ** 2 / 5
    start_point = [1, 2]
    output = nonlinear_cgm(paraboloid, start_point, keep_history=True, verbose=True)
    print(output[0], output[1])
