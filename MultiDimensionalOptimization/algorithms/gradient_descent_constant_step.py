from __future__ import annotations

from typing import Tuple
from MultiDimensionalOptimization.algorithms.support import *


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
    The idea is to move in the opposite direction to x_{k + 1} where f(x_{k + 1}) < f(x_{k}).
    But, if we add a gradient to x_{k} without changes, our method will often diverge.
    So we need to add a gradient with some weight gamma.

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
    x_k = np.array(x0, dtype=float)  # change type to numpy ndarray instead of list, for future working
    grad_k = gradient(function, x_k)
    func_k = function(x_k)
    # if keep_history=True, we will save history. here is initial step
    if keep_history:
        history: HistoryGradDescent = {'iteration': [0],
                                       'f_value': [func_k],
                                       'f_grad_norm': [np.sum(grad_k ** 2) ** 0.5],
                                       'x': [x_k]}
    else:
        history: HistoryGradDescent = {'iteration': [], 'f_value': [], 'x': [], 'f_grad_norm': []}
    # if verbose=True, print the result on each iteration
    if verbose:
        print(f'Iteration: {0} \t|\t point = {x_k} '
              f'\t|\t f(point) = {func_k: 0.3f}')

    try:
        for i in range(max_iter - 1):

            if np.sum(grad_k ** 2) ** 0.5 < epsilon:  # comparing of norm 2 with optimization accuracy
                history['message'] = 'Optimization terminated successfully. code 0'
                break
            else:
                x_k = x_k - gamma * grad_k  # updating the point for next iter and repeat
                grad_k = gradient(function, x_k)
                func_k = function(x_k)

            # again, if keep_history=True add the result of the iter
            if keep_history:
                history = update_history_grad_descent(history, values=[i + 1, func_k, np.sum(grad_k ** 2) ** 0.5, x_k])

            # again, if verbose=True, print the result of the iter
            if verbose:
                round_precision = -int(np.log10(epsilon))
                print(f'Iteration: {i+1} \t|\t point = {np.round(x_k, round_precision)} '
                      f'\t|\t f(point) = {np.round(func_k, round_precision)}')

        else:
            history['message'] = 'Optimization terminated. Max steps. code 1'

    except Exception as e:
        history['message'] = f'Optimization failed. {e}. code 2'

    return {'point': x_k, 'f_value': function(x_k)}, history


if __name__ == '__main__':
    def paraboloid(x): return x[0] ** 2 + x[1] ** 2
    start_point = [1, 2]
    output = gradient_descent_constant_step(paraboloid, start_point, keep_history=True, verbose=True)
    print(output[1], output[0])
