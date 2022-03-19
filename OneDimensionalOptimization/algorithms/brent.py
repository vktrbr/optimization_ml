from __future__ import annotations

from typing import Tuple, Callable, Any, Literal, Sequence
from OneDimensionalOptimization.algorithms.support import *


def brent(function: Callable[[Real, Any], Real],
          bounds: Tuple[Real, Real],
          epsilon: Real = 1e-5,
          type_optimization: Literal['min', 'max'] = 'min',
          max_iter: int = 500,
          verbose: bool = False,
          keep_history: bool = False,
          **kwargs) -> Tuple[Point, HistoryGSS]:
    """
    Brent's algorithm.
    Brent, R. P., Algorithms for Minimization Without Derivatives. Englewood Cliffs, NJ: Prentice-Hall, 1973 pp.72-80

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param bounds: tuple with two numbers. This is left and right bound optimization. [a, b]
    :param epsilon: optimization accuracy
    :param type_optimization: 'min' / 'max' - type of required value
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history

    :var gold_const: b - (b - a) / phi = a + (b - a) * gold_const
    :var type_opt_const: This value unifies the optimization for each type of min and max
    :return: tuple with point and history.

    """

    type_optimization = type_optimization.lower().strip()
    assert type_optimization in ['min', 'max'], 'Invalid type optimization. Enter "min" or "max"'

    if type_optimization == 'max':
        type_opt_const = -1
    else:
        type_opt_const = 1

    gold_const = (3 - 5 ** 0.5) / 2
    remainder = 0.0  # p / q when we calculate x_new

    # initial values
    a, b = sorted(bounds)
    x_largest = x_middle = x_least = a + gold_const * (b - a)
    f_largest = f_middle = f_least = type_opt_const * function(x_least, **kwargs)

    history = {'iteration': [], 'f_least': [], 'f_middle': [], 'f_largest': [],  'x_least': [], 'x_middle': [],
               'x_largest': [], 'left_bound': [], 'right_bound': [], 'type_step': []}

    if keep_history:
        history = update_history(history, [0, f_least, f_middle, f_largest,
                                           x_least, x_middle, x_largest, a, b, 'initial'])
    if verbose:
        print(f'iteration 0\tx = {x_least:0.6f},\tf(x) = {f_least:0.6f}\ttype : initial')

    for i in range(1, max_iter + 1):
        middle_point = (a + b) / 2
        tolerance = epsilon * abs(x_least) + 1e-9  # f is never evaluated at two points closer together than tolerance

        # Check stopping criterion
        if abs(x_least - middle_point) > 2 * tolerance - (b - a) / 2:
            p = q = previous_remainder = 0
            if abs(remainder) > tolerance:
                
                # fit parabola
                p = ((x_least - x_largest) ** 2 * (f_least - f_middle) -
                     (x_least - x_middle) ** 2 * (f_least - f_largest))

                q = 2 * ((x_least - x_largest) * (f_least - f_middle) -
                         (x_least - x_middle) * (f_least - f_largest))

                # change q sign to positive
                if q > 0:
                    p = -p
                else:
                    q = -q
                # r stores the previous value of remainder
                previous_remainder = remainder

            # Check conditions for parabolic step:
            # tol - x_new must not be close to x_least, so we check the step
            # previous_remainder - the value of p / q at the second-last cycle
            # |previous_remainder| > tol - is checked above
            # q != 0 - includes in next conditions
            # x_least + p / q in (a, b). New point in interval
            # p / q < previous(p / q) / 2. Control the divergence

            if abs(p) < 0.5 * abs(q * previous_remainder) and a * q < x_least * q + p < b * q:
                remainder = p / q
                x_new = x_least + remainder
                name_step = 'parabolic'

                # Check that f not be evaluated too close to a or b
                if x_new - a < 2 * tolerance or b - x_new < 2 * tolerance:
                    if x_least < middle_point:
                        remainder = tolerance
                    else:
                        remainder = -tolerance

            # If conditions above is false we do golden section step
            else:
                name_step = 'golden'
                if x_least < middle_point:
                    remainder = (b - x_least) * gold_const
                else:
                    remainder = (a - x_least) * gold_const

            # Check that f not be evaluated too close to x_least
            if abs(remainder) > tolerance:
                x_new = x_least + remainder
            elif remainder > 0:
                x_new = x_least + tolerance
            else:
                x_new = x_least - tolerance

            f_new = type_opt_const * function(x_new, **kwargs)

            # Update a, b, x_largest, x_middle, x_leas
            if f_new <= f_least:
                if x_new < x_least:
                    b = x_least
                else:
                    a = x_least

                x_largest = x_middle
                f_largest = f_middle

                x_middle = x_least
                f_middle = f_least

                x_least = x_new
                f_least = f_new

            else:
                if x_new < x_least:
                    a = x_new
                else:
                    b = x_new

                if f_new <= f_middle:
                    x_largest = x_middle
                    f_largest = f_middle

                    x_middle = x_new
                    f_middle = f_new

                elif f_new <= f_largest:
                    x_largest = x_new
                    f_largest = f_new

        else:
            print('Searching finished. Successfully. code 0')
            return {'point': x_least, 'f_value': f_least}, history

        if keep_history:
            history = update_history(history, [i, f_least, f_middle, f_largest,
                                               x_least, x_middle, x_largest, a, b, name_step])
        if verbose:
            print(f'iteration {i}\tx = {x_least:0.6f},\tf(x) = {f_least:0.6f}\ttype : {name_step}')

    else:
        print('Searching finished. Max iterations have been reached. code 1')
        return {'point': x_least, 'f_value': f_least}, history


def update_history(history: HistoryBrent, values: Sequence[Any]) -> HistoryBrent:
    """
    Updates brent history
    :param history: HistoryBrent object in which the update is required
    :param values: Sequence with values: 'iteration', 'f_least', 'f_middle', 'f_largest',  'x_least',
                                         'x_middle', 'x_largest', 'left_bound', 'right_bound', 'type_step'
    :return: updated HistoryBrent
    """
    name: Literal['iteration', 'f_least', 'f_middle', 'f_largest', 'x_least', 'x_middle',
                  'x_largest', 'left_bound', 'right_bound', 'type_step']

    for i, name in enumerate(['iteration', 'f_least', 'f_middle', 'f_largest', 'x_least', 'x_middle',
                              'x_largest', 'left_bound', 'right_bound', 'type_step']):
        history[name].append(values[i])

    return history


if __name__ == '__main__':
    def func(x): return x ** 3 - x ** 2 - x
    bs = [0, 1.5]
    print(brent(func, bounds=bs, type_optimization='min', keep_history=True, verbose=True))
