from __future__ import annotations

from typing import Tuple, Callable, Any, Literal
from OneDimensionalOptimization.algorithms.support import *
import numpy as np


def brant(function: Callable[[Real, Any], Real],
          bounds: Tuple[Real, Real],
          epsilon: Real = 1e-5,
          type_optimization: Literal['min', 'max'] = 'min',
          max_iter: int = 500,
          verbose: bool = False,
          keep_history: bool = False,
          **kwargs) -> Tuple[Point, HistoryGSS]:
    """
    Brant's algorithm

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param bounds: tuple with two numbers. This is left and right bound optimization. [a, b]
    :param epsilon: optimization accuracy
    :param type_optimization: 'min' / 'max' - type of required value
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """

    type_optimization = type_optimization.lower().strip()
    assert type_optimization in ['min', 'max'], 'Invalid type optimization. Enter "min" or "max"'
    
    if type_optimization == 'max':
        type_opt_const = -1
    else:
        type_opt_const = 1
        
    a, b = sorted(bounds)
    const_from_golden = 1 - 2 / (1 + 5 ** 0.5)
    x = w = v = (a + b) / 2
    fw = fv = fx = type_opt_const * function(x, **kwargs)

    deltax = 0.0
    rat = b - a

    for i in range(max_iter):
        tol1 = epsilon * np.abs(x) + 1.e-10
        tol2 = 2.0 * tol1
        x_middle = (a + b) / 2
        # check for convergence

        if np.abs(x - x_middle) < (tol2 - 0.5 * (b - a)):
            return x, type_opt_const * fx, i

        if np.abs(deltax) <= tol1:
            if x >= x_middle:
                deltax = a - x  # do a golden section step
            else:
                deltax = b - x

            rat = const_from_golden * deltax
        else:  # do a parabolic step
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                p = -p
            tmp2 = np.abs(tmp2)
            dx_temp = deltax
            deltax = rat
            # check parabolic fit
            if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                    (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))):
                rat = p * 1.0 / tmp2  # if parabolic step is useful.
                u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    if x_middle - x >= 0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if x >= x_middle:
                    deltax = a - x  # if it's not do a golden section step
                else:
                    deltax = b - x
                rat = const_from_golden * deltax

        if np.abs(rat) < tol1:  # update by at least tol1
            if rat >= 0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu = type_opt_const * function(u, **kwargs)  # calculate new output value

        if fu > fx:  # if it's bigger than current
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
    else:
        print('Searching finished. Max iterations have been reached. code 1')
        return x, fx, i


if __name__ == '__main__':
    def func(x): return x ** 3 - x ** 2 - x
    bs = [0, 1.5]
    print(brant(func, bounds=bs, type_optimization='min'))
