from __future__ import annotations

from typing import Tuple, Callable, Any, Literal
from OneDimensionalOptimization.algorithms.support import *


def brant(function: Callable[[Real, Any], Real],
          bounds: Tuple[Real, Real],
          epsilon: Real = 1e-5,
          tolerance: Real = 1.e-4,
          type_optimization: Literal['min', 'max'] = 'min',
          max_iter: int = 500,
          verbose: bool = False,
          keep_history: bool = False,
          **kwargs) -> Tuple[Point, HistoryGSS]:
    """
    Brant's algorithm. In original work we need to set a constant t - minimum tolerance. Let's set t = 1.e-8

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param bounds: tuple with two numbers. This is left and right bound optimization. [a, b]
    :param epsilon: optimization accuracy
    :param type_optimization: 'min' / 'max' - type of required value
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """
    c: Real = (3 - 5 ** 0.5) / 2
    a, b = bounds
    t = tolerance
    v = w = x = a + c * (b - a)
    e = 0
    fv = fw = fx = function(x, **kwargs)

    for i in range(max_iter):
        m = 0.5 * (a + b)
        tol = epsilon * abs(x) + t
        t2 = 2 * tol
        # check stopping criterion
        if abs(x - m) > t2 - 0.5 * (b - a):
            p = q = r = 0
            if abs(e) > tol:
                print('fit parabola')
                # fit parabola
                r = (x - w) * (fx - fv)
                q = (x - v) * (fx - fw)
                p = (x - v) * q - (x - w) * r
                q = 2 * (q - r)
                if q > 0:
                    p = -p
                else:
                    q = -q
                r = e
                e = d

            if (abs(p) < abs(0.5 * q * r)) and (p < q * (a - x)) and (p < q * (b - x)):
                print('parabolic step')
                # A parabolic interpolation step
                d = p / q
                u = x + d
                # f ust not be evaluated too close to a or b
                if u - a < t2 and b - u < t2:
                    if x < m:
                        d = tol
                        d = -tol
            else:
                print('golden')
                # golden section step
                if x < m:
                    e = b - x
                else:
                    e = a - x
                d = c * e

            # f must not be evaluated too close to x
            if abs(d) >= tol:
                u = x + d
            elif d > 0:
                u = x + tol
            else:
                u = x - tol

            fu = function(u, **kwargs)
            # update a, b, v, w, x
            if fu <= fx:
                if u < x:
                    b = x
                else:
                    a = x
                v = w
                fv = fw
                w = x
                fw = fx
                x = u
                fx = fu

            else:
                if u < x:
                    a = u
                else:
                    b = u
                if fu <= fw or w == x:
                    v = w
                    fv = fw
                    w = u
                    fw = fu
                elif fu <= fv or v == x or v == w:
                    v = u
                    fv = fu

        else:
            return fx
    else:
        print('max')


if __name__ == '__main__':
    def func(x): return x ** 3 - x ** 2 - x
    bounds = [0, 1.5]
    print(brant(func, bounds=bounds))
