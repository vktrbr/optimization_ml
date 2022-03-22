from __future__ import annotations

from typing import Tuple, Callable, Any, Literal
from OneDimensionalOptimization.algorithms.support import *


def successive_parabolic_interpolation(function: Callable[[Real, Any], Real],
                                       bounds: Tuple[Real, Real],
                                       epsilon: Real = 1e-5,
                                       type_optimization: Literal['min', 'max'] = 'min',
                                       max_iter: int = 500,
                                       verbose: bool = False,
                                       keep_history: bool = False,
                                       **kwargs) -> Tuple[Point, HistorySPI]:
    """
    Successive parabolic interpolation algorithm

    **Algorithm:**
        1. Set :math:`x_0, x_2, x_1` and calculate :math:`f_0 = f(x_0), f_1 = f(x_1), f_2 = f(x_2)`
        2. Arrange :math:`x_0, x_1, x_2` so that :math:`f_2 \\leq f_1 \\leq f_0`
        3. Calculate :math:`x_{i + 1}` with the formula below
        4. Repeat step 2-3 until then :math:`|x_{i+1}-x_{i}| \\geq e` or :math:`|f(x_{i+1})-f(x_{i})| \\geq e`

    .. math::

        x_{i+1}=x_{i}+ \\frac{1}{2}\\left[\\frac{\\left(x_{i-1}-x_{i}\\right)^{2}\\left(f_{i}-f_{i-2}\\right)+
        \\left(x_{i-2}-x_{i}\\right)^{2}\\left(f_{i-1}-f_{i}\\right)}{\\left(x_{i-1}-x_{i}\\right)
        \\left(f_{i}-f_{i-2}\\right)+\\left(x_{i-2}-x_{i}\\right)\\left(f_{i-1}-f_{i}\\right)}\\right]

    Example:
        >>> def func1(x): return x ** 3 - x ** 2 - x
        >>> successive_parabolic_interpolation(func1, (0, 1.5), verbose=True)
        Iteration: 0	|	x2 = 0.750	|	f(x2) = -0.891
        Iteration: 1	|	x2 = 0.850	|	f(x2) = -0.958
        Iteration: 2	|	x2 = 0.961	|	f(x2) = -0.997
        Iteration: 3	|	x2 = 1.017	|	f(x2) = -0.999
        Iteration: 4	|	x2 = 1.001	|	f(x2) = -1.000
        ...

        >>> def func2(x): return - (x ** 3 - x ** 2 - x)
        >>> successive_parabolic_interpolation(func2, (0, 1.5), type_optimization='max', verbose=True)
        Iteration: 0	|	x2 = 0.750	|	f(x2) = -0.891
        Iteration: 1	|	x2 = 0.850	|	f(x2) =  0.958
        Iteration: 2	|	x2 = 0.961	|	f(x2) =  0.997
        Iteration: 3	|	x2 = 1.017	|	f(x2) =  0.999
        ...

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

    history: HistorySPI = {'iteration': [], 'f_value': [], 'x0': [], 'x1': [], 'x2': []}
    x0, x1, x2 = bounds[0], bounds[1], (bounds[0] + bounds[1]) / 2
    f0 = type_opt_const * function(x0, **kwargs)
    f1 = type_opt_const * function(x1, **kwargs)
    f2 = type_opt_const * function(x2, **kwargs)
    f_x = {x0: f0, x1: f1, x2: f2}
    x2, x1, x0 = sorted([x0, x1, x2], key=lambda x: f_x[x])

    if keep_history:
        history['iteration'].append(0)
        history['f_value'].append(type_opt_const * f2)
        history['x0'].append(x0)
        history['x1'].append(x1)
        history['x2'].append(x2)

    if verbose:
        print(f'Iteration: {0}\t|\tx2 = {x2:0.3f}\t|\tf(x2) = {f2: 0.3f}')

    try:
        for i in range(1, max_iter):
            f0, f1, f2 = f_x[x0], f_x[x1], f_x[x2]
            p = (x1 - x2) ** 2 * (f2 - f0) + (x0 - x2) ** 2 * (f1 - f2)
            q = 2 * ((x1 - x2) * (f2 - f0) + (x0 - x2) * (f1 - f2))

            assert p != 0, 'Searching finished. Select an another initial state. Numerator is zero. code 2'
            assert q != 0, 'Searching finished. Select an another initial state. Denominator is zero. code 2'

            x_new = x2 + p / q

            if not bounds[0] <= x_new <= bounds[1]:
                print('Searching finished. Out of bounds. code 1')
                return {'point': x2, 'f_value': type_opt_const * f2}, history

            f_new = type_opt_const * function(x_new, **kwargs)
            f_x[x_new] = f_new
            previous_xs = [x0, x1, x2]

            if f_new < f2:
                x0, f0 = x1, f1
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new

            elif f_new < f1:
                x0, f0 = x1, f1
                x1, f1 = x_new, f_new

            elif f_new < f0:
                x0, f0 = x_new, f_new

            if verbose:
                print(f'Iteration: {i}\t|\tx2 = {x2:0.3f}\t|\tf(x2) = {type_opt_const * f2: 0.3f}')

            if keep_history:
                history['iteration'].append(i)
                history['f_value'].append(type_opt_const * f2)
                history['x0'].append(x0)
                history['x1'].append(x1)
                history['x2'].append(x2)

            # In addition, check the criterion when the points don't change
            change_flag = max(map(lambda x, y: abs(x - y), [x0, x1, x2], previous_xs)) < epsilon
            if abs(x1 - x2) < epsilon and abs(f1 - f2) < epsilon or change_flag:
                print('Searching finished. Successfully. code 0')
                return {'point': x2, 'f_value': type_opt_const * f2}, history

        else:
            print('Searching finished. Max iterations have been reached. code 1')
            return {'point': x2, 'f_value': type_opt_const * f2}, history

    except Exception as e:
        print('Error with optimization. code 2')
        raise e


if __name__ == '__main__':
    def func(x): return x ** 3 - x ** 2 - x

    bs = [-1, 1]
    print(successive_parabolic_interpolation(func, bounds=bs, type_optimization='max', verbose=True))
