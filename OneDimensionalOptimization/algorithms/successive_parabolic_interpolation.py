from __future__ import annotations

from typing import Tuple, Callable, Any, Literal
from .support import *


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
        Iteration: 0 	|	 x0 = 0.000 	|	 x1 = 1.500 	|	 x2 = 0.750 	|	 f(x2) = -0.891
        Iteration: 1 	|	 new point = x2 = 0.850 	|	 f(x2) = -0.958
        Iteration: 2 	|	 new point = x2 = 0.961 	|	 f(x2) = -0.997
        Iteration: 3 	|	 new point = x2 = 1.017 	|	 f(x2) = -0.999
        ...

        >>> def func2(x): return - (x ** 3 - x ** 2 - x)
        >>> successive_parabolic_interpolation(func2, (0, 1.5), type_optimization='max', verbose=True)
        Iteration: 0 	|	 x0 = 0.000 	|	 x1 = 1.500 	|	 x2 = 0.750 	|	 f(x2) =  0.891
        Iteration: 1 	|	 new point = x2 = 0.850 	|	 f(x2) =  0.958
        Iteration: 2 	|	 new point = x2 = 0.961 	|	 f(x2) =  0.997
        Iteration: 3 	|	 new point = x2 = 1.017 	|	 f(x2) =  0.999
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
    x2, x1, x0 = sorted([x0, x1, x2], key=lambda x: type_opt_const * function(x, **kwargs))

    if keep_history:
        history['iteration'].append(0)
        history['f_value'].append(function(x2, **kwargs))
        history['x0'].append(x0)
        history['x1'].append(x1)
        history['x2'].append(x2)

    if verbose:
        print(f'Iteration: {0} \t|\t x0 = {x0:0.3f} '
              f'\t|\t x1 = {x1:0.3f} '
              f'\t|\t x2 = {x2:0.3f} '
              f'\t|\t f(x2) = {function(x2, **kwargs): 0.3f}')

    try:
        for i in range(1, max_iter):
            f0, f1, f2 = list(map(lambda x: type_opt_const * function(x, **kwargs), [x0, x1, x2]))
            x_new = x2 + 0.5 * ((x1 - x2) ** 2 * (f2 - f0) + (x0 - x2) ** 2 * (f1 - f2)) / ((x1 - x2) * (f2 - f0) +
                                                                                            (x0 - x2) * (f1 - f2))
            if x_new == x2:
                print('Searching finished. Successfully. code 0')
                return {'point': x2, 'f_value': function(x2, **kwargs)}, history

            if not bounds[0] <= x_new <= bounds[1]:
                print('Searching finished. Out of bounds. code 3. ')
                return {'point': x2, 'f_value': function(x2, **kwargs)}, history

            x_list_old = [x2, x1, x0]
            x2, x1, x0 = sorted([x1, x2, x_new], key=lambda x: type_opt_const * function(x, **kwargs))

            if verbose:
                print(f'Iteration: {i} \t|\t new point = x2 = {x_new:0.3f} '
                      f'\t|\t f(x2) = {function(x_new, **kwargs): 0.3f}')

            if keep_history:
                history['iteration'].append(i)
                history['f_value'].append(function(x2, **kwargs))
                history['x0'].append(x0)
                history['x1'].append(x1)
                history['x2'].append(x2)

            if abs(x1 - x2) < epsilon and abs(function(x1, **kwargs) - function(x2, **kwargs)) < epsilon or \
                    abs(sum(map(lambda x, y: abs(x - y), x_list_old, [x2, x1, x0]))) < epsilon:
                print('Searching finished. Successfully. code 0')
                return {'point': x2, 'f_value': function(x2, **kwargs)}, history
        else:
            print('Searching finished. Max iterations have been reached. code 1')
            return {'point': x2, 'f_value': function(x2, **kwargs)}, history

    except Exception as e:
        print('Error with optimization. code 2')
        raise e
