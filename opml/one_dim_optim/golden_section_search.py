from __future__ import annotations

from typing import Tuple, Callable, Any, Literal
from .auxiliary_objects import *


def golden_section_search(function: Callable[[Real, Any], Real],
                          bounds: Tuple[Real, Real],
                          epsilon: Real = 1e-5,
                          type_optimization: Literal['min', 'max'] = 'min',
                          max_iter: int = 500,
                          verbose: bool = False,
                          keep_history: bool = False,
                          **kwargs) -> Tuple[Point, History]:
    """
    Golden-section search

    **If optimization fails golden_section_search will return the last point**

    Algorithm::
        phi = (1 + 5 ** 0.5) / 2

        1. a, b = bounds
        2. Calculate:
            x1 = b - (b - a) / phi,
            x2 = a + (b - a) / phi
        3. if `f(x1) > f(x2)` (for `min`) | if `f(x1) > f(x2)` (for `max`) then a = x1 else b = x2
        4. Repeat 2, 3 steps while `|a - b| > e`
    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param bounds: tuple with two numbers. This is left and right bound optimization. [a, b]
    :param epsilon: optimization accuracy
    :param type_optimization: 'min' / 'max' - type of required value
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :returns: tuple with point and history.

    """

    phi: Real = (1 + 5 ** 0.5) / 2

    type_optimization = type_optimization.lower().strip()
    assert type_optimization in ['min', 'max'], 'Invalid type optimization. Enter "min" or "max"'

    a: Real = bounds[0]
    b: Real = bounds[1]

    history: History = {'iteration': [0],
                        'middle_point': [(a + b) / 2],
                        'f_value': [function((a + b) / 2, **kwargs)],
                        'left_point': [a],
                        'right_point': [b]}

    try:
        for i in range(max_iter):
            x1: Real = b - (b - a) / phi
            x2: Real = a + (b - a) / phi

            if type_optimization == 'min':
                if function(x1, **kwargs) > function(x2, **kwargs):
                    a = x1
                else:
                    b = x2
            else:
                if function(x1, **kwargs) < function(x2, **kwargs):
                    a = x1
                else:
                    b = x2

            middle_point = (a + b) / 2
            if verbose:
                print(f'Iteration: {i} \t|\t point = {middle_point :0.3f} '
                      f'\t|\t f(point) = {function(middle_point, **kwargs): 0.3f}')

            if keep_history:
                history['iteration'].append(i)
                history['middle_point'].append(middle_point)
                history['f_value'].append(function(middle_point, **kwargs))
                history['left_point'].append(a)
                history['right_point'].append(b)

            if abs(x1 - x2) < epsilon:
                print('Searching finished. Successfully. code 0')
                return {'point': middle_point, 'f_value': function(middle_point, **kwargs)}, history
        else:
            middle_point = (a + b) / 2
            print('Searching finished. Max iterations have been reached. code 1')
            return {'point': middle_point, 'f_value': function(middle_point, **kwargs)}, history

    except Exception as e:
        print('Error with optimization. code 2')
        raise e


if __name__ == '__main__':

    def func(x): return 2.71828 ** (3 * x) + 5 * 2.71828 ** (-2 * x)
    point, data = golden_section_search(func, (-10, 10), type_optimization='min', verbose=True, keep_history=True)

    print(data['f_value'][:3])
    print(point)
