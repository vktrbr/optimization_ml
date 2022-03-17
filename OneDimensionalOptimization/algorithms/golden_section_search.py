from __future__ import annotations

from typing import Tuple, Callable, Any, Literal
from OneDimensionalOptimization.algorithms.support import *


def golden_section_search(function: Callable[[Real, Any], Real],
                          bounds: Tuple[Real, Real],
                          epsilon: Real = 1e-5,
                          type_optimization: Literal['min', 'max'] = 'min',
                          max_iter: int = 500,
                          verbose: bool = False,
                          keep_history: bool = False,
                          **kwargs) -> Tuple[Point, HistoryGSS]:
    """
    Golden-section search

    **Algorithm:**

        :math:`\\displaystyle \\varphi = \\frac{(1 + \\sqrt{5})}{2}`

        1. :math:`a, b` - left and right bounds

        2. | :math:`\\displaystyle x_1 = b - \\frac{b - a}{\\varphi}`
           | :math:`\\displaystyle x_2 = a + \\frac{b - a}{\\varphi}`

        3. | if :math:`\\displaystyle f(x_1) > f(x_2)` (for min)
                :math:`\\displaystyle [ f(x_1) < f(x_2)` (for max) :math:`]`
           | then :math:`a = x_1` else  :math:`b = x_2`

        4. Repeat  :math:`2, 3` steps while :math:`|a - b| > e`

    **If optimization fails golden_section_search will return the last point**

    Code example::

        >>> def func(x): return 2.71828 ** (3 * x) + 5 * 2.71828 ** (-2 * x)
        >>> point, data = golden_section_search(func, (-10, 10), type_optimization='min', keep_history=True)

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param bounds: tuple with two numbers. This is left and right bound optimization. [a, b]
    :param epsilon: optimization accuracy
    :param type_optimization: 'min' / 'max' - type of required value
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.

    """

    phi: Real = (1 + 5 ** 0.5) / 2

    type_optimization = type_optimization.lower().strip()
    assert type_optimization in ['min', 'max'], 'Invalid type optimization. Enter "min" or "max"'

    a: Real = bounds[0]
    b: Real = bounds[1]
    if keep_history:
        history: HistoryGSS = {'iteration': [0],
                               'middle_point': [(a + b) / 2],
                               'f_value': [function((a + b) / 2, **kwargs)],
                               'left_point': [a],
                               'right_point': [b]}

    else:
        history: HistoryGSS = {'iteration': [], 'middle_point': [], 'f_value': [], 'left_point': [], 'right_point': []}

    if verbose:
        print(f'Iteration: {0} \t|\t point = {(a + b) / 2 :0.3f} '
              f'\t|\t f(point) = {function((a + b) / 2, **kwargs): 0.3f}')

    try:
        for i in range(1, max_iter):
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
