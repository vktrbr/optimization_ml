from __future__ import annotations

from typing import List, Literal, Tuple, TypedDict
from numbers import Real
import sympy as sp


def golden_section_search(function: sp.core.expr.Expr,
                          bounds: Tuple[Real, Real],
                          epsilon: Real = 1e-5,
                          type_optimization: Literal['min', 'max'] = 'min',
                          max_iter: int = 500,
                          verbose: bool = False,
                          keep_history: bool = False) -> Tuple[Dot, History]:
    """
    Golden-section search
    **If optimize will fail golden_section_search will return last point**

    Algorithm::
        phi = (1 + 5 ** 0.5) / 2

        1. a, b = bounds
        2. Calculate:
            x1 = b - (b - a) / phi,
            x2 = a + (b - a) / phi
        3. if `f(x1) > f(x2)` (for `min`) | if `f(x1) > f(x2)` (for `max`) then a = x1 else b = x2
        4. Repeat 2, 3 steps while `|a - b| > e`
    :param function: a sympy expression that depends on a one variable
    :param bounds: tuple with two numbers. This is left and right bound optimization. [a, b]
    :param epsilon: optimization accuracy
    :param type_optimization: 'min' / 'max' - type of required value
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :returns: tuple with dot and history.


    """
    assert isinstance(function, sp.core.expr.Expr), 'Function is not sympy'
    phi: Real = (1 + 5 ** 0.5) / 2

    type_optimization = type_optimization.lower().strip()
    assert type_optimization in ['min', 'max'], 'Invalid type optimization. Enter "min" or "max"'

    assert len(function.free_symbols) <= 1, 'Function depend on more than 1 variables'

    a, b = bounds

    history: History = {'iteration': [], 'point': [], 'f_value': []}

    if len(function.free_symbols) == 0:
        print('Function independent on variables. code 0')
        middle_point = (a + b) / 2
        return {'point': middle_point, 'f_value': float(function)}, history

    name_var = str(list(function.free_symbols)[0])

    function = sp.lambdify(list(function.free_symbols), function, 'numpy')

    try:
        for i in range(max_iter):
            x1: Real = b - (b - a) / phi
            x2: Real = a + (b - a) / phi
            if type_optimization == 'min':
                if function(x1) > function(x2):
                    a = x1
                else:
                    b = x2
            else:
                if function(x1) < function(x2):
                    a = x1
                else:
                    b = x2

            middle_point = (a + b) / 2
            if verbose:
                print(f'Iteration: {i} \t|\t {name_var} = {middle_point :0.3f} '
                      f'\t|\t f({name_var}) = {function(middle_point): 0.3f}')

            if keep_history:
                history['iteration'].append(i)
                history['point'].append(middle_point)
                history['f_value'].append(function(middle_point))

            if abs(x1 - x2) < epsilon:
                print('Searching finished. Successfully. code 0')
                return {'point': middle_point, 'f_value': function(middle_point)}, history
        else:
            middle_point = (a + b) / 2
            print('Searching finished. Max iterations have been reached. code 1')
            return {'point': middle_point, 'f_value': function(middle_point)}, history

    except Exception as e:
        print('Error with optimization. code 2')
        raise e


class History(TypedDict):
    iteration: List[int]
    point: List[Real]
    f_value: List[Real]


class Dot(TypedDict):
    point: Real
    f_value: Real


if __name__ == '__main__':

    x = sp.symbols('x')
    func = sp.exp(3 * x) + 5 * sp.exp(-2 * x)
    point, data = golden_section_search(x ** 2, (0, 1), type_optimization='min')
    print(data['f_value'][:3])
    print(point)
