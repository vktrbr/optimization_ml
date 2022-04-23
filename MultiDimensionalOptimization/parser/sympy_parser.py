from OneDimensionalOptimization.parser.sympy_parser import parse_func
import sympy
from typing import Callable, Tuple, List


def sympy_to_callable(function_sympy: sympy.core.expr.Expr) -> Tuple[Callable, List]:
    """
    Convert sympy expression to callable function, for example::

        >>> f = parse_func('x ** 2 + y ** 2')
        >>> f, _ = sympy_to_callable(f)
        >>> f
        <function_lambdifygenerated(x)>
        >>> f(2, 2)
        8
    :param function_sympy: sympy expression
    :except: AssertionError. If function depends on more one variable
    :return: callable function from number of varialbes arguments and number of variables

    """
    var = list(function_sympy.free_symbols)
    var = sorted(var, key=lambda x: str(x))

    return sympy.lambdify(var, function_sympy), var
