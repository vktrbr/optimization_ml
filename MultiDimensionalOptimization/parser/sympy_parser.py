from OneDimensionalOptimization.parser.sympy_parser import parse_func
import sympy
from typing import Callable, Tuple
from numbers import Integral


def sympy_to_callable(function_sympy: sympy.core.expr.Expr) -> Tuple[Callable, Integral]:
    """
    Convert sympy expression to callable function, for example::

        >>> f = parse_func('x**2')
        >>> f, _ = sympy_to_callable(f)
        >>> f
        <function_lambdifygenerated(x)>
        >>> f(2)
        4
    :param function_sympy: sympy expression
    :except: AssertionError. If function depends on more one variable
    :return: callable function from one argument and number of variables

    """
    var = list(function_sympy.free_symbols)
    return sympy.lambdify(var, function_sympy), len(var)