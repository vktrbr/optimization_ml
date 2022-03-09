from __future__ import annotations
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from typing import AnyStr

# assert isinstance(function, sp.core.expr.Expr), 'Function is not sympy'
# assert len(function.free_symbols) <= 1, 'Function depend on more than 1 variables'
# if len(function.free_symbols) == 0:
#     print('Function independent on variables. code 0')
#     middle_point = (a + b) / 2
#     return {'point': middle_point, 'f_value': float(function)}, history
#
# function = sp.lambdify(list(function.free_symbols), function, 'numpy')


class Text2Sympy:

    @staticmethod
    def parse_func(function_string: AnyStr) -> sp.core.expr.Expr:
        """
        Convert the string to sympy.core.expr.Expr
        :param function_string: a string with function that is written by python rules
        :return: function as sympy Expression
        """
        function_sympy = parse_expr(function_string)
        return function_sympy
