from __future__ import annotations
import sympy as sp
from sympy.parsing.sympy_parser import standard_transformations
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
        try:
            transformations = standard_transformations
            transformations += (logarithm_notation, )
            function_string = parse_expr(function_string,
                                         transformations=transformations)
            check_var = function_string.free_symbols
            if len(check_var) > 1:
                print('Too many variables. Please enter function that depend on the one variable')

        except Exception as e:
            raise e
        return function_string


def logarithm_notation(tokens, local_dict, global_dict):
    for toknum, tokval in tokens:
        print(toknum, tokval)


if __name__ == '__main__':
    print(Text2Sympy.parse_func('log3(x) + x'))