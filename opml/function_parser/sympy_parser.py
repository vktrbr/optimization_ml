from __future__ import annotations
import sympy
from sympy.parsing.sympy_parser import parse_expr
from typing import AnyStr, Callable
from numbers import Integral


class Text2Sympy:

    @staticmethod
    def parse_func(function_string: AnyStr) -> sympy.core.expr.Expr:
        """
        Convert the string to sympy.core.expr.Expr
        :param function_string: a string with function that is written by python rules
        :return: function as sympy Expression
        """
        function_string = logarithm_replace(function_string)
        function_string = ru_names_to_sympy(function_string)
        function_sympy = parse_expr(function_string)
        function_sympy = function_sympy.subs({sympy.symbols('e'): sympy.exp(1)})
        return function_sympy


def logarithm_replace(string: AnyStr) -> AnyStr:
    """
    Replace logN(A) on log(A, N), where N is the sequence of symbols before '('.
    A - is the symbols between '(' and ')', including other '(', ')' symbols at lower levels.

    examples::
        In [0]:  logarithm_replace('log3(x) + 2 * log5(4)')
        Out [0]: 'log(x, 3) + 2 * log(4, 5)'

        In [1]:  logarithm_replace('logA(log5(4 * x + 1)) + 8')
        Out [1]: 'log(log(4 * x + 1, 5), A) + 8'
    """
    dict_replaces = {}
    i = 0
    while i < len(string):

        if string[i:i + 3] == 'log' and string[i + 3] != '(':
            open_bracket_i: Integral = i + string[i:].find('(')
            print(open_bracket_i)
            n: AnyStr = string[i+3:open_bracket_i]
            j = open_bracket_i
            cnt_open_br = 1

            while cnt_open_br != 0:
                j += 1
                cur_symbol = string[j]

                if cur_symbol == ')':
                    cnt_open_br -= 1

                elif cur_symbol == '(':
                    cnt_open_br += 1

            x = string[open_bracket_i + 1:j]
            x = logarithm_replace(x)
            dict_replaces[string[i:j]] = f'log({x}, {n}'

            i = j + 1

        else:
            i += 1

    for pattern in dict_replaces.keys():
        string = string.replace(pattern, dict_replaces[pattern])

    return string


def ru_names_to_sympy(string: AnyStr) -> AnyStr:
    """
    Replace russian names of function like tg to tan
    """
    dictionary_ru_func = {'tg': 'tan', 'ctg': 'cot', 'arcsin': 'asin',
                          'arccos': 'acos', 'arctg': 'atan', 'arcctg': 'acot'}
    for ru_f in dictionary_ru_func.keys():
        string = string.replace(ru_f, dictionary_ru_func[ru_f])

    return string


def sympy_to_callable(function_sympy: sympy.core.expr.Expr) -> Callable:
    """
    Conver sympy expression to callable function
    :param function_sympy: sympy expression
    :return: callable function from one argument
    """
    assert len(function_sympy.free_symbols) <= 1, 'Too many arguments in function'
    var = function_sympy.free_symbols
    return sympy.lambdify(var, function_sympy)


if __name__ == '__main__':
    str_example = 'log3(x) + 2 * log5(4)'
    print(logarithm_replace(str_example))
    str_example = 'logA(log5(4 * x + 1)) + 8'
    print(logarithm_replace(str_example))
