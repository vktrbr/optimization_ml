from __future__ import annotations
import sympy
from sympy.parsing.sympy_parser import parse_expr
from typing import AnyStr
import re


class Text2Sympy:

    @staticmethod
    def parse_func(function_string: AnyStr) -> sympy.core.expr.Expr:
        """
        Convert the string to sympy.core.expr.Expr
        :param function_string: a string with function that is written by python rules
        :return: function as sympy Expression
        """
        log_pattern = re.compile(r'log.+\(.+\)')
        logarithms = re.findall(log_pattern, function_string)
        if len(logarithms) > 0:
            for log in logarithms:
                log: AnyStr
                log_base = log[3:log.find('(')]
                log_arg = log[log.find('(') + 1:log.find(')')]
                log_right = f'log({log_arg}, {log_base})'
                function_string = function_string.replace(log, log_right)

        function_sympy = parse_expr(function_string)
        function_sympy = function_sympy.subs({sympy.symbols('e'): sympy.exp(1)})
        return function_sympy
