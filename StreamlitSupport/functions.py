import re
from numbers import Integral
from tokenize import TokenError
from typing import NamedTuple, Text, Callable, Tuple

import streamlit as st
import sympy
from sympy import SympifyError

from MultiDimensionalOptimization.parser.sympy_parser import parse_func, sympy_to_callable
from StreamlitSupport.constants import help_function_string


class FunctionKeeper(NamedTuple):
    latex: Text
    sympy: sympy.core.expr.Expr
    call: Callable
    n_vars: Integral
    variables: Tuple
    flag_empty_func: bool


def parse_function(input_text: Text = 'Enter the input_text here',
                   default_value: Text = 'x1 ** 3 - x1 ** 2 - x1 + x2 ** 2') -> FunctionKeeper:
    input_text = st.text_input(input_text, default_value, help=help_function_string)
    function = FunctionKeeper(None, None, None, None, None, True)
    if re.sub(r'\s', '', input_text) != '':
        if '|' in input_text:
            st.write('Use abs(h) instead of |h|')
            st.stop()
        else:
            try:
                function_latex = sympy.latex(sympy.sympify(input_text))
                function_sympy = parse_func(input_text)
                function_callable_initial, var = sympy_to_callable(function_sympy)
                n_vars = len(var)

                def function_callable(x):
                    return function_callable_initial(*x)

                flag_empty_func = False
                function = FunctionKeeper(function_latex, function_sympy, function_callable, n_vars, var,
                                          flag_empty_func)

            except (SyntaxError, TypeError, NameError):
                st.write('Check syntax. Wrong input :(')
            except (TokenError, SympifyError):
                st.write('**Error**')
                st.write('Write logarithms as: `log(x**2, x+1)` '
                         'variables as `pi * x` instead of `pi x`')

        return function
    else:
        st.stop()
