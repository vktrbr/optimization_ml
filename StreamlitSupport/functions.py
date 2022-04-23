from typing import AnyStr

from tokenize import TokenError
import sympy
import streamlit as st
from MultiDimensionalOptimization.parser.sympy_parser import parse_func, sympy_to_callable

from sympy import SympifyError
from StreamlitSupport.constants import help_function_string
import re


def parse_function(input_text: AnyStr = 'Enter the function here',
                   default_value: AnyStr = 'x1 ** 3 - x1 ** 2 - x1 + x2 ** 2'):
    function = st.text_input(input_text, default_value, help=help_function_string)
    n_vars = 0
    var = 0
    if re.sub(r'\s', '', function) != '':
        if '|' in function:
            st.write('Use abs(h) instead of |h|')
        else:
            try:
                function_latex = sympy.latex(sympy.sympify(function))
                function_sympy = parse_func(function)
                function_callable_initial, var = sympy_to_callable(function_sympy)
                n_vars = len(var)

                def function_callable(x):
                    return function_callable_initial(*x)

                flag_empty_func = False
            except (SyntaxError, TypeError, NameError):
                st.write('Check syntax. Wrong input :(')
            except (TokenError, SympifyError):
                st.write('**Error**')
                st.write('Write logarithms as: `log(x**2, x+1)` '
                         'variables as `pi * x` instead of `pi x`')
        return function, n_vars, var
    else:
        st.stop()
