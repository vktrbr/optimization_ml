"""
For local running streamlit app : streamlit run opml/streamlit_app.py
"""
from tokenize import TokenError

import sympy
import streamlit as st
from sympy import SympifyError

from function_parser.sympy_parser import Text2Sympy
import re

st.set_page_config(
    page_title="OneD optimization",
    page_icon=":balloon:",
)

algorithms_list = ["Golden-section search", "Successive parabolic interpolation",
                   "Brent's method", "Broyden–Fletcher–Goldfarb–Shanno algorithm"]

st.title('1-D Optimization')

st.sidebar.markdown('## Task conditions:')
function = st.sidebar.text_input('Enter the function here', 'log3(x) * e ** (x + pi) + x ** 2 + sin(x)', )

if re.sub(r'\s', '', function) != '':

    try:
        function_sympy = Text2Sympy.parse_func(function)
        st.sidebar.write(r'> $ \displaystyle f(x) = ', rf'\ {sympy.latex(sympy.sympify(function))} $')
        flag_empty_func = False
    except (SyntaxError, TypeError):
        st.sidebar.write('Check syntax. Wrong input :(')
    except (TokenError, SympifyError):
        st.sidebar.write('Error')
        st.sidebar.write('Try to write logarithms as: `log(x**2, x+1)`')
