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
    page_title=r"OneD optimization",
    page_icon=":balloon:",
)
st.title(r"$ \text{1-D optimization} $")

algorithms_list = ["Golden-section search", "Successive parabolic interpolation",
                   "Brent's method", "BFGS algorithm"]

with st.sidebar.form('input_data'):
    st.markdown('## Task conditions:')
    function = st.text_input('Enter the function here', 'e ** (-x**2)')
    function_latex = sympy.latex(sympy.sympify(function))

    if re.sub(r'\s', '', function) != '':
        if '|' in function:
            st.write('Use abs(h) instead of |h|')
        else:
            try:
                function_sympy = Text2Sympy.parse_func(function)
                flag_empty_func = False
            except (SyntaxError, TypeError):
                st.write('Check syntax. Wrong input :(')
            except (TokenError, SympifyError):
                st.write('Error')
                st.write('Try to write logarithms as: `log(x**2, x+1)`')

        type_opt = st.radio('Optimization aim', ['min', 'max'])

        bounds_a = st.number_input('Left bound', value=0)
        bounds_b = st.number_input('Right bound', value=1)
        type_alg = st.selectbox('Algorithm of optimization', algorithms_list)
        verbose = st.checkbox('Verbose', True)

    submit_button = st.form_submit_button(label='Solve!')

if not submit_button:
    st.stop()

else:
    problem_latex = r'$ \text{Function:} \qquad'
    problem_latex += rf' \displaystyle f(x) = {function_latex}, \quad'
    problem_latex += rf' x \in [{bounds_a}, {bounds_b}]'
    problem_latex += '$'

    method_latex = r'$ \text{Using the ``' + type_alg + '" to solve: }'
    method_latex += rf'\quad f \longrightarrow \{type_opt}'
    method_latex += '$'

    st.write(problem_latex)
    st.write(method_latex)
