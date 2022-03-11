"""
For local running streamlit app : streamlit run opml/streamlit_app.py
"""
import timeit
from tokenize import TokenError

import pandas as pd
import sympy
import streamlit as st

from sympy import SympifyError
from com_func import solve_task
from plot_funcs.simple_plot import gen_lineplot
from function_parser.sympy_parser import Text2Sympy, sympy_to_callable
import re


st.set_page_config(
    page_title=r"OneD optimization",
    page_icon=":two:",
)

algorithms_list = ["Golden-section search", "Successive parabolic interpolation",
                   "Brent's method", "BFGS algorithm"]

help_function_string = r'''
Available functions: $ \\ 
\log(u, v) \ (v - \text{base}), \ \exp(u),\ \operatorname{abs}(u), \ \tan(u), \ \cot(u), \\
\sin(u), \ \cos(u), \ \operatorname{asin}(u), \ \operatorname{sec}(u) \ \operatorname{csc}(u),
\ \operatorname{sinc}(u), \\ \ \operatorname{asin}(u), \ \operatorname{acos}(u), 
\ \operatorname{atan}(u), \ \operatorname{acot}(u), \ \operatorname{asec}(u), 
\ \operatorname{acsc}(u), u!, \operatorname{sqrt}(u)$
'''


with st.sidebar.form('input_data'):
    flag_empty_func = True
    st.markdown('# Conditions:')
    function = st.text_input('Enter the function here', '10 + x ** 2 - 10 * cos(2 * pi * x)',
                             help=help_function_string)

    if re.sub(r'\s', '', function) != '':
        if '|' in function:
            st.write('Use abs(h) instead of |h|')
        else:
            try:
                function_latex = sympy.latex(sympy.sympify(function))
                function_sympy = Text2Sympy.parse_func(function)
                flag_empty_func = False
            except (SyntaxError, TypeError):
                st.write('Check syntax. Wrong input :(')
            except (TokenError, SympifyError):
                st.write('**Error**')
                st.write('Write logarithms as: `log(x**2, x+1)` '
                         'variables as `pi * x` instead of `pi x`')

        type_opt = st.radio('Optimization aim', ['min', 'max'])

        bounds_a = st.number_input('Left bound', value=-5.12, format='%.3f')
        bounds_b = st.number_input('Right bound', value=5.12, format='%.3f')
        bounds = sorted([bounds_a, bounds_b])
        type_alg = st.selectbox('Algorithm of optimization', algorithms_list)
        cnt_iterations = int(st.number_input('Maximum number of iterations', value=500, min_value=0), )

    submit_button = st.form_submit_button(label='Solve!')

if not submit_button or flag_empty_func:
    title = st.title(r"1-D optimization")
    st.write('**Hello!** \n\n'
             'This app demonstrates 1d optimization algorithms \n\n '
             'You can specify **function**, **bounds** and **method**')
    st.write('### Available methods: ')
    for alg in algorithms_list:
        st.write(f'- **{alg}**')
    st.stop()

else:
    # Generate first raw
    st.write('### Problem:')
    problem_latex = rf'$ \displaystyle f(x) = {function_latex}, \quad'
    problem_latex += rf' x \in [{bounds_a: 0.3f}, {bounds_b :0.3f}] $'
    st.write(problem_latex)

    # Generate second row
    st.write(f'Using the **{type_alg}** to solve:'
             rf'$\displaystyle \quad f \longrightarrow \{type_opt}' + r'_{x} $')

    # Drawing function
    function_callable = sympy_to_callable(function_sympy)

    time_start = timeit.timeit()
    point, history = solve_task(type_alg, function=function_callable, bounds=bounds, keep_history=True,
                                type_optimization=type_opt, max_iter=cnt_iterations)
    total_time = timeit.timeit() - time_start

    point_screen, f_value_screen, time_screen, iteration_screen = st.columns(4)
    point_screen.write(r'$ x_{\ ' + f'{type_opt}' + '} = ' + f'{point["point"]: 0.4f} $')
    f_value_screen.write(r'$ f_{\ ' + f'{type_opt}' + '} = ' + f'{function_callable(point["point"]): 0.4f} $' )
    time_screen.write(f'**Time** = {abs(total_time): 0.6f} s.')
    iteration_screen.write(f'**Iterations**: {len(history["iteration"])}')

    plotly_figure = gen_lineplot(function_callable,
                                 [bounds_a, bounds_b],
                                 [[point['point']], [point['f_value']]])
    figure = st.write(plotly_figure)

    information_cols = st.columns([1, 4, 4])
    history = pd.DataFrame(history).set_index('iteration')
    history = history.rename_axis('iteration', axis=0)

    info_df = pd.DataFrame({'': [point['point'], point['f_value'], history.index.max(), total_time]},
                           index=[f'x_{type_opt}', 'f(x)', r'n of iter', r'time'])