"""
For local running streamlit app : streamlit run scripts/streamlit_app.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import timeit
from tokenize import TokenError
import sympy
import streamlit as st

from sympy import SympifyError
from OneDimensionalOptimization.algorithms.combine_function import solve_task
from OneDimensionalOptimization.drawing.gss_visualizer import gen_animation_gss
from OneDimensionalOptimization.drawing.spi_visualizer import gen_animation_spi
from OneDimensionalOptimization.parser.sympy_parser import parse_func, sympy_to_callable
import re

from OneDimensionalOptimization.drawing.simple_plot import gen_lineplot

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

type_alg = st.sidebar.selectbox(r'Algorithm of optimization', algorithms_list)

if type_alg in ["Golden-section search", "Successive parabolic interpolation", "Brent's method"]:
    with st.sidebar.form('input_data'):
        flag_empty_func = True
        st.markdown('# Conditions:')
        function = st.text_input('Enter the function here', 'x ** 3 - x ** 2 - x - log2(x + 2)',
                                 help=help_function_string)

        if re.sub(r'\s', '', function) != '':
            if '|' in function:
                st.write('Use abs(h) instead of |h|')
            else:
                try:
                    function_latex = sympy.latex(sympy.sympify(function))
                    function_sympy = parse_func(function)
                    flag_empty_func = False
                except (SyntaxError, TypeError, NameError):
                    st.write('Check syntax. Wrong input :(')
                except (TokenError, SympifyError):
                    st.write('**Error**')
                    st.write('Write logarithms as: `log(x**2, x+1)` '
                             'variables as `pi * x` instead of `pi x`')

            type_opt = st.radio('Optimization aim', ['min', 'max'])
            bounds_a = st.number_input('Left bound', value=0., format='%.3f')
            bounds_b = st.number_input('Right bound', value=2., format='%.3f')
            bounds = sorted([bounds_a, bounds_b])
            cnt_iterations = int(st.number_input('Maximum number of iterations', value=500, min_value=0), )
            epsilon = float(st.number_input('epsilon', value=1e-6, min_value=1e-6,
                                            max_value=1., step=1e-6, format='%.6f'))
            if type_alg != "Brent's method":
                type_plot = st.radio('Type visualization', ['step-by-step', 'static'])
            else:
                type_plot = 'static'

        submit_button = st.form_submit_button(label='Solve!')

elif type_alg == 'BFGS algorithm':
    with st.sidebar.form('input_data'):
        flag_empty_func = True
        st.markdown('# Conditions:')
        function = st.text_input('Enter the function here', 'x ** 3 - x ** 2 - x - log2(x + 2)',
                                 help=help_function_string)

        if re.sub(r'\s', '', function) != '':
            if '|' in function:
                st.write('Use abs(h) instead of |h|')
            else:
                try:
                    function_latex = sympy.latex(sympy.sympify(function))
                    function_sympy = parse_func(function)
                    flag_empty_func = False
                except (SyntaxError, TypeError, NameError):
                    st.write('Check syntax. Wrong input :(')
                except (TokenError, SympifyError):
                    st.write('**Error**')
                    st.write('Write logarithms as: `log(x**2, x+1)` '
                             'variables as `pi * x` instead of `pi x`')
            type_opt = 'min'
            x0 = st.number_input('Start search point', value=2., format='%.3f')
            cnt_iterations = int(st.number_input('Maximum number of iterations', value=100, min_value=0))
            c1 = float(st.number_input('First Wolfe constant', value=1e-5,
                                       min_value=1e-5, max_value=1., format='%.5f'))
            c2 = float(st.number_input('Second Wolfe constant', value=0.9,
                                       min_value=1e-2, max_value=1., format='%.2f'))
            tolerance = float(
                st.number_input('tolerance', value=1e-6, min_value=1e-6, max_value=1., step=1e-6, format='%.6f'))
            type_plot = 'static'  # st.radio('Type visualization', ['step-by-step', 'static'])

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
    if type_alg != 'BFGS algorithm':
        problem_latex = rf'$ \displaystyle f(x) = {function_latex}, \quad'
        problem_latex += rf' x \in [{bounds_a: 0.3f}, {bounds_b :0.3f}] $'
    else:
        problem_latex = rf'$ \displaystyle f(x) = {function_latex}, \quad'
        problem_latex += rf' x_0 = {x0: 0.3f}$'
    st.write(problem_latex)

    # Generate second row
    st.write(f'Using the **{type_alg}** to solve:'
             rf'$\displaystyle \quad f \longrightarrow \{type_opt}' + r'_{x} $')

    # Drawing function
    function_callable = sympy_to_callable(function_sympy)

    time_start = timeit.timeit()
    try:
        try:
            if type_alg != 'BFGS algorithm':
                point, history = solve_task(type_alg, function=function_callable, bounds=bounds, keep_history=True,
                                            type_optimization=type_opt, max_iter=cnt_iterations, epsilon=epsilon)
            else:
                def function_callable_to_bfgs(x): return function_callable(x[0])
                point, history = solve_task(type_alg, function=function_callable_to_bfgs, x0=x0,
                                            keep_history=True, max_iter=cnt_iterations,
                                            tolerance=tolerance, c1=c1, c2=c2)
                diff_x = abs(x0 - point['point'])
                bounds_a = min(x0, point['point']) - diff_x * 0.15
                bounds_b = max(x0, point['point']) + diff_x * 0.15

            total_time = timeit.timeit() - time_start

            point_screen, f_value_screen, time_screen, iteration_screen = st.columns(4)
            point_screen.write(r'$ x_{\ ' + f'{type_opt}' + '} = ' + f'{point["point"]: 0.4f} $')
            f_value_screen.write(r'$ f_{\ ' + f'{type_opt}' + '} = ' + f'{point["f_value"]: 0.4f} $')
            time_screen.write(f'**Time** = {abs(total_time): 0.6f} s.')
            iteration_screen.write(f'**Iterations**: {len(history["iteration"])}')

            if type_plot == 'static':
                plotly_figure = gen_lineplot(function_callable,
                                             [bounds_a, bounds_b],
                                             [[point['point']], [point['f_value']]])

            if len(history['iteration']) == 1:
                st.write('**The solution found in 1 step!**')
                plotly_figure = gen_lineplot(function_callable,
                                             [bounds_a, bounds_b],
                                             [[point['point']], [point['f_value']]])

            elif type_plot == 'step-by-step' and type_alg == 'Golden-section search':
                plotly_figure = gen_animation_gss(function_callable, bounds, history)

            elif type_plot == 'step-by-step' and type_alg == 'Successive parabolic interpolation':
                plotly_figure = gen_animation_spi(function_callable, bounds, history)

            else:
                plotly_figure = gen_lineplot(function_callable,
                                             [bounds_a, bounds_b],
                                             [[point['point']], [point['f_value']]])
            figure = st.write(plotly_figure)

        except AssertionError:
            st.write('The method has diverged. Set new initial state.')
            st.stop()
    except NameError:
        st.write('Check syntax. Wrong input :(')
        st.stop()
