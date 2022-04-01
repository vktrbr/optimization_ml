import os
import sys

sys.path.insert(0, os.path.abspath('..'))

from tokenize import TokenError
import sympy
import streamlit as st
from MultiDimensionalOptimization.parser.sympy_parser import parse_func, sympy_to_callable

from MultiDimensionalOptimization.algorithms.combine_function import solve_task_nd_minimize
from sympy import SympifyError
from StreamlitSupport.constants import help_function_string
import re
from time import time_ns

st.set_page_config(
    page_title=r"MultiD optimization",
    page_icon=":three:",
)

algorithms_list = ["Gradient Descent", "Nonlinear conjugate gradient method"]

epsilon = 1e-5
gamma = 0.1
delta = 0.1
lambda0 = 0.1

type_alg = st.sidebar.selectbox(r'Algorithm of descent', algorithms_list)
if type_alg == 'Gradient Descent':
    type_step = st.sidebar.selectbox(r'Type step', ['Fixed', 'Descent', 'Optimal'])
else:
    type_step = ''

with st.sidebar.form('input_data'):
    flag_empty_func = True
    st.markdown('# Conditions:')
    function = st.text_input('Enter the function here', 'e ** (-(x ** 2 + y ** 2))',
                             help=help_function_string)

    if re.sub(r'\s', '', function) != '':
        if '|' in function:
            st.write('Use abs(h) instead of |h|')
        else:
            try:
                function_latex = sympy.latex(sympy.sympify(function))
                function_sympy = parse_func(function)
                function_callable, n_vars = sympy_to_callable(function_sympy)
                flag_empty_func = False
            except (SyntaxError, TypeError, NameError):
                st.write('Check syntax. Wrong input :(')
            except (TokenError, SympifyError):
                st.write('**Error**')
                st.write('Write logarithms as: `log(x**2, x+1)` '
                         'variables as `pi * x` instead of `pi x`')

        x0 = st.text_input('Start search point', '1, -1')
        try:
            x0 = x0.replace(' ', '').split(',')
            print(x0)
            x0 = tuple(map(float, x0))
        except Exception as e:
            st.write('**Error** with start point', str(e))

        cnt_iterations = int(st.number_input('Maximum number of iterations', value=500, min_value=0), )
        epsilon = float(st.number_input('epsilon', value=1e-6, min_value=1e-6,
                                        max_value=1., step=1e-6, format='%.6f'))

        if type_alg == 'Gradient Descent':

            if type_step in ['Descent', 'Fixed']:
                gamma = float(st.number_input("gamma. This gradient multiplier", value=1e-6, min_value=1e-6,
                                              max_value=1., step=1e-6, format='%.6f'))
            if type_step == 'Descent':
                lambda0 = float(
                    st.number_input("lambda. This multiplier that reduces the gamma", value=9e-1, min_value=1e-4,
                                    max_value=1., step=1e-4, format='%.4f'))

                delta = float(st.number_input(r"delta. Meaning in documents", value=9e-1,
                                              min_value=1e-4,
                                              max_value=1., step=1e-4, format='%.4f'))

    submit_button = st.form_submit_button(label='Solve!')

if not submit_button or flag_empty_func:
    title = st.title(r"N-D optimization")
    st.write('**Hello!** \n\n'
             'This app demonstrates Nd optimization algorithms \n\n '
             'You can specify a **function**, a **start point** and a **method**')

    st.write('### Available methods: ')
    for alg in algorithms_list:
        st.write(f'- **{alg}**')
    st.stop()

else:
    # Generate first raw
    st.write('### Problem:')
    problem_latex = rf'$ \displaystyle f(x) = {function_latex}, \quad'
    problem_latex += rf' x_0 = {x0}$'
    st.write(problem_latex)

    # Generate second row
    st.write(f'Using the **{type_alg}** to solve:'
             rf'$\displaystyle \quad f \longrightarrow \min' + r'_{x} $')

    time_start = time_ns()
    try:
        try:
            type_alg += ' ' + type_step
            point, history = solve_task_nd_minimize(type_alg, function=function_callable, x0=x0,
                                                    keep_history=True, max_iter=cnt_iterations)

            total_time = time_ns() - time_start

            point_screen, f_value_screen, time_screen, iteration_screen = st.columns(4)
            point_screen.write(r'$ x_{\min} = ' + f'{point["point"]: 0.4f} $')
            f_value_screen.write(r'$ f_{\min} = ' + f'{point["f_value"]: 0.4f} $')
            time_screen.write(f'**Time** = {abs(total_time): 0.6f} s.')
            iteration_screen.write(f'**Iterations**: {len(history["iteration"])}')

        except AssertionError:
            st.write('The method has diverged. Set new initial state.')
            st.stop()

    except NameError:
        st.write('Check syntax. Wrong input :(')
        st.stop()
