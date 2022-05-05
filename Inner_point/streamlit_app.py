import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import streamlit as st
from Inner_point.algorithms import log_barrier_solver, bound_constrained_lagrangian_method
from Inner_point.visualizer import contour_log_barrier
from StreamlitSupport.functions import parse_function
import numpy

numpy.seterr('ignore')

st.set_page_config(
    page_title=r"Inner point",
    page_icon=":five:",
)

# --- Part of main settings at sidebar --- #
st.sidebar.markdown('# Settings:')

# "Primal-Dual Interior-Point Methods"
methods_list = ["Newton’s method under equality constrains", "Log Barrier Method"]

equality_functions = []
inequality_functions = []
type_alg = st.sidebar.selectbox(r'Method', methods_list)

with st.sidebar:
    function = parse_function(input_text='Enter the function here',
                              default_value='x1 ** 3 - x1 ** 2 - x1 + x2 ** 2')

    if type_alg == "Newton’s method under equality constrains":
        equality_number = int(
            st.number_input('Number of equality type constraints ', value=1,
                            min_value=1, max_value=function.n_vars - 1))
    else:
        # equality_number = int(
        #     st.number_input('Number of equality type constraints ', value=1,
        #                     min_value=0, max_value=function.n_vars - 1))
        inequality_number = int(st.number_input('Number of inequality type constraints ', value=1, min_value=1))

with st.sidebar.form('input_data'):
    st.markdown('# Conditions:')
    if type_alg == "Newton’s method under equality constrains":
        for i in range(equality_number):
            equality_function = parse_function(input_text='Enter ' + str(i + 1) + ' equality function here',
                                               default_value='x1 + x2 - 1')
            equality_functions.append(equality_function)
    else:
        # for i in range(equality_number):
        #     equality_function = parse_function(input_text='Enter ' + str(i + 1) + ' equality function here',
        #                                              default_value='x1 ** 3 - x1 ** 2 - x1 + x2 ** 2')
        for i in range(inequality_number):
            inequality_function = parse_function(
                input_text='Enter ' + str(i + 1) + ' inequality function . Format example: g(x)>0',
                default_value='x1 ** 2 + x2 + 1')
            inequality_functions.append(inequality_function)

    x0 = st.text_input('Start search point. ' + str(tuple(function.variables)), ('1.2, ' * function.n_vars)[:-2])
    # epsilon = float(st.number_input('epsilon', value=1e-6, min_value=1e-6,
    #                                 max_value=1., step=1e-6, format='%.6f'))
    try:
        x0 = x0.replace(' ', '').split(',')
        x0 = tuple(map(float, x0))
        flag_wrong_x0 = False
    except Exception as e:
        st.write('**Error** with start point. ', str(e).capitalize())
        flag_wrong_x0 = True

    submit_button = st.form_submit_button(label='Solve!')

if submit_button is False or function.flag_empty_func:
    title = st.title(r"Inner point ")
    st.write('**Hello!** \n\n'
             'This app demonstrates methods of the inner point \n\n '
             'You can specify a **function**, a **start point** and a **method**')

    st.write('### Available methods: ')
    for alg in methods_list:
        st.write(f'- **{alg}**')
    st.stop()
else:
    # --- Print initial problem --- #
    st.write('### Problem:')
    problem_latex = rf'$ \displaystyle f(x) = {function.latex}, \quad'
    problem_latex += rf' x_0 = {x0}$'
    st.write(problem_latex)

    # --- Print constaints and solution --- #
    if type_alg == "Newton’s method under equality constrains":
        const_latex = rf'$\displaystyle G(x) = 0, \quad G(x) = '
        const_latex_matrix = r'\displaystyle \begin{bmatrix}'
        for func in equality_functions:
            const_latex_matrix += func.latex + r'\\'

        const_latex_matrix += r'\end{bmatrix} $'
        st.write(const_latex + const_latex_matrix)

        constraints = list(map(lambda x: x.call, equality_functions))
        solution, history = bound_constrained_lagrangian_method(function.call, x0, constraints, max_iter=100)
        g_x_min_latex = rf'$G(x_{{\min}}) = \begin{{bmatrix}}'
        for func in equality_functions:
            g_x_min_latex += f'{(func.call(solution)): .3f}' + '&'

        else:
            g_x_min_latex = g_x_min_latex[:-2]
            g_x_min_latex += rf'\end{{bmatrix}}$'
            st.write(g_x_min_latex)
        st.write(rf'$f(x_{{\min}}) = {function.call(solution): 0.4f}$')

    elif type_alg == 'Log Barrier Method':
        # --- CONDITION --- #
        const_latex = rf'$\displaystyle G(x) \succeq 0, \quad G(x) = '
        ineq_latex_matrix = r'\displaystyle \begin{bmatrix}'
        for func in inequality_functions:
            ineq_latex_matrix += func.latex + r'\\'

        ineq_latex_matrix += r'\end{bmatrix} $'
        st.write(const_latex + ineq_latex_matrix)

        log_barrier_function_latex = rf'$ \qquad \displaystyle P(x, \mu) = ' \
                                     rf'f(x) - \mu \sum_{{i=1}}^{inequality_number}\ln g_i(x)$'
        st.write('**Log barrier function:**' + log_barrier_function_latex)

        # --- SOLUTION --- #
        constraints = list(map(lambda x: x.call, inequality_functions))
        solution, history = log_barrier_solver(function.call, x0, constraints, max_iter=100)
        st.write(r'$ x_{\min} = ' + f'{list(numpy.round(solution, 3))}$')

        g_x_min_latex = rf'$G(x_{{\min}}) = \begin{{bmatrix}}'
        for func in inequality_functions:
            g_x_min_latex += f'{(func.call(solution)): .3f}' + '&'

        else:
            g_x_min_latex = g_x_min_latex[:-2]
            g_x_min_latex += rf'\end{{bmatrix}}$'
        st.write(g_x_min_latex)
        st.write(rf'$f(x_{{\min}}) = {function.call(solution): 0.4f}$')
        # --- PLOTLY CHART --- #
        st.plotly_chart(contour_log_barrier(function.call, history, constraints))
