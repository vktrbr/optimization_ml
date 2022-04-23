import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import streamlit as st
# import numpy as np

from StreamlitSupport.functions import parse_function

st.set_page_config(
    page_title=r"Inner point",
    page_icon="🦖",
)
st.sidebar.markdown('# Settings:')
methods_list = ["Newton’s method under equality constrains", "Log Barrier Method", "Primal-Dual Interior-Point Methods"]
equality_functions = []
inequality_functions = []
type_alg = st.sidebar.selectbox(r'Method', methods_list)

with st.sidebar:
    function, n_vars, var = parse_function(input_text='Enter the function here',
                                           default_value='x1 ** 3 - x1 ** 2 - x1 + x2 ** 2')

    if type_alg == "Newton’s method under equality constrains":
        equality_number = int(
            st.number_input('Number of equality type constraints ', value=1, min_value=0, max_value=n_vars - 1), )
    else:
        equality_number = int(
            st.number_input('Number of equality type constraints ', value=1, min_value=0, max_value=n_vars - 1), )
        inequality_number = int(st.number_input('Number of inequality type constraints ', value=1, min_value=0), )

with st.sidebar.form('input_data'):
    flag_empty_func = True
    st.markdown('# Conditions:')
    if type_alg == "Newton’s method under equality constrains":
        for i in range(equality_number):
            equality_function, _, _ = parse_function(input_text='Enter ' + str(i + 1) + ' equality function here',
                                                     default_value='x1 ** 3 - x1 ** 2 - x1 + x2 ** 2')
    else:
        for i in range(equality_number):

            equality_function, _, _ = parse_function(input_text='Enter ' + str(i + 1) + ' equality function here',
                                                     default_value='x1 ** 3 - x1 ** 2 - x1 + x2 ** 2')
        for i in range(inequality_number):

            inequality_function, _, _ = parse_function(
                input_text='Enter ' + str(i + 1) + ' inequality function . Format example: g(x)>0',
                default_value='x1 ** 3 - x1 ** 2 - x1 + x2 ** 2')

    x0 = st.text_input('Start search point. ' + str(tuple(var)), ('1.2, ' * n_vars)[:-2])
    epsilon = float(st.number_input('epsilon', value=1e-6, min_value=1e-6,
                                    max_value=1., step=1e-6, format='%.6f'))
    try:
        x0 = x0.replace(' ', '').split(',')
        x0 = tuple(map(float, x0))
        flag_wrong_x0 = False
    except Exception as e:
        st.write('**Error** with start point. ', str(e).capitalize())
        flag_wrong_x0 = True

    submit_button = st.form_submit_button(label='Solve!')

title = st.title(r"Inner point ")
st.write('**Hello!** \n\n'
         'This app demonstrates methods of the inner point \n\n '
         'You can specify a **function**, a **start point** and a **method**')

st.write('### Available methods: ')
for alg in methods_list:
    st.write(f'- **{alg}**')
st.stop()
