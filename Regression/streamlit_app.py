import streamlit as st

st.set_page_config(
    page_title=r"Regression",
    page_icon=":four:", )  # make page name
column_names = ['x1', 'x2', 'y']  # name of GIVEN columns
regression_types = ['Linear', 'Polynomial', 'Exponential']
regulators_types = ['L1', 'Tikhonov (L2)', 'None']
opt_alg = ['By gradient methods']  # matrix?
data = st.write('data')  # place for getting excel file


y_name = st.sidebar.selectbox(r'Y name', column_names)
regression_type = st.sidebar.selectbox(r'Regression model', regression_types)

with st.sidebar.form('input_data'):

    if regression_type == 'Polynomial':
        degree = int(st.number_input('degree of polynomial regression', value=1, min_value=1,
                     max_value=10, step=1))
    regulators_type = st.selectbox(r'Regulators type', regulators_types)
    optim_alg = st.selectbox(r'Optimization algorithm', opt_alg)

    submit_button = st.form_submit_button(label='Solve!')
