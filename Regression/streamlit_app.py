import streamlit as st
import io
import pandas as pd

st.set_page_config(
    page_title=r"Regression",
    page_icon=":four:", )  # make page name
regression_types = ['Linear', 'Polynomial', 'Exponential']
regulators_types = ['L1', 'Tikhonov (L2)', 'None']
opt_alg = ['By gradient methods']  # matrix?

regression_type = st.sidebar.selectbox(r'Regression model', regression_types)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # To convert to a string based IO:
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    # st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)

    column_names = dataframe.columns.tolist()
if uploaded_file is not None:

    with st.sidebar.form('input_data'):
        if regression_type == 'Polynomial':
            degree = int(st.number_input('degree of polynomial regression', value=1, min_value=1,
                                         max_value=10, step=1))
        regulators_type = st.selectbox(r'Regulators type', regulators_types)
        optim_alg = st.selectbox(r'Optimization algorithm', opt_alg)
        y_name = st.selectbox(r'Y name', column_names)

        x_name = st.multiselect(r'X name', column_names)

        submit_button = st.form_submit_button(label='Solve!')


    def convert_df(dataframe):
        return dataframe.to_csv().encode('utf-8')


    csv = convert_df(dataframe)

    st.download_button(
        "Press to Download",
        csv,
        "results.csv",
        "text/csv",
        key='download-csv'
    )
    print(x_name)
