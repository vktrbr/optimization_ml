import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import streamlit as st
from datetime import date
# import io
import pandas as pd
import yfinance as yf
import numpy as np
from typing import List
from Regression.algorithms.linear_model import linear_regression

# from Regression.algorithms.exponential_regression import exponential_regression
# from Regression.algorithms.polynomial_regression import polynomial_regression

st.set_page_config(
    page_title=r"Regression",
    page_icon=":four:")  # make page name

regression_types = ['Linear', 'Polynomial', 'Exponential']
regulators_types = ['None', 'Lasso (L1)', 'Tikhonov (L2)']

# ------ Initial variables ------ #
if 'has_been_uploaded' not in st.session_state:
    st.session_state.has_been_uploaded = False

if 'placeholder' not in st.session_state:
    st.session_state.placeholder = st.empty()

if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame([])


# ------ Main Page 1 Settings ------ #
def main_page():
    with st.container():
        st.title('Regression')
        left, right = st.columns([7, 1])
        left.markdown('<h4>Linear, Polynomial, Exponential</h4>', unsafe_allow_html=True)
        _reset_button = right.button('Reset', key=True)

    st.write('Hello! Welcome to regression app. You can upload your **own data** or check some **stock index**.')
    _type_data = st.selectbox('', ['ticker', 'own data'])
    return _type_data, _reset_button


def ticker_way():
    # form for stock exchange data upload settings
    _placeholder = st.empty()
    with _placeholder.container():
        with st.form('ticker'):
            st.markdown("Check the tickers on the [https://finance.yahoo.com/moex]"
                        "(https://finance.yahoo.com/quote/IMOEX.ME?p=IMOEX.ME&.tsrc=fin-srch). "
                        "Ticker name in parentheses")

            ticker = st.text_input('Select a ticker', 'IMOEX.ME')
            start_date: date = st.date_input('Start date', value=date(year=2022, month=2, day=20))
            end_date: date = st.date_input('End date')
            interval = st.selectbox('Intervals', options=['1d', '5m', '15m', '30m', '60m',
                                                          '1h', '90m', '5d', '1wk', '1mo'])
            submit_button = st.form_submit_button(label='Download data')

        if submit_button:
            # Check dates
            if start_date > end_date:
                st.markdown("<h6 style='text-align: center;'>Check dates. End less Start</h6>", unsafe_allow_html=True)
                st.stop()

            # Try downloading stock exchange data. If everything is correct, returns the dataset and closes the form
            try:
                dataset: pd.DataFrame = yf.download(ticker, start_date, end_date, interval=interval)
                if dataset.empty:
                    raise ValueError('no data in ticker dataset')
                dataset['Normalized date'] = np.linspace(0, 1, dataset.shape[0])
                dataset = dataset[['Normalized date'] + dataset.columns.to_list()[:-1]]
                print(dataset.dtypes)
            except ValueError:
                st.markdown("<h6 style='text-align: center;'>"
                            "Error. Check the settings, e.g. intervals. Or maybe the ticker is incorrect</h6>",
                            unsafe_allow_html=True)
                st.stop()
        else:
            st.stop()

    return _placeholder, dataset


def regression(df: pd.DataFrame, column_names: List):
    with st.sidebar:
        st.title('Regression settings')
        regression_type = st.sidebar.selectbox(r'Regression model', regression_types)
        regularization_type = st.selectbox(r'Regulators type', regulators_types)
        regularization_type = {'None': None, 'Lasso (L1)': 'L1', 'Tikhonov (L2)': 'L2'}[regularization_type]

        with st.form('solve!'):
            if regression_type == 'Polynomial':
                degree = int(st.number_input('degree of polynomial regression', value=1, min_value=1,
                                             max_value=10, step=1))
            const_l1, const_l2 = 0, 0
            if regularization_type == 'L1':
                const_l1 = float(st.number_input('l1 constant', value=1e-1, min_value=0.,
                                                 max_value=1., step=1e-2, format='%.2f'))

            elif regularization_type == 'L2':
                const_l2 = float(st.number_input('l2 constant', value=1e-1, min_value=0.,
                                                 max_value=1., step=1e-2, format='%.2f'))

            y_name = st.selectbox(r'Y name', column_names[::-1])
            x_names = st.multiselect(r'X names', column_names, default=column_names[:-1])
            submit_button = st.form_submit_button(label='Solve!')

    if submit_button:
        x = df[x_names].values
        y = df[y_name].values
        if regression_type == 'Linear':
            w = linear_regression(x, y, reg_type=regularization_type, const_l1=const_l1, const_l2=const_l2,
                                  flag_constant=True, max_iter=1500)

    else:
        st.stop()

    st.write(f'{w}')


type_data, reset_button = main_page()

# Reset button. Move to initial state when data don't loaded
if reset_button:
    st.session_state.has_been_uploaded = False
    st.session_state.df = pd.DataFrame([])

# Reset to the initial state too. If there are any errors in the data
if st.session_state.df.shape[0] == 0:
    st.session_state.has_been_uploaded = False

# Branch with ticker. We download data if has_been_uploaded is False
if type_data == 'ticker' and not st.session_state.has_been_uploaded:
    placeholder, exchange_df = ticker_way()
    st.session_state.placeholder = placeholder
    st.session_state.has_been_uploaded = True
    st.session_state.df = exchange_df

# Branch with own data. We request data from user if flag has_been_uploaded is False
# ...

# If the data has been uploaded, we will work with it and make a regression
if st.session_state.has_been_uploaded and st.session_state.df.shape[0] != 0:
    st.session_state.placeholder.empty()
    st.dataframe(st.session_state.df)
    regression(st.session_state.df, st.session_state.df.columns.to_list())