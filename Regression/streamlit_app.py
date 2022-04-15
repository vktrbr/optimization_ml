import io
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import streamlit as st
from datetime import date
import yfinance as yf
from typing import List, Tuple
from Regression.algorithms import *
from Regression.visualization import *
from math import comb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MAX_NUMBER_FEATURES = 500

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

if 'data_source' not in st.session_state:
    st.session_state.data_source = None

# ------ Main header Settings ------ #
st.title('Regression')
left, right = st.columns([7, 1])
left.markdown('<h4>Linear, Polynomial, Exponential</h4>', unsafe_allow_html=True)
reset_button = right.button('Reset', key=True)
st.write('Hello! Welcome to regression app. You can upload your **own data** or check some **stock index**.')


# ------ Part with source type of data ------ #
def type_source() -> Literal['ticker', 'own data']:
    """
    A function that creates a small form with source options, "ticker" or "own data"

    :return: source option
    """
    # creates a container that will be cleaned when the user selects the source type
    _placeholder = st.empty()

    # fills container
    with _placeholder:
        _type_source_place, _button_place = st.columns([7, 1])  # places the selection form and the button together

        # The line below removes the header of the selectbox
        _type_source_place.markdown('<style> [data-baseweb="select"] {margin-top: -50px;}</style>',
                                    unsafe_allow_html=True)
        _type_source = _type_source_place.selectbox('', ['ticker', 'own data'])

        _button = _button_place.button('ok')

    if _button:
        _placeholder.empty()  # cleans container
        return _type_source
    else:
        st.stop()  # waits for the user


def reset() -> None:
    """
    Resets the page
    """
    st.session_state.has_been_uploaded = False
    st.session_state.df = pd.DataFrame([])
    st.session_state.data_source = None
    st.session_state.regression_done = False
    st.session_state.data_source = type_source()


@st.cache
def convert_df_excel(df: pd.DataFrame):
    """
    Function to save excel file

    :param df:
    :return: excel
    """
    output = io.BytesIO()
    writer = pd.ExcelWriter(output)
    df.to_excel(writer, index=False, sheet_name='Weights')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def ticker_way() -> Tuple[st.delta_generator.DeltaGenerator, pd.DataFrame]:
    # form for stock exchange data upload settings
    st.markdown('<style> [data-baseweb="select"] {margin-top: 0px;}</style>', unsafe_allow_html=True)
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
                dataset = dataset[['Normalized date', 'Open', 'High', 'Low', 'Adj Close', 'Volume', 'Close']]
                dataset = dataset.dropna(axis=0)
            except ValueError:
                st.markdown("<h6 style='text-align: center;'>"
                            "Error. Check the settings, e.g. intervals or maybe the ticker is incorrect</h6>",
                            unsafe_allow_html=True)
                st.stop()
        else:
            st.stop()

    return _placeholder, dataset


def own_data_way():
    _placeholder = st.empty()
    with _placeholder.container():
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            with st.form('own_data'):
                # Try read data. If everything is correct, returns the dataset and closes the form
                try:
                    if uploaded_file.type[-3:] == 'csv':
                        dataset: pd.DataFrame = pd.read_csv(uploaded_file)
                    else:
                        dataset: pd.DataFrame = pd.read_excel(uploaded_file)

                    dataset = dataset.fillna(0)
                    st.dataframe(dataset)

                    if dataset.empty:
                        st.markdown("<h6 style='text-align: center;'>"
                                    "Empty dataset</h6>",
                                    unsafe_allow_html=True)
                        st.stop()

                    st.subheader('Check datatypes:')
                    st.dataframe(pd.DataFrame(dataset.dtypes).astype(str).T)
                    st.write('**We will leave only numeric columns and fill NaN with 0**')
                    dataset = dataset.select_dtypes(include=['number'])

                except ValueError:
                    st.markdown("<h6 style='text-align: center;'>"
                                "Error with dataset. Try check encoding or dataset located askew</h6>",
                                unsafe_allow_html=True)
                    st.stop()

                flag_uploaded_data = st.form_submit_button('Download data')
            if flag_uploaded_data:
                return _placeholder, dataset
            else:
                st.stop()
        else:
            st.stop()


def regression(df: pd.DataFrame, column_names: List):
    _placeholder, x, y, features, _warning = st.container(), [], [], [], None

    with _placeholder:

        df_container = st.empty()
        with df_container:
            buff_container = st.container()
            buff_container.subheader('Downloaded data')
            buff_container.dataframe(df)

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
                                                     step=1e-2, format='%.2f'))

                elif regularization_type == 'L2':
                    const_l2 = float(st.number_input('l2 constant', value=1e-1, min_value=0.,
                                                     step=1e-2, format='%.2f'))

                if st.session_state.data_source == 'ticker':
                    y_name = st.selectbox(r'Y name', ['Close', 'Open', 'High', 'Low'])
                    x_names = st.multiselect(r'X names', ['Normalized date'], default=['Normalized date'])

                else:
                    y_name = st.selectbox(r'Y name', column_names[::-1])
                    x_names = st.multiselect(r'X names', column_names, default=column_names[:-1])

                submit_button = st.form_submit_button(label='Solve!')

        if submit_button:
            df_container.empty()

            # ----- Calculations part ----- #
            x = df[x_names]
            y = df[y_name]
            y_true = y.copy()
            x_true = x.copy()

            message_degree = False

            if regression_type in ['Linear', 'Exponential']:
                degree = 1

            if regression_type == 'Polynomial' and degree > 1:
                # The formula for calculating the number of the polynomial features is N(n, d) = C(n + d, d)
                # We have limited the number of features to MAX_NUMBER_FEATURES

                flag_change_degree = False
                while comb(x.shape[1] + degree, degree) > MAX_NUMBER_FEATURES:
                    degree -= 1
                    flag_change_degree = True

                if flag_change_degree:
                    message_degree = f'The degree has been reduced to {degree}. The maximum number of features is ' + \
                                     f'{MAX_NUMBER_FEATURES}. ' + \
                                     f'With degree = {degree}, num of features = {comb(x.shape[1] + degree, degree)}'

            if regression_type == 'Exponential':
                if np.all(y.values > 0):
                    y_true = y.copy()
                    y = np.log(y)

                else:
                    st.warning('For exponential regression, y must be positive')
                    st.stop()

            poly_transformer = PolynomialFeatures(degree)
            with st.spinner('Wait for the features to be created and the weights to be calculated'):
                x = poly_transformer.fit_transform(x)
                weights: np.ndarray = linear_regression(x, y.values, reg_type=regularization_type,
                                                        const_l1=const_l1, const_l2=const_l2, flag_constant=False)

            feature_names = poly_transformer.get_feature_names_out()
            features = pd.DataFrame({'name': feature_names, 'weight': weights})

            if regression_type == 'Exponential':
                y_pred = np.exp(x @ weights)
            else:
                y_pred = x @ weights

            # ----- Conclusion part ----- #
            # 1. Evaluate metrics
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = mse ** 0.5

            if r2 < 0:
                st.markdown("<h6 style='text-align: center;'>"
                            "Error. Check the settings, e.g. x_names and degree</h6>",
                            unsafe_allow_html=True)
                st.stop()

            # ----- metrics output ----- #
            with st.container():
                st.subheader(regression_type + ' regression done!')
                if regression_type == 'Linear':
                    st.write(r'Model: $Y = X @ W + \varepsilon$')
                elif regularization_type == 'Polynomial':
                    st.write(r'Model: $Y = X_{poly_transformed} @ W + \varepsilon$')
                elif regularization_type == 'Exponential':
                    st.write(r'Model: Y = e^{X @ W} + \varepsilon')

                if message_degree:
                    st.warning(message_degree)

                r2_place, mae_place = st.columns([1, 1])
                mse_place, rmse_place = st.columns([1, 1])
                r2_place.write(rf'$\operatorname{{R}}^2 = {r2 : 0.3f}$')
                mse_place.write(rf'$\operatorname{{MSE}} = {mse : 0.3f}$')
                rmse_place.write(rf'$\sqrt{{\operatorname{{MSE}} }}= {rmse : 0.3f}$')
                mae_place.write(rf'$\operatorname{{MAE}} = {mae : 0.3f}$')

            # ----- plot output ----- #
            mode = 'markers'
            if st.session_state.data_source == 'ticker' and x_true.shape[0] > 30:
                mode = 'lines'
            if x_true.shape[1] <= 2:
                if regression_type != "Exponential":
                    x_true = x_true[x_true.columns[0]]
                    st.plotly_chart(gen_polyplot(x_true, y_true, weights, mode=mode))
                else:
                    x_true = x_true[x_true.columns[0]]
                    st.plotly_chart(gen_exponential_plot(x_true, y_true, weights, mode=mode))

            # ---- weights output ----- #
            st.subheader('Coefficients for features')
            features_show = features.copy()
            features_show['abs weight'] = features.weight.map(abs)
            features_show = features_show.sort_values('abs weight', ascending=False)
            features_show = features_show[['name', 'weight']].head(10)
            _, dataframe_place, _ = st.columns([1, 3, 1])
            dataframe_place.dataframe(features_show.style.format(precision=2, thousands=' '))

            dataframe_place.download_button(label='Download excel with weights', data=convert_df_excel(features),
                                            file_name='regression_weights.xlsx')

        else:
            st.stop()


if st.session_state.data_source is None and not st.session_state.has_been_uploaded or reset_button:
    reset()

# Branch with ticker. We download data if flag has_been_uploaded is False
if st.session_state.data_source == 'ticker' and not st.session_state.has_been_uploaded:
    st.session_state.placeholder, st.session_state.df = ticker_way()
    st.session_state.has_been_uploaded = True

# Branch with own data. We request data from user if flag has_been_uploaded is False
if st.session_state.data_source == 'own data' and not st.session_state.has_been_uploaded or reset_button:
    st.session_state.placeholder, st.session_state.df = own_data_way()
    st.session_state.has_been_uploaded = True

# If the data has been uploaded, we will work with it and make a regression
if st.session_state.has_been_uploaded and st.session_state.df.shape[0] != 0:
    st.session_state.placeholder.empty()
    regression(st.session_state.df, st.session_state.df.columns.to_list())
