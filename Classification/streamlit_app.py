import os
import sys

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import torch
from Regression.streamlit_app import own_data_way
from Classification.algorithms import LogisticRegression, LogisticRegressionRBF, SVM
import streamlit as st
from visualizer import make_distribution_plot
from metrics import roc_curve_plot, auc_roc


# ------ Reset button ------ #
def reset() -> None:
    """
    Resets the page
    """
    st.session_state.has_been_uploaded = False
    st.session_state.df = pd.DataFrame([])
    st.session_state.data_source = None
    st.session_state.regression_done = False


def classification(df: pd.DataFrame, column_names: list):
    _placeholder, x, y, features, _warning = st.container(), [], [], [], None

    with _placeholder:
        st.write('\n')
        df_container = st.empty()
        with df_container:
            buff_container = st.container()
            buff_container.subheader('Downloaded data')
            buff_container.dataframe(df)

        with st.sidebar:
            st.title('Classification settings')
            classification_type = st.sidebar.selectbox(r'Classification model', classification_types)

            with st.form('solve!'):
                y_name = st.selectbox(r'Y name', column_names[::-1])
                x_names = st.multiselect(r'X names', column_names, default=column_names[:-1])
                epochs = int(st.number_input('Amount epochs', value=1500, min_value=1, max_value=3000, step=1))
                submit_button = st.form_submit_button(label='Solve!')

        if submit_button:
            df_container.empty()

            # ----- Calculations part ----- #
            x = df[x_names]
            y = df[y_name]
            x_torch = torch.FloatTensor(x.values)
            y_torch = torch.FloatTensor(y.values)

            if len(np.unique(y)) != 2:
                st.warning('Only binary classification. Change dataset')
                st.stop()

            with st.spinner('Wait for the features to be created and the weights to be calculated'):
                if classification_type == 'Soft margin SVM':
                    model = SVM(x_torch.shape[1], st.caption, show_epoch=7)
                    if 0 in y or 0. in y:
                        y_torch = y_torch * 2 - 1

                elif classification_type == 'Logistic':
                    model = LogisticRegression(x_torch.shape[1], 'linear', st.caption, show_epoch=7)
                elif classification_type == 'Logistic with RBF':
                    model = LogisticRegressionRBF(x_torch[:10], 'gaussian', st.caption, show_epoch=7)

            _, middle_col, _ = st.columns((1, 1, 1))
            with middle_col:
                model = model.fit(x_torch, y_torch, epochs)

            # ----- metrics output ----- #
            with st.container():

                if classification_type == 'Soft margin SVM':
                    metrics = model.metrics_tab(x_torch, y_torch, metric='f1')
                    tab = pd.DataFrame({'-1': metrics['-1.0'], '1': metrics['1.0']}).T.round(3)
                    st.write(f'Accuracy: {metrics["accuracy"]: 0.2%}')
                    st.write(tab)
                    if x_torch.shape[1] > 1:
                        with st.spinner('Making plot'):
                            try:
                                fig = make_distribution_plot(x_torch, y_torch, model, 0, k=0, cnt_points=600,
                                                             epsilon=1e-3)
                                st.plotly_chart(fig)
                            except ValueError:
                                st.write('Try to change training or dataset.')
                else:
                    metrics = model.metrics_tab(x_torch, y_torch, metric='by_roc')
                    tab = pd.DataFrame({'0': metrics['0.0'], '1': metrics['1.0']}).T.round(3)

                    y_prob = model(x_torch).detach().cpu().numpy().flatten()

                    col_acc, col_auc, _ = st.columns((1, 1, 1))
                    col_acc.write(f'Accuracy: {metrics["accuracy"]: 0.2%}')
                    col_auc.write(f'AUC-ROC: {auc_roc(y_torch, y_prob): 0.2f}')
                    st.write(tab)
                    st.plotly_chart(roc_curve_plot(y_torch, y_prob, fill=True))
                    if classification_type == 'Logistic' and x_torch.shape[1] > 1:
                        with st.spinner('Making plot'):
                            try:
                                fig = make_distribution_plot(x_torch, y_torch, model, metrics['threshold'], k=0.,
                                                             cnt_points=1000, insert_na=True, epsilon=1e-4)
                                st.plotly_chart(fig)
                            except ValueError:
                                st.write('Try to change training or dataset.')
                    elif classification_type == 'Logistic with RBF' and x_torch.shape[1] > 1:
                        with st.spinner('Making plot'):
                            try:
                                fig = make_distribution_plot(x_torch, y_torch, model, metrics['threshold'], k=0,
                                                             cnt_points=500, insert_na=True, epsilon=7e-4)
                                st.plotly_chart(fig)
                            except ValueError:
                                st.write('Try to change training or dataset.')

        else:
            st.stop()


if __name__ == '__main__':
    st.set_page_config(
        page_title=r"Classification",
        page_icon=":six:")  # make page name

    classification_types = ['Soft margin SVM', 'Logistic', 'Logistic with RBF']

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
    st.title('Classification')
    left, right = st.columns([7, 1])
    left.markdown('<h4>Soft margin SVM, Logistic, Logistic with RBF</h4>', unsafe_allow_html=True)
    reset_button = right.button('ReStart', key=True)
    st.write('Hello! Welcome to regression app. You can upload your **own data**.')

    if st.session_state.data_source is None and not st.session_state.has_been_uploaded or reset_button:
        reset()
        # Branch with own data. We request data from user if flag has_been_uploaded is False

    if not st.session_state.has_been_uploaded or reset_button:
        st.session_state.placeholder, st.session_state.df = own_data_way()
        st.session_state.has_been_uploaded = True

    # If the data has been uploaded, we will work with it and make a regression
    if st.session_state.has_been_uploaded and st.session_state.df.shape[0] != 0:
        st.session_state.placeholder.empty()
        classification(st.session_state.df, st.session_state.df.columns.to_list())
