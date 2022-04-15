from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures


def gen_polyplot(x: pd.Series, y: pd.Series, w: np.ndarray, mode: Literal['markers', 'lines'] = 'markers') -> go.Figure:
    """
    Return figure with initial distribution and regression. Scatter plot of initial one and line plot of regression.

    :param x: array of predictor
    :param y: array of variable to predict
    :param w: array of regression coefficients
    :param mode: mode of initial distribution
    :return: figure with two-colored plot
    """
    degree = len(w) - 1
    min_max_delta = 0.05 * (np.max(x) - np.min(x))
    x_axis = np.linspace(np.min(x) - min_max_delta, np.max(x) + min_max_delta, 50)
    y_axis = PolynomialFeatures(degree).fit_transform(x_axis.reshape(-1, 1)) @ w

    if mode == 'lines':
        df = pd.concat([x, y], axis=1).sort_values(x.name)
        x = df[x.name]
        y = df[y.name]

    data_scatter = go.Scatter(x=x, y=y, name='initial distr', mode=mode,
                              marker={'color': '#62001C', 'size': max(3, 10 - np.log(len(x)))})

    data_line = go.Scatter(x=x_axis, y=y_axis.reshape(-1), mode='lines', name='regression', line={'color': '#C30037'})

    fig: go.Figure = go.Figure(data=[data_scatter, data_line])
    fig.update_layout(title='<b>Regression plot</b>',
                      xaxis_title=rf'<b>{x.name}</b>',
                      yaxis_title=rf'<b>{y.name}</b>',
                      font=dict(size=14))
    return fig


def gen_exponential_plot(x: pd.Series, y: pd.Series, w: np.ndarray,
                         mode: Literal['markers', 'lines'] = 'markers') -> go.Figure:
    """
    y = w * exp(x @ w)

    :param x: array of predictor
    :param y: array of variable to predict
    :param w: array of regression coefficients
    :param mode: mode of initial distribution
    :return: figure with two-colored plot
    """

    degree = 1
    min_max_delta = 0.05 * (np.max(x) - np.min(x))
    x_axis = np.linspace(np.min(x) - min_max_delta, np.max(x) + min_max_delta, 50)
    y_axis = np.exp(PolynomialFeatures(degree).fit_transform(x_axis.reshape(-1, 1)) @ w)

    if mode == 'lines':
        df = pd.concat([x, y], axis=1).sort_values(x.name)
        x = df[x.name]
        y = df[y.name]

    data_scatter = go.Scatter(x=x, y=y, name='initial distr', mode=mode,
                              marker={'color': '#62001C', 'size': max(3, 10 - np.log(len(x)))})

    data_line = go.Scatter(x=x_axis, y=y_axis.reshape(-1), mode='lines', name='regression', line={'color': '#C30037'})

    fig: go.Figure = go.Figure(data=[data_scatter, data_line])
    fig.update_layout(title='<b>Regression plot</b>',
                      xaxis_title=rf'<b>{x.name}</b>',
                      yaxis_title=rf'<b>{y.name}</b>',
                      font=dict(size=14))
    return fig
