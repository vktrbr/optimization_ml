import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures


def gen_polyplot(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> go.Figure:
    """
    Return figure with initial distribution and regression. Scatter plot of initial one and line plot of regression.

    :param x: array of predictor
    :param y: array of variable to predict
    :param w: array of regression coefficients
    :return: figure with two-colored plot
    """
    degree = len(w) - 1
    min_max_delta = 0.05 * (np.max(x) - np.min(x))
    x_axis = np.linspace(np.min(x) - min_max_delta, np.max(x) + min_max_delta, 50)
    y_axis = PolynomialFeatures(degree).fit_transform(x_axis.reshape(-1, 1)) @ w.astype(float)
    fig: go.Figure = px.scatter({'x': x.reshape(-1), 'initial distr': y}, x='x', y=['initial distr'],
                                title='<b>Regression plot</b>')
    fig.add_trace(go.Scatter(x=x_axis, y=y_axis.reshape(-1), mode='lines', name='regression'))
    fig.update_layout(xaxis_title=r'<b>x</b>',
                      yaxis_title=r'<b>y</b>',
                      font=dict(size=14))
    return fig
