import numpy as np

np.seterr('ignore')

from typing import Callable, Sequence
from numbers import Real
from MultiDimensionalOptimization.drawing.visualizer import make_contour, make_ranges, make_descent_history
from plotly.subplots import make_subplots
from algorithms import History, log_barrier_function
import plotly.graph_objects as go


def contour_log_barrier(function: Callable[[np.ndarray], Real], history: History,
                        inequality_constraints: Sequence[Callable[[np.ndarray], Real]]) -> go.Figure:
    """
    Returns go.Figure with two plots. Left is contour with main function and way.

    :param function:
    :param history:
    :param inequality_constraints:
    :return:
    """
    fig = make_subplots(1, 2, subplot_titles=["<b>Log-barrier function's contour. mu = 1</b>",
                                              "<b>Function's contour</b>"])
    fig.add_trace(make_contour(lambda x: log_barrier_function(function, x, 1, inequality_constraints),
                               make_ranges(history, 0.1), cnt_dots=200, showlegend=False), 1, 1)
    fig.add_trace(make_contour(function, make_ranges(history, 0.2)), 1, 2)

    descent_history = make_descent_history(history)
    descending_way = go.Scatter(x=descent_history.x, y=descent_history.y, name='descent',
                                mode='lines', line={'width': 3, 'color': 'rgb(202, 40, 22)'})

    point_indexes = np.unique(np.geomspace(1, len(descent_history), 10, dtype=int) - 1)
    descending_points = go.Scatter(x=descent_history.loc[point_indexes, 'x'], y=descent_history.loc[point_indexes, 'y'],
                                   name='descent', mode='markers', marker={'size': 7, 'color': 'rgb(202, 40, 22)'})

    fig.add_trace(descending_way, row=1, col=1)
    fig.update_layout(title='<b>Contour plot for the primal barrier method</b>',
                      xaxis={'title': r'<b>x</b>'}, yaxis={'title': r'<b>y</b>'}, font=dict(size=14),
                      xaxis2={'title': r'<b>x</b>'}, )

    fig.add_trace(descending_way, row=1, col=2)
    fig.add_trace(descending_points, row=1, col=1)
    fig.add_trace(descending_points, row=1, col=2)

    fig.layout = fig.layout.update(showlegend=False)

    return fig
