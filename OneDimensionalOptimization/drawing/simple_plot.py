from numbers import Real
from typing import Callable, Tuple, Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def gen_lineplot(function: Callable,
                 bounds: Tuple[Real, Real],
                 found_point: Tuple[Sequence[Real], Sequence[Real]]) -> go.Figure:
    """
    Generates a graph of the function between the bounds

        >>> def f(x): return x ** 2
        >>> gen_lineplot(f, (-1, 2), ([0], [0]))


    :param function: callable that depends on the first positional argument
    :param bounds: tuple with left and right points on the x-axis
    :param found_point: points that was found by the method. A tulpe with two list / np.ndarray / tuple

    :return: go.Figure with graph
    """
    x_axis = np.linspace(bounds[0], bounds[1], 500)
    f_axis = np.zeros_like(x_axis)
    for i, xi in enumerate(x_axis):
        f_axis[i] = function(xi)

    fig = px.line({'x': x_axis, 'f(x)': f_axis},
                  x='x',
                  y='f(x)',
                  title='<b>Function plot</b>')
    fig.update_layout(
        xaxis_title=r'<b>x</b>, c.u.',
        yaxis_title=r'<b>f(x)</b>, c.u.',
        font=dict(size=14)
    )
    fig.add_scatter(x=found_point[0], y=found_point[1],
                    name='found <br>point', mode='markers',
                    hovertemplate='x: %{x}<br><extra></extra>'
                                  'f(x): %{y}<br>',
                    marker=dict(size=10))
    return fig
