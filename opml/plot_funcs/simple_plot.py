from numbers import Real
from typing import Callable, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def gen_lineplot(func: Callable,
                 bounds: Tuple[Real, Real]) -> go.Figure:
    """
    Generates a graph of the `func` between the `bounds`
    :param func: callable that depends on the first positional argument
    :param bounds: tuple with left and right points on the x-axis
    :return: go.Figure with graph
    """
    x_axis = np.linspace(bounds[0], bounds[1], 500)
    f_axis = np.zeros_like(x_axis)
    for i, x in enumerate(x_axis):
        f_axis[i] = func(x)

    fig = px.line({'x': x_axis, 'f': f_axis}, x='x', y='f')
    fig.update_layout(
        xaxis_title=r'x, c.u.',
        yaxis_title=r'f(x), c.u.'
    )
    return fig
