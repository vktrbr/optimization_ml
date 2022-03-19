from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from OneDimensionalOptimization.algorithms.support import HistoryGSS
from typing import Callable, Tuple, Any
from numbers import Real


def transfer_history_gss(history: HistoryGSS,
                         func: Callable[[Real, Any], Real]) -> pd.DataFrame:
    """
    Generate data for plotly express with using animation_frame for animate
        >>> from OneDimensionalOptimization.algorithms.golden_section_search import golden_section_search
        >>> _, hist = golden_section_search(lambda x: x ** 2, (-1, 2))
        >>> data_for_plot = transfer_history_gss(hist, lambda x: x ** 2)
        >>> data_for_plot[::50]
            index  iteration    type        x             y  size
        0       0          0  middle  0.50000  2.500000e-01     3
        50     24         24    left -0.00001  9.302363e-11     3

    :param history: a history object. a dict with lists. keys iteration, f_value, middle_point, left_point, right_point
    :param func: the functions for which the story was created

    :return: pd.DataFrame for px.scatter
    """
    n = len(history['middle_point'])

    df_middle = pd.DataFrame({'iteration': history['iteration'],
                              'type': ['middle'] * n,
                              'x': history['middle_point'],
                              'y': history['f_value'],
                              'size': [3] * n})

    df_left = pd.DataFrame({'iteration': history['iteration'],
                            'type': ['left'] * n,
                            'x': history['left_point'],
                            'y': list(map(func, history['left_point'])),
                            'size': [3] * n})

    df_right = pd.DataFrame({'iteration': history['iteration'],
                             'type': ['right'] * n,
                             'x': history['right_point'],
                             'y': list(map(func, history['right_point'])),
                             'size': [3] * n})

    df = pd.concat([df_middle, df_left, df_right]).reset_index()

    return df


def gen_animation_gss(func: Callable,
                      bounds: Tuple[Real, Real],
                      history: HistoryGSS,
                      **kwargs) -> go.Figure:
    """
    Generates an animation of the golden-section search on `func` between the `bounds`

    :param func: callable that depends on the first positional argument
    :param bounds: tuple with left and right points on the x-axis
    :param history: a history object. a dict with lists. keys iteration, f_value, middle_point, left_point, right_point

    :return: go.Figure with graph
    """
    x_axis = np.linspace(bounds[0], bounds[1], 500)
    f_axis = np.zeros_like(x_axis)
    diff_x = max(x_axis) - min(x_axis)

    for i, x in enumerate(x_axis):
        f_axis[i] = func(x, **kwargs)
    diff_f = max(f_axis) - min(f_axis)

    df = transfer_history_gss(history, func)
    fig = px.scatter(df, x='x', y='y', size='size', color='type',
                     animation_frame='iteration',
                     range_x=[min(x_axis) - diff_x * 0.15, max(x_axis) + diff_x * 0.15],
                     range_y=[min(f_axis) - diff_f * 0.15, max(f_axis) + diff_f * 0.15],
                     size_max=10,
                     title='<b>Function plot</b>')

    fig.add_trace(go.Scatter(x=x_axis, y=f_axis, name='function'))

    fig.update_layout(
        xaxis_title=r'<b>x</b>',
        yaxis_title=r'<b>f(x)</b>',
        font=dict(size=14)
    )

    return fig
