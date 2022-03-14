from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from OneDimensionalOptimization.algorithms.support import HistorySPI
from typing import Callable, Tuple, Any
from numbers import Real


def gen_animation_spi(func: Callable[[Real, Any], Real],
                      bounds: Tuple[Real, Real],
                      history: HistorySPI,
                      **kwargs) -> go.Figure:
    """
    Generate animation. Per each iteration we create a go.Frame with parabola plot passing through three points

    :param history: a history object. a dict with lists. keys iteration, f_value, middle_point, left_point, right_point
    :param bounds: tuple with left and right points on the x-axis
    :param func: the functions for which the story was created
    """
    print(history)
    n = len(history['iteration'])
    x_axis = np.linspace(bounds[0], bounds[1], 200)
    f_axis = np.zeros_like(x_axis)
    diff_x = max(x_axis) - min(x_axis)
    for i, x in enumerate(x_axis):
        f_axis[i] = func(x, **kwargs)

    diff_f = max(f_axis) - min(f_axis)
    x_range = [x_axis[0] - diff_x * 0.1, x_axis[-1] + diff_x * 0.1]
    f_range = [min(f_axis) - diff_f * 0.1, max(f_axis) + diff_f * 0.1]

    history = pd.DataFrame(history)

    a, b, c = make_parabolic_function(history.loc[0, 'x_left'],
                                      history.loc[0, 'x_middle'],
                                      history.loc[0, 'x_right'],
                                      func, **kwargs)

    data = [go.Scatter(x=x_axis, y=a * x_axis ** 2 + b * x_axis + c, name='parabola'),
            go.Scatter(x=x_axis, y=f_axis, name='function')]

    layout = go.Layout({'font': {'size': 14},
                        # 'legend': {'itemsizing': 'constant', 'title': {'text': 'type'}, 'tracegroupgap': 0},
                        'sliders': [{'active': 0,
                                     'currentvalue': {'prefix': 'iteration='},
                                     'len': 0.9,
                                     'pad': {'b': 10, 't': 60},
                                     'steps': [{'args': [[f'{i}'], {'frame': {'duration': 300, 'redraw': True},
                                                                    'mode': 'immediate',
                                                                    'fromcurrent': True,
                                                                    'transition': {'duration': 300,
                                                                                   'easing': 'cubic-in-out'}}],
                                                'label': f'{i}',
                                                'method': 'animate'} for i in range(n)],

                                     'x': 0.1,
                                     'xanchor': 'left',
                                     'y': 0,
                                     'yanchor': 'top'}],
                        # 'template': '...',
                        'title': {'text': '<b>Function plot</b>'},
                        'updatemenus': [{'buttons': [{'args': [None, {'frame': {'duration': 1000,
                                                                                'redraw': True},
                                                                      'mode': 'immediate',
                                                                      'fromcurrent': True,
                                                                      'transition':
                                                                          {'duration': 500, 'easing': 'cubic-in-out'}}],
                                                      'label': '&#9654;',
                                                      'method': 'animate'},
                                                     {'args': [[None], {'frame': {'duration': 0,
                                                                                  'redraw': True},
                                                                        'mode': 'immediate',
                                                                        'fromcurrent': True,
                                                                        'transition':
                                                                            {'duration': 0, 'easing': 'cubic-in-out'}}],
                                                      'label': '&#9724;',
                                                      'method': 'animate'}],
                                         'direction': 'left',
                                         'pad': {'r': 10, 't': 70},
                                         'showactive': False,
                                         'type': 'buttons',
                                         'x': 0.1,
                                         'xanchor': 'right',
                                         'y': 0,
                                         'yanchor': 'top'}],
                        'xaxis': {'anchor': 'y',
                                  'domain': [0.0, 1.0],
                                  'range': x_range,
                                  'title': {'text': '<b>x</b>, c.u.'}},
                        'yaxis': {'anchor': 'x',
                                  'domain': [0.0, 1.0],
                                  'range': f_range,
                                  'title': {'text': '<b>f(x)</b>, c.u.'}}
                        })

    frames = []
    for i in range(1, history.shape[0]):

        x0, x1, x2 = history.loc[i, ['x_left', 'x_middle', 'x_right']].values
        a, b, c = make_parabolic_function(x0, x1, x2, func, **kwargs)
        parabola = a * x_axis ** 2 + b * x_axis + c
        frames.append(go.Frame({
               'data': [go.Scatter(x=x_axis, y=parabola, name='parabola'),
                        go.Scatter(x=[x0], y=[func(x0, **kwargs)], name='left', mode='markers'),
                        go.Scatter(x=[x1], y=[func(x1, **kwargs)], name='middle', mode='markers'),
                        go.Scatter(x=[x2], y=[func(x2, **kwargs)], name='right', mode='markers')],

               'name': f'{i}'}))

    fig = go.Figure(data=data, layout=layout, frames=frames)
    return fig


def make_parabolic_function(x0: Tuple[Real, Real],
                            x1: Tuple[Real, Real],
                            x2: Tuple[Real, Real],
                            func: Callable[[Real, Any], Real],
                            **kwargs) -> Tuple[Real, Real, Real]:
    """
    Creates a parabolic function passing through the specified points
    :param x0: first point
    :param x1: second point
    :param x2: third point
    :param func: the functions for which the story was created
    :return: Coefficients of the parabolic function
    """
    f0 = func(x0, **kwargs)
    f1 = func(x1, **kwargs)
    f2 = func(x2, **kwargs)
    a = (f0 * x1 - f0 * x2 - f1 * x0 + f1 * x2 + f2 * x0 - f2 * x1) / (
                x0 ** 2 * x1 - x0 ** 2 * x2 - x0 * x1 ** 2 + x0 * x2 ** 2 + x1 ** 2 * x2 - x1 * x2 ** 2)
    b = (-f0 * x1 ** 2 + f0 * x2 ** 2 + f1 * x0 ** 2 - f1 * x2 ** 2 - f2 * x0 ** 2 + f2 * x1 ** 2) / (
                x0 ** 2 * x1 - x0 ** 2 * x2 - x0 * x1 ** 2 + x0 * x2 ** 2 + x1 ** 2 * x2 - x1 * x2 ** 2)
    c = (f0 * x1 ** 2 * x2 - f0 * x1 * x2 ** 2 - f1 * x0 ** 2 * x2 + f1 * x0 * x2 ** 2 + f2 * x0 ** 2 * x1 - f2 * x0 *
         x1 ** 2) / (x0 ** 2 * x1 - x0 ** 2 * x2 - x0 * x1 ** 2 + x0 * x2 ** 2 + x1 ** 2 * x2 - x1 * x2 ** 2)

    return a, b, c
