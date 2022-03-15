from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from OneDimensionalOptimization.algorithms.support import HistorySPI
from typing import Callable, Tuple, Any
from numbers import Real
from decimal import Decimal


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
    a, b, c = make_parabolic_function(history.loc[0, 'x0'],
                                      history.loc[0, 'x1'],
                                      history.loc[0, 'x2'],
                                      func, **kwargs)

    x0, x1, x2 = history.loc[0, ['x0', 'x1', 'x2']].values

    data = [go.Scatter(x=x_axis, y=float(a) * x_axis ** 2 + float(b) * x_axis + float(c),
                       name='parabola 1', marker={'color': 'rgba(55, 101, 164, 1)'}),
            go.Scatter(x=[float(x0)], y=[func(float(x0), **kwargs)],
                       name='x0', mode='markers', marker={'size': 10, 'color': 'rgba(66, 122, 161, 1)'}),
            go.Scatter(x=[float(x1)], y=[func(float(x1), **kwargs)],
                       name='x1', mode='markers', marker={'size': 10, 'color': 'rgba(66, 122, 161, 1)'}),
            go.Scatter(x=[float(x2)], y=[func(float(x2), **kwargs)],
                       name='x2', mode='markers', marker={'size': 10, 'color': 'rgba(231, 29, 54, 1)'}),
            go.Scatter(x=x_axis, y=f_axis, name='function', marker={'color': 'rgba(0, 0, 0, 0.8)'})]

    layout = go.Layout({'font': {'size': 14},
                        # 'legend': {'itemsizing': 'constant', 'title': {'text': 'type'}, 'tracegroupgap': 0},
                        'sliders': [{'active': 0,
                                     'currentvalue': {'prefix': 'iteration='},
                                     'len': 0.9,
                                     'pad': {'b': 10, 't': 60},
                                     'steps': [{'args': [[f'{i}'], {'frame': {'duration': 500, 'redraw': False,
                                                                              'mode': 'immediate'},
                                                                    'mode': 'immediate',
                                                                    'fromcurrent': False,
                                                                    'transition': {'duration': 200,
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
                                                                          {'duration': 100, 'easing': 'cubic-in-out'}}],
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
                                         'showactive': True,
                                         'type': 'buttons',
                                         'x': 0.1,
                                         'xanchor': 'right',
                                         'y': 0,
                                         'yanchor': 'top'}],
                        'xaxis': {'anchor': 'y',
                                  'domain': [0.0, 1.0],
                                  'range': x_range,
                                  'autorange': False,
                                  'title': {'text': '<b>x</b>, c.u.'}},
                        'yaxis': {'anchor': 'x',
                                  'domain': [0.0, 1.0],
                                  'range': f_range,
                                  'autorange': False,
                                  'title': {'text': '<b>f(x)</b>, c.u.'}}
                        })

    frames = []
    for i in range(1, history.shape[0]):

        x0, x1, x2 = history.loc[i, ['x0', 'x1', 'x2']].values
        a, b, c = make_parabolic_function(x0, x1, x2, func, **kwargs)

        parabola = float(a) * x_axis ** 2 + float(b) * x_axis + float(c)
        frames.append(go.Frame({
               'data': [go.Scatter(x=x_axis, y=parabola, name=f'parabola {i + 1}',
                                   marker={'color': 'rgba(55, 101, 164, 1)'}),
                        go.Scatter(x=[float(x0)], y=[func(float(x0), **kwargs)],
                                   name='x0', mode='markers', marker={'size': 10, 'color': 'rgba(66, 122, 161, 1)'}),
                        go.Scatter(x=[float(x1)], y=[func(float(x1), **kwargs)],
                                   name='x1', mode='markers', marker={'size': 10, 'color': 'rgba(66, 122, 161, 1)'}),
                        go.Scatter(x=[float(x2)], y=[func(float(x2), **kwargs)],
                                   name='x2', mode='markers', marker={'size': 10, 'color': 'rgba(231, 29, 54, 1)'})],

               'name': f'{i}'}))

    fig = go.Figure(data=data, layout=layout, frames=frames)
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=f_range)

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
    x0, x1, x2 = Decimal(x0), Decimal(x1), Decimal(x2)
    f0 = Decimal(func(float(x0), **kwargs))
    f1 = Decimal(func(float(x1), **kwargs))
    f2 = Decimal(func(float(x2), **kwargs))
    a = (f0 * x1 - f0 * x2 - f1 * x0 + f1 * x2 + f2 * x0 - f2 * x1) / (
                x0 ** 2 * x1 - x0 ** 2 * x2 - x0 * x1 ** 2 + x0 * x2 ** 2 + x1 ** 2 * x2 - x1 * x2 ** 2)
    b = (-f0 * x1 ** 2 + f0 * x2 ** 2 + f1 * x0 ** 2 - f1 * x2 ** 2 - f2 * x0 ** 2 + f2 * x1 ** 2) / (
                x0 ** 2 * x1 - x0 ** 2 * x2 - x0 * x1 ** 2 + x0 * x2 ** 2 + x1 ** 2 * x2 - x1 * x2 ** 2)
    c = (f0 * x1 ** 2 * x2 - f0 * x1 * x2 ** 2 - f1 * x0 ** 2 * x2 + f1 * x0 * x2 ** 2 + f2 * x0 ** 2 * x1 - f2 * x0 *
         x1 ** 2) / (x0 ** 2 * x1 - x0 ** 2 * x2 - x0 * x1 ** 2 + x0 * x2 ** 2 + x1 ** 2 * x2 - x1 * x2 ** 2)

    return a, b, c
