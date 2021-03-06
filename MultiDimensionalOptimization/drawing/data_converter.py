from numbers import Real, Integral
from typing import Callable, Tuple, List, Text

from MultiDimensionalOptimization.algorithms.support import HistoryMDO
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def make_descent_history(history: HistoryMDO) -> pd.DataFrame:
    """
    Return converted HistoryMDO object into pd.DataFrame with columsn ['x', 'y', 'z', 'iteration']::

        >>> from MultiDimensionalOptimization.algorithms.gd_optimal_step import gradient_descent_optimal_step
        >>> point, hist = gradient_descent_optimal_step(x[0] ** 2 + x[1] ** 2 * 1.01, [10, 10], keep_history=True)
        >>> make_descent_history(hist).round(4)
        +---+----------+----------+----------+-------------+
        |   | x        | y        | z        | iteration   |
        +===+==========+==========+==========+=============+
        | 0 | 10.0000  | 10.0000  | 201.000  | 0           |
        | 1 | 0.0502   | -0.0493  | 0.005    | 1           |
        | 2 | 0.0002   | 0.0002   | 0.000    | 2           |
        | 3 | 0.0000   | -0.0000  | 0.000    | 3           |
        +---+----------+----------+----------+-------------+

    :param history: History after some gradient method
    :return: pd.DataFrame

    """
    x, y = np.array(history['x']).T
    output_data = pd.DataFrame({'x': x, 'y': y, 'z': history['f_value'], 'iteration': history['iteration']})
    return output_data


def make_contour(function: Callable[[np.ndarray], Real],
                 bounds: Tuple[Tuple[Real, Real], Tuple[Real, Real]],
                 cnt_dots: Integral = 100,
                 colorscale='ice',
                 showlegend: bool = True) -> go.Contour:
    """
    Return go.Contour for draw by go.Figure. Evaluate function per each point in the 2d grid

    :param function: callable that depends on the first positional argument
    :param bounds: two tuples with constraints for x- and y-axis
    :param cnt_dots: number of point per each axis
    :param colorscale: plotly colorscale for go.Contour
    :param showlegend: showlegend flag
    :return: go.Contour
    """

    assert len(bounds) == 2, 'two tuples are required'
    assert len(bounds[0]) == 2 and len(bounds[1]) == 2, 'both tuples must have 2 numbers'
    x_axis = np.linspace(bounds[0][0], bounds[0][1], cnt_dots)
    y_axis = np.linspace(bounds[1][0], bounds[1][1], cnt_dots)
    z_axis = []
    for x in x_axis:
        z_axis_i = []
        for y in y_axis:
            z_axis_i.append(function([x, y]))
        z_axis.append(z_axis_i)

    return go.Contour(x=x_axis, y=y_axis, z=np.transpose(z_axis), colorscale=colorscale,
                      name='f(x, y)', showscale=showlegend, showlegend=showlegend)


def make_surface(function: Callable[[np.ndarray], Real],
                 bounds: Tuple[Tuple[Real, Real], Tuple[Real, Real]],
                 cnt_dots: Integral = 100,
                 colorscale: Text = 'ice',
                 showlegend: bool = True) -> go.Contour:
    """
    Return go.Surface for draw by go.Figure. Evaluate function per each point in the 2d grid

    :param function: callable that depends on the first positional argument
    :param bounds: two tuples with constraints for x- and y-axis
    :param cnt_dots: number of point per each axis
    :param colorscale: plotly colorscale for go.Contour
    :param showlegend: showlegend flag
    :return: go.Surface
    """

    assert len(bounds) == 2, 'two tuples are required'
    assert len(bounds[0]) == 2 and len(bounds[1]) == 2, 'both tuples must have 2 numbers'
    x_axis = np.linspace(bounds[0][0], bounds[0][1], cnt_dots)
    y_axis = np.linspace(bounds[1][0], bounds[1][1], cnt_dots)
    z_axis = []
    for x in x_axis:
        z_axis_i = []
        for y in y_axis:
            z_axis_i.append(function([x, y]))
        z_axis.append(z_axis_i)

    return go.Surface(x=x_axis, y=y_axis, z=np.transpose(z_axis), colorscale=colorscale,
                      name='f(x, y)', opacity=0.75, showlegend=showlegend)


def make_ranges(history: HistoryMDO, k: Real = 0.5) -> Tuple[List, List]:
    """
    Return bounds for the x-axis and the y-axis.
    1. Find a min_x and max_x and then x_range = [min_x - (max_x - min_x) * k, max_x + (max_x - min_x) * k]
    2. Similarly for y_axis

    :param history: History after some gradient method
    :param k: [min_x - (max_x - min_x) * k, max_x + (max_x - min_x) * k]
    :return: [x_range, y_range]
    """

    descent_history = make_descent_history(history)
    min_x = descent_history.x.min() - 0.1
    max_x = descent_history.x.max() + 0.1

    min_y = descent_history.y.min() - 0.1
    max_y = descent_history.y.max() + 0.1

    x_range = [min_x - (max_x - min_x) * k, max_x + (max_x - min_x) * k]
    y_range = [min_y - (max_y - min_y) * k, max_y + (max_y - min_y) * k]

    return x_range, y_range
