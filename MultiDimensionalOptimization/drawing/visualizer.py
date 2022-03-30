import plotly.express as px

from MultiDimensionalOptimization.drawing.data_converter import *


def simple_gradient(function: Callable[[np.ndarray], Real],
                    history: HistoryMDO,
                    cnt_dots: Integral = 100) -> go.Figure:
    """
    Return go.Figure with

    :param function: callable that depends on the first positional argument
    :param history: History after some gradient method
    :param cnt_dots: the numbers of point per each axis
    :return: go.Figure with contour and line of gradient steps

    """
    descent_history = make_descent_history(history)
    bounds = make_ranges(history)

    layout = go.Layout(title='<b>Contour plot with optimization steps</b>',
                       xaxis={'title': r'<b>x</b>'},
                       yaxis={'title': r'<b>y</b>'})

    contour = make_contour(function=function, bounds=bounds, cnt_dots=cnt_dots)
    descending_way = go.Scatter(x=descent_history.x,
                                y=descent_history.y,
                                name='descent',
                                mode='lines+markers',
                                line={'width': 3, 'color': 'rgb(202, 40, 22)'},
                                marker={'size': 10, 'color': 'rgb(202, 40, 22)'})

    fig = go.Figure(data=[contour, descending_way], layout=layout)

    return fig


def make_descent_frames_3d(function: Callable[[np.ndarray], Real],
                           history: HistoryMDO) -> List[go.Frame]:
    """
    Make sequence of go.Frame which contain frame for each step of descent with a previous history

    :param function: callable that depends on the first positional argument
    :param history: History after some gradient method
    :return: List[go.Frame]
    """
    frames = []
    descent_history = make_descent_history(history)

    draw_descent = [[], [], []]

    for i in range(descent_history.shape[0]):

        if i > 0:
            x0, x1 = descent_history.x[i - 1], descent_history.x[i]
            y0, y1 = descent_history.y[i - 1], descent_history.y[i]

            for alpha in np.linspace(0, 1, 10):
                draw_descent[0].append(x0 * alpha + x1 * (1 - alpha))
                draw_descent[1].append(y0 * alpha + y1 * (1 - alpha))
                draw_descent[2].append(function([draw_descent[0][-1], draw_descent[1][-1]]))
            else:
                draw_descent[0].append(np.nan)
                draw_descent[1].append(np.nan)
                draw_descent[2].append(np.nan)

        scatter_line = go.Scatter3d(x=draw_descent[0],
                                    y=draw_descent[1],
                                    z=draw_descent[2],
                                    name='descent',
                                    mode='lines',
                                    line={'width': 4, 'color': 'rgb(1, 23, 47)'})

        scatter_points = go.Scatter3d(x=descent_history.x[:max(1, i)],
                                      y=descent_history.y[:max(1, i)],
                                      z=descent_history.z[:max(1, i)],
                                      name='descent',
                                      mode='markers',
                                      marker={'size': 5, 'color': 'rgb(1, 23, 47)'},
                                      showlegend=False)

        frames.append(go.Frame(data=[scatter_points, scatter_line], name=i, traces=[1, 2]))

    return frames


def animated_surface(function: Callable[[np.ndarray], Real],
                     history: HistoryMDO,
                     cnt_dots: Integral = 100) -> go.Figure:
    """
    Return go.Figure with animation per each step of descent

    :param function: callable that depends on the first positional argument
    :param history: History after some gradient method
    :param cnt_dots: the numbers of point per each axis
    :return: go.Figure with animation steps on surface
    """
    descent_history = make_descent_history(history)
    bounds = make_ranges(history)

    first_point = go.Scatter3d(x=descent_history.x[:1],
                               y=descent_history.y[:1],
                               z=descent_history.z[:1],
                               mode='markers',
                               marker={'size': 5, 'color': 'rgb(1, 23, 47)'},
                               showlegend=False)

    surface = make_surface(function, bounds, cnt_dots=cnt_dots)
    layout = px.scatter_3d(descent_history, x='x', y='y', z='z', animation_frame='iteration').layout
    frames = make_descent_frames_3d(function, history)

    fig = go.Figure(data=[surface, first_point, first_point],
                    layout=layout, frames=frames)
    fig.update_scenes(
        xaxis_title=r'<b>x</b>',
        yaxis_title=r'<b>y</b>',
        zaxis_title=r'<b>z</b>'
    )
    fig.update_layout({'title': r'<b>Surface with optimization steps</b>'})
    return fig
