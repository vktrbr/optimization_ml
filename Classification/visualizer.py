from typing import Optional
import torch
import plotly.express as px
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from metrics import roc_curve_plot


def make_distribution_plot(x: torch.Tensor, y_true: torch.Tensor, model: Optional[torch.nn.Module] = None,
                           threshold: float = 0.5, cnt_points: int = 1000, k: float = 0.2,
                           title: Optional[str] = None, epsilon: float = 1e-4, insert_na: bool = False) -> go.Figure:
    """
    Returns a graph with a distribution and an optional line. If dim(x) = 2, then you can get model. If dim(x) > 2,
    then returns graph of TSNE from sklearn.manifold with default settings. dim(x) is not support

    :param x: training tensor
    :param y_true: target tensor. array with true values of binary classification
    :param model: some model that returns a torch tensor with class 1 probabilities using the call: model(x)
    :param threshold: if model(xi) >= threshold, then yi = 1
    :param cnt_points: number of points on each of the two axes when dim(x) = 2
    :param k: constant for draw on section: [x.min() - (x.max() - x.min()) * k, x.max() + (x.max() - x.min()) * k]
    :param title: title of plots
    :param epsilon: contour line points: :math:`\\{x\\in \\mathbb{R}^2 \\, | \\,
                \\text{threshold} - \\text{epsilon} \\le \\text{model}(x) \\le \\text{threshold} + \\text{epsilon}\\}`
    :param insert_na: na insertion flag when two points too far away
    :return: scatter plot go.Figure
    """
    colors = list(map(lambda e: str(int(e)), y_true))

    if x.shape[1] < 2:
        raise AssertionError('x.shape[1] must be >= 2')

    elif x.shape[1] == 2:
        title = '<b>Initial Distribution</b>' if title is None else title
        fig = px.scatter(x=x[:, 0], y=x[:, 1], title=title, color=colors)

        if model is not None:
            x1 = torch.tensor([x[:, 0].min() - (x[:, 0].max() - x[:, 0].min()) * k,
                               x[:, 1].min() - (x[:, 1].max() - x[:, 1].min()) * k])

            x2 = torch.tensor([x[:, 0].max() + (x[:, 0].max() - x[:, 0].min()) * k,
                               x[:, 1].max() + (x[:, 1].max() - x[:, 1].min()) * k])

            grid = make_line(x1, x2, model, threshold, cnt_points, epsilon, insert_na).detach().cpu()
            line_x, line_y = grid.T
            fig.add_scatter(x=line_x, y=line_y, name='sep plane', mode='lines')

    else:
        title = '<b>TSNE of Distribution</b>' if title is None else title
        tsne_x = TSNE().fit_transform(x)
        fig = px.scatter(x=tsne_x[:, 0], y=tsne_x[:, 1], title=title, color=colors)

    fig.update_layout(font={'size': 18}, autosize=False, width=1200, height=800)

    return fig


def sort_points(line: torch.Tensor, epsilon: float = 1e-3, metric: int = 2, insert_na: bool = True) -> torch.Tensor:
    """
    Returns tensor sorted by closeness between each other. if || lines[i] - closest{lines[j]} ||_metric > epsilon
    insert [nan, nan]

    :param line: tensor n x 2
    :param epsilon: maximum closeness
    :param metric: l1, l2, or some other metric
    :param insert_na: na insertion flag
    :return: sorted tensor line with probably added nan values
    """

    copy_line = [line[0, :]]
    mask = torch.tile(torch.tensor([True]), line.shape[:1])
    mask[0] = False
    for i in range(line.shape[0] - 1):
        distances = torch.norm(line - copy_line[-1], p=metric, dim=1)
        distances[mask == False] = torch.inf

        min_d, argmin_d = distances.min(), distances.argmin()
        if min_d <= epsilon ** 0.3 or insert_na is False:
            copy_line.append(line[[argmin_d]])
        else:
            copy_line.append(torch.tensor([torch.nan, torch.nan]))
            copy_line.append(line[[argmin_d]])

        mask[argmin_d] = False

    line = torch.zeros(len(copy_line), 2)
    for i in range(line.shape[0]):
        line[i, :] = copy_line[i]
    return line


roc_curve_plot = roc_curve_plot


def make_line(x1: torch.Tensor, x2: torch.Tensor, model: torch.nn.Module, threshold: float = 0.5,
              cnt_points: int = 25, epsilon: float = 1e-3, insert_na: bool = True) -> torch.Tensor:
    """
    Returns x in [x1, x2] : threshold - epsilon <= model(x) <= threshold + epsilon

    :param x1: 2-dim tensor start
    :param x2: 2-dim tensor end
    :param model: some model that returns a torch tensor with class 1 probabilities using the call: model(x)
    :param threshold: if model(xi) >= threshold, then yi = 1
    :param cnt_points: number of points on each of the two axes
    :param epsilon: contour line points: :math:`\\{x\\in \\mathbb{R}^2 \\, | \\,
                \\text{threshold} - \\text{epsilon} \\le \\text{model}(x) \\le \\text{threshold} + \\text{epsilon}\\}`
    :param insert_na: na insertion flag
    :return: scatter plot go.Figure
    """
    if torch.isnan(x1[0]) or torch.isnan(x1[1]) or torch.isnan(x2[0]) or torch.isnan(x2[1]):
        return torch.tensor([[torch.nan, torch.nan]])

    lin_settings_1 = (min(x1[0], x2[0]), max(x1[0], x2[0]), cnt_points)
    lin_settings_2 = (min(x1[1], x2[1]), max(x1[1], x2[1]), cnt_points)

    grid = torch.cartesian_prod(torch.linspace(*lin_settings_1), torch.linspace(*lin_settings_2))

    with torch.no_grad():
        grid_pred = model(grid)

    mask = (threshold - epsilon <= grid_pred) & (grid_pred <= threshold + epsilon)
    if sum(mask) > 0:
        if sum(mask) > 1000:
            grid = grid[mask.flatten(), :]
            grid = grid[torch.linspace(0, grid.shape[0], 1000, dtype=torch.int64), :]
        else:
            grid = grid[mask.flatten(), :]
        grid = sort_points(grid, epsilon=epsilon, insert_na=insert_na)
    else:
        grid = torch.tensor([torch.nan, torch.nan])
    return grid
