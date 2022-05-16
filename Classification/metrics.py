import plotly.graph_objs as go
import plotly.express as px
from typing import Dict, Union, Literal
from numbers import Real
import numpy as np
from sklearn.metrics import classification_report
import torch


def tpr(y_true: np.array, y_pred: np.array) -> Real:
    """
    Return True Positive Rate. TPR = TP / P = TP / (TP + FN). Alias is Recall

    .. note:: if P == 0, then TPR = 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return:
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    tp = sum((y_true == y_pred) & (y_true == 1))
    p = sum(y_true == 1)
    return tp / max(p, 1)


def fpr(y_true: np.array, y_pred: np.array) -> Real:
    """
    Return False Positive Rate. FPR = FP / N = FP / (FP + TN).

    .. note:: if N == 0, then FPR = 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return:
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    fp = sum((y_true != y_pred) & (y_true == 0))
    n = sum(y_true == 0)
    return fp / max(n, 1)


def precision(y_true: np.array, y_pred: np.array) -> Real:
    """
    Return Positive Predictive Value . PPV = TP / (TP + FN). Alias is precision

    .. note:: if TP + FN == 0, then PPV = 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return:
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    tp = sum((y_true == y_pred) & (y_true == 1))
    fp = sum((y_true != y_pred) & (y_true == 0))
    return tp / max(tp + fp, 1)


def f_score(y_true: np.array, y_pred: np.array, beta: Real = 1) -> Real:
    """
    Return F_score. https://en.wikipedia.org/wiki/F-score

    .. math::
        F_\\beta = (1 + \\beta^2) \\cdot \\frac{\\mathrm{precision} \\cdot \\mathrm{recall}}
        {(\\beta^2 \\cdot \\mathrm{precision}) + \\mathrm{recall}}.

    .. note:: if beta ** 2 * _precision + _recall == 0, then f_score = 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :param beta: is chosen such that recall is considered beta times as important as precision
    :return:
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    _precision = precision(y_true, y_pred)
    _recall = tpr(y_true, y_pred)

    numerator = (1 + beta ** 2) * (_precision * _recall)
    denominator = max(beta ** 2 * _precision + _recall, 1e-12)

    return numerator / denominator


def best_threshold(x: torch.Tensor, y_true: torch.Tensor, model: torch.nn.Module,
                   metric: Literal['f1', 'by_roc'] = 'f1', step_size: Real = 0.01):
    """
    Returns best threshold by metric by linear search

    :param x: training tensor
    :param y_true: target tensor. array with true values of binary classification
    :param model: some model that returns a torch tensor with class 1 probabilities using the call: model(x)
    :param metric: name of the target metric that we need to maximize. by_roc - difference between TPR and FPR
    :param step_size: step size of linear search
    :return:
    """
    metric = {'f1': f_score, 'by_roc': lambda y1, y2: tpr(y1, y2) - fpr(y1, y2)}[metric]
    best_t = 0
    best_metric = 0

    for threshold in np.arange(0, 1 + step_size, step_size):
        y_prob = model(x).detach().flatten().cpu().numpy()
        y_pred = (y_prob >= threshold) * 1
        metric_i = metric(y_true, y_pred)

        if metric_i > best_metric:
            best_metric = metric_i
            best_t = threshold

    return best_t


def roc_curve(y_true: np.array, y_prob: np.array, n_thresholds: Union[int, None] = None) -> Dict:
    """
    Return dict with points at TPR - FPR coordinates

    :param y_true: array with true values of binary classification
    :param y_prob: array of probabilities of confidence of belonging to the 1st class
    :param n_thresholds: if len(y_true) is too large, you can limit the number of threshold values
    :return: dict with values of TPR and FPR
    """
    tpr_array = []
    fpr_array = []
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob)

    thresholds = np.sort(np.unique(y_prob))[::-1]
    if n_thresholds is not None:
        thresholds = thresholds[np.linspace(0, len(thresholds) - 1, n_thresholds, dtype=int)]

    for threshold in thresholds:
        tpr_array.append(tpr(y_true, (y_prob >= threshold) * 1))
        fpr_array.append(fpr(y_true, (y_prob >= threshold) * 1))

    return {'TPR': tpr_array, 'FPR': fpr_array}


def roc_curve_plot(y_true: np.array, y_prob: np.array, fill: bool = False) -> go.Figure:
    """
    Return figure with plotly.Figure ROC curve

    :param y_true: array with true values of binary classification
    :param y_prob: array of probabilities of confidence of belonging to the 1st class
    :param fill: flag for filling the area under the curve
    :return: go.Figure
    """
    if fill:
        fig = px.area(roc_curve(y_true, y_prob, None if len(y_true) < 1000 else 1000), x='FPR', y='TPR',
                      title='<b>ROC curve</b>')
    else:
        fig = px.line(roc_curve(y_true, y_prob, None if len(y_true) < 1000 else 1000), x='FPR', y='TPR',
                      title='<b>ROC curve</b>')

    fig.update_layout(font={'size': 18}, autosize=False, width=700, height=600)
    fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line={'dash': 'dash'}, showlegend=False)
    return fig


def auc_roc(y_true: np.array, y_prob: np.array) -> Real:
    """
    Return area under curve ROC (AUC-ROC metric)

    :param y_true: array with true values of binary classification
    :param y_prob: array of probabilities of confidence of belonging to the 1st class
    :return: Real value of are
    """
    tpr_array, fpr_array = roc_curve(y_true, y_prob).values()
    auc = 0
    for i in range(len(fpr_array) - 1):  # Integrating by Trapezoidal rule
        auc += (tpr_array[i] + tpr_array[i + 1]) * (fpr_array[i + 1] - fpr_array[i]) / 2
    return auc


def make_metrics_tab(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, threshold: Real = 0.5):
    """
    Return classification_report from sklearn
    :param model:
    :param x:
    :param y:
    :param threshold:
    :return:
    """
    y_prob = model.forward(x)
    y_pred = (y_prob > threshold) * 1
    output = classification_report(y, y_pred, output_dict=True, zero_division=0)
    output['threshold'] = threshold
    return output
