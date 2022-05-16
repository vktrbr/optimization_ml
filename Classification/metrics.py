import plotly.graph_objs as go
import plotly.express as px
from typing import Dict, Union
from numbers import Real
import numpy as np


def tpr(y_true: np.array, y_pred: np.array) -> Real:
    """
    Return True Positive Rate. TPR = TP / P = TP / (TP + FN).

    .. note:: if P == 0, then TPR == 0

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

    .. note:: if N == 0, then FPR == 0

    :param y_true: array with true values of binary classification
    :param y_pred: array with prediction values of binary classification
    :return:
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    fp = sum((y_true != y_pred) & (y_true == 0))
    n = sum(y_true == 0)
    return fp / max(n, 1)


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
