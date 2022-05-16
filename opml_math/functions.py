from numbers import Real
from typing import Union
import torch
import numpy as np


def linear_rbf(x: np.ndarray, xi: np.ndarray = 0) -> np.ndarray:
    """ just return || x - xi || """

    return min(1, max(-1, ((x - xi) ** 2).sum(axis=1) ** 0.5))


def gaussian_rbf(x: np.ndarray, xi: np.ndarray = np.array([0]), e: Real = 1) -> np.ndarray:
    """
    Returns phi(||x - xi||) = exp(- e * || x - xi ||^2) .. math::

        {\\displaystyle \\varphi(r)=e^{-(\\varepsilon \\Vert x - x_i \\Vert^2)^{2}}}

        >>> x_1 = np.array([[1, 2, 3], [2, 2, 1]])
        >>> x_2 = np.array([[1.1, 2.1, 2.9], [2, 2, 1]])
        >>> print(gaussian_rbf(x_1, x_2))
        [0.97044553 1.        ]

    :param x: a specific point in the space R^n
    :param xi: a specific point in the space R^n
    :param e: scale parameter
    :return: Real number with value of rbf
    """
    return np.exp(- e * ((x - xi) ** 2).sum(axis=1))


def logistic_func(x: Union[np.ndarray, Real]) -> Real:
    """
    Returns logistic function at point x .. math::

        {\\displaystyle f(x)={\\frac{ \\displaystyle 1}{\\displaystyle 1+e^{x)}}}}

    :param x: a specific point in the space R^n
    :return: real number, logistic function at point x
    """
    return 1 / (1 + np.exp(x))


if __name__ == '__main__':
    x1 = np.array([[1, 2, 3], [2, 2, 1]])
    x2 = np.array([[1.1, 2.1, 2.9], [2, 2, 1]])
    print(gaussian_rbf(x1, x2))
