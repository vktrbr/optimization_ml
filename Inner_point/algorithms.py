from numbers import Real
from typing import Callable, Sequence

import numpy as np


def gradient(f: Callable[[np.ndarray], Real], x: np.ndarray, delta_x: Real = 1e-8) -> np.ndarray:
    """
    Returns the gradient of the function at a specific point x
    
    A two-point finite difference formula that approximates the derivative

    .. math::

        \\displaystyle \\frac{\\partial f}{\\partial x} \\approx {\\frac {f(x+h)-f(x-h)}{2h}}

    Gradient

    .. math::

         \\displaystyle \\nabla f = \\left[\\frac{\\partial f}{\\partial x_1} \\enspace \\frac{\\partial f}{\\partial x_2}
         \\enspace \\dots \\enspace \\frac{\\partial f}{\\partial x_n}\\right]^\\top

    :param f: function which depends on n variables from x
    :param x: n - dimensional array
    :param delta_x: precision of two-point formula above (delta_x = h)
    :return:
    """


def jacobian(f_vector: Sequence[Callable[[np.ndarray], Real]], x: np.ndarray, delta_x: Real = 1e-8) -> np.ndarray:
    """
    Returns the Jacobian matrix of a sequence of m functions from f_vector by n variables from x.

    .. math::

    СЮДА ВСТАВИТЬ ЯКОБИНА

    :param f_vector: a flat sequence, list or tuple or other containing m functions
    :param x: an n-dimensional array. The specific point at which we will calculate the Jacobian
    :param delta_x: precision of gradient
    :return: the Jacobian matrix according to the above formula. Matrix n x m
    """
    pass
