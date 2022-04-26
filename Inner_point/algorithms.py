from numbers import Real
from typing import Callable, Sequence

import numpy as np


def jacobian(f: Sequence[Callable[[np.ndarray], Real]], x: np.ndarray, delta_x: Real=1e-8) -> np.ndarray:
    """
    Returns the Jacobian matrix of a sequence of functions from f by variables from x.

    .. math::
        {\displaystyle \mathbf {J} ={\begin{bmatrix}{\dfrac {\partial f_{1}}{\partial x_{1}}}
        &\cdots &{\dfrac {\partial f_{1}}{\partial x_{n}}}\\\vdots &\ddots &\vdots \\{\dfrac {\partial f_{m}}
        {\partial x_{1}}}&\cdots &{\dfrac {\partial f_{m}}{\partial x_{n}}}\end{bmatrix}}}

    :param f:
    :param x:
    :param delta_x:
    :return:
    """
    """
    Function calculate jaciian
    
    :param f: 
    :return: 
    """