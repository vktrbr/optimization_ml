from numbers import Real
from typing import Callable, Sequence
import numpy as np


def gradient(function: Callable[[np.ndarray], Real], x0: np.ndarray, delta_x: Real = 1e-8) -> np.ndarray:
    """
    Returns the gradient of the function at a specific point x0
    A two-point finite difference formula that approximates the derivative

    .. math::

        \\displaystyle \\frac{\\partial f}{\\partial x} \\approx {\\frac {f(x+h)-f(x-h)}{2h}}

    Gradient

    .. math::

         \\displaystyle \\nabla f = \\left[\\frac{\\partial f}{\\partial x_1} \\enspace
         \\frac{\\partial f}{\\partial x_2}
         \\enspace \\dots \\enspace \\frac{\\partial f}{\\partial x_n}\\right]^\\top

    :param function: function which depends on n variables from x
    :param x0: n - dimensional array
    :param delta_x: precision of two-point formula above (delta_x = h)
    :return: vector of partial derivatives
    """
    grad = []
    if not isinstance(x0, np.ndarray):
        x0 = np.array(x0, dtype=float)

    for i in range(len(x0)):
        delta = np.zeros_like(x0)
        delta[i] += delta_x
        delta_x_vec_plus = x0 + delta
        delta_x_vec_minus = x0 - delta
        grad_i = (function(delta_x_vec_plus) - function(delta_x_vec_minus)) / (2 * delta_x)
        grad.append(grad_i)

    grad = np.array(grad)
    return grad


def jacobian(f_vector: Sequence[Callable[[np.ndarray], Real]], x0: np.ndarray, delta_x: Real = 1e-8) -> np.ndarray:
    """
    Returns the Jacobian matrix of a sequence of m functions from f_vector by n variables from x.

        >>> func_3 = [lambda x: x[0] ** 2 + x[1], lambda x: 2 * x[0] + 5 * x[1], lambda x: x[0] * x[1]]
        >>> print(jacobian(func_3, [-1, 2]).round())
        [[-2.  1.]
         [ 2.  5.]
         [ 2. -1.]]

    :param f_vector: a flat sequence, list or tuple or other containing m functions
    :param x0: an n-dimensional array. The specific point at which we will calculate the Jacobian
    :param delta_x: precision of gradient
    :return: the Jacobian matrix according to the above formula. Matrix n x m
    """
    assert isinstance(f_vector, Sequence), 'f_vector must be sequence'
    jac = []
    for j in range(len(f_vector)):
        jac.append(gradient(f_vector[j], x0, delta_x))

    return np.array(jac)


def hessian(function: Callable[[np.ndarray], Real], x0: np.ndarray, delta_x: Real = 1e-4) -> np.ndarray:
    """
    Returns a hessian of function at point x0

        >>> def paraboloid(x): return x[0] ** 2 + 2 * x[1] ** 2
        >>> print(hessian(paraboloid, [1, 1]).round())
        [[2. 0.]
         [0. 4.]]

    :param function: function which depends on n variables from x
    :param x0: n - dimensional array
    :param delta_x: precision of two-point formula above (delta_x = h)
    :return: the hessian of function

    .. note::
        If we make delta_x :math:`\\leq` 1e-6 hessian returns matrix with large error rate

    """
    delta_x = max(delta_x, 1e-6)  # Check note
    if delta_x > 1e-4 - 1e-8:
        x0 = np.array(x0, dtype=np.longfloat)  # Make longdouble for more precision calculations
    elif not isinstance(x0, np.ndarray):
        x0 = np.array(x0)

    hes = []

    for i in range(len(x0)):
        delta_i = np.zeros_like(x0, dtype=np.longfloat)
        delta_i[i] += delta_x
        print(delta_x, delta_i)

        def partial_i(x: np.ndarray) -> Real:
            return (function(x + delta_i) - function(x - delta_i)) / (2 * delta_x)

        hes.append(gradient(partial_i, x0, delta_x))

    return np.array(hes)


if __name__ == '__main__':
    funcs = [lambda x: x[0] ** 2 + x[1], lambda x: 2 * x[0] + 5 * x[1], lambda x: x[0] * x[1]]
    print(jacobian(funcs, [-1, 2]).round())

    print(hessian(funcs[2], [1, 1]).round())
