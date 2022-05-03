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


def log_barrier_solver(function: Callable[[np.ndarray], Real],
                       x0: np.ndarray,
                       inequality_constraints: Sequence[Callable[[np.ndarray], Real]],
                       epsilon: Real = 1e-5,
                       delta_x: Real = 1e-4) -> Tuple[Point, HistoryMDO]:
    """
    Returns optimal point of optimization with inequality constraints by Log Barrier method.


    :param function:
    :param x0:
    :param epsilon:
    :param inequality_constraints:
    :param delta_x:
    :return:
    """
    m = len(inequality_constraints)  # Amount of inequality constraints

    try:
        for i in range(m):
            np.log(inequality_constraints[i](x0))

    except FloatingPointError:
        return 'Точка мимо'

    def log_barrier_function(x: np.ndarray, mu: Real) -> Real:
        """
        Support function. Compute log-barrier function .. math::

            P(x, \\mu) = f(x) - \\mu \\sum_{i\\in\\mathcal{I}}\\ln c_i(x)

        :param x: some specific point x
        :param mu:
        :return:
        """
        output_lb = function(x)
        for i in range(m):
            output_lb -= mu * np.log(inequality_constraints[i](x))
        return output_lb

    tau = 1  # The tau sequence will be geometric
    x_k = x0
    _, history = gradient_descent_frac_step(lambda x: log_barrier_function(x, tau), x_k, gamma=tau,
                                            epsilon=tau, keep_history=True, max_iter=100)
    while tau > epsilon:
        mu_k = tau ** 0.5
        point, history_step = gradient_descent_frac_step(lambda x: log_barrier_function(x, mu_k), x_k, gamma=mu_k,
                                                         epsilon=tau, keep_history=True, max_iter=10)

        history['f_value'].extend(history_step['f_value'])
        history['x'].extend(history_step['x'])
        x_k = point['point']
        tau *= 0.9

    return pd.DataFrame({'log-barrier function': history['f_value'], 'x': history['x']})


print(log_barrier_solver(lambda x: (x[0] + 0.5) ** 2 + (x[1] - 0.5) ** 2, [0.9, 0.1], [lambda x: x[0],
                                                                                       lambda x: 1 - x[0],
                                                                                       lambda x: x[1],
                                                                                       lambda x: 1 - x[1]]))
