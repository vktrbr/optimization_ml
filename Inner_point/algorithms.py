from numbers import Integral
from typing import Tuple, Optional, TypedDict, List

import pandas as pd

from MultiDimensionalOptimization.algorithms.gd_frac_step import gradient_descent_frac_step
from MultiDimensionalOptimization.algorithms.support import HistoryMDO
from opml_math.calculations import *

np.seterr(invalid='raise')


class History(TypedDict):
    x: List[np.ndarray]
    f_value: List[np.ndarray]


def bound_constrained_lagrangian_method(function: Callable[[np.ndarray], Real],
                                        x0: np.ndarray,
                                        constraints: Sequence[Callable[[np.ndarray], Real]],
                                        x_bounds: Optional[Sequence[Tuple[Real, Real]]] = None,
                                        epsilon: Real = 1e-4,
                                        delta_x: Real = 1e-4,
                                        max_iter: Integral = 2000) -> np.ndarray:
    """
    Returns solution of minimization by newton_eq_const. Alias of <<Newton’s method under equality constrains>>
    Nocedal, J., &amp; Wright, S. J. (2006). 17.4 PRACTICAL AUGMENTED LAGRANGIAN METHODS.
    In Numerical optimization (pp. 519–521). essay, Springer.

    Example for :math:`f(x, y) = (x + 0.5)^2 + (y - 0.5)^2, \\quad x = 1`

        >>> bound_constrained_lagrangian_method(lambda x: (x[0] + 0.5) ** 2 + (x[1] - 0.5) ** 2, [0.1, 0.1],
        >>>                                     [lambda x: x[0] - 1]))

    :param function:
    :param x0:
    :param constraints: list of equality constraints
    :param x_bounds: bounds on x. e.g. 0 <= x[i] <= 1, then x_bounds[i] = (0, 1)
    :param epsilon:
    :param delta_x:
    :param max_iter:
    :return:
    """
    m = len(constraints)
    if x_bounds is None:
        x_bounds = []
        for i in range(len(x0)):
            x_bounds.append((-np.inf, np.inf))


    def c(x: np.ndarray):
        """Returns vector of constraints at specific x"""
        _c = []
        for j in range(m):
            _c.append(constraints[j](x))
        return np.array(_c)

    def lagrangian_a(x: np.ndarray, lam: np.ndarray, mu: Real) -> Real:
        """
        Returns :math:`\\mathcal{L}_a`
        Nocedal, J., &amp; Wright, S. J. (2006). Numerical optimization (p. 520)
        """
        # assert np.all(lam >= 0), 'lambdas must be non negative'
        # assert mu >= 0, 'mu must ne non negative'

        output = function(x)

        for j in range(m):
            output += -lam[j] * constraints[j](x) + mu / 2 * constraints[j](x) ** 2
        return output

    def p_function(g: np.ndarray, u_l_bounds: Sequence[Tuple[Real, Real]]):
        """
        P(g, l, u) is the projection of the vector g :math:`\\in` IRn onto the rectangular box :math:`[l, u]`
        Nocedal, J., &amp; Wright, S. J. (2006). Numerical optimization (p. 520)
        """

        for j in range(len(g)):
            if g[j] >= u_l_bounds[j][1]:
                g[j] = u_l_bounds[j][1]
            elif g[j] <= u_l_bounds[j][0]:
                g[j] = u_l_bounds[j][0]

        return g

    try:
        function(x0)
        for i in range(m):
            constraints[i](x0)
    except ArithmeticError:
        return 'Point out of domain'

    x_k = x0
    lambdas_k = np.repeat(abs(x0[0]), m)
    eta = epsilon  # Main tolerance for constraints
    omega = eta  # Main tolerance for lagrange function
    mu_k = 10
    omega_k = 1 / mu_k
    eta_k = 1 / mu_k ** 0.1

    history: History = {'x': [], 'f_value': []}

    for i in range(max_iter):
        def local_min_function(x):
            grad_lagrangian = x - gradient(lambda y: lagrangian_a(y, lambdas_k, mu_k), x, delta_x)
            p = p_function(grad_lagrangian, x_bounds)
            return sum((x - p) ** 2)

        point, history_step = gradient_descent_frac_step(local_min_function, x_k, epsilon=omega_k,
                                                         max_iter=max_iter, keep_history=True)
        history['x'].extend(history_step['x'])
        history['f_value'].extend(history_step['f_value'])

        x_k = point['point']
        lmf_k = point['f_value']  # x_k, local_min_function(x_k)

        c_k = c(x_k)
        if sum(c_k ** 2) ** 0.5 <= eta_k:
            # test for convergence
            if sum(c_k ** 2) ** 0.5 <= eta and lmf_k ** 0.5 <= omega:
                break

            # update multipliers, tighten tolerances
            lambdas_k = lambdas_k - mu_k * c_k
            mu_k = mu_k
            eta_k = 1 / mu_k ** 0.1
            omega = 1 / mu_k
        else:
            # increase penalty parameter, tighten tolerances
            lambdas_k = lambdas_k - mu_k * c_k
            mu_k = mu_k * 100
            eta_k = 1 / mu_k ** 0.1
            omega = 1 / mu_k

    return x_k, pd.DataFrame({'log-barrier function': history['f_value'], 'x': history['x']})


def log_barrier_solver(function: Callable[[np.ndarray], Real],
                       x0: np.ndarray,
                       inequality_constraints: Sequence[Callable[[np.ndarray], Real]],
                       epsilon: Real = 1e-5) -> Tuple[np.ndarray, HistoryMDO]:
    """
    Returns optimal point of optimization with inequality constraints by Log Barrier method.

    Example for :math:`f(x) = x^2 + y^2, \\quad 0 \\le x \\le 1, 0 \\le y \\le 1`

            >>> log_barrier_solver(lambda x: (x[0] + 0.5) ** 2 + (x[1] - 0.5) ** 2, [0.9, 0.1],
            >>>                    [lambda x: x[0], lambda x: 1 - x[0], lambda x: x[1], lambda x: 1 - x[1]])

    :param function:
    :param x0:
    :param epsilon:
    :param inequality_constraints:
    :return:
    """
    m = len(inequality_constraints)  # Amount of inequality constraints

    try:
        function(x0)
        for i in range(m):
            inequality_constraints[i](x0)
    except ArithmeticError:
        return 'Point out of domain'

    def log_barrier_function(x: np.ndarray, mu: Real) -> Real:
        """
        Support function. Compute log-barrier function .. math::

            P(x, \\mu) = f(x) - \\mu \\sum_{i\\in\\mathcal{I}}\\ln c_i(x)

        :param x: some specific point x
        :param mu:
        :return:
        """
        output_lb = function(x)
        for j in range(m):
            output_lb -= mu * np.log(inequality_constraints[j](x))
        return output_lb

    tau = 1  # The tau sequence will be geometric
    x_k = x0
    history: History = {'x': [], 'f_value': []}
    while tau > epsilon:
        mu_k = tau ** 0.5
        point, history_step = gradient_descent_frac_step(lambda x: log_barrier_function(x, mu_k), x_k, gamma=mu_k,
                                                         epsilon=tau, keep_history=True, max_iter=100)

        history['f_value'].extend(history_step['f_value'])
        history['x'].extend(history_step['x'])

        x_k = point['point']
        tau *= 0.9

    return x_k, pd.DataFrame({'log-barrier function': history['f_value'], 'x': history['x']})


if __name__ == '__main__':
    print(log_barrier_solver(lambda x: (x[0] + 0.5) ** 2 + (x[1] - 0.5) ** 2, [0.9, 0.1], [lambda x: x[0],
                                                                                           lambda x: 1 - x[0],
                                                                                           lambda x: x[1],
                                                                                           lambda x: 1 - x[1]]))
    print(bound_constrained_lagrangian_method(lambda x: (x[0] + 0.5) ** 2 + (x[1] - 0.5) ** 2,
                                              [0.1, 0.1], [lambda x: x[0] - 1]))
