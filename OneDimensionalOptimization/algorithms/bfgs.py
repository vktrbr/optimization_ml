from typing import Callable, Any, Tuple
from numbers import Real, Integral
import numpy as np
from scipy.optimize import line_search
from OneDimensionalOptimization.algorithms.support import HistoryBFGS, PointNd


def bfgs(function: Callable[[np.ndarray, Any], Real],
         x0: np.ndarray,
         c1: Real = 1e-4,
         c2: Real = 9e-1,
         tolerance: Real = 1e-8,
         max_iter: Integral = 500,
         verbose: bool = False,
         keep_history: bool = False,
         **kwargs) -> Tuple[PointNd, HistoryBFGS]:
    """
    BFGS method.
    Wright and Nocedal, 'Numerical Optimization', 1999, pp. 56-60 - alpha search;
    pp.136-140 BFGS algorithm.
    **Modernize: ** if linesearch of alpha if fallen, alpha_k = min(tolerance * 10, 0.01)

    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param x0: start minimization point
    :param c1: first wolfe's constant
    :param c2: second wolfe's constant
    :param tolerance: criterion of stop os l2 norm(grad f) < tolerance
    :param max_iter: maximum number of iterations
    :param verbose: flag of printing iteration logs
    :param keep_history: flag of return history
    :return: tuple with point and history.
    """

    x_k = np.array(x0).reshape(-1, 1).astype(float)
    h_k = np.eye(len(x_k)) * tolerance ** 0.5  # This need set another value pp. 143-144, it will in future versions.
    grad_f_k = gradient(function, x_k).reshape(-1, 1)
    f_k = function(x_k, **kwargs)

    history = {'iteration': [], 'point': [], 'function': []}
    if keep_history:
        history['iteration'].append(0)
        history['point'].append(x_k.reshape(-1))
        history['function'].append(f_k)

    if verbose:
        print(f'iteration: {0} \t x = {np.round(x_k.reshape(-1), 3)} \t f(x) = {f_k : 0.3f}')

    for k in range(max_iter):

        if sum(grad_f_k ** 2) ** 0.5 < tolerance:
            print('Searching finished. Successfully. code 0')
            return {'point': x_k.reshape(-1), 'f_value': f_k}, history

        p_k = -h_k @ grad_f_k

        alpha_k = line_search(function,
                              lambda x: gradient(function, x, **kwargs).reshape(1, -1),
                              x_k, p_k,
                              c1=c1, c2=c2, maxiter=max_iter * 10)[0]

        if alpha_k is None:
            alpha_k = min(tolerance * 10, 0.01)

        x_k_plus1 = x_k + alpha_k * p_k
        grad_f_k_plus1 = gradient(function, x_k_plus1, **kwargs)
        s_k = x_k_plus1 - x_k
        y_k = grad_f_k_plus1 - grad_f_k

        h_k = calc_h_new(h_k, s_k, y_k)
        grad_f_k = grad_f_k_plus1
        x_k = x_k_plus1
        f_k = function(x_k, **kwargs)

        if verbose:
            print(f'iteration: {k + 1} \t x = {np.round(x_k.reshape(-1), 3)} \t f(x) = {f_k: 0.3f}')

        if keep_history:
            history['iteration'].append(k + 1)
            history['point'].append(x_k.reshape(-1))
            history['function'].append(f_k)

    print('Searching finished. Max iterations have been reached. code 1')
    return {'point': x_k.reshape(-1), 'f_value': f_k}, history


def gradient(function: Callable,
             x0: np.ndarray,
             delta_x=1e-8,
             **kwargs) -> np.ndarray:
    """
    Calculate gradient
    :param function: callable that depends on the first positional argument. Other arguments are passed through kwargs
    :param x0: the point at which we calculate the gradient
    :param delta_x: precision of differentiation
    :return: vector np.ndarray with the gradient at the point

    """
    grad = []
    for i in range(len(x0)):
        delta_x_vec_plus = x0.copy()
        delta_x_vec_minus = x0.copy()
        delta_x_vec_plus[i] += delta_x
        delta_x_vec_minus[i] -= delta_x
        grad_i = (function(delta_x_vec_plus, **kwargs) - function(delta_x_vec_minus, **kwargs)) / (2 * delta_x)
        grad.append(grad_i)

    grad = np.array(grad)
    return grad


def calc_h_new(h: np.ndarray,
               s: np.ndarray,
               y: np.ndarray) -> np.ndarray:
    """
    Calculates a new approximation of the inverse Hessian matrix
    :param h: The previous approximation of the H matrix
    :param s: the difference x_{k+1} - x_{k}
    :param y: the difference f'_{k+1} - f'_{k}
    :return: The new approximation of inverse Hessian matrix
    """

    ro = 1 / (y.T @ s)
    i = np.eye(h.shape[0])

    h_new = (i - ro * s @ y.T) @ h @ (i - ro * s @ y.T) + ro * s @ s.T

    return h_new


if __name__ == '__main__':
    def func1(x):
        return (- x[0] / (x[0] ** 2 + 2))[0]


    def func2(x):
        return ((x[0] + 0.004) ** 5 - 2 * (x[0] + 0.004) ** 4)[0]


    def phi(alpha):
        if alpha <= 1 - 0.01:
            return 1 - alpha
        elif 1 - 0.01 <= alpha <= 1 + 0.01:
            return 1 / (2 * 0.01) * (alpha - 1) ** 2 + 0.01 / 2
        else:
            return alpha - 1


    def func3(x):
        return (phi(x[0]) + 2 * (1 - 0.01) / (39 * np.pi) * np.sin(39 * np.pi / 2 * x[0]))[0]


    funcs = [func1, func2, func3]

    for j in range(3):
        x_solve, hist = bfgs(funcs[j], 10, max_iter=100, keep_history=True)
        print(x_solve['point'], x_solve['f_value'])
        print(hist)
