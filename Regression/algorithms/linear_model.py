from __future__ import annotations

import numpy as np
from numbers import Real, Integral
from typing import Literal, Sequence
from MultiDimensionalOptimization.algorithms.gd_optimal_step import gradient_descent_optimal_step


def linear_regression(x: np.ndarray,
                      y: np.ndarray,
                      reg_type: Literal['l1', 'l2', None] = None,
                      epsilon: Real = 1e-4,
                      const_l1: Real = 1e-2,
                      const_l2: Real = 1e-1,
                      flag_constant: bool = True,
                      max_iter: Integral = 1000) -> Sequence[Real]:
    """
    Make linear regression with tikhonov or lasso regularization or without regularization::

        >>> from sklearn.datasets import make_regression
        >>> x_, y_ = make_regression(30, 5, random_state=1)
        >>> np.round(linear_regression(x_data, y_data), 2)
        [-0.   21.17 47.25  3.29 13.33 14.85]

    :param x: array of predictors
    :param y: array of variable to predict
    :param reg_type: type of regularization
    :param epsilon: accuracy for optimization methods
    :param const_l1: constant for L1 regularization
    :param const_l2: constant for L2 regularization
    :param flag_constant: flag of the need to add columns with ones to find for a permanent term
    :param max_iter: maximum of gradient descent steps
    :return: array of regression coefficients
    """
    assert isinstance(x, np.ndarray), 'x must be numpy ndarray'  # check format of inputs
    assert isinstance(y, np.ndarray), 'y must be numpy ndarray'
    assert len(x.shape) == 2, 'x must be 2-d array'

    if flag_constant:
        x = np.hstack([np.ones((x.shape[0], 1)), x])

    if reg_type is None:  # without regularization
        def loss_function(w):
            return ((x @ w - y) ** 2).sum()

        w0 = np.random.random(size=x.shape[1])
        return np.array(gradient_descent_optimal_step(loss_function, w0,
                                                      epsilon=epsilon, max_iter=max_iter)[0]['point'])
    if reg_type == 'l1':  # with L1
        def loss_function(w):
            return ((x @ w - y) ** 2).sum() + const_l1 * abs(w).mean()

        w0 = np.zeros(x.shape[1]) + 0.01
        return np.array(gradient_descent_optimal_step(loss_function, w0,
                                                      epsilon=epsilon, max_iter=max_iter)[0]['point'])

    if reg_type == 'l2':  # with L2
        def loss_function(w):
            return ((x @ w - y) ** 2).sum() + const_l2 * (w ** 2).sum()

        w0 = np.random.random(size=x.shape[1])
        return np.array(gradient_descent_optimal_step(loss_function, w0,
                                                      epsilon=epsilon, max_iter=max_iter)[0]['point'])


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    print('Test without regularization')
    print('--------------------------------')
    x_data, y_data = make_regression(30, 5)
    print('Our method:\t', np.round(linear_regression(x_data, y_data), 2))

    model: LinearRegression = LinearRegression()
    model.fit(x_data, y_data)
    print('Sklearn:\t', np.round([model.intercept_] + list(model.coef_), 2))
    print('--------------------------------')

    print('Test with l1')
    print('--------------------------------')
    x_data, y_data = make_regression(30, 5)
    print('Our method:\t', np.round(linear_regression(x_data, y_data, reg_type='l1', const_l1=.1), 2))

    model: Lasso = Lasso(alpha=.1)
    model.fit(x_data, y_data)
    print('Sklearn:\t', np.round([model.intercept_] + list(model.coef_), 2))
    print('--------------------------------')

    print('Test with l2')
    print('--------------------------------')
    x_data, y_data = make_regression(30, 5)
    print('Our method:\t', np.round(linear_regression(x_data, y_data, reg_type='l2', const_l2=.5), 2))

    model: Ridge = Ridge(alpha=.5)
    model.fit(x_data, y_data)
    print('Sklearn:\t', np.round([model.intercept_] + list(model.coef_), 2))
    print('--------------------------------')
