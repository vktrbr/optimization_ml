from __future__ import annotations

import numpy as np
from numbers import Real, Integral
from typing import Literal, Sequence
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures


def linear_regression(x: np.ndarray,
                      y: np.ndarray,
                      reg_type: Literal['l1', 'l2', None] = None,
                      epsilon: Real = 1e-4,
                      const_l1: Real = 1e-2,
                      const_l2: Real = 1e-1,
                      flag_constant: bool = True) -> Sequence[Real]:
    """
    Make linear regression with tikhonov or lasso regularization or without regularization::

        >>> from sklearn.datasets import make_regression
        >>> import numpy
        >>> numpy.random.seed(1)
        >>> x_, y_ = make_regression(30, 5)
        >>> numpy.round(linear_regression(x_data, y_data), 2)
        [ 0.   54.95 48.6  92.62 39.   60.43]

    :param x: array of predictors
    :param y: array of variable to predict
    :param reg_type: type of regularization
    :param epsilon: accuracy for optimization methods
    :param const_l1: constant for L1 regularization
    :param const_l2: constant for L2 regularization
    :param flag_constant: flag of the need to add columns with ones to find for a permanent term
    :return: array of regression coefficients
    """
    assert isinstance(x, np.ndarray), 'x must be numpy ndarray'  # check format of inputs
    assert isinstance(y, np.ndarray), 'y must be numpy ndarray'
    assert len(x.shape) == 2, 'x must be 2-d array'

    if flag_constant:
        x = np.hstack([np.ones((x.shape[0], 1)), x])

    def loss_function(w):
        return ((x @ w - y) ** 2).mean()

    if reg_type == 'l1':  # with L1
        def loss_function(w):
            return ((x @ w - y) ** 2).mean() + const_l1 * abs(w).sum()

    if reg_type == 'l2':  # with L2
        def loss_function(w):
            return ((x @ w - y) ** 2).mean() + const_l2 * (w ** 2).sum()

    w0 = (np.random.random(size=x.shape[1]) - 0.5) / 100
    return minimize(loss_function, w0, tol=epsilon)['x']


def exponential_regression(x: np.ndarray,
                           y: np.ndarray,
                           reg_type: Literal['l1', 'l2', None] = None,
                           epsilon: Real = 1e-4,
                           const_l1: Real = 1e-1,
                           const_l2: Real = 1e-1) -> np.ndarray:
    """
    Make exponential regression with a tikhonov or a lasso regularization or without any regularization.
    Model: y = exp(x @ w) + eps ~ N(0, 1)
    Step1: ln(y) = x @ w
    Step2: make linear regression

        >>> from sklearn.preprocessing import PolynomialFeatures

        >>> x_ = np.linspace(-2, 3, 50).reshape(-1, 1)
        >>> y_ = np.exp(2 * x_data).flatten()
        >>> w_ = exponential_regression(x_data, y_data)

        >>> x_ = PolynomialFeatures(1).fit_transform(x_)
        >>> print('Our method:\t', np.round(w_, 2))
        >>> print('Precision:\t', ((np.exp(x_data @ w_opt) - y_data) ** 2).sum())
        Our method:	 [-0.  2.]
        Precision:	 2.4924700315739753e-10

    :param x: array of predictors
    :param y: array of variable to predict
    :param reg_type: type of regularization
    :param epsilon: accuracy for optimization methods
    :param const_l1: constant for L1 regularization
    :param const_l2: constant for L2 regularization
    :return: array of regression coefficients
    """
    assert isinstance(x, np.ndarray), 'x must be numpy ndarray'  # check format of inputs
    assert isinstance(y, np.ndarray), 'y must be numpy ndarray'
    assert np.all(y > 0), 'y elements must positive'
    assert len(x.shape) == 2, 'x must be 2-d array'

    lny = np.log(y.flatten())

    return linear_regression(x, lny, reg_type=reg_type, const_l1=const_l1, const_l2=const_l2, epsilon=epsilon)


def polynomial_regression(x: np.ndarray,
                          y: np.ndarray,
                          degree: Integral,
                          reg_type: Literal['l1', 'l2', None] = None,
                          epsilon: Real = 1e-4,
                          const_l1: Real = 1e-1,
                          const_l2: Real = 1e-1) -> np.ndarray:
    """
    Make polynomial regression with a tikhonov or a lasso regularization or without any regularization.
    Step: 1. Get x, 2. Make polynomial features, 3. Make linear regression::

        >>> x_ = np.array([[-1], [0], [1]])
        >>> y_ = np.array([1, 0, 1])
        >>> np.round(polynomial_regression(x_data, y_data, 2))
        [ 0. -0.  1.]

    :param x: array of predictors
    :param y: array of variable to predict
    :param degree: degree for PolynomialFeatures
    :param reg_type: type of regularization
    :param epsilon: accuracy for optimization methods
    :param const_l1: constant for L1 regularization
    :param const_l2: constant for L2 regularization
    :return: array of regression coefficients
    """
    assert isinstance(x, np.ndarray), 'x must be numpy ndarray'  # check format of inputs
    assert isinstance(y, np.ndarray), 'y must be numpy ndarray'
    assert len(x.shape) == 2, 'x must be 2-d array'
    x = x.astype(np.longdouble)  # ability to use numbers longer than default float64
    y = y.astype(np.longdouble)

    # Generate a new feature matrix consisting of all polynomial combinations
    x = PolynomialFeatures(degree=degree).fit_transform(x)

    return linear_regression(x, y,
                             reg_type=reg_type,
                             epsilon=epsilon,
                             const_l1=const_l1,
                             const_l2=const_l2,
                             flag_constant=False)


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    print('LINEAR PART')
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

    print('\n\n--------------------------------')
    print('POLYNOMIAL PART')
    print('how to tests idk')
    print('plot looks good')
    x_data = np.array([[-1], [0], [1]])
    y_data = np.array([1, 0, 1])
    print('Simple parabola test y = x**2: ', np.round(polynomial_regression(x_data, y_data, 2)))

    print('\n\n--------------------------------')
    print('EXPONENTIAL PART')
    print('Test without regularization')
    print('--------------------------------')
    print('f(x) = e**(2 * x)')
    x_data = np.linspace(-2, 3, 50).reshape(-1, 1)
    y_data = np.exp(2 * x_data).flatten()
    w_opt = exponential_regression(x_data, y_data)

    x_data = PolynomialFeatures(1).fit_transform(x_data)
    print('Our method:\t', np.round(w_opt, 2))
    print('Precision:\t', ((np.exp(x_data @ w_opt) - y_data) ** 2).sum())
