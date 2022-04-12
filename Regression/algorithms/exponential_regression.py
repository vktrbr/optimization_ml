from scipy.optimize import minimize

from Regression.algorithms.linear_model import *


def exponential_regression(x: np.ndarray,
                           y: np.ndarray,
                           reg_type: Literal['l1', 'l2', None] = None,
                           epsilon: Real = 1e-4,
                           const_l1: Real = 1e-1,
                           const_l2: Real = 1e-1) -> np.ndarray:
    """
    Make exponential regression with a tikhonov or a lasso regularization or without any regularization.
    Step: 1. Get x, 2. Y = w * exp(X @ W) + w1 minimize SSE loss::

        >>> x_ = np.linspace(-2, 3, 50).reshape(-1, 1)
        >>> y_ = 5 * np.exp(2 * x_)
        >>> w_ = exponential_regression(x_, y_)
        >>> print('Our method:\t', np.round(w_, 2))
        >>> print(((w_[0] * np.exp(x_ @ w_[2:].reshape(-1, 1)) + w_[1] - y_) ** 2).sum())
        Our method:	 [ 5. -0.  2.]
        3.203750731599574e-07

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
    assert len(x.shape) == 2, 'x must be 2-d array'
    x = x.astype(np.longdouble)  # it's a magic maan
    y = y.astype(np.longdouble)  # ability to use numbers longer than default float64

    def loss_function(w):
        reg = 0
        if reg_type == 'l1':
            reg = const_l1 * abs(w).sum()
        elif reg_type == 'l2':
            reg = const_l2 * (w ** 2).sum()

        return (((w[0] * np.e ** (x @ w[2:].reshape(-1, 1)) + w[1]) - y) ** 2).sum() + reg

    w0 = (np.zeros(x.shape[1] + 2) + 0.001).astype(np.longdouble)
    return minimize(loss_function, w0, tol=epsilon, method='L-BFGS-B')['x']


if __name__ == '__main__':
    print('Test without regularization')
    print('--------------------------------')
    x_data = np.linspace(-2, 3, 50).reshape(-1, 1)
    y_data = 5 * np.exp(2 * x_data)
    w_opt = exponential_regression(x_data, y_data)
    print('Our method:\t', np.round(w_opt, 2))
    print(((w_opt[0] * np.exp(x_data @ w_opt[2:].reshape(-1, 1)) + w_opt[1] - y_data) ** 2).sum())
