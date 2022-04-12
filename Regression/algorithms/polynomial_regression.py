from Regression.algorithms.linear_model import *
from numbers import Integral
from sklearn.preprocessing import PolynomialFeatures


def polynomial_regression(x: np.ndarray,
                          y: np.ndarray,
                          degree: Integral,
                          reg_type: Literal['l1', 'l2', None] = None,
                          epsilon: Real = 1e-4,
                          const_l1: Real = 1e-1,
                          const_l2: Real = 1e-1,
                          max_iter: Integral = 1000) -> np.ndarray:
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
    :param max_iter: maximum of gradient descent steps
    :return: array of regression coefficients
    """
    assert isinstance(x, np.ndarray), 'x must be numpy ndarray'  # check format of inputs
    assert isinstance(y, np.ndarray), 'y must be numpy ndarray'
    assert len(x.shape) == 2, 'x must be 2-d array'
    x = x.astype(np.longdouble)  # ability to use numbers longer than default float64
    y = y.astype(np.longdouble)

    x = PolynomialFeatures(degree=degree).fit_transform(x)  # Generate a new feature matrix consisting of all
                                                            # polynomial combinations
    return linear_regression(x, y,
                             reg_type=reg_type,
                             epsilon=epsilon,
                             const_l1=const_l1,
                             const_l2=const_l2,
                             flag_constant=False,
                             max_iter=max_iter)


if __name__ == '__main__':
    print('how to tests idk')
    print('plot looks good')
    x_data = np.array([[-1], [0], [1]])
    y_data = np.array([1, 0, 1])
    print('Simple parabola test y = x**2: ', np.round(polynomial_regression(x_data, y_data, 2)))
