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
    Step: 1. Get x, 2. Make polynomial features, 3. Make linear regression

    :param x:
    :param y:
    :param degree:
    :param reg_type:
    :param epsilon:
    :param const_l1:
    :param const_l2:
    :param max_iter: maximum of gradient descent steps
    :return:
    """
    assert isinstance(x, np.ndarray), 'x must be numpy ndarray'
    assert len(x.shape) == 2, 'x must be 2-d array'

    x = PolynomialFeatures(degree=degree).fit_transform(x)
    return linear_regression(x, y,
                             reg_type=reg_type,
                             epsilon=epsilon,
                             const_l1=const_l1,
                             const_l2=const_l2,
                             flag_constant=False,
                             max_iter=max_iter)


if __name__ == '__main__':
    print('how to tests i dnk')
    print('plot looks good')
