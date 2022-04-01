from MultiDimensionalOptimization.algorithms.gd_constant_step import gradient_descent_constant_step
from MultiDimensionalOptimization.algorithms.gd_frac_step import gradient_descent_frac_step
from MultiDimensionalOptimization.algorithms.gd_optimal_step import gradient_descent_optimal_step
from MultiDimensionalOptimization.algorithms.nonlinear_cgm import nonlinear_cgm
from typing import Tuple
from MultiDimensionalOptimization.algorithms.support import *


def solve_task_nd_minimize(algorithm: Literal["Gradient Descent Fixed",
                                              "Gradient Descent Descent",
                                              "Gradient Descent Optimal",
                                              "Nonlinear conjugate gradient method"] = "Gradient Descent Fixed",
                           **kwargs) -> Tuple[Point, HistoryMDO]:
    """
    A function that calls one of 4 multidimensional optimization algorithms from the current directory, example with
    Golden-section search algorithm::

        >>> def func(x): return x[0] ** 2
        >>> print(solve_task_nd_minimize('Gradient Descent Fixed', function=func, x0=[1])[0])
        {'point': array([4.67680523e-06]), 'f_value': 2.1872507145833953e-11}

    :param algorithm: name of type optimization algorithm
    :param kwargs: arguments requested by the algorithm
    :return: tuple with point and history.
    """
    if algorithm == 'Gradient Descent Fixed':
        return gradient_descent_constant_step(**kwargs)

    if algorithm == 'Gradient Descent Descent':
        return gradient_descent_frac_step(**kwargs)

    if algorithm == "Gradient Descent Optimal":
        return gradient_descent_optimal_step(**kwargs)

    if algorithm == "Nonlinear conjugate gradient method":
        return nonlinear_cgm(**kwargs)


if __name__ == '__main__':
    def parabola(x): return x[0] ** 2


    print(solve_task_nd_minimize('Gradient Descent Fixed', function=parabola, x0=[1])[0])
