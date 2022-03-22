from OneDimensionalOptimization.algorithms.golden_section_search import golden_section_search
from OneDimensionalOptimization.algorithms.successive_parabolic_interpolation import successive_parabolic_interpolation
from OneDimensionalOptimization.algorithms.brent import brent
from OneDimensionalOptimization.algorithms.bfgs import bfgs
from typing import Literal, Tuple
from OneDimensionalOptimization.algorithms.support import Point, HistoryGSS
import numpy as np


def solve_task(algorithm: Literal["Golden-section search",
                                  "Successive parabolic interpolation",
                                  "Brent's method",
                                  "BFGS algorithm"] = "Golden-section search",
               **kwargs) -> Tuple[Point, HistoryGSS]:
    """
    A function that calls one of 4 one-dimensional optimization algorithms from the current directory, example with
    Golden-section search algorithm::

        >>> def f(x): return x ** 2
        >>> solve_task('Golden-section search', function=f, bounds=[-1, 1])
        ({'point': -7.538932043742175e-17, 'f_value': 5.6835496360162564e-33},
         {'iteration': [0], 'middle_point': [0], 'f_value': [], 'left_point': [0], 'right_point': [0]})

    :param algorithm: name of type optimization algorithm
    :param kwargs: arguments requested by the algorithm
    :return: tuple with point and history.
    """
    if algorithm == 'Golden-section search':
        return golden_section_search(**kwargs)

    if algorithm == 'Successive parabolic interpolation':
        return successive_parabolic_interpolation(**kwargs)

    if algorithm == "Brent's method":
        return brent(**kwargs)

    if algorithm == 'BFGS algorithm':
        point, history = bfgs(**kwargs)
        x_min = point['point'][0]
        fx = point['f_value'][0]
        history['point'] = list(np.array(history['point']).reshape(-1))

        return {'point': x_min, 'f_value': fx}, history


if __name__ == '__main__':
    def func(t): return t ** 2
    solve_task('Golden-section search', function=func, bounds=[-1, 1])
