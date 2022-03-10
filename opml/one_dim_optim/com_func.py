from golden_section_search import golden_section_search
from typing import Literal, Tuple
from auxiliary_objects import *


def solve_task(algorithm: Literal["Golden-section search",
                                  "Successive parabolic interpolation",
                                  "Brent's method",
                                  "BFGS algorithm"] = "Golden-section search",
               **kwargs) -> Tuple[Point, History]:
    """
    A function that calls one of 4 one-dimensional optimization algorithms from the current directory
    :param algorithm: name of type optimization algorithm
    :param kwargs: arguments requested by the algorithm
    :return: tuple with point and history.
    """
    if algorithm == 'Golden-section search':
        return golden_section_search(**kwargs)
