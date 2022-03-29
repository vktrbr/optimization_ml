from MultiDimensionalOptimization.algorithms.support import HistoryMDO
import pandas as pd
import numpy as np


def make_data_for_contour(history: HistoryMDO) -> pd.DataFrame:
    """
    Return converted HistoryMDO object into pd.DataFrame with columsn ['x', 'y', 'z', 'iteration']::

        >>> from MultiDimensionalOptimization.algorithms.gd_optimal_step import gradient_descent_optimal_step
        >>> point, hist = gradient_descent_optimal_step(x[0] ** 2 + x[1] ** 2 * 1.01, [10, 10], keep_history=True)
        >>> make_data_for_contour(hist).round(4)
        +---+----------+----------+----------+-------------+
        |   | x        | y        | z        | iteration   |
        +===+==========+==========+==========+=============+
        | 0 | 10.0000  | 10.0000  | 201.000  | 0           |
        | 1 | 0.0502   | -0.0493  | 0.005    | 1           |
        | 2 | 0.0002   | 0.0002   | 0.000    | 2           |
        | 3 | 0.0000   | -0.0000  | 0.000    | 3           |
        +---+----------+----------+----------+-------------+

    :param history: History after some gradient method
    :return: pd.DataFrame

    """
    x, y = np.array(history['x']).T
    output_data = pd.DataFrame({'x': x, 'y': y, 'z': history['f_value'], 'iteration': history['iteration']})
    return output_data
