import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=777)

x = x / np.abs(x).max()

pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1).to_csv('test_100_2.csv', index=False)
