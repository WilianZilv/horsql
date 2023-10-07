import pandas as pd
import numpy as np

def is_iterable(x):
    return isinstance(x, (pd.Series, np.ndarray, list, tuple))