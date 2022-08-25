import numpy as np
import pandas as pd


def replace_nan(value_dict):
    """ There probably shouldn't be nans at all. Use this to find them """
    for k, v in value_dict.items():
        if isinstance(v, dict):
            value_dict[k] = replace_nan(v)
        elif np.isnan(v):
            value_dict[k] = "nanner"
    return value_dict

test_dict = {'a': np.nan, 'b': {'aa': 1, 'bb': np.nan}, 'c': 2}

td = replace_nan(test_dict)
print(td)