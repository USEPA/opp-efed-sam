import numpy as np
import pandas as pd

a = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [True, False, False, True, False, False], 'c':np.arange(10,16)})
selected = a[a.b][['a', 'c']]
print(selected)