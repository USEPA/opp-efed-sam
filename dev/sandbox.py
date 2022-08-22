import numpy as np
import pandas as pd

a = np.array([1, 2, 4])
b = pd.DataFrame({'a': [np.nan, 15, np.nan, 24, 23, 23]})


print(b)
b['chemical_applied'] = b['a'].notna()
b.loc[a, 'confine'] = True
b['active'] = b.chemical_applied & b.confine
print(b)