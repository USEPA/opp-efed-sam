import numpy as np
import pandas as pd

a = np.zeros((4, 4018, 78))
b = pd.Series(np.zeros((78,))).values
print(a.shape)
print(b.shape)
print(a[:2].shape)
a[:2] *= b