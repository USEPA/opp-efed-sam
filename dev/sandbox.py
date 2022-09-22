import pandas as pd
import numpy as np
columns = ['a', 'b', 'c']
types = [np.int32, np.float32, np.int32]
dtypes = dict(zip(columns, types))
print(dtypes)
a = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns, dtype=dtypes)

print(a)