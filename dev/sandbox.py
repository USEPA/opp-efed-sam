import numpy as np
import pandas as pd

a = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5]})
keepers = [1, 2, 4]
a['keep'] = True
a.loc[np.array(keepers), 'keep'] = False

print(a)