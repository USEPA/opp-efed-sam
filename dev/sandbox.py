import numpy as np
import pandas as pd
mean_runoff = 6
q = np.array([3, 4, 5, 6, 7, 6, 5, 4, 3, 2])
n_dates = q.size
baseflow = q - np.subtract(q, mean_runoff, out=np.zeros(n_dates), where=(q > mean_runoff))
print(baseflow)