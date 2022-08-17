import numpy as np
import pandas as pd

active_scenarios = [1, 3, 4]

test_df = pd.DataFrame(np.array([[False, False], [True, True], [False, False], [False, False], [False, False], [False, False]]), columns=['a', 'b'])

test_df.loc[np.array(active_scenarios), 'b'] *= True

print(test_df)