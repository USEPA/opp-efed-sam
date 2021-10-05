import pandas as pd
import numpy as np

a = pd.read_csv("E:\opp-efed-data\sam\Results\summary.csv", index_col=0)

print(a.T)

b = a.T.pivot(index=['cdl_1', 'cdl_13'], columns=['erosion', 'runoff'])

print(b)