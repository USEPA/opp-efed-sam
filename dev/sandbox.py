import numpy as np

a = np.array([0, 0, 1, 0, 1, 0])
b = np.arange(20)
c = b[np.where(a)]

print(c)