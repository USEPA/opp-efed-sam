import pandas as pd
import numpy as np
import datetime as dt
import time

a = set(range(14))
b = [np.random.randint(0, 100) for _ in range(3000)]
start = time.time()
for c in b:
    if c in a:
        print("yes")
print(time.time() - start)