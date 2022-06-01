import os
import time


for f in os.listdir(r"A:"):
    x =
    result = (time.time() - os.stat(f).st_mtime) / 86400
    print(f, result)