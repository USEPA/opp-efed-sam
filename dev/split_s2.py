"""
A Stage Two Scenario consists of a bunch of different time series data (runoff, erosion, leaching, soil_water, rain)

All of the data are needed to build Stage 3 Scenarios, but only 2 (runoff and erosion) are needed for the SAM modeling

The initial idea was to create a separate s2 array with just runoff and erosion upon model execution. It turns out that
that takes a pretty long time.

New idea - create separate runoff/erosion and leaching/soil_water/rain arrays when creating s2

Since it takes like 10 hours to run s2, this script will manually split the existing s2 array into two

If this proof of concept works, modify the s1_to_s2 process to do this every time
"""
import numpy as np

from tools.efed_lib import MemoryMatrix
#[1016554, 5, 11323] E:/opp-efed-data/sam\Intermediate\StageTwoScenarios/r07_arrays.dat 4018
time_series_shape = [1016554, 5, 11323]
re_shape = [1016554, 2, 11323]
other_shape = [1016554, 3, 11323]
array_path_old = r"E:/opp-efed-data/sam\Intermediate\StageTwoScenarios/r07_arrays.dat"
array_path_re = r"E:/opp-efed-data/sam\Intermediate\StageTwoScenarios/r07_runoff_erosion.dat"
array_path_other = r"E:/opp-efed-data/sam\Intermediate\StageTwoScenarios/r07_other.dat"

n_dates = 4018
old_array = MemoryMatrix(time_series_shape, path=array_path_old, existing=True, name='s2 scenario')
new_re = MemoryMatrix(re_shape, path=array_path_re, dtype=np.float32, name='s2 runoff erosion')
new_other = MemoryMatrix(other_shape, path=array_path_other, dtype=np.float32, name='s2 runoff erosion')

wre = new_re.writer
no = new_other.writer
read = old_array.reader
for i in range(time_series_shape[0]):
    old_record = read[i]
    wre[i] = old_record[[0, 1]]
    no[i] = old_record[2:]
    if not i % 100:
        print(i, time_series_shape[0])
