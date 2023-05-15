import pandas as pd
import numpy as np
s1_names = ["plant_date", "emergence_date", "maxcover_date", "harvest_date", "max_canopy",
                                    "orgC_5", "bd_5", "season"]

s1_values = [[i * 10] for i in np.arange(len(s1_names))]
s1 = pd.DataFrame(dict(zip(s1_names, s1_values))).T