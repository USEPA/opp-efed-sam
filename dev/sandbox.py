import numpy as np
import pandas as pd


batch = []  # This will hold all the dask calls for each batch
batch_index = []  # This is only used to retain the scenario id for writing sample csv outputs
batch_count = 0  # Num of batches processed - used for identifying position in array

# Iterate scenarios
n_scenarios = 5000
lookup = []
n_selected = selected.shape[0]
for count, (s1_index, scenario_id) in enumerate(selected[['s1_index', 'scenario_id']].values):
    params = np.random.randint(0, 1)
    if params:

        time_series = np.random.randint(0, 10, (4, 365))
        if len(batch) == batch_size or (count + 1) == n_scenarios:
            arrays = np.array(batch)
            # [(vars, dates)*batch_size]
            start_pos = batch_count * self.sim.batch_size
            self.writer[start_pos:start_pos + len(batch)] = arrays
            batch_count += 1
            report(f'Processed {count + 1} of {n_selected} scenarios...', 1)
            # write_sample(self.dates, self.sim, arrays, batch_index, 3)
            batch = []
            batch_index = []