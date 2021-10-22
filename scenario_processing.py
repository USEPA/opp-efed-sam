import os
import numpy as np
import pandas as pd

from .tools.efed_lib import DateManager, MemoryMatrix
from .field import plant_growth, initialize_soil, process_erosion
from .hydrology import surface_hydrology
from .transport import pesticide_to_field, field_to_soil, soil_to_water
from .utilities import report

# For QAQC or debugging purposes - write the nth scenario from each batch to file (turn off with None)
sample_row = 17  # None

"""
Scenario indexing:
* recipe_index (int, iloc=False) - Unique integer alias for s1 scenarios, generated in scenarios_and_recipes.py
* scenario_id (str, iloc=False) - Name for each s1 scenario, same as used in PWC
* s1_index (int, iloc=True for s1) - a unique integer alias for s1 scenarios
* s3_index (int, iloc=True for s3) - a unique integer alias for s3 scenarios

"""


class StageOneScenarios(MemoryMatrix):
    """
    This class exists primarily to read a table of Stage 1 Scenarios (tabular, same as used for PWC) into
    a MemoryMatrix class object for faster recall of indexed data without holding in memory

    If 'build scenarios' mode is on and custom intakes are provided, this class will also create a smaller
    subset of scenarios that are used only in the affected reaches. This will limit the scope of Stage 2
    Scenario processing for faster testing of the model.
    """

    # TODO - I used to have this set up to read in chunks, but taken out for now. Worth looking into
    #  whether it's worth putting back in. scenarios_and_recipes.py should be adjusted to turn this off

    def __init__(self, region, sim, recipes=None, overwrite_subset=False):
        self.region_id = region.id
        self.sim = sim
        self.table_root = self.sim.s1_scenarios_table_path
        self.array_root = self.sim.s1_scenarios_path

        # Designate the fields that carry through to higher-level scenarios
        self.s2_fields = self.sim.fields.fetch('s1_to_s2')
        self.s3_fields = [self.sim.crop_group_field] + list(self.sim.fields.fetch('s1_to_s3'))

        # If building scenarios and a subset of reaches is specified, confine to a subset of scenarios
        if self.sim.custom_intakes is not None and not self.sim.random:
            self.array_path, self.table_path = \
                self.confine(recipes, region.local_reaches, self.sim.tag, self.sim.build_scenarios)
        else:
            self.array_path = self.array_root.format(self.region_id)
            self.table_path = self.table_root.format(self.region_id, 1)

        # Create a tabular index of core scenario identifiers
        self.lookup, self.array_fields = self.build_index()

        # Find the cdl aliases for all the crops that will have pesticide applications based on user input
        self.sim.active_crops = self.get_active_crops()

        # No need to allocate memory if just generating random output
        if not self.sim.random:
            MemoryMatrix.__init__(self, [self.lookup.s1_index, self.array_fields], name='s1 scenario',
                                  dtype=np.float32, path=self.array_path, persistent_read=True)
            self.csv_to_mem()

    def build_index(self):
        # Get all the column headings from the input table
        columns = pd.read_csv(self.table_path, nrows=1).columns.values

        # Sort columns into those that go in the lookup table, and those that go in the memory array
        lookup_fields = list(self.sim.fields.fetch('s1_lookup'))
        array_fields = [c for c in columns if c not in lookup_fields]

        # Read the lookup table and add a unique 's1_index'
        lookup = pd.read_csv(self.table_path, usecols=lookup_fields + [self.sim.crop_group_field])
        lookup['s1_index'] = np.arange(lookup.shape[0])

        # Set recipe_index as the index field, it makes processing the reaches faster
        lookup = lookup.set_index("recipe_index")
        return lookup, array_fields

    def confine(self, recipes, reaches, tag, overwrite=False):
        region_table_path = self.table_root.format(self.region_id, 1)
        confined_table_path = self.table_root.format(self.region_id, tag)
        confined_array_path = self.array_root.format(self.region_id, tag)
        if overwrite or not os.path.exists(confined_table_path):
            report(f"Confining Stage One Scenarios...")
            # Loop through the watershed recipes for the region and
            # select only the Stage 1 scenarios that exist in the local reaches
            scenario_indices = set()
            for reach in reaches:
                scenarios = recipes.fetch(reach)
                if scenarios is not None:
                    scenario_indices |= set(scenarios.index.values)
            selected = pd.DataFrame({'recipe_index': sorted(scenario_indices)})

            # Read all the s1 tables and extract the selected scenarios
            selected_rows = []
            full_size = 0
            for chunk in pd.read_csv(region_table_path, chunksize=100000):
                full_size += chunk.shape[0]
                chunk = chunk.merge(selected, on='recipe_index', how='inner')
                if not chunk.empty:
                    selected_rows.append(chunk)
            selected = pd.concat(selected_rows, axis=0)
            selected.to_csv(confined_table_path, index=None)

            report(f'Confined Stage One Scenarios table for "{tag}" from {full_size} to {selected.shape[0]}')
            report(f'Confined table written to {confined_table_path}')
        else:
            report(f'Reading confined Stage One Scenarios from {confined_table_path}')
        return confined_array_path, confined_table_path

    def csv_to_mem(self):
        """
        Iteratively loop through all scenarios in chunks
        :param subset: Only return scenarios with a scenario id in the subset (list)
        :return:
        """
        cursor = 0
        writer = self.writer
        report(f'Reading Stage One Scenarios into memory...')
        for chunk in pd.read_csv(self.table_path, usecols=self.array_fields, chunksize=self.sim.stage_one_chunksize):
            writer[cursor:cursor + chunk.shape[0]] = chunk
            cursor += chunk.shape[0]
        del writer

    def fetch(self, index, fields=None, iloc=True):
        fields = {'s2': self.s2_fields, 's3': self.s3_fields}.get(fields, self.array_fields)
        field_index = [self.array_fields.index(f) for f in fields]
        row = super(StageOneScenarios, self).fetch(index, iloc=iloc)
        return list(row[field_index])

    def get_active_crops(self):
        user_selected_crops = \
            pd.DataFrame({self.sim.crop_group_field: list(self.sim.selected_crops)}, dtype=np.int32)
        selected = self.lookup[[self.sim.crop_group_field, 'cdl_alias']] \
            .reset_index().merge(user_selected_crops, on=self.sim.crop_group_field, how='inner')
        return sorted(selected.cdl_alias.unique())


class StageTwoScenarios(DateManager, MemoryMatrix):
    def __init__(self, region, sim, stage_one, met):
        self.sim = sim
        self.s1 = stage_one
        self.met = met
        self.fields = sim.fields
        self.keyfile_path, self.array_path, self.index_path = self.set_paths(region.id)

        # If build is True, create the Stage 2 Scenarios by running model routines on Stage 1 scenario inputs
        if sim.build_scenarios:
            report(f'Building Stage Two Scenarios at {self.array_path}...')
            self.arrays = self.fields.fetch('s2_arrays')
            DateManager.__init__(self, self.sim.scenario_start_date, self.sim.scenario_end_date)
            MemoryMatrix.__init__(self, [self.s1.lookup.index, self.arrays, self.n_dates],
                                  dtype=np.float32, path=self.array_path, persistent_read=True)

            # Create key
            self.create_keyfile()

            # Run scenarios
            self.build_from_stage_one()

        # If build is False, load the saved Stage 2 Scenario array
        else:
            report(f'Loading Stage Two Scenarios in array {self.array_path}...')
            # The key contains a list of array names, a start date, and the shape of the array
            self.arrays, array_start_date, time_series_shape = self.load_key()
            self.runoff_erosion = [self.arrays.index('runoff'), self.arrays.index('erosion')]
            n_dates = time_series_shape[2]
            array_end_date = array_start_date + n_dates - 1
            DateManager.__init__(self, array_start_date, array_end_date)
            self.start_offset, self.end_offset = \
                self.date_offset(self.sim.start_date, self.sim.end_date, n_dates=n_dates)

            # Initialize MemoryMatrix
            MemoryMatrix.__init__(self, time_series_shape, path=self.array_path, existing=True, name='s2 scenario')

    def build_from_stage_one(self):
        """
        Stage 2 Scenarios (s2) are built by running plant growth, hydrology, and erosion simulations
        on each Stage 1 Scenario (s1), using a combination of global parameters and parameters unique to each
        s1. None of these parameters are specified by the user, so these scenarios can be generated ahead of time.
        Because it's a time-consuming process, Dask is used to parallelize the runs. Dask processes the runs in
        batches, and batch_size is set in params.csv
        """

        # In debug mode, the processing will not use Dask or occur in parallel
        debug_mode = False

        batch = []  # This will hold all the dask calls for each batch
        batch_index = []  # This is only used to retain the scenario id for writing sample csv outputs
        batch_count = 0  # Num of batches processed - used for identifying position in array

        # Initialize a list of the simulation parameters used to process Stage 1 Scenarios
        sim_params = [self.sim.cn_min, self.sim.delta_x, self.sim.bins, self.sim.depth, self.sim.anetd,
                      self.sim.n_increments, self.sim.sfac, self.sim.types, self.fields.fetch('s2_arrays')]

        # Group by weather grid to reduce the overhead from fetching met data
        weather_grid = None
        for _, row in self.s1.lookup.iterrows():
            # Stage 1 scenarios are sorted by weather grid. When it changes, update the time series data
            if row.weather_grid != weather_grid:
                weather_grid = row.weather_grid
                precip, pet, temp, *_ = self.met.fetch_station(weather_grid)
                time_series_data = [precip, pet, temp, self.new_year]

            # Unpack the needed parameters from the Stage 1 scenario
            # The parameters that are fetched here can be found in fields_and_qc.csv by sorting by 's1_to_s2'
            s1_params = self.s1.fetch(row.s1_index, 's2')

            # Combine needed input parameters and add a call to the processing function (stage_one_to_two)
            s2_input = time_series_data + s1_params + sim_params
            if not debug_mode:
                batch.append(self.sim.dask_client.submit(stage_one_to_two, *s2_input))
                batch_index.append(row.scenario_id)
            else:
                results = stage_one_to_two(*s2_input, debug=True)
                runoff, erosion, leaching, soil_water, rain = map(float, results.sum(axis=1))

            # Submit the batch for asynchronous processing
            # TODO - how do the weather and scenario arrays match up?
            n_scenarios = self.s1.lookup.shape[0]
            if len(batch) == self.sim.batch_size or row.s1_index == n_scenarios:
                results = np.float32(self.sim.dask_client.gather(batch))
                batch_count += 1
                start_pos = (batch_count - 1) * self.sim.batch_size
                self.writer[start_pos:start_pos + results.shape[0]] = np.array(results)
                report(f'Processed {row.s1_index + 1} of {n_scenarios} scenarios', 1)
                write_sample(self.dates, self.sim, results, batch_index)
                batch = []
                batch_index = []

    def create_keyfile(self):
        with open(self.keyfile_path, 'w') as f:
            f.write(','.join(self.arrays) + '\n')
            f.write(pd.to_datetime(self.start_date).strftime('%Y-%m-%d') + '\n')
            f.write(','.join(map(str, self.shape)) + '\n')

    def fetch_single(self, index, copy=False, verbose=False, iloc=False, pop=False, return_alias=False):
        result = self.fetch(index, copy, verbose, iloc, pop, return_alias)
        return result[:, self.start_offset:-self.end_offset]

    def fetch_multiple(self, index, row_index=None, copy=False, verbose=False, iloc=False, pop=False,
                       return_alias=False):
        result = self.fetch(index, copy, verbose, iloc, pop, return_alias)
        if row_index is not None:
            return result[:, row_index, self.start_offset:-self.end_offset]
        else:
            return result[:, :, self.start_offset:-self.end_offset]

    def fetch_from_recipe(self, recipe):
        found = recipe.join(self.s1.lookup)
        if found.empty:
            raise ValueError("Mismatch between s2 and s1 scenarios")
        arrays = self.fetch_multiple(found.s1_index)[:, self.runoff_erosion]
        return arrays, found.dropna()

    def load_key(self):
        with open(self.keyfile_path) as f:
            time_series = next(f).strip().split(',')
            start_date = np.datetime64(next(f).strip())
            time_series_shape = [int(val) for val in next(f).strip().split(',')]
        return time_series, start_date, time_series_shape

    def set_paths(self, region):
        root_path = self.sim.s2_scenarios_path.format(region)
        if self.sim.tag is not None:
            root_path += f'_{self.sim.tag}'
        keyfile_path = root_path + '_key.txt'
        array_path = root_path + '_arrays.dat'
        index_path = root_path + '_index.csv'
        return keyfile_path, array_path, index_path


class StageThreeScenarios(DateManager, MemoryMatrix):
    def __init__(self, sim, stage_one, stage_two):
        self.s1 = stage_one
        self.s2 = stage_two
        self.sim = sim
        self.array_path = sim.s3_scenarios_path.format(self.s1.region_id)
        self.lookup = self.select_scenarios(self.sim.selected_crops)
        self.n_scenarios = self.lookup.shape[0]

        # Set dates
        DateManager.__init__(self, stage_two.start_date, stage_two.end_date)

        # Initialize memory matrix
        # arrays - runoff_mass, erosion_mass
        disable_build = (self.sim.retain_s3 and os.path.exists(self.array_path))
        MemoryMatrix.__init__(self, [self.lookup.s3_index, 2, self.n_dates], name='pesticide mass',
                              dtype=np.float32, path=self.array_path, existing=disable_build,
                              persistent_read=True, persistent_write=True)

        if not disable_build:
            report(f'Building Stage 3 scenarios...')
            self.build_from_stage_two()

    def build_from_stage_two(self):
        batch = []  # This will hold all the dask calls for each batch
        batch_index = []  # This is only used to retain the scenario id for writing sample csv outputs
        batch_count = 0  # Num of batches processed - used for identifying position in array

        # Initialize some params now
        sim_params = [self.sim.runoff_effic, self.sim.erosion_effic, self.sim.surface_dx,
                      self.sim.cm_2, self.sim.soil_depth, self.sim.deg_foliar, self.sim.washoff_coeff,
                      self.sim.koc, self.sim.deg_aqueous, self.new_year, self.sim.kd_flag]

        # Iterate scenarios
        for count, (s1_index, s3_index, scenario_id) in \
                enumerate(self.lookup[['s1_index', 's3_index', 'scenario_id']].values):

            # Stage 1 params, specified in fields_and_qc
            # Currently plant_date, emergence_date, maxcover_date, harvest_date, max_canopy, orgC_5, bd_5, season

            # These fields should match the order of the parameters used by stage_two_to_three
            # Currently: [plant_date, emergence_date, maxcover_date, harvest_date, max_canopy, orgC_5, bd_5, season]
            crop_group, *s1_params = self.s1.fetch(s1_index, 's3')

            # Get application information for the active crop
            crop_applications = self.sim.applications[self.sim.applications.crop == crop_group]
            if not crop_applications.empty:
                # Extract stored data
                # runoff, erosion, leaching, soil_water, rain
                s2_time_series = list(self.s2.fetch_single(s1_index, iloc=True))

                scenario = [crop_applications.values] + sim_params + s2_time_series + s1_params

                batch.append(self.sim.dask_client.submit(stage_two_to_three, *scenario))
                batch_index.append(scenario_id)
                if len(batch) == self.sim.batch_size or (count + 1) == self.n_scenarios:
                    arrays = self.sim.dask_client.gather(batch)
                    start_pos = batch_count * self.sim.batch_size
                    self.writer[start_pos:start_pos + len(batch)] = arrays
                    batch_count += 1
                    report(f'Processed {count + 1} of {self.n_scenarios} scenarios...', 1)
                    if sample_row is not None:
                        write_sample(self.dates, self.sim, arrays, batch_index, 3)
                    batch = []
                    batch_index = []

    def fetch_from_recipe(self, recipe, verbose=False):
        found = recipe.join(self.lookup, how='inner')
        if found.empty:
            return None, None
        arrays = super(StageThreeScenarios, self).fetch(found.s3_index, iloc=True, verbose=verbose)
        return arrays, found.dropna()

    def select_scenarios(self, crops):

        # Select only the s1 scenarios that represent crops with pesticide applied
        selection = self.s1.lookup[self.sim.crop_group_field].isin(crops)
        selected = self.s1.lookup[selection][['scenario_id', 's1_index', 'cdl_alias']]
        selected['s3_index'] = np.arange(selected.shape[0])

        # A sub-alias for cdl_alias, enables faster bincount operation (e.g., [1, 5, 21, 143] - > [0, 1, 2, 3])
        selected['contribution_index'] = selected.cdl_alias.map(
            {val: i for i, val in enumerate(self.sim.active_crops)})

        return selected


def write_sample(dates, sim, results, batch_index, s=2):
    if sample_row is not None:
        sample_id = batch_index[sample_row]
        sample_path = os.path.join(sim.scratch_path, f"{sample_id}_s{s}.csv")
        sample_data = pd.DataFrame(results[sample_row].T, dates, sim.fields.fetch(f's{s}_arrays'))
        sample_data.to_csv(sample_path)
        report(f'Wrote a Stage {s} sample to {sample_path}', 2)


def stage_one_to_two(precip, pet, temp, new_year,  # weather params
                     plant_date, emergence_date, maxcover_date, harvest_date,  # crop dates
                     max_root_depth, crop_intercept,  # crop properties
                     slope, slope_length,  # field properties
                     fc_5, wp_5, fc_20, wp_20,  # soil properties
                     cn_cov, cn_fallow, usle_k, usle_ls, usle_c_cov, usle_c_fal, usle_p,  # usle params
                     irrigation_type, ireg, depletion_allowed, leaching_fraction,  # irrigation params
                     cn_min, delta_x, bins, depth, anetd, n_increments, sfac,  # simulation soil params
                     types, array_fields, debug=False):
    # Model the growth of plant between emergence and maturity (defined as full canopy cover)
    plant_factor = plant_growth(precip.size, new_year, plant_date, emergence_date, maxcover_date, harvest_date)

    # Initialize soil properties for depth
    cn, field_capacity, wilting_point, usle_klscp = \
        initialize_soil(plant_factor, cn_cov, cn_fallow, usle_c_cov, usle_c_fal, fc_5, wp_5, fc_20,
                        wp_20, usle_k, usle_ls, usle_p, cn_min, delta_x, bins)

    runoff, rain, effective_rain, soil_water, leaching = \
        surface_hydrology(field_capacity, wilting_point, plant_factor, cn, depth,
                          irrigation_type, depletion_allowed, anetd, max_root_depth, leaching_fraction,
                          crop_intercept, precip, temp, pet, n_increments, delta_x,
                          sfac)

    # Calculate erosion loss
    type_matrix = types[types.index == ireg].values  # ireg parameter
    erosion = process_erosion(slope, runoff, effective_rain, cn, usle_klscp, type_matrix, slope_length)

    # Output array order is specified in fields_and_qc.py
    arrays = []
    # self.fields.fetch('s2_arrays')
    for field in array_fields:
        arrays.append(eval(field))
    return np.float32(arrays)


def stage_two_to_three(application_matrix,
                       runoff_effic, erosion_effic, surface_dx, cm_2, soil_depth, deg_foliar,
                       washoff_coeff, koc, deg_aqueous, new_year, kd_flag,
                       runoff, erosion, leaching, soil_water, rain,
                       plant_date, emergence_date, maxcover_date, harvest_date, covmax, org_carbon,
                       bulk_density, season):
    try:
        # TODO - season?

        # Use Kd instead of Koc if flag is on. Kd = Koc * organic C in the top layer of soil
        # Reference: PRZM5 Manual(Young and Fry, 2016), Section 4.13
        if kd_flag:
            koc *= org_carbon

        # Calculate the application of pesticide to the landscape
        plant_dates = [plant_date, emergence_date, maxcover_date, harvest_date]
        application_mass = pesticide_to_field(application_matrix, new_year, plant_dates, rain)

        # Calculate plant factor (could have this info for s2 scenarios, but if it's quick then it saves space)
        plant_factor = plant_growth(runoff.size, new_year, plant_date, emergence_date, maxcover_date, harvest_date)

        # Calculate the daily mass of applied pesticide that reaches the soil (crop intercept, runoff)
        pesticide_mass_soil = field_to_soil(application_mass, rain, plant_factor, cm_2,
                                            deg_foliar, washoff_coeff, covmax)

        # Determine the loading of pesticide into runoff and eroded sediment
        # Also need to add deg_soil, deg_benthic here - NT 8/28/18
        aquatic_pesticide_mass = \
            soil_to_water(pesticide_mass_soil, runoff, erosion, leaching, bulk_density, soil_water, koc,
                          deg_aqueous, runoff_effic, surface_dx, erosion_effic,
                          soil_depth)
    except Exception as e:
        print(e)
        aquatic_pesticide_mass = np.zeros((2, rain.shape[0]))

    return aquatic_pesticide_mass
