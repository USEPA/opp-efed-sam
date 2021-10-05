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


class StageOneScenarios(MemoryMatrix):

    def __init__(self, region, sim, recipes=None, overwrite_subset=False):
        self.region_id = region.id
        self.sim = sim
        self.table_path = sim.s1_scenarios_table_path
        self.array_path = sim.s1_scenarios_path.format(self.region_id)

        # Objects that are generated on-the-fly by the @property methods.
        # TODO - is this necessary?
        self._paths = None
        self._index = None
        self._columns = None
        self._active_crops = None

        """
        If the simulation is in s2 scenario building mode, and the s2 scenarios are subset as indicated by a 'tag',
         (e.g., Mark Twain Basin), create a confined s1 scenario table for more limited iteration.
        """

        # The lookup table for s1 scenarios is needed for generating random output, but nothing else
        if not self.sim.random:

            if self.sim.tag is not None:
                self._paths = self.confine(recipes, region.local_reaches, self.sim.tag, overwrite_subset)

            MemoryMatrix.__init__(self, [self.lookup.index, self.columns], name='s1 scenario',
                                  dtype=np.float32, path=self.array_path, persistent_read=True)
            self.csv_to_mem()

    def column_index(self, indices):
        return [self.columns.index(f) for f in indices]

    @property
    def active_crops(self):
        if self._active_crops is None:
            crops = pd.DataFrame({self.sim.crop_group_field: list(self.sim.selected_crops)}, dtype=np.int32)
            selected = self.lookup[[self.sim.crop_group_field, 'cdl_alias']] \
                .reset_index().merge(crops, on=self.sim.crop_group_field, how='inner')
            self._active_crops = sorted(selected.cdl_alias.unique())
        return self._active_crops

    @property
    def n_active_crops(self):
        return len(self._active_crops)

    @property
    def columns(self):
        if self._columns is None:
            self._columns = [c for c in pd.read_csv(self.paths[0], nrows=1).columns if c != 'scenario_id']
        return self._columns

    @property
    def lookup(self):
        if self._index is None:
            index = []
            keep_cols = list(self.sim.fields.fetch('s1_lookup')) + [self.sim.crop_group_field]
            for path in self.paths:
                for chunk in pd.read_csv(path, usecols=keep_cols, chunksize=self.sim.stage_one_chunksize):
                    for _, scenarios in chunk.groupby('weather_grid'):
                        index.append(scenarios)
            self._index = pd.concat(index).set_index('scenario_index')
            self._index['s1_index'] = np.arange(self._index.shape[0])
        return self._index

    def confine(self, recipes, reaches, tag, overwrite=False):
        temp_path = self.sim.s1_scenarios_table_path.format(self.region_id, self.sim.tag)
        if overwrite or not os.path.exists(temp_path):
            # Select only the Stage 1 scenarios that are components in the local reaches
            scenario_indices = set()
            for reach in reaches:
                scenarios = recipes.fetch(reach)
                if scenarios is not None:
                    scenario_indices |= set(scenarios.index.values)
            confiner = pd.DataFrame({'scenario_index': sorted(scenario_indices)})

            # Read all the old tables and filter out all scenario ids not in the confiner
            new_table = []
            old_rows = 0
            for p in self.paths:
                old_table = pd.read_csv(p)
                old_rows += old_table.shape[0]
                new_table.append(confiner.merge(old_table, on='scenario_index', how='inner'))
            new_table = pd.concat(new_table, axis=0)
            new_table.to_csv(temp_path, index=None)
            report(f'Confined Stage One Scenarios table for "{tag}" from {old_rows} to {new_table.shape[0]}')
            report(f'Confined table written to {temp_path}')
        return [temp_path]

    def csv_to_mem(self):
        """
        Iteratively loop through all scenarios in chunks
        :param subset: Only return scenarios with a scenario id in the subset (list)
        :return:
        """
        cursor = 0
        writer = self.writer
        for path in self.paths:
            report(f'Reading scenario table {path}...')
            for chunk in pd.read_csv(path, chunksize=self.sim.stage_one_chunksize):
                chunk = self.modify_array(chunk)
                writer[cursor:cursor + chunk.shape[0]] = chunk
                cursor += chunk.shape[0]
        del writer

    def modify_array(self, array):

        # TODO - can we clean this up? what needs to be here vs in scenarios project?
        for var in ('orgC_5', 'crop_intercept', 'slope', 'max_canopy', 'root_depth'):
            array[var] /= 100.  # cm -> m
        # TODO - confirm that this still jibes with the updates
        # for var in ('anetd', 'amxdr'):
        #    array[var] = np.min((array[var], array['root_zone_max'] / 100.))
        for var in ['bd_5', 'bd_20']:
            array[var] *= 1000  # kg/m3
        for var, min_val in (('usle_k', 0.2), ('usle_p', 0.25), ('usle_ls', 1.0)):
            array.loc[array[var] == 0, var] = min_val  # TODO - Why are so many zeros?
        # TODO - do I still need to fix dates?
        array.loc[array.ireg == 0, 'ireg'] = 1
        array.slope = np.minimum(array.slope, 0.01)

        # TODO - move this to the qc part of fields and qc.
        #  Also, this probably isn't right
        for var in ['bd_5', 'bd_20']:
            array.loc[array[var] <= 0, var] = 1000000.

        return array.sort_values(['weather_grid', 'scenario_index'])

    @property
    def n_scenarios(self):
        return self._index.shape[0]

    @property
    def paths(self):
        if self._paths is None:
            max_chunks = 10  # this will rarely if ever be more than 10
            paths = list(filter(os.path.exists, [self.table_path.format(self.region_id, i) for i in range(max_chunks)]))
            if not paths:
                raise FileNotFoundError(f'No Stage 1 scenarios found at {self.table_path}')
            return paths
        else:
            return self._paths


class StageTwoScenarios(DateManager, MemoryMatrix):
    def __init__(self, region, sim, stage_one, met):
        self.sim = sim
        self.s1 = stage_one
        self.met = met
        self.region_id = region.id
        self.path = sim.s2_scenarios_path.format(region.id)
        self.fields = sim.fields
        if sim.tag is not None:
            self.path += f'_{sim.tag}'
        self.keyfile_path = self.path + '_key.txt'
        self.array_path = self.path + '_arrays.dat'
        self.index_path = self.path + '_index.csv'

        # If build is True, create the Stage 2 Scenarios by running model routines on Stage 1 scenario inputs
        if sim.build_scenarios:
            report('Building Stage Two Scenarios from Stage One...')
            # Getting rid of this, replace with what?
            # DateManager.__init__(self, self.sim.scenario_start_date, self.sim.scenario_end_date)
            self.arrays = self.fields.fetch('s2_arrays')
            DateManager.__init__(self, self.sim.scenario_start_date, self.sim.scenario_end_date)
            MemoryMatrix.__init__(self, [self.s1.lookup.index, self.arrays, self.n_dates],
                                  dtype=np.float32, path=self.array_path, persistent_read=True)

            # Create key
            self.create_keyfile()

            # Run scenarios
            self.build_from_stage_one()
        else:
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
        batch = []
        batch_index = []
        batch_count = 0

        # Initialize a list of the simulation parameters used to process Stage 1 Scenarios
        sim_params = [self.sim.cn_min, self.sim.delta_x, self.sim.bins, self.sim.depth, self.sim.anetd,
                      self.sim.n_increments, self.sim.sfac, self.sim.types, self.fields.fetch('s2_arrays')]

        # Check fields_and_qc.py to ensure that the field order matches the order of the input parameters
        s1_fields = self.sim.fields.fetch('s1_to_s2')
        s1_field_map = self.s1.column_index(s1_fields)

        # Group by weather grid to reduce the overhead from fetching met data
        weather_grid = None
        for _, row in self.s1.lookup.iterrows():
            # Stage 1 scenarios are sorted by weather grid. When it changes, update the time series data
            if row.weather_grid != weather_grid:
                weather_grid = row.weather_grid
                precip, pet, temp, *_ = self.met.fetch_station(weather_grid)
                time_series_data = [precip, pet, temp, self.new_year]

            # Unpack the needed parameters from the Stage 1 scenario
            s1_params = list(self.s1.fetch(row.s1_index, iloc=True)[s1_field_map])

            # Combine needed input parameters and add a call to the processing function (stage_one_to_two)
            s2_input = time_series_data + s1_params + sim_params
            batch.append(self.sim.dask_client.submit(stage_one_to_two, *s2_input))
            batch_index.append(row.scenario_id)

            # Submit the batch for asynchronous processing
            # TODO - how do the weather and scenario arrays match up?
            if len(batch) == self.sim.batch_size or row.s1_index == self.s1.n_scenarios:
                results = np.float32(self.sim.dask_client.gather(batch))
                batch_count += 1
                start_pos = (batch_count - 1) * self.sim.batch_size
                self.writer[start_pos:start_pos + results.shape[0]] = np.array(results)
                report(f'Processed {row.s1_index + 1} of {self.s1.n_scenarios} scenarios', 1)
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
        arrays = self.fetch_multiple(found.s1_index)[:, self.runoff_erosion]
        return arrays, found.dropna()

    def load_key(self):
        with open(self.keyfile_path) as f:
            time_series = next(f).strip().split(',')
            start_date = np.datetime64(next(f).strip())
            time_series_shape = [int(val) for val in next(f).strip().split(',')]
        return time_series, start_date, time_series_shape


class StageThreeScenarios(DateManager, MemoryMatrix):
    def __init__(self, sim, stage_one, stage_two, disable_build=False, retain=None):
        self.s1 = stage_one
        self.s2 = stage_two
        self.sim = sim
        self.array_path = sim.s3_scenarios_path.format(self.s2.region_id)
        self.lookup = self.select_scenarios(self.sim.selected_crops)
        self.n_scenarios = self.lookup.shape[0]

        # Set dates
        DateManager.__init__(self, stage_two.start_date, stage_two.end_date)

        # Initialize memory matrix
        # arrays - runoff_mass, erosion_mass
        MemoryMatrix.__init__(self, [self.lookup.s3_index, 2, self.n_dates], name='pesticide mass',
                              dtype=np.float32, path=self.array_path, persistent_read=True, persistent_write=True)

        report(f'Building Stage 3 scenarios...')
        if not disable_build:
            self.build_from_stage_two()

    def build_from_stage_two(self):
        batch = []
        batch_index = []
        batch_count = 0

        # Initialize some params now
        sim_params = [self.sim.runoff_effic, self.sim.erosion_effic, self.sim.surface_dx,
                      self.sim.cm_2, self.sim.soil_depth, self.sim.deg_foliar, self.sim.washoff_coeff,
                      self.sim.koc, self.sim.deg_aqueous, self.new_year, self.sim.kd_flag]

        # These fields should match the order of the parameters used by stage_two_to_three
        # Currently: [plant_date, emergence_date, maxcover_date, harvest_date, max_canopy, orgC_5, bd_5, season]
        s1_fields = [self.sim.crop_group_field] + list(self.sim.fields.fetch('s1_to_s3'))
        s1_field_map = self.s1.column_index(s1_fields)

        # Iterate scenarios
        for count, (s1_index, s3_index, scenario_id) in \
                enumerate(self.lookup[['s1_index', 's3_index', 'scenario_id']].values):

            # Stage 1 params, specified in fields_and_qc
            # Currently plant_date, emergence_date, maxcover_date, harvest_date, max_canopy, orgC_5, bd_5, season

            crop_group, *s1_params = self.s1.fetch(s1_index, iloc=True)[s1_field_map]

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
        arrays = super(StageThreeScenarios, self).fetch(found.s3_index, verbose=verbose)
        return arrays, found.dropna()

    def select_scenarios(self, crops):
        crops = pd.DataFrame({self.sim.crop_group_field: list(crops)}, dtype=np.int32)
        selected = self.s1.lookup.reset_index().merge(crops, on=self.sim.crop_group_field, how='inner')
        selected['s3_index'] = np.arange(selected.shape[0])

        # The 'contributions' array created in the outputs uses 'numpy.bincount' to add things up. We can reduce
        # the size of the array by creating an alias for the CDL classes (e.g., [1, 5, 21, 143] - > [1, 2, 3, 4])
        # This alias is called 'contribution_id' to avoid re-using 'cdl_alias'
        active_crops = pd.DataFrame({'cdl_alias': sorted(selected.cdl_alias.unique())})
        active_crops['contribution_id'] = active_crops.index + 1
        lookup = selected[['scenario_index', 'scenario_id', 's1_index', 's3_index', 'cdl_alias']] \
            .sort_values('s3_index') \
            .set_index('scenario_index') \
            .merge(active_crops, on='cdl_alias')
        return lookup


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
                     types, array_fields):
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
