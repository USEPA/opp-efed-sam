import os
import numpy as np
import pandas as pd

from .tools.efed_lib import report, DateManager, MemoryMatrix
from .field import plant_growth, initialize_soil, process_erosion
from .hydrology import surface_hydrology
from .transport import pesticide_to_field, field_to_soil, soil_to_water


class StageOneScenarios(object):
    def __init__(self, region, sim, recipes=None, subset_outlets=None, subset_year=None):
        self.region = region
        self.sim = sim
        self.path = sim.paths.s1_scenarios
        self._subset = False
        self._scenario_id = None
        self.scratch_path = os.path.join(sim.paths.scratch_path, f"s1_subset.csv")

        if subset_outlets is not None:
            self.build_subset(subset_outlets, subset_year, recipes)
            self._subset = True

    def build_subset(self, outlets, year, recipes):
        report(f"Subsetting Stage 1 Scenarios...", 1)

        # Build an index of all the scenarios in the subset
        all_scenarios = set()
        for reach in outlets:
            recipe = recipes.fetch(reach, year)
            all_scenarios |= set(recipe.index)
        all_scenarios = pd.DataFrame({'scenario_index': sorted(all_scenarios)})

        # Select only the rows in the Stage 1 scenarios table that belong to the subset
        full_table = []
        for _, table in self.iterate():
            selected = table.merge(all_scenarios, on='scenario_index', how='inner')
            full_table.append(selected)
        full_table = pd.concat(full_table, axis=0)

        # Write the subset to a temporary csv file and update the paths value
        report(f"Writing subset to {self.scratch_path}")
        full_table.to_csv(self.scratch_path, index=None)

    def fetch(self, field_names, return_array=False):
        singular = type(field_names) == str
        if singular:
            field_name = field_names
            field_names = [field_name]
        selection = pd.concat([pd.read_csv(p)[field_names] for p in self.paths], axis=0)
        if singular:
            selection = selection[field_name]
        if return_array:
            return selection.values
        else:
            return selection

    def iterate(self):
        for path in self.paths:
            report(f"Reading scenario table {path}...")
            for chunk in pd.read_csv(path, chunksize=self.sim.stage_one_chunksize):
                chunk = self.modify_array(chunk)
                for weather_grid, scenarios in chunk.groupby('weather_grid'):
                    yield weather_grid, scenarios

    @property
    def names(self):
        if self._scenario_id is None:
            self._scenario_id = self.fetch('scenario_id', True)
        return self._scenario_id

    def modify_array(self, array):
        # TODO - can we clean this up? what needs to be here vs in scenarios project?

        for field, val in self.sim.scenario_defaults.items():
            array[field] = val
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

        return array

    @property
    def n_scenarios(self):
        return len(self.names)

    @property
    def paths(self):
        if not self._subset:
            paths = []
            i = 0
            while True:
                i += 1
                path = self.path.format(self.region, i)
                if os.path.exists(path):
                    paths.append(path)
                else:
                    break
            if not paths:
                raise FileNotFoundError(f"No Stage 1 scenarios found at {path}")
            return paths
        else:
            return [self.scratch_path]


class StageTwoScenarios(DateManager, MemoryMatrix):
    def __init__(self, region, sim, stage_one, met, tag=None, build=False):
        self.region = region
        self.sim = sim
        self.s1 = stage_one
        self.met = met

        self.fields = sim.fields
        self.path = sim.paths.s2_scenarios.format(region)
        if tag is not None:
            self.path += f"_{tag}"
        self.keyfile_path = self.path + "_key.txt"
        self.array_path = self.path + "_arrays.dat"
        self.index_path = self.path + "_index.csv"

        # If build is True, create the Stage 2 Scenarios by running model routines on Stage 1 scenario inputs
        if build:
            # TODO - at this point, check and see if the stage 2 scenarios (1) exist locally, or (2) exist on s3.
            #  if not, build them and send a copy to s3
            report("Building Stage Two Scenarios from Stage One...")
            DateManager.__init__(self, self.sim.scenario_start_date, self.sim.scenario_end_date)
            self.arrays = self.fields.fetch('s2_arrays')
            self.align_met_dates()
            scenario_index = self.s1.fetch('scenario_id')
            MemoryMatrix.__init__(self, [scenario_index, self.arrays, self.n_dates],
                                  dtype=np.float32, path=self.array_path, persistent_read=True)

            # Create key
            self.create_keyfile()

        else:
            self.arrays, self.array_start_date, time_series_shape = self.load_key()
            self.runoff_erosion = [self.arrays.index('runoff'), self.arrays.index('erosion')]
            self.scenario_vars, self.lookup = self.create_lookup()
            self.n_dates_array = time_series_shape[2]
            self.array_end_date = self.array_start_date + self.n_dates_array - 1
            self.align_sim_dates()
            DateManager.__init__(self, self.array_start_date, self.array_end_date)
            self.start_offset, self.end_offset = self.date_offset(self.sim.start_date, self.sim.end_date,
                                                                  n_dates=self.n_dates_array)

            # Initialize MemoryMatrix
            MemoryMatrix.__init__(self, time_series_shape, path=self.array_path, existing=True, name='scenario')

    def align_sim_dates(self):
        """ Get offset between scenario and simulation start dates """
        messages = []
        if self.sim.start_date < self.array_start_date:
            self.sim.start_date = self.array_start_date
            messages.append("start date is earlier")
        if self.array_end_date < self.sim.end_date:
            self.sim.end_date = self.array_end_date
            messages.append("end date is later")
        if any(messages):
            report(f"Simulation {' and '.join(messages)} than range of available scenario data. "
                   f"Date range has been truncated at {self.sim.start_date} to {self.sim.end_date}.")

    def align_met_dates(self):
        # TODO - this should be combined with align_sim_dates and probably put into the parent DateManager class
        messages = []
        if self.start_date < self.met.start_date:
            messages.append("start date is earlier")
            self.start_date = self.met.start_date
        else:
            self.met.start_offset = (self.start_date - self.met.start_date).astype(np.int32)
        if self.met.end_date < self.end_date:
            messages.append("end date is later")
            self.end_date = self.met.end_date
        else:
            self.met.end_offset = (self.end_date - self.met.end_date).astype(np.int32)

    def build_from_stage_one(self):
        batch = []
        scenario_count = 0
        batch_count = 0
        keep_fields = list(self.fields.fetch('s1_keep_cols')) + [self.sim.crop_group_field]
        stage_two_index = []
        soil = self.sim.soil
        types = pd.read_csv(self.sim.types_path).set_index('type')

        # Group by weather grid to reduce the overhead from fetching met data
        for weather_grid, scenarios in self.s1.iterate():
            precip, pet, temp, *_ = self.met.fetch_station(weather_grid)
            for _, s in scenarios.iterrows():
                # Result - arrays of runoff, erosion, leaching, soil_water, rain
                scenario = [precip, pet, temp, self.new_year,
                            s.plant_date, s.emergence_date, s.maxcover_date, s.harvest_date,
                            s.max_root_depth, s.crop_intercept, s.slope, s.slope_length,
                            s.water_max_5, s.water_min_5, s.water_max_20, s.water_min_20,
                            s.cn_cov, s.cn_fal, s.usle_k, s.usle_ls, s.usle_c_cov, s.usle_c_fal, s.usle_p,
                            s.irrigation_type, s.ireg, s.depletion_allowed, s.leaching_fraction, types,
                            soil.cn_min, soil.delta_x, soil.bins, soil.depth, soil.anetd, soil.n_increments,
                            soil.sfac, self.sim.fields.fetch('s2_arrays')]
                batch.append(self.sim.dask_client.submit(stage_one_to_two, *scenario))
                scenario_count += 1
                scenario_vars = s[keep_fields]
                stage_two_index.append(scenario_vars.values)
                if len(batch) == self.sim.batch_size or scenario_count == self.s1.n_scenarios:
                    results = self.sim.dask_client.gather(batch)
                    batch_count += 1
                    batch = []
                    report(f"Processed {scenario_count} of {self.s1.n_scenarios} scenarios", 1)
                    self.write(batch_count, results)

        own_index = pd.DataFrame(stage_two_index, columns=keep_fields)
        self.write('index', own_index)

    def create_keyfile(self):
        with open(self.keyfile_path, 'w') as f:
            f.write(",".join(self.arrays) + "\n")
            f.write(pd.to_datetime(self.start_date).strftime('%Y-%m-%d') + "\n")
            f.write(",".join(map(str, self.shape)) + "\n")

    def create_lookup(self):
        scenario_vars = pd.read_csv(self.index_path)
        scenario_vars['s2_index'] = scenario_vars.index
        lookup = scenario_vars[['scenario_index', 'scenario_id', 's2_index']].set_index('scenario_index')
        return scenario_vars, lookup

    def fetch(self, index, copy=False, verbose=False, iloc=False, pop=False, return_alias=False):
        result = super(StageTwoScenarios, self).fetch(index, copy, iloc, pop, return_alias)
        return result[:, self.start_offset:-self.end_offset]

    def fetch_from_recipe(self, recipe, verbose=True):
        found = recipe.join(self.lookup)
        arrays = super(StageTwoScenarios, self).fetch(found.s2_index, verbose=verbose)
        arrays = arrays[:, self.runoff_erosion, self.start_offset:-self.end_offset]
        return arrays, found.dropna()

    def load_key(self):
        with open(self.keyfile_path) as f:
            time_series = next(f).strip().split(",")
            start_date = np.datetime64(next(f).strip())
            time_series_shape = [int(val) for val in next(f).strip().split(",")]
        return time_series, start_date, time_series_shape

    def write(self, batch_num, data):
        if batch_num == 'index':
            data.to_csv(self.index_path, index=None)
        else:
            batch_size_actual = len(data)
            start_pos = (batch_num - 1) * self.sim.batch_size
            self.writer[start_pos:start_pos + batch_size_actual] = np.array(data)


class StageThreeScenarios(DateManager, MemoryMatrix):
    def __init__(self, sim, stage_two, disable_processing=False):
        self.s2 = stage_two
        self.region = self.s2.region
        self.sim = sim
        self.array_path = sim.paths.s3_scenarios.format(self.region)
        self.scenario_vars, self.lookup = self.select_scenarios(self.sim.crops)

        # Set dates
        DateManager.__init__(self, stage_two.start_date, stage_two.end_date)

        # Initialize memory matrix
        # arrays - runoff_mass, erosion_mass
        MemoryMatrix.__init__(self, [len(self.scenario_vars.s3_index), 2, self.n_dates], name='pesticide mass',
                              dtype=np.float32, path=self.array_path, persistent_read=True, persistent_write=True)

        report(f"Building Stage 3 scenarios...")
        if not disable_processing:
            self.build_from_stage_two()

    def build_from_stage_two(self):
        # TODO - can the dask allocation part of this be put into a function or wrapper?
        #  it's also used in s1->s2
        soil = self.sim.soil
        plant = self.sim.plant

        var_table = self.scenario_vars.set_index('s2_index')

        batch = []
        batch_count = 0
        n_scenarios = var_table.shape[0]
        dask_client = self.sim.dask_client

        # Iterate scenarios
        for count, (s2_index, s3_index) in enumerate(self.lookup[['s2_index', 's3_index']].values):

            # Get the non-array values associated with the scenario
            s2 = var_table.loc[s2_index]

            # Extract stored data
            runoff, erosion, leaching, soil_water, rain = self.s2.fetch(s2_index, iloc=True)

            # Get application information for the active crop
            crop_applications = \
                self.sim.applications[self.sim.applications.crop == s2[self.sim.crop_group_field]]

            if not crop_applications.empty:

                # Get crop ID of scenario and find all associated crops in group
                scenario = [crop_applications.values,
                            self.new_year, self.sim.kd_flag, self.sim.koc, self.sim.deg_aqueous,
                            leaching, runoff, erosion, soil_water, rain,
                            s2.plant_date, s2.emergence_date, s2.maxcover_date, s2.harvest_date,
                            s2.max_canopy, s2.orgC_5, s2.bd_5, s2.season,
                            soil.runoff_effic, soil.erosion_effic, soil.surface_dx, soil.cm_2, soil.soil_depth,
                            plant.deg_foliar, plant.washoff_coeff]

                batch.append(dask_client.submit(stage_two_to_three, *scenario))
                if len(batch) == self.sim.batch_size or (count + 1) == n_scenarios:
                    arrays = dask_client.gather(batch)
                    start_pos = batch_count * self.sim.batch_size
                    self.writer[start_pos:start_pos + len(batch)] = arrays
                    batch_count += 1
                    batch = []
                    report(f"Processed {count + 1} of {n_scenarios} scenarios...", 1)

    def fetch_from_recipe(self, recipe, verbose=False):
        found = recipe.join(self.lookup, how='inner')
        arrays = super(StageThreeScenarios, self).fetch(found.s3_index, verbose=verbose)
        return arrays, found.dropna()

    def select_scenarios(self, crops):
        """ Use the Stage Two Scenarios index, but filter by crops """
        crops = pd.DataFrame({self.sim.crop_group_field: list(crops)}, dtype=np.int32)
        selected = self.s2.scenario_vars.merge(crops, on=self.sim.crop_group_field, how='inner')
        selected['s3_index'] = np.arange(selected.shape[0])
        scenario_vars = selected.sort_values('s3_index')
        # JCH - for diagnostic purposes, 'scenario_id' can be added to the lookup table
        lookup = scenario_vars[['scenario_index', 's2_index', 's3_index']].set_index('scenario_index')
        return scenario_vars, lookup


def stage_one_to_two(precip, pet, temp, new_year,  # weather params
                     plant_date, emergence_date, maxcover_date, harvest_date,  # crop dates
                     max_root_depth, crop_intercept,  # crop properties
                     slope, slope_length,  # field properties
                     fc_5, wp_5, fc_20, wp_20,  # soil properties
                     cn_cov, cn_fallow, usle_k, usle_ls, usle_c_cov, usle_c_fal, usle_p,  # usle params
                     irrigation_type, ireg, depletion_allowed, leaching_fraction,  # irrigation params
                     cn_min, delta_x, bins, depth, anetd, n_increments, sfac,  # simulation soil params
                     types, array_names):
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
    type_matrix = types[types.index == ireg].values.astype(np.float32)  # ireg parameter
    erosion = process_erosion(slope, runoff, effective_rain, cn, usle_klscp, type_matrix, slope_length)

    # Output array order is specified in fields_and_qc.py
    arrays = []
    for array_name in array_names:
        arrays.append(eval(array_name))
    return np.float32(arrays)


def stage_two_to_three(application_matrix, new_year, kd_flag, koc, deg_aqueous, leaching, runoff, erosion,
                       soil_water, rain, plant_date, emergence_date, maxcover_date, harvest_date, covmax,
                       org_carbon, bulk_density, season,
                       runoff_effic, erosion_effic, surface_dx, cm_2, soil_depth,
                       deg_foliar, washoff_coeff):
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

    return aquatic_pesticide_mass
