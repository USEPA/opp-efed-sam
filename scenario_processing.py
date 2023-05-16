import os
import time
import numpy as np
import pandas as pd
from ast import literal_eval

from .tools.efed_lib import DateManager, MemoryMatrix
from .field import plant_growth, initialize_soil, process_erosion
from .hydrology import surface_hydrology
from .transport import pesticide_to_field, field_to_soil, soil_to_water
from .utilities import report

# For QAQC or debugging purposes - write the nth scenario from each batch to file (turn off with None)
sample_row = 17

"""
Scenario indexing:
* scenario_id (str, iloc=False) - Name for each s1 scenario, same as used in PWC
* s1_index (int, iloc=True for s1) - a unique integer alias for s1 scenarios
* s3_index (int, iloc=True for s3) - a unique integer alias for s3 scenarios

Scenario shapes:
* s1: (scenarios, vars)
* s2: (scenarios, [runoff, erosion, leaching, soil_water, rain], dates)
* s3: (scenarios, [runoff_mass, erosion_mass], dates)


Most expensive functionality in SAM is fetching s2 and s3 scenarios from the matrix. Can this be sped up?
* Changing the array shapes probably won't help. I think the costly part is creating a copy. 
* Can we avoid making a copy? Create an intermediate memory map?
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

    def __init__(self, region, sim, recipes=None):
        self.region_id = region.id
        self.sim = sim

        # Set paths
        self.array_path = self.sim.s1_scenarios_array_path.format(self.region_id)
        self.table_path = self.sim.s1_scenarios_table_path.format(self.region_id, 1)

        # Designate the fields that carry through to higher-level scenarios
        self.s2_fields = self.sim.fields.fetch('s1_to_s2')
        self.s3_fields = [self.sim.crop_group_field] + list(self.sim.fields.fetch('s1_to_s3'))

        # Create a tabular index of core scenario identifiers
        self.lookup, self.array_fields = self.build_index()
        self.n_scenarios = self.lookup.shape[0]

        # Find the cdl aliases for all the crops that will have pesticide applications based on user input
        self.sim.active_crops = self.get_active_crops()

        # No need to allocate memory if just generating random output
        MemoryMatrix.__init__(self, [self.lookup.s1_index, self.array_fields], name='s1 scenario',
                              dtype=np.float32, path=self.array_path, persistent_read=True)
        self.csv_to_memmap()

    def build_index(self):
        # Get all the column headings from the input table
        columns = pd.read_csv(self.table_path, nrows=1).columns.values

        # Sort columns into those that go in the lookup table, and those that go in the memory array
        lookup_fields = list(self.sim.fields.fetch('s1_lookup'))
        array_fields = [c for c in columns if c not in lookup_fields]

        # Read the lookup table and add a unique 's1_index'
        lookup = pd.read_csv(self.table_path, usecols=lookup_fields + [self.sim.crop_group_field])
        lookup['s1_index'] = np.arange(lookup.shape[0])

        return lookup, array_fields

    def csv_to_memmap(self):
        """ Iteratively loop through all scenarios in chunks and read into memmap array """
        cursor = 0
        writer = self.writer
        report(f'Reading Stage One Scenarios into memory...')
        for chunk in pd.read_csv(self.table_path, usecols=self.array_fields, chunksize=self.sim.scenario_chunksize):
            writer[cursor:cursor + chunk.shape[0]] = chunk
            cursor += chunk.shape[0]
        del writer

    def fetch(self, index, field_set=None, iloc=True, return_fields=False):
        fields = {'s2': self.s2_fields, 's3': self.s3_fields}.get(field_set, self.array_fields)
        field_index = [self.array_fields.index(f) for f in fields]
        row = super(StageOneScenarios, self).fetch(index, iloc=iloc)
        if not return_fields:
            row = np.array(row[field_index])
            nans = np.isnan(row)
            if nans.any():
                print(np.array(fields)[nans])
            return list(row)
        else:
            return pd.Series(row[field_index], index=field_set)

    def get_active_crops(self):
        # Read the lookup table to send all active crops to the simulation
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
        self.keyfile_path, self.array_path = self.set_paths(region.id)
        self.n_scenarios = self.s1.n_scenarios

        # If build is True, create the Stage 2 Scenarios by running model routines on Stage 1 scenario inputs
        if sim.build_scenarios:
            report(f'Building Stage Two Scenarios at {self.array_path}...')

            self.vars = self.fields.fetch('s2_arrays')  # runoff, erosion, leaching, soil_water, rain
            DateManager.__init__(self, self.sim.scenario_start_date, self.sim.scenario_end_date)
            MemoryMatrix.__init__(self, [self.s1.lookup.index, self.vars, self.n_dates],
                                  dtype=np.float32, path=self.array_path, persistent_read=True)

            # Create key
            self.create_keyfile()

            # Run scenarios
            self.build_from_stage_one()

        # If build is False, load the saved Stage 2 Scenario array
        else:
            report(f'Loading Stage Two Scenarios in array {self.array_path}...')
            self.vars, start_date, end_date, n_dates = self.load_key()
            DateManager.__init__(self, start_date, end_date)
            MemoryMatrix.__init__(self, [self.s1.lookup.index, self.vars, self.n_dates],
                                  dtype=np.float32, path=self.array_path, persistent_read=True)
            self.start_offset, self.end_offset = \
                self.date_offset(self.sim.start_date, self.sim.end_date, n_dates=n_dates)

    def build_from_stage_one(self):
        """
        Stage 2 Scenarios (s2) are built by running plant growth, hydrology, and erosion simulations
        on each Stage 1 Scenario (s1), using a combination of global parameters and parameters unique to each
        s1. None of these parameters are specified by the user, so these scenarios can be generated ahead of time.
        Because it's a time-consuming process, Dask is used to parallelize the runs. Dask processes the runs in
        batches, and batch_size is set in params.csv
        """

        batch = []  # This will hold all the dask calls for each batch
        batch_index = []  # This is only used to retain the scenario id for writing sample csv outputs
        batch_count = 0  # Num of batches processed - used for identifying position in array

        # Initialize a list of the simulation parameters used to process Stage 1 Scenarios
        sim_params = [self.sim.cn_min, self.sim.delta_x, self.sim.bins, self.sim.depth, self.sim.anetd,
                      self.sim.n_increments, self.sim.sfac, self.sim.types]

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
            scenario_inputs = time_series_data + s1_params + sim_params

            # In debug mode, the processing will not use Dask or occur in parallel
            batch.append(self.sim.dask_client.submit(stage_one_to_two, *scenario_inputs))
            batch_index.append(row.scenario_id)

            # Submit the batch for asynchronous processing
            # TODO - how do the weather and scenario arrays match up?
            if len(batch) == self.sim.batch_size or row.s1_index == self.n_scenarios:
                results = np.float32(self.sim.dask_client.gather(batch))
                batch_count += 1
                start_pos = (batch_count - 1) * self.sim.batch_size
                self.writer[start_pos:start_pos + results.shape[0]] = results
                report(f'Processed {row.s1_index + 1} of {self.n_scenarios} scenarios', 1)
                #write_sample(self.dates, self.sim, results, batch_index)
                batch = []
                batch_index = []

    def create_keyfile(self):
        with open(self.keyfile_path, 'w') as f:
            f.write(','.join(self.vars) + '\n')
            f.write(pd.to_datetime(self.start_date).strftime('%Y-%m-%d') + '\n')
            f.write(','.join(map(str, self.shape)) + '\n')

    def load_key(self):
        with open(self.keyfile_path) as f:
            names = next(f).strip().split(',')
            start_date = np.datetime64(next(f).strip())
            time_series_shape = [int(val) for val in next(f).strip().split(',')]
            n_dates = time_series_shape[2]
            end_date = start_date + n_dates - 1
        return names, start_date, end_date, n_dates

    def set_paths(self, region):
        root_path = self.sim.s2_scenarios_path.format(region)
        keyfile_path = root_path + '_key.txt'
        array_path = root_path + '_arrays.dat'
        return keyfile_path, array_path

    def fetch(self, index):
        ts = super(StageTwoScenarios, self).fetch(index, iloc=True)
        return list(ts[:, self.start_offset:-self.end_offset])


class StageThreeScenarios(DateManager, MemoryMatrix):
    def __init__(self, sim, stage_one, stage_two, active_reaches, recipes):
        self.s1 = stage_one
        self.s2 = stage_two
        self.sim = sim
        self.array_path = sim.s3_scenarios_path.format(self.sim.token)
        self.sample_path = os.path.join(os.path.dirname(self.array_path), "{}_s{}.csv")
        self.lookup = self.build_lookup(active_reaches, recipes)
        self.vars = sim.fields.fetch('s3_arrays')  # runoff, runoff_mass, erosion, erosion_mass

        # Set dates
        DateManager.__init__(self, stage_two.start_date, stage_two.end_date)

        # Initialize memory matrix
        # arrays - runoff_mass, erosion_mass
        MemoryMatrix.__init__(self, [self.lookup.s1_index, self.vars, self.n_dates], name='pesticide mass',
                              dtype=np.float32, path=self.array_path, persistent_read=True, persistent_write=True)

        report(f'Building Stage 3 scenarios...')
        self.build_from_stage_two()

    def confine(self, active_reaches, recipes):
        import time
        start = time.time()
        area = 0
        active_scenarios = set()
        for reach_id in active_reaches:
            for i, (year, recipe) in enumerate(recipes.fetch(reach_id, df=True)):
                active_scenarios |= set(recipe.s1_index)
                area += recipe.area.sum()
        print(f"Found {len(active_scenarios)} active scenarios in {int(time.time() - start)} seconds")
        print(f"Confining analysis to an area of {area * 1e-6} sq km")
        return sorted(active_scenarios)

    def build_lookup(self, active_reaches, recipes):
        # Carry over the index table from the s1 scenarios
        lookup = self.s1.lookup

        # Create a simple numeric index for each crop type. Crops receiving chemical are active
        lookup['contribution_index'] = lookup.cdl_alias.map({val: i for i, val in enumerate(self.sim.active_crops)})
        lookup['chemical_applied'] = lookup['contribution_index'].notna()

        # Confine processing if not running the whole region

        if self.sim.confine_reaches is not None:
            lookup['in_confine'] = False
            active_scenarios = self.confine(active_reaches, recipes)
            lookup.loc[np.array(active_scenarios), 'in_confine'] = True
        else:
            lookup['in_confine'] = True

        return lookup

    def build_from_stage_two(self):
        batch = []  # This will hold all the dask calls for each batch
        batch_index = []  # This is only used to retain the scenario id for writing sample csv outputs
        batch_count = 0  # Num of batches processed - used for identifying position in array

        # Initialize some params now
        sim_params = [self.sim.runoff_effic, self.sim.erosion_effic, self.sim.surface_dx,
                      self.sim.cm_2, self.sim.soil_depth, self.sim.deg_foliar, self.sim.washoff_coeff,
                      self.sim.koc, self.sim.deg_aqueous, self.new_year, self.sim.kd_flag]

        # Subset the scenarios if necessary
        selected = self.lookup[self.lookup.in_confine][['s1_index', 'scenario_id', 'chemical_applied']]
        n_selected = selected.shape[0]

        # Iterate scenarios
        nochem = 0
        badvars = set()
        success = 0
        for count, (s1_index, scenario_id, chemical_applied) in enumerate(selected.values):
            # self.shape = [scenarios, vars, dates]
            s2_time_series = self.s2.fetch(s1_index)  # runoff, erosion, leaching, soil_water, rain

            if not chemical_applied:
                job = self.sim.dask_client.submit(pass_s2_to_s3, *s2_time_series[:2])
                nochem += 1
            else:
                # These fields should match the order of the parameters used by stage_two_to_three
                # Currently: [plant_date, emergence_date, maxcover_date, harvest_date, max_canopy, orgC_5, bd_5, season]
                crop_group, *s1_params = self.s1.fetch(s1_index, 's3')
                if not np.isnan(np.array(s1_params)).any():

                    # Get application information for the active crop
                    crop_applications = self.sim.applications[self.sim.applications.crop == crop_group]

                    # Extract stored data
                    scenario_inputs = [crop_applications.values] + sim_params + s2_time_series + s1_params

                    # Turn this on for testing
                    if sample_row is not None and sample_row == count:
                        results = stage_two_to_three(*scenario_inputs) # np.array([runoff, runoff_mass, erosion, erosion_mass])
                        write_sample(scenario_id, s1_params, s2_time_series, results, self.sample_path)

                    job = self.sim.dask_client.submit(stage_two_to_three, *scenario_inputs)
                    success += 1
                else:
                    report(f"Unable to process {scenario_id} due to missing data")
                    # TODO - do i bother with this, or just continue?
                    job = self.sim.dask_client.submit(invalid_s2_scenario, s2_time_series)
                    soil, weather, landcover = scenario_id.split("-")
                    badvars.add(f"{weather}, {landcover}")

            batch.append(job)
            batch_index.append(s1_index)
            if len(batch) == self.sim.batch_size or (count + 1) == n_selected:
                report(f"Processed {count + 1} of {n_selected} scenarios...")
                report(f"Good:{success} Bad:{len(badvars)} Nochem:{nochem}")
                arrays = self.sim.dask_client.gather(batch)
                self.writer[batch_index] = np.array(arrays)
                batch_count += 1
                batch = []
                batch_index = []
        badvars = ",".join(badvars)
        print(f"Unable to process the following scenarios:{badvars}")

    def fetch_from_recipe(self, recipe, verbose=False):
        found = self.lookup.iloc[recipe]
        arrays = super(StageThreeScenarios, self).fetch(found.s1_index, iloc=True, verbose=verbose)
        return arrays, found


def stage_one_to_two(precip, pet, temp, new_year,  # weather params
                     plant_date, emergence_date, maxcover_date, harvest_date,  # crop dates
                     max_root_depth, crop_intercept,  # crop properties
                     slope, slope_length,  # field properties
                     fc_5, wp_5, fc_20, wp_20,  # soil properties
                     cn_cov, cn_fallow, usle_k, usle_ls, usle_c_cov, usle_c_fal, usle_p,  # usle params
                     irrigation_type, ireg, depletion_allowed, leaching_fraction,  # irrigation params
                     cn_min, delta_x, bins, depth, anetd, n_increments, sfac,  # simulation soil params
                     types):
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
    return np.array([runoff, erosion, leaching, soil_water, rain])

def write_sample(scenario_id, s1, s2, s3, out_path):
    s1_names = ["plant_date", "emergence_date", "maxcover_date", "harvest_date", "max_canopy",
                "orgC_5", "bd_5", "season"]
    s2_names = ["runoff", "erosion", "leaching", "soil_water", "rain"]
    s3_names = ['runoff', 'runoff_mass', 'erosion', 'erosion_mass']

    print(f"Saving sample scenario to {out_path.format(scenario_id, 'x')}")
    pd.DataFrame({s1_names[i]: [val] for i, val in enumerate(s1)}).T.to_csv(out_path.format(scenario_id, 1))
    pd.DataFrame(s2.T, columns=s2_names).to_csv(out_path.format(scenario_id, 2))
    pd.DataFrame(s3.T, columns=s3_names).to_csv(out_path.format(scenario_id, 3))

def pass_s2_to_s3(runoff, erosion):
    out_array = np.zeros((4, runoff.size), dtype=np.float64)
    out_array[0] = runoff
    out_array[2] = erosion
    return out_array


def invalid_s2_scenario(s2_time_series):
    runoff, erosion = s2_time_series[:2]
    output = np.zeros([4, len(runoff)])
    output[0] = runoff
    output[2] = erosion
    return output

def stage_two_to_three(application_matrix,
                       runoff_effic, erosion_effic, surface_dx, cm_2, soil_depth, deg_foliar,
                       washoff_coeff, koc, deg_aqueous, new_year, kd_flag,
                       runoff, erosion, leaching, soil_water, rain,
                       plant_date, emergence_date, maxcover_date, harvest_date, covmax, org_carbon,
                       bulk_density, season):
    # TODO - season?

    # Use Kd instead of Koc if flag is on. Kd = Koc * organic C in the top layer of soil
    # Reference: PRZM5 Manual(Young and Fry, 2016), Section 4.13
    if kd_flag:
        koc *= org_carbon

    erosion_nans = np.isnan(erosion).sum()
    if erosion_nans > 0:
        print(f"{erosion_nans} NaN values found in erosion for scenario")
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
    runoff_mass, erosion_mass = \
        soil_to_water(pesticide_mass_soil, runoff, erosion, leaching, bulk_density, soil_water, koc,
                      deg_aqueous, runoff_effic, surface_dx, erosion_effic,
                      soil_depth)

    return np.array([runoff, runoff_mass, erosion, erosion_mass])
