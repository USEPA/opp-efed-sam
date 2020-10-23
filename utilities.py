import os
import re
import dask
import math
import numpy as np
import pandas as pd
import json
from ast import literal_eval

from sam.tools.efed_lib import MemoryMatrix, FieldManager, DateManager, report
from sam.hydro.nhd.params_nhd import nhd_regions
from sam.hydro.nhd.navigator import Navigator
from sam.hydro.nhd.process_nhd import identify_waterbody_outlets, calculate_surface_area
from sam.paths import weather_path, stage_one_scenario_path, stage_two_scenario_path, recipe_path, scratch_path, \
    dwi_path, manual_points_path, output_path, fields_and_qc_path, endpoint_format_path, condensed_nhd_path, \
    navigator_path
from sam.parameters import hydrology_params, soil_params, plant_params, output_params, fields
from sam.parameters import scenario_defaults, scenario_start_date, scenario_end_date, batch_size, stage_one_chunksize, \
    crop_group_field
from sam.aquatic_concentration import compute_concentration, partition_benthic, exceedance_probability
from sam.scenario_processing import stage_two_to_three

# Initialize endpoints
endpoint_format = pd.read_csv(endpoint_format_path)

# If true, only return 5 time series fields, otherwise, return 13
# TODO - there's a better way to do this
compact_out = True


class HydroRegion(Navigator):
    """
    Contains all datasets and functions related to the NHD Plus region, including all hydrological features and links
    outlet_reaches: reaches with an outlet (used to delineate)
    active_reaches: reaches upstream of the outlet (or for eco, all of them)
    """

    def __init__(self, sim, region):

        self.id = region

        # Assign a watershed navigator to the class
        super(HydroRegion, self).__init__(self.id)

        # Read hydrological input files
        self.reach_table = pd.read_csv(condensed_nhd_path.format('sam', region, 'reach'))
        self.lake_table = pd.read_csv(condensed_nhd_path.format('sam', region, 'waterbody'))
        self.process_nhd()

        # Initialize the fields that will be used to pull flows based on month
        self.flow_fields = [f'q_{str(month).zfill(2)}' for month in sim.month_index]

        # Select which stream reaches will be fully processed, partially processed, or excluded
        self.intake_reaches, self.active_reaches, self.output_reaches, self.reservoir_outlets = \
            self.sort_reaches(sim.intake_reaches, sim.intakes_only)

        # Holder for reaches that have been processed
        self.burned_reaches = set()

    def sort_reaches(self, intakes, intakes_only):
        """
        intakes - reaches corresponding to an intake
        active - all reaches upstream of an intake
        output - reaches for which a full suite of outputs is computed
        intakes_only - do we do the full monty for the intakes only, or all upstream?
        lake_outlets - reaches that correspond to the outlet of a lake
        """

        # Confine to available reaches and assess what's missing
        if intakes is None:
            active = output = self.reach_table.comid
        else:
            active = self.confine(intakes)
            if intakes_only:
                output = intakes
            else:
                output = active
        reservoir_outlets = \
            self.lake_table.loc[np.in1d(self.lake_table.outlet_comid, active)][['outlet_comid', 'wb_comid']]

        return intakes, active, output, reservoir_outlets

    def process_nhd(self):
        self.lake_table = \
            identify_waterbody_outlets(self.lake_table, self.reach_table)

        # Calculate average surface area of a reach segment
        self.reach_table['surface_area'] = calculate_surface_area(self.reach_table)

        # Calculate residence times of reservoirs
        self.lake_table = self.lake_table.merge(self.reach_table[['comid', 'q_ma']],
                                                left_on='outlet_comid', right_on='comid', how='left')
        self.lake_table['residence_time'] = self.lake_table.wb_volume / self.lake_table.q_ma

        # Convert units
        self.reach_table['length'] = self.reach_table.pop('lengthkm') * 1000.  # km -> m
        for month in list(map(lambda x: str(x).zfill(2), range(1, 13))) + ['ma']:
            self.reach_table["q_{}".format(month)] *= 2446.58  # cfs -> cmd
            self.reach_table["v_{}".format(month)] *= 26334.7  # f/s -> md
        self.reach_table = self.reach_table.drop_duplicates().set_index('comid')

    @property
    def cascade(self):
        # Tier the reaches by counting the number of outlets (lakes) upstream of each lake outlet
        reach_counts = []
        lake_outlets = set(self.reservoir_outlets.outlet_comid)
        for outlet in lake_outlets:
            upstream_lakes = len((set(self.upstream_watershed(outlet)) - {outlet}) & lake_outlets)
            reach_counts.append([outlet, upstream_lakes])
        reach_counts = pd.DataFrame(reach_counts, columns=['comid', 'n_upstream']).sort_values('n_upstream')

        # Cascade downward through tiers
        upstream_outlets = set()  # outlets from previous tier
        for tier, lake_outlets in reach_counts.groupby('n_upstream')['comid']:
            lakes = self.lake_table[np.in1d(self.lake_table.outlet_comid, lake_outlets)]
            all_upstream = {reach for outlet in lake_outlets for reach in self.upstream_watershed(outlet)}
            reaches = (all_upstream - set(lake_outlets)) | upstream_outlets
            reaches &= set(self.active_reaches)
            reaches -= self.burned_reaches
            yield tier, reaches, lakes
            self.burned_reaches |= reaches
            upstream_outlets = set(lake_outlets)
        all_upstream = {reach for outlet in upstream_outlets for reach in self.upstream_watershed(outlet)}
        yield -1, all_upstream - self.burned_reaches, pd.DataFrame([])

    def confine(self, outlets):
        """ If running a series of intakes or reaches, confine analysis to upstream areas only """
        upstream_reaches = \
            list({upstream for outlet in outlets for upstream in self.upstream_watershed(outlet)})
        return pd.Series(upstream_reaches, name='comid')

    def flow_table(self, reach_id):
        return self.reach_table.loc[reach_id]

    def daily_flows(self, reach_id):
        # TODO - this is taking a little time, maybe a one-time month-to-field comprehension
        selected = self.flow_table(reach_id)
        return selected[self.flow_fields].values.astype(np.float32)


class ImpulseResponseMatrix(MemoryMatrix):
    """ A matrix designed to hold the results of an impulse response function for 50 day offsets """

    def __init__(self, n_dates, size=50):
        self.n_dates = n_dates
        self.size = size
        super(ImpulseResponseMatrix, self).__init__([size, n_dates], name='impulse response')
        for i in range(size):
            irf = self.generate(i, 1, self.n_dates)
            self.update(i, irf)

    def fetch(self, index):
        if index <= self.size:
            irf = super(ImpulseResponseMatrix, self).fetch(index, verbose=False)
        else:
            irf = self.generate(index, 1, self.n_dates)
        return irf

    @staticmethod
    def generate(alpha, beta, length):
        def gamma_distribution(t, a, b):
            a, b = map(float, (a, b))
            tau = a * b
            return ((t ** (a - 1)) / (((tau / a) ** a) * math.gamma(a))) * math.exp(-(a / tau) * t)

        return np.array([gamma_distribution(i, alpha, beta) for i in range(length)])


class Simulation(DateManager):
    """
    User-specified parameters and parameters derived from hem.
    This class is used to hold parameters and small datasets that are global in nature and apply to the entire model
    run including Endpoints, Crops, Dates, Intake reaches, and Impulse Response Functions
    """

    def __init__(self, input_dict):

        # Read input dictionary
        self.__dict__.update(input_dict)

        # Initialize file structure
        self.initialize()

        # Read endpoints and applications
        self.endpoints = self.read_endpoints()
        self.applications = self.read_applications()
        self.crops = set(self.applications.crop)

        # Dates
        DateManager.__init__(self, np.datetime64(self.sim_date_start), np.datetime64(self.sim_date_end))

        # Select regions and reaches that will be run
        self.intakes, self.run_regions, self.intake_reaches = self.processing_extent()

        # Initialize an impulse response matrix if convolving timesheds
        self.irf = None if not hydrology_params.gamma_convolve else ImpulseResponseMatrix(self.dates.size)

        # Make numerical adjustments (units etc)
        self.adjust_data()

        # Read token
        self.token = \
            self.simulation_name if not hasattr(self, 'csrfmiddlewaretoken') else self.csrfmiddlewaretoken

    @staticmethod
    def initialize():
        # Make sure needed subdirectories exist
        preexisting_subdirs = (os.path.join("..", "bin", d) for d in ("Results", "temp"))
        for subdir in preexisting_subdirs:
            if not os.path.exists(subdir):
                os.makedirs(subdir)

        # Purge temp folder
        temp_folder = os.path.join("..", "bin", "temp")
        for f in os.listdir(temp_folder):
            os.remove(os.path.join(temp_folder, f))

    def adjust_data(self):
        """ Convert half-lives to degradation rates """

        # NOTE that not all half-life inputs are included below: 'aqueous' should not be used for 'soil'
        # soil_hl applies to the half-life of the pesticide in the soil/on the field
        # wc_metabolism_hl applies to the pesticide when it reaches the water column
        # ben_metabolism_hl applies to the pesticide that is sorbed to sediment in the benthic layer of water body
        # aq_photolysis_hl and hydrolysis_hl are abiotic degradation routes that occur when metabolism is stable
        # Following the input naming convention in fields_and_qc.csv for the 'old', here is a suggested revision:
        # for old, new in [('soil_hl', 'deg_soil'), ('wc_metabolism_hl', 'deg_wc_metabolism'),
        #                  ('ben_metabolism_hl', 'deg_ben_metabolism'), ('photolysis_hl', 'aq_photolysis'),
        #                 ('hydrolysis_hl', 'hydrolysis')] - NT: 8/28/18

        def adjust(x):
            return 0.693 / x if x else np.inf  # half-life of 'zero' indicates stability

        for old, new in [('aqueous', 'soil'), ('photolysis', 'aq_photolysis'),
                         ('hydrolysis', 'hydrolysis'), ('wc', 'wc_metabolism')]:
            setattr(self, 'deg_{}'.format(old), adjust(getattr(self, f"{new}_hl")))

        self.applications.rate *= 0.0001  # convert kg/ha -> kg/m2 (1 ha = 10,000 m2)

    def read_applications(self):
        applications = pd.DataFrame(self.applications, columns=fields.fetch('applications'))

        # Create a float array of the applications table for faster use
        for i, row in applications.iterrows():
            applications.loc[i, 'event'] = ['plant', 'harvest', 'maxcover', 'harvest'].index(row.event)
            applications.loc[i, 'dist'] = ['canopy', 'ground'].index(row.dist)
            applications.loc[i, 'method'] = ['uniform', 'step'].index(row.method)
        return applications.astype(np.float32)

    def read_endpoints(self):
        endpoints = pd.DataFrame(self.endpoints.T, columns=('acute_tox', 'chronic_tox', 'overall_tox'))
        return pd.concat([endpoint_format, endpoints], axis=1)

    def processing_extent(self):
        """ Determine which NHD regions need to be run to process the specified reacches """

        assert self.sim_type in ('eco', 'drinking_water', 'manual'), \
            "Invalid simulation type '{}'".format(self.sim_type)

        # Get the path of the table used for intakes
        intake_file = {'drinking_water': dwi_path, 'manual': manual_points_path}.get(self.sim_type)

        # Get reaches and regions to be processed if not running eco
        # Intake files must contain 'comid' and 'region' fields
        if intake_file is not None:
            intakes = pd.read_csv(intake_file)
            intakes['region'] = [str(r).zfill(2) for r in intakes.region]
            run_regions = sorted(np.unique(intakes.region))
            intake_reaches = sorted(np.unique(intakes.comid))
        else:  # Run everything if running eco
            intake_reaches = None
            run_regions = nhd_regions

        return intakes, run_regions, intake_reaches


class ModelOutputs(DateManager):
    def __init__(self, i, output_reaches, start_date, end_date):
        self.input = i
        self.output_dir = os.path.join(output_path, i.token)
        self.output_reaches = output_reaches
        DateManager.__init__(self, start_date, end_date)

        # Initialize output JSON dict
        self.out_json = {}

        # Initialize output matrices
        self.output_fields = fields.fetch("time_series_compact" if compact_out else "time_series")
        self.time_series = MemoryMatrix([self.output_reaches, self.output_fields, self.n_dates],
                                        name='output time series')

        # Initialize exceedances matrix: the probability that concentration exceeds endpoint thresholds
        self.exceedances = MemoryMatrix([self.output_reaches, self.input.endpoints.shape[0], 3], name='exceedance')

        # Initialize contributions matrix: loading data broken down by crop and runoff v. erosion source
        self.contributions = MemoryMatrix([2, self.output_reaches, self.input.crops], name='contributions')
        self.contributions.columns = self.input.crops
        self.contributions.header = ["cls" + str(c) for c in self.contributions.columns]

    def update_contributions(self, recipe_id, scenario_names, loads):
        """ Sum the total contribution by land cover class and add to running total """
        classes = [int(re.match('s[A-Za-z\d]{10,12}w\d{2,8}lc(\d+?)', name).group(1)) for name in scenario_names]
        contributions = np.zeros((2, 255))
        for i in range(2):  # Runoff Mass, Erosion Mass
            contributions[i] += np.bincount(classes, weights=loads[i], minlength=255)

        self.contributions.update(recipe_id, contributions[:, self.contributions.columns])

    def update_exceedances(self, recipe_id, concentration):

        # Extract exceedance durations and corresponding thresholds from endpoints table
        durations = \
            np.int16(self.input.endpoints[['acute_duration', 'chronic_duration', 'overall_duration']].values)
        thresholds = \
            np.int16(self.input.endpoints[['acute_tox', 'chronic_tox', 'overall_tox']].values)

        # Calculate excedance probabilities
        exceed = exceedance_probability(concentration, durations.flatten(), thresholds.flatten(), self.year_index)

        self.exceedances.update(recipe_id, exceed.reshape(durations.shape))

    def update_time_series(self, recipe_id, update_rows):
        new_rows = np.vstack(update_rows)
        self.time_series.update(recipe_id, new_rows)

    def write_json(self, write_exceedances=False, write_contributions=False):

        json.encoder.FLOAT_REPR = lambda o: format(o, '.4f')
        out_file = os.path.join(self.output_dir, "{}_json.csv".format(self.input.chemical_name))
        out_json = {"COMID": {}}
        for recipe_id in self.output_reaches:
            out_json["COMID"][str(recipe_id)] = {}
            if write_exceedances:
                labels = ["{}_{}".format(species, level)
                          for species in self.input.endpoints.species for level in ('acute', 'chronic', 'overall')]
                exceedance_dict = dict(zip(labels, np.float64(self.exceedances.fetch(recipe_id)).flatten()))
                out_json["COMID"][str(recipe_id)].update(exceedance_dict)
            if write_contributions:
                contributions = self.contributions.fetch(recipe_id)
                for i, category in enumerate(("runoff", "erosion")):
                    labels = ["{}_load_{}".format(category, label) for label in self.contributions.header]
                    contribution_dict = dict(zip(labels, np.float64(contributions[i])))
                    out_json["COMID"][str(recipe_id)].update(contribution_dict)

        out_json = json.dumps(dict(out_json), sort_keys=True, indent=4, separators=(',', ': '))
        with open(out_file, 'w') as f:
            f.write(out_json)

    def write_output(self):

        # Create output directory
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Write JSON output
        self.write_json(output_params.write_exceedances, output_params.write_contributions)

        # Write time series
        if self.write_time_series:
            self.write_time_series()

    def write_time_series(self):
        for recipe_id in self.output_reaches:
            out_file = os.path.join(self.output_dir, "time_series_{}.csv".format(recipe_id))
            out_data = self.time_series.fetch(recipe_id).T
            df = pd.DataFrame(data=out_data, index=self.dates, columns=self.output_fields)
            df.to_csv(out_file)


class StageOneScenarios(object):
    def __init__(self, region, subset_outlets=None, subset_year=None, recipes=None):
        self.region = region
        self._subset = False
        self._scenario_id = None
        self.scratch_path = os.path.join(scratch_path, f"s1_subset.csv")

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
            for chunk in pd.read_csv(path, chunksize=stage_one_chunksize):
                chunk = self.modify_array(chunk)
                for weather_grid, scenarios in chunk.groupby('weather_grid'):
                    yield weather_grid, scenarios

    @property
    def names(self):
        if self._scenario_id is None:
            self._scenario_id = self.fetch('scenario_id', True)
        return self._scenario_id

    @staticmethod
    def modify_array(array):
        # TODO - can we clean this up? what needs to be here vs in scenarios project?

        for field, val in scenario_defaults.items():
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
                path = stage_one_scenario_path.format(self.region, i)
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
    def __init__(self, region, met=None, scenario_index=None, sim=None, tag=None):
        self.path = stage_two_scenario_path.format(region)
        if tag is not None:
            self.path += f"_{tag}"
        self.keyfile_path = self.path + "_key.txt"
        self.array_path = self.path + "_arrays.dat"
        self.index_path = self.path + "_index.csv"
        self.sim = sim
        self.met = met

        build = met is not None and scenario_index is not None and sim is None
        # If build is True, create the Stage 2 Scenarios by running model routines on Stage 1 scenario inputs
        if build:
            self.arrays = fields.fetch('s2_arrays')
            DateManager.__init__(self, scenario_start_date, scenario_end_date)
            self.align_met_dates()
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

    def clip(self, arrays):
        return arrays[:, self.runoff_erosion, self.start_offset:-self.end_offset]

    def fetch(self, index, copy=False, verbose=False, iloc=False, pop=False, return_alias=False):
        result = super(StageTwoScenarios, self).fetch(index, copy, iloc, pop, return_alias)
        return result[:, self.start_offset:-self.end_offset]

    def fetch_from_recipe(self, recipe, verbose=True):
        found = recipe.join(self.lookup)
        arrays = super(StageTwoScenarios, self).fetch(found.s2_index, verbose=verbose)
        arrays = self.clip(arrays)
        return arrays, found.dropna()

    def load_key(self):
        with open(self.keyfile_path) as f:
            time_series = next(f).strip().split(",")
            start_date = np.datetime64(next(f).strip())
            time_series_shape = [int(val) for val in next(f).strip().split(",")]
        return time_series, start_date, time_series_shape

    def write(self, batch_num, data):
        d = np.array(data)
        if batch_num == 'index':
            data.to_csv(self.index_path, index=None)
        else:
            batch_size_actual = len(data)
            start_pos = (batch_num - 1) * batch_size
            self.writer[start_pos:start_pos + batch_size_actual] = np.array(data)


class StageThreeScenarios(DateManager, MemoryMatrix):
    def __init__(self, inputs, stage_one, stage_two):
        self.s1 = stage_one
        self.s2 = stage_two
        self.i = inputs
        self.scenario_vars, self.lookup = self.select_scenarios(self.i.crops)

        # Set dates
        DateManager.__init__(self, stage_two.start_date, stage_two.end_date)

        # Initialize memory matrix
        # arrays - runoff_mass, erosion_mass
        MemoryMatrix.__init__(self, [len(self.scenario_vars.s3_index), 2, self.n_dates], name='pesticide mass',
                              persistent_read=True, persistent_write=True)

    def select_scenarios(self, crops):
        """ Use the Stage Two Scenarios index, but filter by crops """
        crops = pd.DataFrame({crop_group_field: list(crops)}, dtype=np.int32)
        selected = self.s2.scenario_vars.merge(crops, on=crop_group_field, how='inner')
        selected['s3_index'] = np.arange(selected.shape[0])
        scenario_vars = selected.sort_values('s3_index')
        # JCH - for diagnostic purposes, 'scenario_id' can be added to the lookup table
        lookup = scenario_vars[['scenario_index', 's2_index', 's3_index']].set_index('scenario_index')
        return scenario_vars, lookup

    def build_from_stage_two(self):
        # TODO - can the dask allocation part of this be put into a function or wrapper?
        #  it's also used in s1->s2
        # Reminder: array_matrix.shape, processed_matrix.shape = (scenario, variable, date)
        var_table = self.scenario_vars.set_index('s2_index')

        batch = []
        batch_count = 0
        n_scenarios = var_table.shape[0]

        # Iterate scenarios
        for count, (s2_index, s3_index) in enumerate(self.lookup[['s2_index', 's3_index']].values):
            if not (count + 1) % 100:
                report(f"Processed {count + 1} of {n_scenarios} scenarios...", 1)

            # Get the non-array values associated with the scenario
            s = var_table.loc[s2_index]

            # Extract stored data
            runoff, erosion, leaching, soil_water, rain = self.s2.fetch(s2_index, iloc=True)

            # Get application information for the active crop
            crop_applications = self.i.applications[self.i.applications.crop == s[crop_group_field]]

            if not crop_applications.empty:

                # Get crop ID of scenario and find all associated crops in group
                scenario = \
                    stage_two_to_three(crop_applications.values,
                                       self.new_year, self.i.kd_flag, self.i.koc, self.i.deg_aqueous,
                                       leaching, runoff, erosion, soil_water, rain,
                                       soil_params.cm_2, soil_params.surface_dx, soil_params.erosion_effic,
                                       soil_params.soil_depth, plant_params.deg_foliar, plant_params.washoff_coeff,
                                       soil_params.runoff_effic, s.plant_date, s.emergence_date, s.maxcover_date,
                                       s.harvest_date, s.max_canopy, s.orgC_5, s.bd_5, s.season)

                batch.append(scenario)
                if len(batch) == batch_size or (count + 1) == n_scenarios:
                    try:
                        arrays = dask.delayed()(batch).compute()
                        start_pos = batch_count * batch_size
                        self.writer[start_pos:start_pos + len(batch)] = arrays
                        batch_count += 1
                    except Exception as e:
                        raise e
                        report(e)
                    batch = []

    def fetch_from_recipe(self, recipe, verbose=False):

        found = recipe.join(self.lookup, how='inner')
        arrays = super(StageThreeScenarios, self).fetch(found.s3_index, verbose=verbose)
        return arrays, found.dropna()


class ReachManager(DateManager, MemoryMatrix):
    def __init__(self, s2_scenarios, s3_scenarios, recipes, region, output, progress_interval=10000):
        self.output = output
        self.s2 = s2_scenarios
        self.s3 = s3_scenarios  # stage 3
        self.recipes = recipes
        self.region = region
        self.progress_interval = progress_interval

        # Initialize dates
        DateManager.__init__(self, s3_scenarios.start_date, s3_scenarios.end_date)

        # Initialize a matrix to store time series data for reaches (crunched scenarios)
        # vars: runoff, runoff_mass, erosion, erosion_mass
        MemoryMatrix.__init__(self, [region.active_reaches, 4, self.n_dates], name='reach manager')

        # Keep track of which reaches have been run
        self.burned_reaches = set()  # reaches that have been processed

    def process_local(self, reach_id, year, verbose=False):
        """  Fetch all scenarios and multiply by area. For erosion, area is adjusted. """

        def weight_and_combine(time_series, areas):
            areas = areas.values
            time_series = np.moveaxis(time_series, 0, 2)  # (scenarios, vars, dates) -> (vars, dates, scenarios)
            time_series[0] *= areas
            time_series[1] *= np.power(areas / 10000., .12)
            return time_series.sum(axis=2)

        # JCH - this pulls up a table of ['scenario_index', 'area'] index is used here to keep recipe files small
        recipe = self.recipes.fetch(reach_id, year)  # recipe is indexed by scenario_index
        if not recipe.empty:
            # Pull runoff and erosion from Stage 2 Scenarios
            transport, found_s2 = self.s2.fetch_from_recipe(recipe)
            runoff, erosion = weight_and_combine(transport, found_s2.area)

            # Pull chemical mass from Stage 3 scenarios
            pesticide_mass, found_s3 = self.s3.fetch_from_recipe(recipe, verbose=False)
            runoff_mass, erosion_mass = weight_and_combine(pesticide_mass, found_s3.area)
            out_array = np.array([runoff, runoff_mass, erosion, erosion_mass])
            self.update(reach_id, out_array)

            # Assess the contributions to the recipe from ach source (runoff/erosion) and crop
            # self.o.update_contributions(recipe_id, scenarios, time_series[[1, 3]].sum(axis=1))

        elif verbose:
            report("No scenarios found for {}".format(reach_id))

    def report(self, reach_id):
        # Get flow values for reach
        flow = self.region.daily_flows(reach_id)

        # Get local runoff, erosion, and pesticide masses
        local_runoff, local_runoff_mass, local_erosion, local_erosion_mass = self.fetch(reach_id)

        # Process upstream contributions
        upstream_runoff, upstream_runoff_mass = self.upstream_loading(reach_id)

        # Compute concentrations
        surface_area = self.region.flow_table(reach_id)['surface_area']
        total_flow, (concentration, runoff_conc) = \
            compute_concentration(upstream_runoff_mass, upstream_runoff, self.n_dates, flow)
        benthic_conc = partition_benthic(local_erosion, local_erosion_mass, surface_area)

        # Store results in output array
        self.output.update_exceedances(reach_id, concentration)
        self.output.update_time_series(reach_id, [total_flow, upstream_runoff, upstream_runoff_mass,
                                                  concentration, benthic_conc])

    def upstream_loading(self, reach_id):
        """ Identify all upstream reaches, pull data and offset in time """

        # Fetch all upstream reaches and corresponding travel times
        upstream_reaches, travel_times, warning = \
            self.region.upstream_watershed(reach_id, return_times=True, return_warning=True)

        # Filter out reaches (and corresponding times) that have already been burned
        indices = np.int16([i for i, r in enumerate(upstream_reaches) if r not in self.burned_reaches])
        reaches, reach_times = upstream_reaches[indices], travel_times[indices]

        # Don't need to do proceed if there's nothing upstream
        if len(reaches) > 1:

            # Initialize the output array
            totals = np.zeros((4, self.n_dates))  # (mass/runoff, dates)

            # Fetch time series data for each upstream reach
            reach_array, found_reaches = self.fetch(reaches, verbose=True, return_alias=True)  # (reaches, vars, dates)

            # Stagger time series by dayshed
            for tank in range(np.max(reach_times) + 1):
                tank_array = reach_array[reach_times == tank]
                in_tank = reach_array[reach_times == tank].sum(axis=0)
                if tank > 0:
                    if hydrology_params.gamma_convolve:
                        irf = self.region.irf.fetch(tank)  # Get the convolution function
                        in_tank[0] = np.convolve(in_tank[0], irf)[:self.n_dates]  # mass
                        in_tank[1] = np.convolve(in_tank[1], irf)[:self.n_dates]  # runoff
                    else:
                        in_tank = np.pad(in_tank[:, :-tank], ((0, 0), (tank, 0)), mode='constant')
                totals += in_tank  # Add the convolved tank time series to the total for the reach

            runoff, runoff_mass, erosion, erosion_mass = totals
        else:
            result = self.fetch(reach_id)
            runoff, runoff_mass, erosion, erosion_mass = result

        # TODO - erosion mass here?
        return runoff, runoff_mass

    def burn(self, lake):

        irf = ImpulseResponseMatrix.generate(1, lake.residence_time, self.n_dates)

        # Get the convolution function
        # Get mass and runoff for the reach
        total_mass, total_runoff = self.upstream_loading(lake.outlet_comid)

        # Modify combined time series to reflect lake
        new_mass = np.convolve(total_mass, irf)[:self.n_dates]
        if hydrology_params.convolve_runoff:  # Convolve runoff
            new_runoff = np.convolve(total_runoff, irf)[:self.n_dates]
        else:  # Flatten runoff
            new_runoff = np.repeat(np.mean(total_runoff), self.n_dates)

        # Retain old erosion numbers
        _, _, erosion, erosion_mass = self.fetch(lake.outlet_comid)

        # Add all lake mass and runoff to outlet
        self.update(lake.outlet_comid, np.array([new_runoff, new_mass, erosion, erosion_mass]))


class WatershedRecipes(object):
    def __init__(self, region):
        self.path = recipe_path.format(region)

        # Read shape
        with open(f"{self.path}_key.txt") as f:
            self.shape = literal_eval(next(f))

        # Read lookup map
        self.map = pd.read_csv(f"{self.path}_map.csv")

    def fetch(self, reach_id, year='all'):
        # TODO - 'all' doesn't work yet
        start, end = self.lookup(reach_id, year)
        if all((start, end)):
            fp = np.memmap(f"{self.path}", dtype=np.int64, mode='r', shape=self.shape)
            result = fp[start:end]
        else:
            result = None
        return pd.DataFrame(result, columns=['scenario_index', 'area']).set_index('scenario_index')

    def lookup(self, reach_id, year, verbose=False):
        if year != 'all':
            result = self.map[(self.map.comid == reach_id) & (self.map.year == year)]
        else:
            result = self.map[(self.map.comid == reach_id)]
        if result.shape[0] < 1:
            if verbose:
                report(f"No recipes found for {reach_id}, {year}", 2)
            return None, None
        elif result.shape[0] > 1:
            raise KeyError(f"Lookup error for {reach_id}, {year}")
        else:
            return result.iloc[0][['start', 'end']].values


class WeatherArray(MemoryMatrix, DateManager):
    def __init__(self):
        # TODO - sync this class with opp-efed-weather
        array_path = weather_path.format('array.dat')
        key_path = weather_path.format('key.npz')

        # Set row/column offsets
        index, header, start_date, end_date = self.load_key(key_path)

        # Set dates
        # TODO - this could be an intrinsic function of the date manager class
        self.start_offset = 0
        self.end_offset = 0
        DateManager.__init__(self, start_date, end_date)

        # Initialize memory matrix
        MemoryMatrix.__init__(self, [index, self.n_dates, header], np.float32, array_path, True)

    @staticmethod
    def load_key(key_path):
        key = np.load(key_path)
        points, years, header = key['points'], key['years'], key['header']
        start_date = np.datetime64(f"{years[0]}-01-01")
        end_date = np.datetime64(f"{years[-1]}-12-31")
        return points.T[0], header, start_date, end_date

    def fetch_station(self, station_id):
        if self.end_offset == 0:
            self.end_offset = self.n_dates
        data = self.fetch(station_id, copy=True, verbose=True).T
        data[:2] /= 100.  # Precip, PET  cm -> m
        return data[:, self.start_offset:self.end_offset]



def report(message, tabs=0):
    """ Display a message with a specified indentation """
    tabs = "\t" * tabs
    print(tabs + str(message))
