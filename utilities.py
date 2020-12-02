import os
import re
import math
import numpy as np
import pandas as pd
import json
from ast import literal_eval
from distributed import Client

from .tools.efed_lib import FieldManager, MemoryMatrix, DateManager, report
from .parameters import ParameterManager
from .paths import PathManager
from .hydro.params_nhd import nhd_regions
from .hydro.navigator import Navigator
from .hydro.process_nhd import identify_waterbody_outlets, calculate_surface_area
from .aquatic_concentration import compute_concentration, partition_benthic, exceedance_probability

# If true, only return 5 time series fields, otherwise, return 13
# TODO - there's a better way to do this
compact_out = True


class HydroRegion(Navigator):
    """
    Contains all datasets and functions related to the NHD Plus region, including all hydrological features and links
    outlet_reaches: reaches with an outlet (used to delineate)
    active_reaches: reaches upstream of the outlet (or for eco, all of them)
    """

    def __init__(self, region, sim):

        self.id = region

        # Assign a watershed navigator to the class
        # TODO - a path should be provided here
        super(HydroRegion, self).__init__(self.id)

        # Read hydrological input files
        self.reach_table = pd.read_csv(sim.paths.condensed_nhd.format('sam', region, 'reach'))
        self.lake_table = pd.read_csv(sim.paths.condensed_nhd.format('sam', region, 'waterbody'))
        self.process_nhd()

        # Initialize the fields that will be used to pull flows based on month
        self.flow_fields = [f'q_{str(month).zfill(2)}' for month in sim.month_index]

        # Select which stream reaches will be fully processed, partially processed, or excluded
        self.partial_reaches, self.full_reaches, self.reservoir_outlets = \
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
            partial = full = self.reach_table.comid
        else:
            partial = self.confine(intakes)
            if intakes_only:
                full = intakes
            else:
                full = partial
        reservoir_outlets = \
            self.lake_table.loc[np.in1d(self.lake_table.outlet_comid, partial)][['outlet_comid', 'wb_comid']]

        return partial, full, reservoir_outlets

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
            reaches &= set(self.partial_reaches)
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

    def __init__(self, input_json):

        # Get paths for model resources
        self.paths = PathManager()
        self.local_run = self.paths.local_run

        # Initialize field manager
        self.fields = FieldManager(self.paths.fields_and_qc)

        # Read the inputs supplied by the user in the GUI
        input_dict = InputDict(input_json, self.paths.endpoint_format, self.fields)
        self.__dict__.update(input_dict)

        # Read the hardwired parameters
        external_params = ParameterManager()
        self.__dict__.update(external_params.__dict__)

        # Initialize file structure
        self.check_directories()

        # Initialize Dask client
        if self.local_run:
            self.dask_client = Client(processes=False)
        else:
            dask_scheduler = os.environ.get("DASK_SCHEDULER")
            self.dask_client = Client(dask_scheduler)

        # Initialize dates
        DateManager.__init__(self, np.datetime64(self.sim_date_start), np.datetime64(self.sim_date_end))

        # Select regions and reaches that will be run
        self.intakes, self.run_regions, self.intake_reaches = self.processing_extent()

        # Initialize an impulse response matrix if convolving time of travel
        self.irf = None if not self.hydrology.gamma_convolve else ImpulseResponseMatrix(self.dates.size)

        # Make numerical adjustments (units etc)
        self.adjust_data()

        # Read token
        self.token = \
            self.simulation_name if not hasattr(self, 'csrfmiddlewaretoken') else self.csrfmiddlewaretoken

    def check_directories(self):
        # Make sure needed subdirectories exist
        for subdir in self.paths.scratch_path, self.paths.output_path:
            if not os.path.exists(subdir):
                os.makedirs(subdir)

        # Purge temp folder
        for f in os.listdir(self.paths.scratch_path):
            os.remove(os.path.join(self.paths.scratch_path, f))

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

        self.applications.apprate *= 0.0001  # convert kg/ha -> kg/m2 (1 ha = 10,000 m2)

    def processing_extent(self):
        """ Determine which NHD regions need to be run to process the specified reacches """

        assert self.sim_type in ('eco', 'drinking_water', 'manual'), \
            "Invalid simulation type '{}'".format(self.sim_type)

        # Get the path of the table used for intakes
        # TODO - there's an issue here. I have to manually assign 'manual' as a value for now and 'eco' isn't working
        self.sim_type = 'manual'
        intake_file = {'drinking_water': self.paths.dw_intakes,
                       'manual': self.paths.manual_intakes}.get(self.sim_type)
        self.intakes_only = (self.sim_type != 'eco')
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


class InputDict(dict):
    """ Processes the input string from the front end into a form usable by tool """

    # TODO - combine this into Simulation?  clean it up
    def __init__(self, pd_obj, endpoint_format_path, fields):

        # Unpack JSON string into dictionary
        # TODO - Taking a dict at the moment, will need json
        super(InputDict, self).__init__((k, v['0']) for k, v in pd_obj.items())
        self.fields = fields
        self.endpoint_format = pd.read_csv(endpoint_format_path)
        self['applications'] = self.process_applications()
        self['endpoints'] = self.process_endpoints()
        self['sim_date_start'], self['sim_date_end'] = self.process_dates()
        self['crops'] = set(self['applications'].crop)

        self.coerce_data_type()
        self.check_fields()

    def check_fields(self):

        # Check if any required input data are missing or extraneous data are provided
        provided_fields = set(self.keys())
        required_fields = set(self.fields.fetch('input_param')) | {'applications', 'endpoints'}
        unknown_fields = provided_fields - required_fields
        missing_fields = required_fields - provided_fields
        if unknown_fields:
            report("Input field(s) \"{}\" not understood".format(", ".join(unknown_fields)))
        assert not missing_fields, "Required input field(s) \"{}\" not provided".format(", ".join(missing_fields))

    def coerce_data_type(self):
        _, data_types = self.fields.fetch('input_param', dtypes=True)
        for field, data_type in data_types.items():
            if data_type != object:
                self[field] = data_type(self[field])

    def process_applications(self):
        # TODO - I'm sure this can be cleaned up drastically
        # Get fields and field types
        app_fields, data_types = self.fields.fetch('applications', dtypes=True)

        # Populate matrix
        matrix = []
        for app_num in range(1, int(self['napps']) + 1):
            crops = self[f"crop_{app_num}"].split(" ")
            for crop in crops:
                row = []
                for field in app_fields:
                    if field == 'crop':
                        val = crop
                    else:
                        val = self[f"{field}_{app_num}"]
                    dtype = data_types[field]
                    if dtype != object:
                        val = dtype(val)
                    row.append(val)
                matrix.append(row)
            for field in app_fields:
                del self[f"{field}_{app_num}"]

        applications = pd.DataFrame(matrix, columns=self.fields.fetch('applications'))

        # Create a float array of the applications table for faster use
        for i, row in applications.iterrows():
            applications.loc[i, 'event'] = ['plant', 'harvest', 'maxcover', 'harvest'].index(row.event)
            applications.loc[i, 'dist'] = ['canopy', 'ground'].index(row.dist)
            applications.loc[i, 'method'] = ['uniform', 'step'].index(row.method)
        return applications.astype(np.float32)

    def process_dates(self):
        date_format = lambda x: np.datetime64("{2}-{0}-{1}".format(*x.split("/")))
        return map(date_format, (self['sim_date_start'], self['sim_date_end']))

    def process_endpoints(self):
        # TODO - this can probably be cleaned up
        endpoints = []
        for level in ('acute', 'chronic', 'overall'):
            endpoints.append([self.pop("{}_{}".format(level, species), np.nan)
                              for species in self.endpoint_format.species])
        endpoints = np.array(endpoints)
        endpoints[endpoints == ''] = np.nan
        endpoints = pd.DataFrame(endpoints.T, columns=('acute_tox', 'chronic_tox', 'overall_tox'))
        return pd.concat([self.endpoint_format, endpoints], axis=1)


class ModelOutputs(DateManager):
    def __init__(self, sim, region):
        self.sim = sim
        self.output_dir = os.path.join(sim.paths.output_path, sim.token)
        self.partial_reaches = region.partial_reaches
        self.full_reaches = region.full_reaches
        self.array_path = os.path.join(sim.paths.scratch_path, "model_out")
        DateManager.__init__(self, sim.start_date, sim.end_date)

        # Initialize output JSON dict
        self.json_output = {}

        # Initialize time series output array
        self.output_fields = sim.fields.fetch("time_series_compact" if compact_out else "time_series")
        self.time_series = MemoryMatrix([self.full_reaches, self.output_fields, self.n_dates],
                                        name='output time series', path=self.array_path + "_ts")

        # Initialize exceedances matrix: the probability that concentration exceeds endpoint thresholds
        self.exceedances = MemoryMatrix([self.partial_reaches, self.sim.endpoints.shape[0], 3], name='exceedance',
                                        path=self.array_path + "_ex")

        # Initialize contributions matrix: loading data broken down by crop and runoff v. erosion source
        self.contributions = MemoryMatrix([2, self.partial_reaches, self.sim.crops], name='contributions',
                                          path=self.array_path + "_cn")
        self.contributions.columns = self.sim.crops
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
            np.int16(self.sim.endpoints[['acute_duration', 'chronic_duration', 'overall_duration']].values)
        thresholds = \
            np.float32(self.sim.endpoints[['acute_tox', 'chronic_tox', 'overall_tox']].values)

        # Calculate excedance probabilities
        exceed = exceedance_probability(concentration, durations.flatten(), thresholds.flatten(), self.year_index)

        self.exceedances.update(recipe_id, exceed.reshape(durations.shape))

    def update_time_series(self, recipe_id, update_rows):
        new_rows = np.vstack(update_rows)
        self.time_series.update(recipe_id, new_rows)

    def write_json(self, write_exceedances=False, write_contributions=False):

        json.encoder.FLOAT_REPR = lambda o: format(o, '.4f')
        out_file = os.path.join(self.output_dir, "{}_json.csv".format(self.sim.chemical_name))
        self.json_output = {"COMID": {}}
        for recipe_id in self.partial_reaches:
            self.json_output["COMID"][str(recipe_id)] = {}
            if write_exceedances:
                labels = ["{}_{}".format(species, level)
                          for species in self.sim.endpoints.species for level in ('acute', 'chronic', 'overall')]
                exceedance_dict = dict(zip(labels, np.float64(self.exceedances.fetch(recipe_id)).flatten()))
                self.json_output["COMID"][str(recipe_id)].update(exceedance_dict)
            if write_contributions:
                contributions = self.contributions.fetch(recipe_id)
                for i, category in enumerate(("runoff", "erosion")):
                    labels = ["{}_load_{}".format(category, label) for label in self.contributions.header]
                    contribution_dict = dict(zip(labels, np.float64(contributions[i])))
                    self.json_output["COMID"][str(recipe_id)].update(contribution_dict)

        #self.json_output = json.dumps(dict(self.json_output), sort_keys=True, indent=4, separators=(',', ': '))
        with open(out_file, 'w') as f:
            f.write(str(self.json_output))

    def write_output(self):

        # Create output directory
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Write JSON output
        self.write_json(self.sim.output.write_exceedances, self.sim.output.write_contributions)

        # Write time series
        if self.write_time_series:
            self.write_time_series()

    def write_time_series(self):
        for recipe_id in self.full_reaches:
            out_file = os.path.join(self.output_dir, "time_series_{}.csv".format(recipe_id))
            out_data = self.time_series.fetch(recipe_id).T
            df = pd.DataFrame(data=out_data, index=self.dates, columns=self.output_fields)
            df.to_csv(out_file)


class ReachManager(DateManager, MemoryMatrix):
    def __init__(self, sim, s2_scenarios, s3_scenarios, recipes, region, output, progress_interval=10000):
        self.output = output
        self.sim = sim
        self.s2 = s2_scenarios
        self.s3 = s3_scenarios  # stage 3
        self.recipes = recipes
        self.region = region
        self.progress_interval = progress_interval
        self.array_path = os.path.join(sim.paths.scratch_path, "_reach_mgr{}".format(self.s2.region))
        self.i = sim

        # Initialize dates
        DateManager.__init__(self, s3_scenarios.start_date, s3_scenarios.end_date)

        # Initialize a matrix to store time series data for reaches (crunched scenarios)
        # vars: runoff, runoff_mass, erosion, erosion_mass
        MemoryMatrix.__init__(self, [region.active_reaches, 4, self.n_dates], name='reach manager',
                              path=self.array_path)

        # Keep track of which reaches have been run
        self.burned_reaches = set()  # reaches that have been processed

    def burn(self, lake):

        irf = ImpulseResponseMatrix.generate(1, lake.residence_time, self.n_dates)

        # Get the convolution function
        # Get mass and runoff for the reach
        total_mass, total_runoff = self.upstream_loading(lake.outlet_comid)

        # Modify combined time series to reflect lake
        new_mass = np.convolve(total_mass, irf)[:self.n_dates]
        if self.sim.hydrology.convolve_runoff:  # Convolve runoff
            new_runoff = np.convolve(total_runoff, irf)[:self.n_dates]
        else:  # Flatten runoff
            new_runoff = np.repeat(np.mean(total_runoff), self.n_dates)

        # Retain old erosion numbers
        _, _, erosion, erosion_mass = self.fetch(lake.outlet_comid)

        # Add all lake mass and runoff to outlet
        self.update(lake.outlet_comid, np.array([new_runoff, new_mass, erosion, erosion_mass]))

    def burn_batch(self, lakes):
        if True or self.sim.local_run:
            for _, lake in lakes.iterrows():
                self.burn(lake)
        else:
            # TODO - this doesn't work yet. Dask doesn't like async calls to class functions
            batch = []
            for _, lake in lakes.iterrows():
                batch.append(self.sim.dask_client.submit(self.burn, lake))
            self.sim.dask_client.gather(batch)

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

    def process_local_batch(self, reach_ids, year):
        if True or self.sim.local_run:
            for reach_id in reach_ids:
                self.process_local(reach_id, year)
        else:
            # TODO - this doesn't work yet. Dask doesn't seem to like the async call on a self. function
            batch = []
            for reach_id in reach_ids:
                batch.append(self.sim.dask_client.submit(self.process_local, reach_id, year))
            self.sim.dask_client.gather(batch)

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
        benthic_conc = partition_benthic(local_erosion, local_erosion_mass, surface_area,
                                         self.sim.benthic.depth, self.sim.benthic.porosity)

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
                in_tank = reach_array[reach_times == tank].sum(axis=0)
                if tank > 0:
                    if self.sim.hydrology.gamma_convolve:
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


class WatershedRecipes(object):
    def __init__(self, region, sim):
        self.path = sim.paths.recipes.format(region)

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
    def __init__(self, sim):
        # TODO - sync this class with opp-efed-weather
        array_path = sim.paths.weather.format('array.dat')
        key_path = sim.paths.weather.format('key.npz')

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
