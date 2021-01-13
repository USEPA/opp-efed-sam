import os
import re
import math
import numpy as np
import pandas as pd
import json
import sys
import pathlib
from ast import literal_eval
from distributed import Client

from .tools.efed_lib import FieldManager, MemoryMatrix, DateManager, report
from .hydro.params_nhd import nhd_regions
from .hydro.navigator import Navigator
from .hydro.process_nhd import identify_waterbody_outlets, calculate_surface_area


class HydroRegion(Navigator):
    """
    Contains all datasets and functions related to the NHD Plus region, including all hydrological features and links
    """

    def __init__(self, region, sim):

        self.id = region

        # Assign a watershed navigator to the class
        # TODO - a path should be provided here
        super(HydroRegion, self).__init__(sim.navigator_path.format(self.id))

        # Read hydrological input files
        self.reach_table = pd.read_csv(sim.condensed_nhd_path.format('sam', region, 'reach'))
        self.lake_table = pd.read_csv(sim.condensed_nhd_path.format('sam', region, 'waterbody'))
        self.huc_crosswalk = pd.read_csv(sim.nhd_wbd_xwalk_path, dtype=object)[['FEATUREID', 'HUC_12']] \
            .rename(columns={'FEATUREID': 'comid'})

        self.process_nhd()

        # Initialize the fields that will be used to pull flows based on month
        self.flow_fields = [f'q_{str(month).zfill(2)}' for month in sim.month_index]

        # Select which stream reaches will be fully processed, locally processed, or excluded
        self.local_reaches, self.full_reaches, self.reservoir_outlets = \
            self.sort_reaches(sim.intake_reaches, sim.intakes_only)

        # Holder for reaches that have been processed
        self.burned_reaches = set()

    def sort_reaches(self, intakes, intakes_only):
        """
        intakes - reaches corresponding to an intake
        local - all reaches upstream of an intake
        full - reaches for which a full suite of outputs is computed
        intakes_only - do we do the full monty for the intakes only, or all upstream?
        lake_outlets - reaches that correspond to the outlet of a lake
        """

        # Confine to available reaches and assess what's missing
        if intakes is None:
            local = full = self.reach_table.lookup.values
        else:
            local = self.confine(intakes)
            if intakes_only:  # eco mode but intakes provided - not a situation that happens yet
                full = intakes
            else:
                full = local
        reservoir_outlets = \
            self.lake_table.loc[np.in1d(self.lake_table.outlet_comid, local)][['outlet_comid', 'wb_comid']]

        return local, full, reservoir_outlets

    def process_nhd(self):
        self.lake_table = \
            identify_waterbody_outlets(self.lake_table, self.reach_table)

        # Add HUC ids to the reach table
        self.huc_crosswalk.comid = self.huc_crosswalk.comid.astype(np.int32)
        self.reach_table = self.reach_table.merge(self.huc_crosswalk, on='comid')
        self.reach_table['HUC_8'] = self.reach_table['HUC_12'].str.slice(0, 8)

        # Calculate average surface area of a reach segment
        self.reach_table['surface_area'] = calculate_surface_area(self.reach_table)

        # Calculate residence times of reservoirs
        self.lake_table = self.lake_table.merge(self.reach_table[['comid', 'q_ma']],
                                                left_on='outlet_comid', right_on='comid', how='left')
        self.lake_table['residence_time'] = self.lake_table.wb_volume / self.lake_table.q_ma

        # Convert units
        self.reach_table['length'] = self.reach_table.pop('lengthkm') * 1000.  # km -> m
        for month in list(map(lambda x: str(x).zfill(2), range(1, 13))) + ['ma']:
            self.reach_table['q_{}'.format(month)] *= 2446.58  # cfs -> cmd
            self.reach_table['v_{}'.format(month)] *= 26334.7  # f/s -> md
        self.reach_table = self.reach_table.drop_duplicates().set_index('comid')

    @property
    def cascade(self):
        # Tier the reaches by counting the number of outlets (lakes) upstream of each lake outlet
        reach_counts = []
        lake_outlets = set(self.reservoir_outlets.outlet_comid)
        for outlet in lake_outlets:
            upstream_lakes = len((set(self.upstream_watershed(outlet)) - {outlet}) & lake_outlets)
            reach_counts.append([outlet, upstream_lakes])
        reach_counts = pd.DataFrame(reach_counts, columns=['comid', 'n_upstream'])

        # Cascade downward through tiers
        upstream_outlets = set()  # outlets from previous tier
        for tier, lake_outlets in reach_counts.groupby('n_upstream')['comid']:
            lakes = self.lake_table[np.in1d(self.lake_table.outlet_comid, lake_outlets)]
            all_upstream = {reach for outlet in lake_outlets for reach in self.upstream_watershed(outlet)}
            reaches = (all_upstream - set(lake_outlets)) | upstream_outlets
            reaches &= set(self.local_reaches)
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
        # TODO - confirm that everything works as intended here, esp. for eco runs
        # Determine whether the simulation is being run on a windows desktop (local for epa devs)
        self.local_run = any([r'C:' in p for p in sys.path])

        # Add paths
        self.__dict__.update(self.initialize_paths())

        # Read the hardwired parameters
        self.__dict__.update(self.initialize_parameters())

        # Initialize field manager
        self.fields = FieldManager(self.fields_and_qc_path)

        # Read the inputs supplied by the user in the GUI
        self.__dict__.update(
            ModelInputs(input_json, self.endpoint_format_path, self.fields, self.output_selection_path))

        # Initialize file structure
        self.check_directories()

        # Initialize Dask client
        if self.local_run:
            self.dask_client = Client(processes=False)
        else:
            dask_scheduler = os.environ.get('DASK_SCHEDULER')
            self.dask_client = Client(dask_scheduler)

        # Select regions and reaches that will be run
        # If the parameter given for 'simulation_name' indicates a 'build mode' run, parse it from there.
        # This will trigger a build of Stage 2 Scenarios and not a SAM run
        self.build_scenarios, self.intakes, self.run_regions, self.intake_reaches, self.tag = \
            self.detect_build_mode()

        if not self.build_scenarios:
            self.intakes, self.run_regions, self.intake_reaches = self.processing_extent()
            # TODO - this is for testing on the mark twain basin. delete when scaling up
            if self.region == 'Mark Twain Demo':
                self.tag = 'mtb'

        # Initialize dates
        DateManager.__init__(self, *self.align_dates())

        self.intakes_only = (self.sim_type != 'eco') or self.intake_reaches is None

        # Initialize an impulse response matrix if convolving time of travel
        self.irf = None if not self.gamma_convolve else ImpulseResponseMatrix(self.dates.size)

        # Make numerical adjustments (units etc)
        self.adjust_data()

        # Read token
        self.token = \
            self.simulation_name if not hasattr(self, 'csrfmiddlewaretoken') else self.csrfmiddlewaretoken

    def align_dates(self):
        # If building scenarios, the simulation start/end dates should be the scenario start/end dates
        # Check to make sure that weather data is available for all years
        if self.build_scenarios:
            if (self.scenario_start_date < self.weather_start_date) or (
                    self.scenario_end_date > self.weather_end_date):
                raise ValueError("Scenario dates must be completely within available weather dates")
            else:
                return self.scenario_start_date, self.scenario_end_date

        # If not building scenarios, set the dates of the simulation to the period of overlap
        # between the user-specified dates and the dates of the available input data
        sim_start = np.datetime64(self.sim_date_start)
        sim_end = np.datetime64(self.sim_date_end)

        data_start = max((self.weather_start_date, self.scenario_start_date))
        data_end = min((self.weather_end_date, self.scenario_end_date))
        messages = []
        if sim_start < data_start:
            sim_start = data_start
            messages.append('start date is earlier')
        if sim_end > data_end:
            sim_end = data_end
            messages.append('end date is later')
        if any(messages):
            report(f'Simulation {" and ".join(messages)} than range of available input data. '
                   f'Date range has been truncated at {sim_start} to {sim_end}.')
        return sim_start, sim_end

    def detect_build_mode(self):
        params = self.simulation_name.lower().split("&")
        if len(params) > 1:
            build_mode = True
            regions = params[1].split(",")
        else:
            build_mode = False
            regions = None
        if len(params) > 2:
            intake_reaches = list(map(int, params[2].split(",")))
        else:
            intake_reaches = None
        if len(params) > 3:
            tag = params[3]
        else:
            tag = None
        return build_mode, None, regions, intake_reaches, tag

    def check_directories(self):
        # Make sure needed subdirectories exist
        for subdir in self.scratch_path, self.output_path:
            if not os.path.exists(subdir):
                os.makedirs(subdir)

        # Purge temp folder
        for f in os.listdir(self.scratch_path):
            os.remove(os.path.join(self.scratch_path, f))

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
            setattr(self, 'deg_{}'.format(old), adjust(getattr(self, f'{new}_hl')))

        self.applications.apprate *= 0.0001  # convert kg/ha -> kg/m2 (1 ha = 10,000 m2)
        # Build a soil profile
        self.depth_bins = np.array(eval(self.depth_bins))
        self.delta_x = np.array(
            [self.surface_dx] + [self.layer_dx] * (self.n_increments - 1))
        self.depth = np.cumsum(self.delta_x)
        self.bins = np.minimum(self.depth_bins.size - 1, np.digitize(self.depth * 100., self.depth_bins))
        self.types = pd.read_csv(self.types_path).set_index('type')

    def processing_extent(self):
        """ Determine which NHD regions need to be run to process the specified reacches """

        assert self.sim_type in ('eco', 'drinking_water', 'manual'), \
            'Invalid simulation type "{}"'.format(self.sim_type)

        # Get the path of the table used for intakes
        # TODO - there's an issue here. I have to manually assign 'manual' as a value for now and 'eco' isn't working
        self.sim_type = 'manual'
        intake_file = {'drinking_water': self.dw_intakes_path,
                       'manual': self.manual_intakes_path}.get(self.sim_type)

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

    def initialize_paths(self):
        paths = {}
        # Identify the path to the table containing all other paths
        if self.local_run:
            paths['data_root'] = r'E:/opp-efed-data/sam'
        else:
            paths['data_root'] = r'/src/app-data/sampreprocessed'
        paths['local_root'] = pathlib.Path(__file__).parent.absolute()
        paths_table = os.path.join(paths['local_root'], 'Tables', 'paths.csv')
        table = pd.read_csv(paths_table).sort_values('level')[['var', 'dir', 'base']]
        for var, dirname, basename in table.values:
            paths[f'{var}_path'] = os.path.join(paths[dirname], basename)
        return paths

    def initialize_parameters(self):
        params = pd.read_csv(self.parameters_path, usecols=['parameter', 'value', 'dtype'])
        return {k: eval(d)(v) for k, v, d in params.values}


class ModelInputs(dict):
    """ Processes the input string from the front end into a form usable by tool """

    # TODO - combine this into Simulation?  clean it up
    def __init__(self, pd_obj, endpoint_format_path, fields, output_selection_path):

        # Unpack JSON string into dictionary
        # TODO - Taking a dict at the moment, will need json
        super(ModelInputs, self).__init__((k, v['0']) for k, v in pd_obj.items())
        self.fields = fields
        self.endpoint_format = pd.read_csv(endpoint_format_path)
        self['applications'] = self.process_applications()
        self['endpoints'] = self.process_endpoints()
        self['sim_date_start'], self['sim_date_end'] = self.process_dates()
        self['selected_crops'] = set(self['applications'].crop)
        self.coerce_data_type()
        self.update(self.output_selection(output_selection_path))

    def coerce_data_type(self):
        _, data_types = self.fields.fetch('input_param', dtypes=True)
        for field, data_type in data_types.items():
            if data_type != object:
                self[field] = data_type(self[field])

    def process_applications(self):
        # TODO - I'm sure this can be cleaned up
        # Get fields and field types
        app_fields, data_types = self.fields.fetch('applications', dtypes=True)

        # Populate matrix
        matrix = []
        for app_num in np.arange(int(float(self['napps']))) + 1:
            crops = self[f'crop_{app_num}'].split(' ')
            for crop in crops:
                row = []
                for field in app_fields:
                    if field == 'crop':
                        val = crop
                    else:
                        val = self[f'{field}_{app_num}']
                    dtype = data_types[field]
                    if dtype != object:
                        val = dtype(val)
                    row.append(val)
                matrix.append(row)
            for field in app_fields:
                del self[f'{field}_{app_num}']

        applications = pd.DataFrame(matrix, columns=self.fields.fetch('applications'))

        # Create a float array of the applications table for faster use
        for i, row in applications.iterrows():
            applications.loc[i, 'event'] = ['plant', 'harvest', 'maxcover', 'harvest'].index(row.event)
            applications.loc[i, 'dist'] = ['canopy', 'ground'].index(row.dist)
            applications.loc[i, 'method'] = ['uniform', 'step'].index(row.method)

        return applications.astype(np.float32)

    def process_dates(self):
        date_format = lambda x: np.datetime64('{2}-{0}-{1}'.format(*x.split('/')))
        return map(date_format, (self['sim_date_start'], self['sim_date_end']))

    def process_endpoints(self):
        endpoints = {'long_name': [], 'short_name': [], 'threshold': [], 'duration': []}
        for _, species in self.endpoint_format.iterrows():
            for level in ('acute', 'chronic', 'overall'):
                endpoints['long_name'].append(f'{level.capitalize()} {species.long_name.lower()}')
                endpoints['short_name'].append(f'{level}_{species.species}')
                endpoints['duration'].append(species[f'{level}_duration'])
                endpoints['threshold'].append(self.pop(f'{level}_{species.species}', np.nan))
        endpoints = pd.DataFrame(endpoints).replace("", np.nan)
        endpoints['duration'] = endpoints.duration.astype(np.float32)
        endpoints['threshold'] = endpoints.threshold.astype(np.float32)

        return endpoints

    def output_selection(self, table_path):
        # TODO - this is hardwired for now - move it to the input page
        t = pd.read_csv(table_path)
        output_dict = {'local_time_series': [var for var, _, sel in t[t.table == 'local'].values if int(sel)],
                       'full_time_series': [var for var, _, sel in t[t.table == 'full'].values if int(sel)]}
        map_dict = t[t.table == 'map'].set_index('var').sel.to_dict()
        output_dict['map_reaches'] = map_dict['reaches']
        output_dict['map_intakes'] = map_dict['intakes']
        output_dict['write_map'] = bool(int(map_dict['write']))
        return output_dict


class ModelOutputs(DateManager):
    """
    A class to hold SAM outputs and postprocessing functions
    """

    def __init__(self, sim, region, s3):
        self.sim = sim
        self.local_reaches = region.local_reaches
        self.full_reaches = region.full_reaches
        self.active_crops = s3.active_crops
        self.huc_crosswalk = region.reach_table[['HUC_8', 'HUC_12']]
        self.huc_crosswalk.index = self.huc_crosswalk.index.astype(str)

        # Initialize dates
        DateManager.__init__(self, sim.start_date, sim.end_date)
        self.array_path = os.path.join(sim.scratch_path, 'r{}_{{}}_out'.format(region.id))

        # Time series data for a single catchment (run everywhere)
        self.local_time_series = MemoryMatrix(
            [region.local_reaches, sim.local_time_series, self.n_dates], name='local',
            path=self.array_path.format('local'))

        # Time series data for full watersheds (run at intakes)
        self.full_time_series = MemoryMatrix(
            [region.full_reaches, sim.full_time_series, self.n_dates], name='full',
            path=self.array_path.format('full'))

        # The relative contribution of pesticide mass broken down by reach, runoff/erosion and crop
        self.contributions_index = []
        self.contributions = np.zeros((len(self.local_reaches), 2, s3.n_active_crops))

        # The probability that concentration exceeds endpoint thresholds
        self.exceedances = pd.DataFrame(np.zeros((len(region.full_reaches), self.sim.endpoints.shape[0])),
                                        index=pd.Series(region.full_reaches, name='comid'),
                                        columns=self.sim.endpoints.short_name)

    def contributions_by_huc(self, df, by_percentile=True):
        # Build a dictionary to hold contributions by reach
        tables = []
        df = df.join(self.huc_crosswalk)
        for field in 'HUC_8', 'HUC_12':
            table = df.groupby(field).agg('sum')
            if by_percentile:
                table = self.percentiles(table, 'total_mass')['percentile']
            tables.append(table)
        return tables

    def percentiles(self, table, sort_col):
        table_index = np.arange(table.shape[0])
        # Assign a percentile rank to each stream reach
        table['original_order'] = table_index
        table = table.sort_values(sort_col)
        table['percentile'] = ((table_index + 1) / table.shape[0]) * 100
        table = table.sort_values('original_order')
        del table['original_order']
        return table

    def prepare_output(self, write=True):
        # Break down terrestrial sources of pesticide mass
        full_table, summary_table, contributions_dict = self.process_contributions()

        # Write outputs
        if write:
            # Write all time series data
            for reach_id in self.local_reaches:
                self.write_time_series(reach_id, 'local')
            for reach_id in self.full_reaches:
                self.write_time_series(reach_id, 'full')

            # Write summary tables
            full_table.to_csv(os.path.join(self.sim.output_path, "full_table.csv"))
            summary_table.to_csv(os.path.join(self.sim.output_path, "summary.csv"))
            self.exceedances.to_csv(os.path.join(self.sim.output_path, "exceedances.csv"))

        # Set mapping dictionaries
        self.exceedances.index = self.exceedances.index.astype(str)
        intake_dict = {'COMID': self.exceedances.T.to_dict()}
        reach_dict = contributions_dict

        return intake_dict, reach_dict

    def process_contributions(self):
        # Build a DataFrame out of the finished contributions table
        # TODO - this is temporary
        made_it_this_far = len(self.contributions_index)
        self.contributions_index = self.contributions_index + list(range(made_it_this_far, self.contributions.shape[0]))

        # Initialize index and headings
        index = pd.Series(np.int64(self.contributions_index).astype(str), name='comid')
        cols = [f'cdl_{cls}' for cls in self.active_crops]

        # Get the total contributions for each crop, source, reach and hc
        by_crop = pd.DataFrame({'total_mass': self.contributions.sum(axis=(0, 1))}, cols)
        by_source = pd.DataFrame({'total_mass': self.contributions.sum(axis=(0, 2))}, ('runoff', 'erosion'))
        by_reach = pd.DataFrame({'total_mass': self.contributions.sum(axis=(1, 2))}, index)
        by_reach = self.percentiles(by_reach, 'total_mass')
        by_huc8, by_huc12 = self.contributions_by_huc(by_reach)

        # Parse outputs into tables and json
        full_table = pd.DataFrame(self.contributions[:, 0], index, cols) \
            .merge(pd.DataFrame(self.contributions[:, 1], index, cols), on='comid', suffixes=("_runoff", "_erosion"))
        summary_table = pd.concat([by_crop, by_source], axis=0)
        map_dict = {
            'comid': by_reach['percentile'].T.to_dict(),
            'huc_8': by_huc8.T.to_dict(),
            'huc_12': by_huc12.T.to_dict()
        }
        return full_table, summary_table, map_dict

    def update_contributions(self, reach_id, contributions):
        self.contributions[len(self.contributions_index)] = contributions
        self.contributions_index.append(reach_id)

    def update_exceedances(self, reach_id, data):
        self.exceedances.loc[reach_id] = data

    def update_full_time_series(self, reach_id, data):
        self.full_time_series.update(reach_id, data)

    def update_local_time_series(self, reach_id, data):
        self.local_time_series.update(reach_id, data)

    def write_time_series(self, reach_id, tag):
        outfile_path = os.path.join(self.sim.output_path, "time_series_{}_{}.csv".format(reach_id, tag))
        if tag == 'local':
            data = pd.DataFrame(self.local_time_series.fetch(reach_id).T, self.sim.dates, self.sim.local_time_series)
        elif tag == 'full':
            data = pd.DataFrame(self.full_time_series.fetch(reach_id).T, self.sim.dates, self.sim.full_time_series)
        data.to_csv(outfile_path)


class WatershedRecipes(object):
    def __init__(self, region, sim):
        self.path = sim.recipes_path.format(region)

        # Read shape
        with open(f'{self.path}_key.txt') as f:
            self.shape = literal_eval(next(f))

        # Read lookup map
        self.map = pd.read_csv(f'{self.path}_map.csv')

        # Get all the available years from the recipe
        self.years = sorted(self.map.year.unique())

    def fetch(self, reach_id, year='all years'):
        address = self.lookup(reach_id, year, verbose=True)
        if address is not None:
            n_blocks = address.shape[0]
            results = []
            fp = np.memmap(f'{self.path}', dtype=np.int64, mode='r', shape=self.shape)
            for start, end in address:
                block = pd.DataFrame(fp[start:end], columns=['scenario_index', 'area']).set_index('scenario_index')
                if n_blocks == 1:
                    return block
                else:
                    results.append(block)
        else:
            return None
        return pd.concat(results, axis=0)

    def lookup(self, reach_id, year, verbose=False):
        if year != 'all years':
            result = self.map[(self.map.comid == reach_id) & (self.map.year == year)]
        else:
            result = self.map[(self.map.comid == reach_id)]
        if result.shape[0] < 1:
            if verbose:
                report(f'No recipes found for {reach_id}, {year}', 2)
            return None
        else:
            return result[['start', 'end']].values


class WeatherArray(MemoryMatrix, DateManager):
    def __init__(self, sim):
        # TODO - sync this class with opp-efed-weather
        array_path = sim.weather_path.format('array.dat')
        key_path = sim.weather_path.format('key.npz')

        # Set row/column offsets
        index, header, start_date, end_date = self.load_key(key_path)

        # Set dates
        DateManager.__init__(self, start_date, end_date)
        self.start_offset, self.end_offset = self.date_offset(sim.start_date, sim.end_date, coerce=False)
        self.end_offset = -self.n_dates if self.end_offset == 0 else self.end_offset

        # Initialize memory matrix
        MemoryMatrix.__init__(self, [index, self.n_dates, header], np.float32, array_path, True, name='weather')

    @staticmethod
    def load_key(key_path):
        key = np.load(key_path)
        points, years, header = key['points'], key['years'], key['header']
        start_date = np.datetime64(f'{years[0]}-01-01')
        end_date = np.datetime64(f'{years[-1]}-12-31')
        return points.T[0], header, start_date, end_date

    def fetch_station(self, station_id):
        data = self.fetch(station_id, copy=True, verbose=True).T
        data[:2] /= 100.  # Precip, PET  cm -> m
        return data[:, self.start_offset:-self.end_offset]
