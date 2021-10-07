import os
import re
import numpy as np
import pandas as pd
import sys
import pathlib
import logging
from distributed import Client

from .hydrology import ImpulseResponseMatrix
from .tools.efed_lib import FieldManager, MemoryMatrix, DateManager


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

        # TODO - get rid of MTB at the frontend?
        # Unpack the 'simulation_name' parameter to detect if a special run is called for
        detected, self.build_scenarios, self.random, self.intake_reaches, self.tag = \
            self.detect_special_run()

        if not self.intake_reaches:
            self.intake_reaches = self.find_intakes()

        # Initialize dates
        DateManager.__init__(self, *self.align_dates())

        self.intakes_only = (self.sim_type != 'eco') or self.intake_reaches is None

        # Initialize an impulse response matrix if convolving time of travel
        self.irf = None if not self.gamma_convolve else ImpulseResponseMatrix(self.dates.size)

        # Make numerical adjustments (units etc)
        self.adjust_data()

        # TODO - placeholder for when running multiple regions is enabled in the frontend
        self.run_regions = [self.region]

        # Read token
        self.token = \
            self.simulation_name if not hasattr(self, 'csrfmiddlewaretoken') else self.csrfmiddlewaretoken

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

        # Make sure that 'region' wasn't read as a numeral
        try:
            self.region = str(int(float(self.region))).zfill(2)
        except ValueError:
            pass

        self.applications.apprate *= 0.0001  # convert kg/ha -> kg/m2 (1 ha = 10,000 m2)
        # Build a soil profile
        self.depth_bins = np.array(eval(self.depth_bins))
        self.delta_x = np.array(
            [self.surface_dx] + [self.layer_dx] * (self.n_increments - 1))
        self.depth = np.cumsum(self.delta_x)
        self.bins = np.minimum(self.depth_bins.size - 1, np.digitize(self.depth * 100., self.depth_bins))
        self.types = pd.read_csv(self.types_path).set_index('type')

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

    def check_directories(self):
        # Make sure needed subdirectories exist
        for subdir in self.scratch_path, self.output_path:
            if not os.path.exists(subdir):
                os.makedirs(subdir)

        # Purge temp folder
        for f in os.listdir(self.scratch_path):
            os.remove(os.path.join(self.scratch_path, f))

    def detect_special_run(self):
        """
        A special run can be triggered from the frontend under the 'Simulation Name' field
        A special run can include the following things:
            1. Random output will be generated, bypassing the modeling functions. This is for output testing
            2. A Stage 1 to Stage 2 scenario build can be triggered
            3. COMIDs can be provided to constrain the model run to an area smaller than a full region
        The Simulation Name string has the format "[special_mode&intake_reaches&tag]
        Random output can be triggered by entering "test" as the special mode
        Scenario building can be triggered by entering "build" as the special mode
        Geographical constraints can be applied by entering outlet COMIDs separated by commas as intake_reaches
        An identifying 'tag' is used to create separate files

        Example build string: 'build&4867727&mtb'
        :return: build_scenarios, intakes, run_regions, intake_reaches, tag, random
        """
        detected = build = random = False
        intake_reaches = tag = None
        params = self.simulation_name.lower().split("&")
        if params[0] == 'test':
            random = True
            tag = "random"
        elif params[0] == 'build':
            build = True
        # 'confine' is the preferred keyword but anything will work
        elif params[0] == 'confine':
            pass

        # Using the 'Mark Twain Demo' selection for region precludes all settings except 'test'
        if self.region == 'Mark Twain Demo':
            self.region = '07'
            intake_reaches = [4867727]
            tag = 'mtb'
        else:
            if len(params) > 1:
                intake_reaches = list(map(int, params[1].split(",")))
            if len(params) > 2:
                tag = params[2]
                if random:
                    tag = f"random_{tag}"
            if any((build, random, intake_reaches, tag)):
                detected = True

        return detected, build, random, intake_reaches, tag

    def find_intakes(self):
        """ Read a hardwired intake file """
        # TODO - The tool is currently only set up for running drinking water intakes.
        #  This line doesn't do anything right now
        assert self.sim_type in ('eco', 'drinking_water'), \
            'Invalid simulation type "{}"'.format(self.sim_type)

        # TODO - this is temporary until eco mode is ready
        assert self.sim_type == 'drinking_water', "'eco' mode is not ready yet"

        intake_file = self.dw_intakes_path
        intakes = pd.read_csv(intake_file)
        intakes['region'] = [str(r).zfill(2) for r in intakes.region]
        intakes = intakes[intakes.region == self.region]
        return sorted(np.unique(intakes.comid))

    def initialize_parameters(self):
        params = pd.read_csv(self.parameters_path, usecols=['parameter', 'value', 'dtype'])
        return {k: eval(d)(v) for k, v, d in params.values}

    def initialize_paths(self):
        paths = {}
        # Identify the path to the table containing all other paths
        if self.local_run:
            paths['data_root'] = r'E:/opp-efed-data/sam'
        else:
            paths['data_root'] = os.getenv('SAM_INPUTS_DIR', r'/src/app-data/sampreprocessed')
        paths['local_root'] = pathlib.Path(__file__).parent.absolute()
        paths_table = os.path.join(paths['local_root'], 'Tables', 'paths.csv')
        table = pd.read_csv(paths_table).sort_values('level')[['var', 'dir', 'base']]
        for var, dirname, basename in table.values:
            paths[f'{var}_path'] = os.path.join(paths[dirname], basename)
        return paths


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

    def __init__(self, sim, region, active_crops):
        self.sim = sim
        self.local_reaches = region.local_reaches
        self.full_reaches = region.full_reaches
        self.active_crops = active_crops
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
        # The reason for using an empty list as an index is so the results can get appended in the order
        # in which they're generated, instead of spending time on indexing. Might not be worth it?
        self.contributions_index = []
        self.contributions = np.zeros((len(self.local_reaches), 2, len(self.active_crops)))

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

    def populate_random(self):
        # Randomly populate the contributions array
        self.contributions_index = self.local_reaches
        self.contributions = np.random.rand(*self.contributions.shape) * 10.

        # Randomly populate the exceedance probabilities
        self.exceedances[:] = np.random.rand(*self.exceedances.shape)

    def prepare_output(self, write_tables=True, write_ts=False):

        if self.sim.random:
            self.populate_random()

        # Break down terrestrial sources of pesticide mass
        full_table, summary_table, contributions_dict = self.process_contributions()

        # Write outputs
        if write_ts:
            self.write_time_series()
        if write_tables:
            self.write_summary_tables(full_table, summary_table)

        # Set mapping dictionaries
        self.exceedances.index = self.exceedances.index.astype(str)
        intake_dict = {'COMID': self.exceedances.T.to_dict()}
        reach_dict = contributions_dict

        return intake_dict, reach_dict

    def process_contributions(self):

        # Initialize index and headings
        index = pd.Series(np.int64(self.contributions_index).astype(str), name='comid')
        cols = [f'cdl_{cls}' for cls in self.active_crops]

        # Get the total contributions for each crop, source, reach and hc
        by_source_and_crop = pd.DataFrame(self.contributions.sum(axis=0), ('runoff', 'erosion'), cols)
        by_reach = pd.DataFrame({'total_mass': self.contributions.sum(axis=(1, 2))}, index)
        by_reach = self.percentiles(by_reach, 'total_mass')
        by_huc8, by_huc12 = self.contributions_by_huc(by_reach)

        # Parse outputs into tables and json
        full_table = pd.DataFrame(self.contributions[:, 0], index, cols) \
            .merge(pd.DataFrame(self.contributions[:, 1], index, cols), on='comid', suffixes=("_runoff", "_erosion"))
        map_dict = {
            'comid': by_reach['percentile'].T.to_dict(),
            'huc_8': by_huc8.T.to_dict(),
            'huc_12': by_huc12.T.to_dict()
        }
        return full_table, by_source_and_crop, map_dict

    def update_contributions(self, reach_id, contributions):
        self.contributions[len(self.contributions_index)] = contributions
        self.contributions_index.append(reach_id)

    def update_exceedances(self, reach_id, data):
        self.exceedances.loc[reach_id] = data

    def update_full_time_series(self, reach_id, data):
        self.full_time_series.update(reach_id, data)

    def update_local_time_series(self, reach_id, data):
        self.local_time_series.update(reach_id, data)

    def write_summary_tables(self, full_table, summary_table):
        # Write summary tables
        full_table.to_csv(os.path.join(self.sim.output_path, "full_table.csv"))
        summary_table.to_csv(os.path.join(self.sim.output_path, "summary.csv"))
        self.exceedances.to_csv(os.path.join(self.sim.output_path, "exceedances.csv"))

    def write_time_series(self):
        outfile_path = os.path.join(self.sim.output_path, "time series", "time_series_{}_{}.csv")
        if not os.path.exists(os.path.dirname(outfile_path)):
            os.makedirs(os.path.dirname(outfile_path))
        for reach_id in self.local_reaches:
            data = pd.DataFrame(self.local_time_series.fetch(reach_id).T, self.sim.dates, self.sim.local_time_series)
            data.to_csv(outfile_path.format(reach_id, 'local'))
        for reach_id in self.full_reaches:
            data = pd.DataFrame(self.full_time_series.fetch(reach_id).T, self.sim.dates, self.sim.full_time_series)
            data.to_csv(outfile_path.format(reach_id, 'full'))


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


def report(message, tabs=0):
    """ Display a message with a specified indentation """
    tabs = "\t" * tabs
    print(tabs + str(message))
    logging.info(tabs + str(message))
