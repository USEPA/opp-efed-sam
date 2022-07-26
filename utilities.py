import os
import re
import time
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

    def __init__(self, input_json, retain_s1=False, retain_s3=False):
        # TODO - confirm that everything works as intended here, esp. for eco runs
        # Determine whether the simulation is being run on a windows desktop (local for epa devs)
        self.local_run = any([r'C:' in p for p in sys.path])
        self.retain_s1 = retain_s1
        self.retain_s3 = retain_s3

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
        self.build_scenarios, self.random, self.tag, self.confine_reaches = self.detect_special_run()

        # Initialize dates
        DateManager.__init__(self, *self.align_dates())

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
            file_path = os.path.join(self.scratch_path, f)
            file_age = (time.time() - os.stat(file_path).st_mtime) / 86400
            if not self.retain_s3 or "_s3" not in f:
                if file_age > 1:
                    os.remove(file_path)

    def detect_special_run(self):
        # TODO - update this so scenarios can't be built for a subset
        """
        A special run can be triggered from the frontend under the 'Simulation Name' field
        A special run can include the following things:
            1. Random output will be generated, bypassing the modeling functions. This is for output testing
            2. A Stage 1 to Stage 2 scenario build can be triggered
            3. A 'tag' can be provided to use a special intake file
        The Simulation Name string has the format "[special_mode&tag]
        Random output can be triggered by entering "test" as the special mode
        Scenario building can be triggered by entering "build" as the special mode
        The run can be confined to areas upstream of a set of reaches by entering "confine" as the special mode
            and providing an identifying 'tag'. For instance, the string 'confine&mtb' will point to the table
            Tables/confine_mtb.csv, which will provide the reaches to constrain the analysis. Random runs can also
            be confined with 'test&[tag]'

        :return: build_scenarios, intakes, run_regions, intake_reaches, tag, random
        """
        build = random = False
        tag, confine_reaches = None, None
        params = self.simulation_name.lower().split("&")
        message = ""

        if params[0] == 'test':
            random = True
            message = f"Generating randomized output"
        elif params[0] == 'build':
            build = True
            message = f"Building new Stage 2 Scenarios. "
        elif params[0] == 'confine':  # Any keyword will work if simply providing custom intakes
            message = f"Special run for custom intakes. "

        if len(params) > 1:
            tag = params[1]

        # Using the 'Mark Twain Demo' selection for region overrides all settings except 'test'
        if self.region == 'Mark Twain Demo':
            self.region = '07'
            tag = 'mtb'

        if tag is not None:
            confined_reaches = pd.read_csv(self.confine_reaches_path.format(tag)).comid.values
            message += f"Confining analysis to areas upstream of reach(es) {', '.join(map(str, confined_reaches))}"
        else:
            tag = self.region
            confined_reaches = None

        report(message)

        return build, random, tag, confined_reaches

    def initialize_parameters(self):
        params = pd.read_csv(self.parameters_path, usecols=['parameter', 'value', 'dtype'])
        return {k: eval(d)(v) for k, v, d in params.values}

    def initialize_paths(self):
        paths = {}
        # Identify the path to the table containing all other paths
        if self.local_run:
            paths['data_root'] = r'D:/opp-efed-data/sam'
        else:
            paths['data_root'] = os.getenv('SAM_INPUTS_DIR', r'/src/app-data/sampreprocessed')
        paths['local_root'] = pathlib.Path(__file__).parent.absolute()
        paths_table = os.path.join(paths['local_root'], 'Tables', 'paths.csv')
        table = pd.read_csv(paths_table).sort_values('level')[['var', 'dir', 'base']]
        for var, dirname, basename in table.values:
            paths[f'{var}_path'] = os.path.join(paths[dirname], basename)
        return paths

    @property
    def n_active_crops(self):
        return len(self.active_crops)


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
        self['selected_output'] = self.output_selection(output_selection_path)

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
            applications.loc[i, 'event'] = ['plant', 'emergence', 'maxcover', 'harvest'].index(row.event)
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
                       'upstream_time_series': [var for var, _, sel in t[t.table == 'upstream'].values if int(sel)]}
        return output_dict


class ModelOutputs(DateManager):
    """
    A class to hold SAM outputs and postprocessing functions
    """

    def __init__(self, sim, region):
        self.sim = sim
        self.active_reaches = region.active_reaches
        self.output_reaches = region.output_reaches
        self.huc_crosswalk = region.reach_table[['HUC_8', 'HUC_12']]
        self.local_time_series = self.sim.selected_output['local_time_series']  # output selection from frontend
        self.upstream_time_series = self.sim.selected_output['upstream_time_series']
        self.all_time_series = self.local_time_series + self.upstream_time_series
        self.n_active = len(self.active_reaches)
        self.run_time_series = (len(self.output_reaches) > 0)
        if self.output_reaches is not None:
            self.n_output = len(self.output_reaches)
            self.lookup = pd.Series(np.arange(len(self.output_reaches)), self.output_reaches)
        else:
            self.n_output = 0
            self.lookup = None

        # Initialize dates
        DateManager.__init__(self, sim.start_date, sim.end_date)
        self.array_path = os.path.join(sim.scratch_path, 'r{}_{{}}_out'.format(region.id))

        # Time series data for a single catchment (run everywhere)
        self.local_index, self.upstream_index, self.time_series_output = self.initialize_time_series()

        # The relative contribution of pesticide mass broken down by reach, runoff/erosion and crop
        header = [f"{lbl}_{crop}" for lbl in ('runoff', 'erosion') for crop in self.sim.active_crops]
        self.contributions = pd.DataFrame(np.zeros((self.n_active, len(header))), self.active_reaches, header)

        # Concentrations
        self.concentrations = pd.DataFrame(np.zeros((self.n_active, 4)), self.active_reaches,
                                           ['wc_conc_mean', 'wc_conc_max', 'benthic_conc_mean', 'benthic_conc_max'])

        # The probability that concentration exceeds endpoint thresholds
        self.exceedances = pd.DataFrame(np.zeros((self.n_active, self.sim.endpoints.shape[0])),
                                        self.active_reaches, self.sim.endpoints.short_name)

    def summarize_by_huc(self):

        # Merge all output data going to the map and add HUC crosswalk
        contributions_by_reach = self.contributions.sum(axis=1).rename("mass")
        df = self.huc_crosswalk.join(contributions_by_reach, how="right") \
            .join(self.exceedances).join(self.concentrations)

        # If we decide to do reach-level output: out_table = {'COMID': df.to_dict()}
        out_table = {}

        # Aggregate everything by HUC
        for field in 'HUC_8', 'HUC_12':
            table = df.groupby(field).agg([np.mean, np.sum]).astype(np.float32)
            for column in table.columns:
                table[column[0], 'pct'] = self.percentiles(table[column])
            table[np.isnan(table)] = -1
            out_table[field] = \
                table.T.unstack().T.groupby(level=0).apply(lambda x: x.xs(x.name).to_dict()).to_dict()

        return out_table

    def summarize_by_reach(self):
        out_dict = {}
        out_table = self.exceedances.join(self.concentrations)
        out_table[np.isnan(out_table)] = -1
        out_dict['comid'] = out_table.T.to_dict()
        return out_dict

    def summarize_by_intake(self):
        # TODO - not sure what we're going to put here
        out_dict = {'comid': {}}
        if self.run_time_series:
            out_dict = {}
            out_table = self.exceedances.join(self.concentrations)
            out_table = pd.Series(self.output_reaches, name='comid')\
                .to_frame()\
                .set_index('comid')\
                .join(out_table)\
                .fillna(-1)
            out_dict['comid'] = out_table.T.to_dict()
        return out_dict

    def initialize_time_series(self):
        # Find the numerical indices for the variables that get carried through to the output
        local_index = \
            [list(self.sim.fields.fetch('local_time_series')).index(f) for f in self.local_time_series]
        upstream_index = \
            [list(self.sim.fields.fetch('upstream_time_series')).index(f) for f in self.upstream_time_series]
        print('Debugging initialize_time_series - START')
        print(self.output_reaches, local_index + upstream_index, self.n_dates, self.array_path.format('all'))
        print('Debugging initialize_time_series - END')
        if self.run_time_series:
            time_series = MemoryMatrix(
                [self.output_reaches, local_index + upstream_index, self.n_dates], name='output time series',
                path=self.array_path.format('all'))
        else:
            time_series = None
        return local_index, upstream_index, time_series

    def percentiles(self, series):
        table = series.reset_index(drop=True) \
            .reset_index(name="val") \
            .rename(columns={'index': 'original'}) \
            .sort_values('val') \
            .reset_index()
        table['percentile'] = ((table.index + 1) / table.shape[0]) * 100
        table = table.sort_values('original')
        return table.percentile.values

    def populate_random(self):
        if self.run_time_series:
            self.time_series_output.writer[:] = np.random.rand(*self.time_series_output.shape) * 10.
        self.contributions[:] = np.random.rand(*self.contributions.shape) * 10.
        self.concentrations[:] = np.random.rand(*self.concentrations.shape) * 100.
        self.exceedances[:] = np.random.rand(*self.exceedances.shape)

    def prepare_output(self):

        if self.sim.random:
            self.populate_random()

        # Summarize pesticide masses and endpoint exceedances by reach and watershed
        huc_dict = self.summarize_by_huc()
        reach_dict = self.summarize_by_reach()
        intake_dict = self.summarize_by_intake()

        # Write output summary tables
        self.write_summary_tables()

        # Get intake time series data to return
        if self.run_time_series:
            intake_time_series = self.get_time_series()
        else:
            intake_time_series = None
        if self.sim.local_run:
            self.write_time_series()

        return reach_dict, huc_dict, intake_dict, intake_time_series

    def update_time_series(self, reach_id, data, mode):
        if self.run_time_series:
            output_index = self.lookup[reach_id]
            writer = self.time_series_output.writer
            if mode == 'local':
                if any(self.local_index):
                    writer[output_index, :len(self.local_index)] = data[self.local_index]
            elif mode == 'upstream':
                print(987, len(self.upstream_index), self.time_series_output.shape)
                if any(self.upstream_index):
                    writer[output_index, :len(self.upstream_index)] = data[self.upstream_index]
            else:
                raise ValueError(f"Invalid mode {mode}, must be 'local' or 'upstream")

    def write_summary_tables(self):
        # Write summary tables
        print(self.contributions)
        print(self.exceedances)
        self.contributions.to_csv(os.path.join(self.sim.output_path, "upstream_table.csv"), index=None)
        self.contributions.sum(axis=0).to_csv(os.path.join(self.sim.output_path, "summary.csv"))
        self.exceedances.to_csv(os.path.join(self.sim.output_path, "exceedances.csv"))

    def write_time_series(self):
        outfile_path = os.path.join(self.sim.output_path, "time series", "time_series_{}_{}.csv")
        if not os.path.exists(os.path.dirname(outfile_path)):
            os.makedirs(os.path.dirname(outfile_path))
        for reach_id in self.output_reaches:
            data = pd.DataFrame(self.time_series_output.fetch(reach_id).T, self.sim.dates, self.all_time_series)
            data.to_csv(outfile_path.format(reach_id, 'all'))

    def get_time_series(self):
        upstream_time_series_dict = {}
        if any(self.all_time_series):
            for reach_id in self.output_reaches:
                data = pd.DataFrame(self.time_series_output.fetch(reach_id).T,
                                    self.sim.dates.astype(str), self.all_time_series)
                upstream_time_series_dict[str(reach_id)] = data.to_dict(orient='split')
        return upstream_time_series_dict


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
