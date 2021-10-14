import os
import numpy as np
import pandas as pd
from ast import literal_eval
from numba import njit
from .tools.efed_lib import MemoryMatrix, DateManager
from .utilities import ImpulseResponseMatrix, report
from .hydrology import benthic_concentration, water_column_concentration


class ReachManager(DateManager, MemoryMatrix):
    """
    A class to hold runoff and runoff mass for each reach as it's processed
    """

    def __init__(self, sim, s2, s3, region, recipes, output):
        self.sim = sim
        self.s2 = s2
        self.s3 = s3
        self.region = region
        self.recipes = recipes
        self.output = output
        self.array_path = os.path.join(sim.scratch_path, 'reach_array')

        # Initialize dates
        DateManager.__init__(self, sim.start_date, sim.end_date)
        self.array_path = os.path.join(sim.scratch_path, 'r{}_{{}}_out'.format(region.id))

        # This is the working array
        MemoryMatrix.__init__(self, [region.local_reaches, self.sim.fields.fetch('local_time_series'), self.n_dates],
                              name='reaches', path=self.array_path)

        # Keep track of which reaches have been run
        self.burned_reaches = set()  # reaches that have been processed

        # Create an 'index array' which tells which year's recipe to use for each day in the time series
        reps, remain = divmod(self.n_dates, len(self.recipes.years))
        self.index_array = np.concatenate((np.tile(self.recipes.years, reps), self.recipes.years[:remain]))

        # Find the numerical indices for the variables that get carried through to the output
        self.local_index = \
            [list(sim.fields.fetch('local_time_series')).index(f) for f in sim.local_time_series]
        self.full_index = \
            [list(sim.fields.fetch('full_time_series')).index(f) for f in sim.full_time_series]

    def burn(self, lakes):

        for _, lake in lakes.iterrows():

            irf = ImpulseResponseMatrix.generate(1, lake.residence_time, self.n_dates)

            # Get the convolution function
            # Get mass and runoff for the reach
            total_mass, total_runoff, erosion, erosion_mass = \
                self.upstream_loading(lake.outlet_comid)

            # Modify combined time series to reflect lake
            new_mass = np.convolve(total_mass, irf)[:self.n_dates]
            if self.sim.convolve_runoff:  # Convolve runoff
                new_runoff = np.convolve(total_runoff, irf)[:self.n_dates]
            else:  # Flatten runoff
                new_runoff = np.repeat(np.mean(total_runoff), self.n_dates)

            # TODO - do we need to do something with erosion here?

            # Add all lake mass and runoff to outlet
            self.update(lake.outlet_comid, np.array([new_runoff, new_mass, erosion, erosion_mass]))

    def process_local(self, reach_ids):
        for i, reach_id in enumerate(reach_ids):
            # Add up the time series for all the combinations in the reach, weighted by area
            time_series, contributions = self.combine_scenarios(reach_id)

            # Store the time series data in the working class array
            self.update(reach_id, time_series)

            # Add the contributions by crop and runoff/erosion to the output array
            self.output.update_contributions(reach_id, contributions)

            # Pick out the data selected for output and send it to
            self.output.update_local_time_series(reach_id, time_series[self.local_index])

    def process_full(self, reach_ids):
        for reach_id in reach_ids:
            # Accumulate runoff, erosion, and pesticide mass from upstream
            runoff, runoff_mass, erosion, erosion_mass = self.upstream_loading(reach_id)

            # Calculate the pesticide concentrations in water and get hydrology time series
            total_flow, baseflow, wc_conc, benthic_conc, runoff_conc = \
                self.compute_concentration(reach_id, runoff, runoff_mass, erosion, erosion_mass)

            # Pick out the time series that will be retained in the output
            upstream_time_series = \
                np.array([runoff, runoff_mass, erosion, erosion_mass,
                          total_flow, baseflow, wc_conc, benthic_conc, runoff_conc])[self.full_index]

            # Store the selected output in the full time series output matrix
            self.output.update_full_time_series(reach_id, upstream_time_series)

            # Calculate excedance probabilities of endpoints
            exceedance = \
                exceedance_probability(wc_conc, self.sim.endpoints.duration.values.astype(np.int32),
                                       self.sim.endpoints.threshold.values, self.year_index)
            self.output.update_exceedances(reach_id, exceedance)

    def combine_scenarios(self, reach_id):
        """
        Combines all the scenarios comprising a given reach and updates the ReachManager matrix
        :param reach_id:
        :return:
        """
        # Initialize a new array to hold runoff, runoff mass, erosion, and erosion mass
        n_active_crops = len(self.sim.active_crops)
        new_array = np.zeros((self.n_dates, 4))
        contributions = np.zeros((2, n_active_crops))
        for year in self.recipes.years:
            # Pull the watershed recipe for the reach and year
            recipe = self.recipes.fetch(reach_id, year)  # scenarios are indexed by recipe_index
            if recipe is not None:
                # Pull runoff and erosion from Stage 2 Scenarios
                transport, found_s2 = self.s2.fetch_from_recipe(recipe)
                runoff, erosion = weight_and_combine(transport, found_s2.area)

                # Pull chemical mass from Stage 3 scenarios
                pesticide_mass, found_s3 = self.s3.fetch_from_recipe(recipe)

                if found_s3 is None:
                    continue

                runoff_mass, erosion_mass = weight_and_combine(pesticide_mass, found_s3.area)

                # Assign the pieces of the time series to the new array based on year
                year_index = (self.index_array == year)
                new_array[year_index] = \
                    np.array([runoff, runoff_mass, erosion, erosion_mass]).T[year_index]

                # Assign contributions
                annual_masses = pesticide_mass[:, :, year_index].sum(axis=2).T
                for i in range(2):
                    contributions[i] += np.bincount(found_s3.contribution_index, weights=annual_masses[i],
                                                    minlength=n_active_crops)
        return new_array.T, contributions

    def upstream_loading(self, reach_id):
        """ Identify all upstream reaches, pull data and offset in time """
        # TODO - how do we handle upstream erosion?
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
                    if self.sim.gamma_convolve:
                        irf = self.region.irf.fetch(tank)  # Get the convolution function
                        in_tank[0] = np.convolve(in_tank[0], irf)[:self.n_dates]  # mass
                        in_tank[1] = np.convolve(in_tank[1], irf)[:self.n_dates]  # runoff
                    else:
                        in_tank = np.pad(in_tank[:, :-tank], ((0, 0), (tank, 0)), mode='constant')
                totals += in_tank  # Add the convolved tank time series to the total for the reach

            # TODO - is this how we're handling erosion?
            _, _, erosion, erosion_mass = self.fetch(reach_id)
            runoff, runoff_mass, erosion, erosion_mass = totals
        else:
            result = self.fetch(reach_id)
            runoff, runoff_mass, erosion, erosion_mass = result

        # TODO - erosion mass here?
        return np.array([runoff, runoff_mass, erosion, erosion_mass])

    def compute_concentration(self, reach_id, runoff, runoff_mass, erosion, erosion_mass):

        # Get hydrologic data for reach
        flow = self.region.daily_flows(reach_id)
        surface_area = self.region.flow_table(reach_id)['surface_area']

        # Get local runoff, erosion, and pesticide masses
        total_flow, baseflow, (wc_conc, runoff_conc) = \
            water_column_concentration(runoff, runoff_mass, self.n_dates, flow)
        benthic_conc = benthic_concentration(
            erosion, erosion_mass, surface_area, self.sim.benthic_depth, self.sim.benthic_porosity)
        # Make sure this matches the order specified in fields_and_qc.csv
        return np.array([total_flow, baseflow, wc_conc, benthic_conc, runoff_conc])


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
                block = pd.DataFrame(fp[start:end], columns=['recipe_index', 'area']).set_index('recipe_index')
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


def weight_and_combine(time_series, areas):
    areas = areas.values
    time_series = np.moveaxis(time_series, 0, 2)  # (scenarios, vars, dates) -> (vars, dates, scenarios)
    time_series[0] *= areas
    time_series[1] *= np.power(areas / 10000., .12)
    return time_series.sum(axis=2)


@njit
def exceedance_probability(time_series, durations, endpoints, years_since_start):
    # Count the number of times the concentration exceeds the test threshold in each year
    result = np.zeros(durations.shape)

    n_years = years_since_start.max() + 1

    # Set up the test for each endpoints
    for test_number in range(durations.size):
        duration = durations[test_number]
        endpoint = endpoints[test_number]

        # If the duration or endpoint isn't set, set the value to 1
        if np.isnan(endpoint) or np.isnan(duration):
            result[test_number] = -1
        else:
            # Initialize an array of exceedance
            exceedances = np.zeros(n_years)
            # Add up all the daily loads for the window. This is used to calculate a daily average
            duration_total = np.sum(time_series[:duration])
            for day in range(duration, len(time_series)):
                year = years_since_start[day]
                duration_total += time_series[day] - time_series[day - duration]
                avg = duration_total / duration
                if avg > endpoint:
                    exceedances[year] = 1
            result[test_number] = exceedances.sum() / n_years
    return result
