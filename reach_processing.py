import os
import numpy as np
import pandas as pd
import time
from ast import literal_eval
from numba import njit
from .tools.efed_lib import MemoryMatrix, DateManager
from .utilities import ImpulseResponseMatrix, report
from .hydrology import benthic_concentration, water_column_concentration


class ReachManager(DateManager, MemoryMatrix):
    """
    A class to hold runoff and runoff mass for each reach as it's processed
    """

    def __init__(self, sim, region, recipes, output):
        self.sim = sim
        self.region = region
        self.recipes = recipes
        self.output = output
        self.array_path = os.path.join(sim.scratch_path, 'reach_array')
        self.reach_index = region.active_reaches
        self.lookup = pd.Series(region.active_reaches.index.values, region.active_reaches)
        self.n_active_crops = int(self.sim.n_active_crops)

        # Initialize dates
        DateManager.__init__(self, sim.start_date, sim.end_date)
        self.array_path = os.path.join(sim.scratch_path, 'r{}_working'.format(region.id))

        # The working array, which contains the same variables as s3, but for a reach instead of a scenario
        MemoryMatrix.__init__(self, [region.active_reaches, self.sim.fields.fetch('local_time_series'), self.n_dates],
                              name='reaches', path=self.array_path)

        # Initialize to zero (test)
        self.set_zero()

        # Keep track of which reaches have been run
        self.burned_reaches = set()  # reaches that have been processed

        # Create an 'index array' which tells which year's recipe to use for each day in the time series
        self.recipe_year_index = self.stagger_years()

    def set_zero(self):
        writer = self.writer
        writer[:] = 0.
        del writer

    def stagger_years(self):
        all_years = self.dates.year.unique()
        recipe_years = np.array(self.recipes.years)
        year_index = ((all_years - recipe_years[0]) % len(recipe_years))
        lookup = pd.Series(recipe_years[year_index], index=all_years)
        index_array = lookup[self.dates.year].values
        index_array_special = np.array([(index_array == year) for year in recipe_years])
        return index_array_special

    def burn(self, lakes):
        reader = self.reader
        writer = self.writer
        for _, lake in lakes.iterrows():
            lake_index = self.lookup[lake.outlet_comid]

            # Get the convolution function
            # Get mass and runoff for the reach
            total_mass, total_runoff, erosion, erosion_mass = \
                self.upstream_loading(lake.outlet_comid, lake_index, reader)

            # Modify combined time series to reflect lake
            if self.sim.convolve_mass or self.sim.convolve_runoff:
                irf = ImpulseResponseMatrix.generate(1, lake.residence_time, self.n_dates)
            else:
                irf = None

            if self.sim.convolve_mass:
                new_mass = np.convolve(total_mass, irf)[:self.n_dates]
            else:
                new_mass = np.repeat(np.mean(total_mass), self.n_dates)
            if self.sim.convolve_runoff:  # Convolve runoff
                new_runoff = np.convolve(total_runoff, irf)[:self.n_dates]
            else:  # Flatten runoff
                new_runoff = np.repeat(np.mean(total_runoff), self.n_dates)

            # TODO - do we need to do something with erosion here?

            # Add all lake mass and runoff to outlet
            writer[lake_index] = np.array([new_runoff, new_mass, erosion, erosion_mass])

        del reader, writer

    def get_contributions(self, found_s3, time_series, output_index):
        if found_s3.chemical_applied.any():
            contribution_idx = found_s3[found_s3.chemical_applied].contribution_index
            runoff_mass, erosion_mass = np.moveaxis(time_series[[1, 3]], 2, 0)[found_s3.chemical_applied].sum(axis=2).T
            runoff_cont = np.bincount(contribution_idx, weights=runoff_mass, minlength=self.n_active_crops)
            erosion_cont = np.bincount(contribution_idx, weights=erosion_mass, minlength=self.n_active_crops)
            return np.concatenate([runoff_cont, erosion_cont])

    def build_time_series(self, time_series, year_index, area):
        time_series *= year_index  # Only keep the data for dates where this recipe year is used
        time_series = np.moveaxis(time_series, 0, 2)  # (scenarios, vars, dates) -> (vars, dates, scenarios)
        time_series[:2] *= area
        time_series[2:] *= np.power(area / 10000., .12)
        return time_series

    def process_local(self, s3, reach_ids, output_reach_ids):
        writer = self.writer
        found = 0
        not_found = 0
        for reach_id in reach_ids:
            combined = np.zeros((4, self.n_dates))
            reach_index = self.lookup[reach_id]
            for i, (year, recipe) in enumerate(self.recipes.fetch(reach_id, df=True)):
                time_series, found_s3 = s3.fetch_from_recipe(recipe.s1_index)
                init_size = time_series.shape
                found += found_s3.shape[0]
                not_found += recipe.shape[0] - found_s3.shape[0]
                time_series = self.build_time_series(time_series, self.recipe_year_index[i], recipe.area.values)
                contributions = self.get_contributions(found_s3, time_series, reach_index)
                try:
                    combined += time_series
                except Exception as e:
                    print("Failing")
                    print(combined.shape, init_size, time_series.shape)
                    raise e
                if contributions is not None:
                    self.output.contributions.iloc[reach_index] += contributions
                if reach_id in output_reach_ids:
                    self.output.update_time_series(reach_id, time_series, 'local')
            if combined.min() < 0:
                print(f"Negative values found in reach {reach_id}: {combined.min(axis=1)}")
            writer[reach_index] = combined
        del writer
        report(f"Sucessfully found {found} scenarios. Unable to find {not_found} of them")

    def process_upstream(self, reach_ids, output_reach_ids):
        reader = self.reader
        for reach_id in reach_ids:
            reach_index = self.lookup[reach_id]

            # Accumulate runoff, erosion, and pesticide mass from upstream
            runoff, runoff_mass, erosion, erosion_mass = self.upstream_loading(reach_id, reach_index, reader)

            # Calculate the pesticide concentrations in water and get hydrology time series
            total_flow, baseflow, wc_conc, benthic_conc, runoff_conc = \
                self.compute_concentration(reach_id, runoff, runoff_mass, erosion, erosion_mass)

            self.output.concentrations.iloc[reach_index] = \
                np.array([wc_conc.mean(), wc_conc.max(), benthic_conc.mean(), benthic_conc.max()])

            # Pick out the time series that will be retained in the output
            if reach_id in output_reach_ids:
                upstream_time_series = \
                    np.array([runoff, runoff_mass, erosion, erosion_mass,
                              total_flow, baseflow, wc_conc, benthic_conc, runoff_conc])
                self.output.update_time_series(reach_id, upstream_time_series, 'upstream')

            # Calculate exceedance probabilities of endpoints
            self.output.exceedances.iloc[reach_index] = \
                exceedance_probability(wc_conc, self.sim.endpoints.duration.values.astype(np.int32),
                                       self.sim.endpoints.threshold.values, self.year_index)

        del reader

    def upstream_loading(self, reach_id, reach_index, reader):
        """ Identify all upstream reaches, pull data and offset in time """
        # TODO - how do we handle upstream erosion?
        # Fetch all upstream reaches and corresponding travel times
        upstream_reaches, travel_times, warning = \
            self.region.upstream_watershed(reach_id, return_times=True, return_warning=True)

        # Filter out reaches (and corresponding times) that have already been burned
        indices = np.int16([i for i, r in enumerate(upstream_reaches) if r not in self.burned_reaches])
        reaches, reach_times = upstream_reaches[indices], travel_times[indices]

        # Start with 'local' data
        time_series = reader[reach_index].copy()

        # Don't need to do proceed if there'snothing upstream
        if len(reaches) > 1:

            # Fetch time series data for each upstream reach
            index = self.lookup[reaches]
            reach_array = reader[index, :2].astype(np.float64)  # (reaches, vars, dates)

            # Stagger time series by dayshed
            for tank in range(np.max(reach_times) + 1):
                in_tank = reach_array[reach_times == tank].sum(axis=0)
                if tank > 0:
                    if self.sim.gamma_convolve:
                        irf = self.region.irf.fetch(tank)  # Get the convolution function
                        in_tank[0] = np.convolve(in_tank[0], irf)[:self.n_dates]  # runoff
                        in_tank[1] = np.convolve(in_tank[1], irf)[:self.n_dates]  # runoff mass
                    else:
                        in_tank = np.pad(in_tank[:2, :-tank], ((0, 0), (tank, 0)), mode='constant')
                if reach_id == 5039952:
                    print(f"Dayshed {tank}: {in_tank.sum(axis=1)}")
                time_series[:2] += in_tank  # Add the convolved tank time series to the total for the reach
        return time_series

    def compute_concentration(self, reach_id, runoff, runoff_mass, erosion, erosion_mass):

        # Get hydrologic data for reach
        predicted_flow, surface_area = self.region.daily_flows(reach_id)

        # Get local runoff, erosion, and pesticide masse
        total_flow, baseflow, (wc_conc, runoff_conc) = \
                water_column_concentration(reach_id, runoff, runoff_mass, self.n_dates, predicted_flow)
        # TODO - why are there nans in the erosion data
        #print(234, np.isnan(np.array([runoff, runoff_mass, erosion, erosion_mass])).sum(axis=1))
        negative_concs = (wc_conc < 0).sum()
        if negative_concs > 0:
            print(f"{negative_concs} negative concentrations found")

        try:
            benthic_conc = benthic_concentration(
                erosion, erosion_mass, surface_area, self.sim.benthic_depth, self.sim.benthic_porosity)
        except Exception as e:
            report(f"{reach_id}: {e}")
            benthic_conc = np.zeros(total_flow.shape)
        # Make sure this matches the order specified in fields_and_qc.csv
        return np.array([total_flow, baseflow, wc_conc, benthic_conc, runoff_conc])


class WatershedRecipes(MemoryMatrix):
    def __init__(self, region, sim):
        self.sim = sim
        self.path = sim.recipes_path.format(region)
        self.map_path = f'{self.path}_map.csv'

        # Read shape
        with open(f'{self.path}_key.txt') as f:
            self.shape = literal_eval(next(f))

        # Read lookup map
        self.map = pd.read_csv(self.map_path).sort_values(['comid', 'year']).set_index('comid')

        # Get all the available years from the recipe
        self.years = sorted(self.map.year.unique())

        # Not using a MemoryMatrix wrapper here since it's fairly simple
        self.matrix = np.memmap(f'{self.path}', dtype=np.int64, mode='r', shape=self.shape)

    def fetch(self, reach_id, get_year='all', df=False):
        try:
            addresses = self.map.loc[reach_id]
        except KeyError:
            report(f'No recipes found for {reach_id}', 2)
            return None, (None, None)
        for comid, (year, start, end) in addresses.iterrows():
            if get_year == 'all' or get_year == year:
                recipe = self.matrix[start:end]
                if df:
                    recipe = pd.DataFrame(recipe, columns=['s1_index', 'area'])
                yield year, recipe


@njit
def pick_indices(upstream_reaches, burned_reaches):
    indices = np.zeros(upstream_reaches.shape)
    counter = 0
    for i, r in enumerate(upstream_reaches):
        if r not in burned_reaches:
            indices[counter] = i
            counter += 1
    return indices[:counter]


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
            result[test_number] = np.nan
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
