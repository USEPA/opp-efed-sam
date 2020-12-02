import os
import numpy as np
from .aquatic_concentration import compute_concentration, partition_benthic
from .tools.efed_lib import MemoryMatrix, DateManager, report
from .utilities import ImpulseResponseMatrix


class ReachManager(DateManager, MemoryMatrix):
    def __init__(self, sim, s3_scenarios, region, year):
        self.region = region
        self.year = year
        self.path = os.path.join(sim.paths.scratch_path, "_reach_mgr{}".format(s3_scenarios.region))

        # Initialize dates
        DateManager.__init__(self, s3_scenarios.start_date, s3_scenarios.end_date)

        # Initialize a matrix to store time series data for reaches (crunched scenarios)
        # vars: runoff, runoff_mass, erosion, erosion_mass
        MemoryMatrix.__init__(self, [region.active_reaches, 4, self.n_dates], name='reach manager',
                              path=self.path)

        # Keep track of which reaches have been run
        self.burned_reaches = set()  # reaches that have been processed


def burn_batch(reaches, sim, region, lakes):
    convolve_runoff = sim.hydrology.convolve_runoff
    dask_client = sim.dask_client
    if sim.local_run:
        for _, lake in lakes.iterrows():
            out_array = burn(reaches, lake, sim, region, convolve_runoff)
            reaches.update(lake.outlet_comid, out_array)
    else:
        batch = []
        for _, lake in lakes.iterrows():
            batch.append(dask_client.submit(burn, reaches, lake, sim, region, convolve_runoff))
        results = sim.dask_client.gather(batch)
        for outlet_comid, out_array in zip(lakes.outlet_comid, results):
            reaches.update(outlet_comid, out_array)


def test(a, b, s2, s3):
    c = s2.plant_date
    d = s3.n_dates
    return a + b + c + d


def process_local_batch(reaches, reach_ids, recipes, s2, s3, sim, year):
    dask_client = sim.dask_client

    if sim.local_run:
        print("No way!")
    else:
        batch = []
        for i, reach_id in enumerate(reach_ids):
            print(reach_id)
            batch.append(dask_client.submit(test, i, 10))
    print("done and now?")
    results = dask_client.gather(batch)
    print(results)
    exit()


def process_local_batch_actual(reaches, reach_ids, recipes, s2, s3, sim, year):
    dask_client = sim.dask_client
    if sim.local_run:
        for reach_id in reach_ids:
            out_array = process_local(reach_id, year, recipes, s2, s3)
            reaches.update(reach_id, out_array)
    else:
        batch = []
        for reach_id in reach_ids:
            batch.append(dask_client.submit(process_local, reach_id, year, recipes, s2, s3))
        results = dask_client.gather(batch)
        for reach_id, out_array in zip(reach_ids, results):
            reaches.update(reach_id, out_array)


def process_full_batch(reaches, reach_ids, sim, region):
    dask_client = sim.dask_client
    reach_ids &= set(region.output_reaches)
    if sim.local_run:
        for reach_id in reach_ids:
            out_array = reaches.process_full(reaches, reach_id, sim, region)
            reaches.update(reach_id, out_array)
    else:
        batch = []
        for reach_id in reach_ids:
            batch.append(dask_client.submit(process_full, reaches, reach_id, sim, region))
        results = sim.dask_client.gather(batch)
        for reach_id, out_array in zip(reach_ids, results):
            reaches.update(reach_id, out_array)


def burn(reaches, lake, sim, region, convolve_runoff):
    irf = ImpulseResponseMatrix.generate(1, lake.residence_time, sim.n_dates)

    # Get the convolution function
    # Get mass and runoff for the reach
    total_mass, total_runoff = upstream_loading(reaches, lake.outlet_comid, sim, region)

    # Modify combined time series to reflect lake
    new_mass = np.convolve(total_mass, irf)[:sim.n_dates]
    if convolve_runoff:  # Convolve runoff
        new_runoff = np.convolve(total_runoff, irf)[:sim.n_dates]
    else:  # Flatten runoff
        new_runoff = np.repeat(np.mean(total_runoff), sim.n_dates)

    # Retain old erosion numbers
    _, _, erosion, erosion_mass = reaches.fetch(lake.outlet_comid)

    # Add all lake mass and runoff to outlet
    return np.array([new_runoff, new_mass, erosion, erosion_mass])


def weight_and_combine(time_series, areas):
    areas = areas.values
    time_series = np.moveaxis(time_series, 0, 2)  # (scenarios, vars, dates) -> (vars, dates, scenarios)
    time_series[0] *= areas
    time_series[1] *= np.power(areas / 10000., .12)
    return time_series.sum(axis=2)


def process_local(reach_id, year, recipes, s2, s3, verbose=False):
    """  Fetch all scenarios and multiply by area. For erosion, area is adjusted. """

    # JCH - this pulls up a table of ['scenario_index', 'area'] index is used here to keep recipe files small
    recipe = recipes.fetch(reach_id, year)  # recipe is indexed by scenario_index
    if not recipe.empty:
        # Pull runoff and erosion from Stage 2 Scenarios
        transport, found_s2 = s2.fetch_from_recipe(recipe)
        runoff, erosion = weight_and_combine(transport, found_s2.area)

        # Pull chemical mass from Stage 3 scenarios
        pesticide_mass, found_s3 = s3.fetch_from_recipe(recipe, verbose=False)
        runoff_mass, erosion_mass = weight_and_combine(pesticide_mass, found_s3.area)
        out_array = np.array([runoff, runoff_mass, erosion, erosion_mass])

        # Assess the contributions to the recipe from ach source (runoff/erosion) and crop
        # self.o.update_contributions(recipe_id, scenarios, time_series[[1, 3]].sum(axis=1))

    else:
        out_array = None
        if verbose:
            report("No scenarios found for {}".format(reach_id))

    return out_array


def upstream_loading(reaches, reach_id, sim, region):
    """ Identify all upstream reaches, pull data and offset in time """

    # Fetch all upstream reaches and corresponding travel times
    upstream_reaches, travel_times, warning = \
        region.upstream_watershed(reach_id, return_times=True, return_warning=True)

    # Filter out reaches (and corresponding times) that have already been burned
    indices = np.int16([i for i, r in enumerate(upstream_reaches) if r not in reaches.burned_reaches])
    reaches, reach_times = upstream_reaches[indices], travel_times[indices]

    # Don't need to do proceed if there's nothing upstream
    if len(reaches) > 1:

        # Initialize the output array
        totals = np.zeros((4, sim.n_dates))  # (mass/runoff, dates)

        # Fetch time series data for each upstream reach
        reach_array, found_reaches = reaches.fetch(reaches, verbose=True, return_alias=True)  # (reaches, vars, dates)

        # Stagger time series by dayshed
        for tank in range(np.max(reach_times) + 1):
            in_tank = reach_array[reach_times == tank].sum(axis=0)
            if tank > 0:
                if sim.hydrology.gamma_convolve:
                    irf = region.irf.fetch(tank)  # Get the convolution function
                    in_tank[0] = np.convolve(in_tank[0], irf)[:sim.n_dates]  # mass
                    in_tank[1] = np.convolve(in_tank[1], irf)[:sim.n_dates]  # runoff
                else:
                    in_tank = np.pad(in_tank[:, :-tank], ((0, 0), (tank, 0)), mode='constant')
            totals += in_tank  # Add the convolved tank time series to the total for the reach

        runoff, runoff_mass, _, _ = totals
    else:
        result = reaches.fetch(reach_id)
        runoff, runoff_mass, _, _ = result

    # TODO - erosion mass here?
    return np.array([runoff, runoff_mass])


def process_full(reaches, reach_id, sim, region):
    # Get flow values for reach
    flow = region.daily_flows(reach_id)

    # Get local runoff, erosion, and pesticide masses
    local_runoff, local_runoff_mass, local_erosion, local_erosion_mass = reaches.fetch(reach_id)

    # Process upstream contributions
    upstream_runoff, upstream_runoff_mass = upstream_loading(reaches, reach_id, sim, region, sim.n_dates)

    # Compute concentrations
    surface_area = region.flow_table(reach_id)['surface_area']
    total_flow, (concentration, runoff_conc) = \
        compute_concentration(upstream_runoff_mass, upstream_runoff, sim.n_dates, flow)
    benthic_conc = partition_benthic(local_erosion, local_erosion_mass, surface_area,
                                     sim.benthic.depth, sim.benthic.porosity)

    return np.array([total_flow, upstream_runoff, upstream_runoff_mass, concentration, benthic_conc])
