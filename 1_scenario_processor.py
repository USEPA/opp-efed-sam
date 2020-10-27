import dask
import pandas as pd
import numpy as np
from .utilities import WeatherArray, StageOneScenarios, StageTwoScenarios, WatershedRecipes
from .hydro.navigator import Navigator
from .parameters import batch_size, crop_group_field
from .scenario_processing import stage_one_to_two
from .utilities import fields, report


# TODO - complete preprocessing script for building condensed nhd (sam and nav), navs, etc
# TODO 9/14 - why are there so few sam scenarios now compared with before?
# TODO - adjust weather and hydro packages so that paths are relative to the project,
#  and make hydro, weather, and efed_lib play nice together (maybe ask Tom?)

# TODO 8/20 - Finish 'season' param. Combines crop dates and double cropping
# TODO - is there overlap between modify.scenarios and funcs here
#  (e.g. evapotranspiration_daily, usle_s) check root_depth, evaporation_depth

# TODO - handling of non-ag classes and non-soil soils
# TODO - confirm that all necessary data are in range
# TODO - confirm that results are written in the correct order by Dask
# TODO - determine exactly which mods are unique to SAM and which can be added to aquatic-model-inputs
# TODO - does SAM take plant_begin, etc or a midpoint?

def process_batch(stage_one, stage_two, met):
    n_scenarios = int(stage_one.n_scenarios)
    batch = []
    scenario_count = 0
    batch_count = 0
    keep_fields = list(fields.fetch('s1_keep_cols')) + [crop_group_field]
    stage_two_index = []

    # Group by weather grid to reduce the overhead from fetching met data
    for weather_grid, scenarios in stage_one.iterate():
        precip, pet, temp, *_ = met.fetch_station(weather_grid)
        for _, s in scenarios.iterrows():
            # Result - arrays of runoff, erosion, leaching, soil_water, rain
            scenario = \
                stage_one_to_two(precip, pet, temp, stage_two.new_year,
                                 s.plant_date, s.emergence_date, s.maxcover_date, s.harvest_date,
                                 s.max_root_depth, s.crop_intercept,
                                 s.slope, s.slope_length,
                                 s.water_max_5, s.water_min_5, s.water_max_20, s.water_min_20,
                                 s.cn_cov, s.cn_fal, s.usle_k, s.usle_ls, s.usle_c_cov, s.usle_c_fal, s.usle_p,
                                 s.irrigation_type, s.ireg, s.depletion_allowed, s.leaching_fraction)
            batch.append(scenario)
            scenario_count += 1
            scenario_vars = s[keep_fields]
            stage_two_index.append(scenario_vars.values)
            if len(batch) == batch_size or scenario_count == n_scenarios:
                run = dask.delayed()(batch)
                results = run.compute()
                batch_count += 1
                batch = []
                report(f"Processed {scenario_count} of {stage_one.n_scenarios} scenarios", 1)
                yield batch_count, results

    yield 'index', pd.DataFrame(stage_two_index, columns=keep_fields)


def build_subset(region, comids):
    if comids is not None:
        nav = Navigator(region)
        all_upstream = nav.batch_upstream(comids)
    else:
        all_upstream = None
    return all_upstream


def main():
    regions = ['07']

    # Initialize input met matrix
    met = WeatherArray()

    for region in regions:
        report("Processing Region {} scenarios...".format(region))

        # Build an experimental subset
        # THIS IS JUST TEMPORARY WHILE I BUILD AN MTB DEMO
        nav = Navigator(region)
        outlets = [4867727]
        active_reaches = nav.batch_upstream(outlets)
        subset_id = 'mtb'
        recipes = WatershedRecipes('07')

        # Initialize scenarios
        stage_one = StageOneScenarios(region, active_reaches, 2015, recipes)  # in
        stage_two = StageTwoScenarios(region, met, stage_one.fetch('scenario_id'), tag=subset_id)  # out
        
        # Run weather simulations to generate stage two scenarios
        for batch_count, result in process_batch(stage_one, stage_two, met):
            stage_two.write(batch_count, result)


if __name__ == "__main__":
    profile = False
    if profile:
        import cProfile
        cProfile.run('main()')
    else:
        main()
