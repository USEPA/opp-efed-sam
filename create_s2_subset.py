from .utilities import WeatherArray, WatershedRecipes
from .scenarios import StageOneScenarios, StageTwoScenarios
from .hydro.navigator import Navigator
from .utilities import report


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
        StageTwoScenarios(region, stage_one, met, tag=subset_id)  # out



if __name__ == "__main__":
    profile = False
    if profile:
        import cProfile
        cProfile.run('main()')
    else:
        main()
