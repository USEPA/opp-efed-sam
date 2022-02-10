from .hydrology import HydroRegion
from .utilities import Simulation, WeatherArray, ModelOutputs, report, scenario_qaqc
from .reach_processing import ReachManager, WatershedRecipes
from .scenario_processing import StageOneScenarios, StageTwoScenarios, StageThreeScenarios

retain_s1 = True
retain_s3 = True
qaqc_scenarios = False


# ISSUES:
#  Attach r/e array to s3
#  0s for erosion in s2
#  'cannot convert float NaN to integer' error after 49800 on s2-s3
#  Should there be a confine function for s3?  Yes. Not s2 but s3

def pesticide_calculator(input_data):
    # Initialize a class with all the simulation parameters (input data, field names, hardwired parameters, dates etc)
    sim = Simulation(input_data, retain_s1, retain_s3)

    # Iterate through each hydroregion that encompasses the run
    for region_id in sim.run_regions:
        report('Processing hydroregion {}...'.format(region_id))

        # Initialize a weather file reader
        met = WeatherArray(sim)

        # Load watershed topology maps and account for necessary files
        region = HydroRegion(region_id, sim)

        # Load recipes for region and year
        recipes = WatershedRecipes(region_id, sim)

        # Initialize Stage 1 scenarios (parameters linked to a unique soil-weather-land cover combination)
        stage_one = StageOneScenarios(region, sim, recipes)

        # Initialize output object
        outputs = ModelOutputs(sim, region)

        if sim.random:
            # No need to do any scenarios processing if generating random output
            continue

        # Initialize Stage 2 scenarios (time series of non-chemical data, e.g., runoff, erosion, rainfall...)
        stage_two = StageTwoScenarios(region, sim, stage_one, met)
        if sim.build_scenarios:  # If 'build' mode is on, skip the rest of the process
            report("Successfully finished building Stage Two Scenarios.")
            return

        # Initialize Stage 3 scenarios (time series of chemical transport data e.g., runoff mass, erosion mass)
        stage_three = StageThreeScenarios(sim, stage_one, stage_two)

        # Examine the scenarios if QAQC is turned on
        if qaqc_scenarios:
            scenario_qaqc(stage_two, stage_three, recipes)

        # Initialize objects to hold results by stream reach and reservoir
        reaches = ReachManager(sim, stage_three, region, recipes, outputs)

        # Combine scenarios to generate data for catchments
        for tier, reach_ids, lakes in region.cascade():  # Traverse downstream in the watershed
            report(f'Running tier {tier}, ({len(reach_ids)} reaches)...')
            upstream_reaches = reach_ids & set(region.upstream_reaches)
            output_reaches = reach_ids & set(region.intake_reaches)

            # Perform analysis within reach catchments
            report("\tProcessing local...")
            reaches.process_local(reach_ids, output_reaches)

            # Perform full upstream analysis including time-of-travel and concentration
            report("\tProcessing upstream...")
            reaches.process_upstream(upstream_reaches, output_reaches)

            # Pass each reach in the tier through a downstream lake
            reaches.burn(lakes)

    # Write output
    report('Writing output...')
    intake_dict, reach_dict, intake_time_series_dict = outputs.prepare_output()
    return {'intakes': intake_dict, 'reaches': reach_dict, 'intake_time_series': intake_time_series_dict}