from .utilities import Simulation, HydroRegion, WatershedRecipes, WeatherArray, ModelOutputs, report
from .reach_processing import ReachManager
from .scenario_processing import StageOneScenarios, StageTwoScenarios, StageThreeScenarios


def pesticide_calculator(input_data):
    # Initialize a class with all the simulation parameters (input data, field names, hardwired parameters, dates etc)
    sim = Simulation(input_data)

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

        # Initialize Stage 2 scenarios (time series of non-chemical data, e.g., runoff, erosion, rainfall...)
        stage_two = StageTwoScenarios(region, sim, stage_one, met)
        if sim.build_scenarios:  # If 'build' mode is on, skip the rest of the process
            continue

        # Initialize Stage 3 scenarios (time series of chemical transport data e.g., runoff mass, erosion mass)
        stage_three = StageThreeScenarios(sim, stage_one, stage_two)

        # Initialize output object
        outputs = ModelOutputs(sim, region, stage_three)

        # Initialize objects to hold results by stream reach and reservoir
        reaches = ReachManager(sim, stage_two, stage_three, region, recipes, outputs)

        # Combine scenarios to generate data for catchments
        # Traverse downstream in the watershed
        for tier, reach_ids, lakes in region.cascade:
            report(f'Running tier {tier}, ({len(reach_ids)} reaches)...')
            # TODO - parallelize
            reaches.process_local(reach_ids)

            # Perform full analysis including time-of-travel and concentration for active reaches
            reaches.process_full(reach_ids & set(region.full_reaches))

            # Pass each reach in the tier through a downstream lake
            reaches.burn(lakes)

        # Write output
        # TODO - spin up results tables on the site?
        report('Writing output...')
        # intake_dict = {'COMID': {4867727: {'acute_human': 1.0,
        # reach_dict = {'comid': {'5640192': 0.0...}, 'huc_8': {'01010101': 0.0,..
        intake_dict, reach_dict = outputs.prepare_output(write=False)
        return {'intakes': intake_dict, 'reaches': reach_dict}
