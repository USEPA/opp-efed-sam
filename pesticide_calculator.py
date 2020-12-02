from .utilities import Simulation, HydroRegion, ModelOutputs, WatershedRecipes, WeatherArray, report
from .scenarios import StageOneScenarios, StageTwoScenarios, StageThreeScenarios
from .reaches import ReachManager, process_local_batch, process_full_batch, burn_batch


def pesticide_calculator(input_data):
    # Initialize parameters from front end
    sim = Simulation(input_data)

    # Loop through all NHD regions included in selected runs
    for region_id in sim.run_regions:
        report("Processing hydroregion {}...".format(region_id))

        # Initialize a weather file reader
        met = WeatherArray(sim)

        # Load recipes for region and year
        recipes = WatershedRecipes(region_id, sim)

        # Initialize Stage 1 scenarios (parameters linked to a unique soil-weather-land cover combination)
        stage_one = StageOneScenarios(region_id, sim, recipes)

        # Initialize Stage 2 scenarios (time series of non-chemical data, e.g., runoff, erosion, rainfall...)
        stage_two = StageTwoScenarios(region_id, sim, stage_one, met, tag='mtb', build=False)

        # Initialize Stage 3 scenarios (time series of chemical transport data e.g., runoff mass, erosion mass)
        stage_three = StageThreeScenarios(sim, stage_two)

        # Load watershed topology maps and account for necessary files
        region = HydroRegion(region_id, sim)

        # Initialize output object
        outputs = ModelOutputs(sim, region.output_reaches, stage_two.start_date, stage_two.end_date)

        # Cascade downstream processing watershed recipes and performing travel time analysis
        for year in [2015]:  # manual years

            # Initialize objects to hold results by stream reach and reservoir
            reaches = ReachManager(sim, stage_three, region, year)

            # Combine scenarios to generate data for catchments
            report("Processing recipes for {}...".format(year))

            # Traverse downstream in the watershed
            for tier, reach_ids, lakes in region.cascade:
                report(f"Running tier {tier}, ({len(reach_ids)} reaches)...")

                # Crunch the scenarios for each reach
                process_local_batch(reaches, reach_ids, recipes, stage_two, stage_three, sim, year)

                # Perform full analysis including time-of-travel and concentration for active reaches
                process_full_batch(reaches, reach_ids, sim, region)

                # Pass each reach in the tier through a downstream lake
                burn_batch(lakes)

        # Write output
        report("Writing output...")
        outputs.write_output()
        return outputs.json_output
