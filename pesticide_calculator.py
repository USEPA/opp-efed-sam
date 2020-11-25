from .utilities import Simulation, HydroRegion, ModelOutputs, WatershedRecipes, ReachManager, report
from .scenarios import StageTwoScenarios, StageThreeScenarios
from .paths import local_run, local_root, data_root
report(f"Local run? {local_run}")
report(f"Local root: {local_root}")
report(f"Data root: {data_root}")


# TODO - why are there fewer scenarios than recipes?
# TODO - what do years do?
# TODO - Check the output arrays
#  Check the local arrays
#  Check the results
# TODO - confirm that the stage 2 scenario data looks good
# TODO - check units, esp NHD stuff (flow et al), check residence times
# TODO - did we change erosion? like, is the area modifier still needed?
# TODO - what are the bottlenecks? faster?
# TODO - should it really raise an error when the application dates start to go out of bounds?
# TODO - confirm that it's handling double-crops and seasons right
#  In fact, design unit tests for all different components - recipe recall, etc.

def pesticide_calculator(input_data):
    # Initialize parameters from front end
    sim = Simulation(input_data)

    # Loop through all NHD regions included in selected runs
    for region_id in sim.run_regions:
        report("Processing hydroregion {}...".format(region_id))

        # Initialize Stage 2 scenarios (time series of non-chemical data, e.g., runoff, erosion, rainfall...)
        stage_two = StageTwoScenarios(region_id, sim=sim, tag='mtb')

        # Initialize Stage 3 scenarios (time series of chemical transport data e.g., runoff mass, erosion mass)
        stage_three = StageThreeScenarios(sim, stage_two)

        report(f"Building Stage 3 scenarios...")
        stage_three.build_from_stage_two()

        # Load watershed topology maps and account for necessary files
        region = HydroRegion(sim, region_id)

        # Load recipes for region and year
        recipes = WatershedRecipes(region_id)

        # Initialize output object
        outputs = ModelOutputs(sim, region.output_reaches, stage_two.start_date, stage_two.end_date)

        # Initialize objects to hold results by stream reach and reservoir
        reaches = ReachManager(stage_two, stage_three, recipes, region, outputs)

        # Cascade downstream processing watershed recipes and performing travel time analysis
        for year in [2015]:  # manual years

            # Combine scenarios to generate data for catchments
            report("Processing recipes for {}...".format(year))

            # Traverse downstream in the watershed
            for tier, reach_ids, lakes in region.cascade:

                report(f"Running tier {tier}, ({len(reach_ids)} reaches)...")

                # Crunch the scenarios for each reach
                reaches.process_local_batch(reach_ids, year)

                # Perform full analysis including time-of-travel and concentration for active reaches
                for reach_id in reach_ids & set(region.output_reaches):
                    reaches.report(reach_id)

                # Pass each reach in the tier through a downstream lake
                reaches.burn_batch(lakes)

        # Write output
        report("Writing output...")
        outputs.write_output()
        return outputs.json_output
