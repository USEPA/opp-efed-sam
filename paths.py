import os
import pathlib

# If running locally (Trip's computer), point to an external hard drive. If in AWS, use a different path
local_run = True
if local_run:
    sam_root = r"E:\opp-efed-data\sam"
    local_root = r"A:\opp-efed\sam"
else:
    sam_root = "/src/app-data/sampreprocessed"
    local_root = pathlib.Path(__file__).parent.absolute()

scenario_root = os.path.join("scenarios", "Production")
input_dir = os.path.join(sam_root, "Inputs")
intermediate_dir = os.path.join(sam_root, "Intermediate")
output_path = os.path.join("Results")
scratch_path = os.path.join("temp")
diagnostic_path = os.path.join(sam_root, "diagnostic")

# Input data
condensed_nhd_path = os.path.join(input_dir, "CondensedNHD", "{}", "r{}_{}.csv")  # 'nav' or 'sam', region, 'reach' or 'waterbody'
weather_path = os.path.join(input_dir, "Weather", "weather_{}")  # 'array' or 'key'
recipe_path = os.path.join(input_dir, "RecipeFiles", "r{}")  # region
stage_one_scenario_path = os.path.join(input_dir, "SamScenarios", "r{}_{}.csv")  # region, i
dwi_path = os.path.join(input_dir, "Intakes", "intake_locations.csv")
manual_points_path = os.path.join(input_dir, "Intakes", "mtb_single_intake.csv")
navigator_path = os.path.join(input_dir, "NavigatorFiles", "nav{}.npz")  # region

# Intermediate data
stage_two_scenario_path = os.path.join(intermediate_dir, "StageTwoScenarios", "r{}")  # region

# Tables
table_root = os.path.join(local_root, "Tables")
endpoint_format_path = os.path.join(table_root, "endpoint_format.csv")
fields_and_qc_path = os.path.join(table_root, "fields_and_qc.csv")
types_path = os.path.join(table_root, "tr_55.csv")
sam_nhd_map = os.path.join(table_root, "nhd_map_sam.csv")

