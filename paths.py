import os
import pathlib
import sys


class PathManager(object):
    def __init__(self):
        # Root directories
        self.data_root = r"E:\opp-efed-data\sam" if self.local_run else "/src/app-data/sampreprocessed"
        self.local_root = pathlib.Path(__file__).parent.absolute()
        self.input_path = os.path.join(self.data_root, "Inputs")
        self.intermediate_path = os.path.join(self.data_root, "Intermediate")
        self.output_path = os.path.join(self.data_root, "Results")
        self.scratch_path = os.path.join(self.data_root, "temp")
        self.table_path = os.path.join(self.local_root, "Tables")

        # Input data
        self.condensed_nhd = os.path.join(self.input_path, "CondensedNHD", "{}", "r{}_{}.csv")
        # 'nav' or 'sam', region, 'reach' or 'waterbody'
        self.weather = os.path.join(self.input_path, "Weather", "weather_{}")  # 'array' or 'key'
        self.recipes = os.path.join(self.input_path, "RecipeFiles", "r{}")  # region
        self.s1_scenarios = os.path.join(self.input_path, "SamScenarios", "r{}_{}.csv")  # region, i
        self.dw_intakes = os.path.join(self.input_path, "Intakes", "intake_locations.csv")
        self.manual_intakes = os.path.join(self.input_path, "Intakes", "mtb_single_intake.csv")
        self.navigator = os.path.join(self.input_path, "NavigatorFiles", "nav{}.npz")  # region
        self.nhd_wbd_xwalk = os.path.join(self.input_path, "NHD_HUC_Crosswalk", "CrosswalkTable_NHDplus_HU12.csv")

        # Intermediate data
        self.s2_scenarios = os.path.join(self.intermediate_path, "StageTwoScenarios", "r{}")  # region
        self.s3_scenarios = os.path.join(self.scratch_path, "r{}")  # region

        # Tables
        self.fields_and_qc = os.path.join(self.table_path, "fields_and_qc.csv")
        self.types = os.path.join(self.table_path, "tr_55.csv")
        self.sam_nhd_map = os.path.join(self.table_path, "nhd_map_sam.csv")
        self.endpoint_format = os.path.join(self.table_path, "endpoint_format.csv")


    @property
    def local_run(self):
        return any([r'C:' in p for p in sys.path])

