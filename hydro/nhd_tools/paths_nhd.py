import os

global_dir = r"E:\opp-efed-data\global"
local_dir = r"E:\opp-efed-data\hydro"
table_dir = r"A:\opp-efed\hydro\Tables"

# Tables
fields_and_qc_path = os.path.join(table_dir, "fields_and_qc.csv")
nhd_map_path = os.path.join(table_dir, "nhd_map.csv")

# HydroFiles
navigator_map_path = os.path.join(table_dir, "nhd_map_nav.csv")
navigator_path = os.path.join(local_dir, "NavigatorFiles", "nav{}.npz")  # region

# Path containing NHD Plus dataset
nhd_dir = os.path.join(global_dir, "NHDPlusV21")
nhd_region_dir = os.path.join(nhd_dir, "NHDPlus{}", "NHDPlus{}")  # vpu, region
catchment_path = os.path.join(nhd_region_dir, "NHDPlusCatchment", "Catchment.shp")

# Intermediate
condensed_nhd_path = os.path.join(local_dir, "CondensedNHD", 'nhd_{}_r{}_{}.csv')  # run_id, region, feature_type

