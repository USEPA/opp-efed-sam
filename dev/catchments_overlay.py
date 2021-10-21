"""
Create a crosswalk between NHD Plus catchments and a set of monitoring sites
"""
import os
import arcpy
import pandas as pd


def overlay(points_file, catchment_file, temp_file, point_id_field):
    # Initialize layers for points and catchments
    arcpy.MakeFeatureLayer_management(points_file, 'points')
    arcpy.MakeFeatureLayer_management(catchment_file, 'catchments')

    # Perform the intersection, writing the output to the temp file
    arcpy.Intersect_analysis(['points', 'catchments'], temp_file)

    # Make an overlay table
    columns = ['comid', point_id_field]
    overlay = \
        pd.DataFrame([r for r in arcpy.da.SearchCursor(temp_file, ["FEATUREID", point_id_field])], columns=columns)
    return overlay


def cleanup(temp_file):
    for f in (temp_file, 'points', 'catchments'):
        try:
            arcpy.Delete_management(f)
        except Exception as e:
            print(e)


def get_crosswalk(crosswalk_file):
    columns = ["comid", "region"]
    return pd.DataFrame([r for r in arcpy.da.SearchCursor(crosswalk_file, ["COMID", "VPUID"])], columns=columns)


def main():
    # Input files
    nhd_dir = r"E:\opp-efed-data\global\NHDPlusV21\NHDPlusNationalData\NHDPlusV21_National_Seamless_Flattened_Lower48.gdb"
    points_path = os.path.join(".", "points.shp")
    points_id_field = 'site_id'
    catchment_path = os.path.join(nhd_dir, r"\NHDPlusCatchment\Catchment")
    region_lookup = os.path.join(nhd_dir, "NHDSnapshot", "NHDFlowline_Network")

    # Output files
    run_id = "monitoring_sites"
    temp_path = os.path.join(".", "intersect.shp")
    output_path = os.path.join(".", "{}_overlay.csv".format(run_id))

    # Get a crosswalk between comids and regions
    region_comid = get_crosswalk(region_lookup)

    # Run the overlay
    overlay_table = overlay(points_path, catchment_path, temp_path, output_path, points_id_field)

    # Assign regions to the overlay
    overlay_table = overlay_table.merge(region_comid, on="comid", how="left")

    # Write to file
    overlay_table.to_csv(output_path)


main()
