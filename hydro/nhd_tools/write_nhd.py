from paths_nhd import condensed_nhd_path, navigator_path
import numpy as np
import os


def create_dir(outfile):
    directory = os.path.dirname(outfile)
    if not os.path.exists(directory):
        os.makedirs(directory)


def condensed_nhd(run_id, region, reach_table, lake_table=None, out_dir=None):
    out_dir = condensed_nhd_path if out_dir is None else out_dir
    create_dir(condensed_nhd_path)
    for feature_type, table in (('reach', reach_table), ('waterbody', lake_table)):
        if table is not None:
            out_path = out_dir.format(run_id, region, feature_type)
            table.to_csv(out_path, index=None)


def navigator_file(region, paths, times, length, path_map, conversion):
    create_dir(navigator_path)
    outfile = navigator_path.format(region)
    np.savez_compressed(outfile, paths=paths, time=times, length=length, path_map=path_map,
                        alias_index=conversion)
