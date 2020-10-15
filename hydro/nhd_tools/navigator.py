import os
import numpy as np
import pandas as pd
import read_nhd
import write_nhd
from paths_nhd import navigator_path, navigator_map_path, condensed_nhd_path
from efed_lib_hydro.efed_lib import report
from process_nhd import identify_outlet_reaches, process_divergence, condense_nhd
from params_nhd import nhd_regions


class Navigator(object):
    def __init__(self, region_id, upstream_path=None):
        if upstream_path is None:
            upstream_path = navigator_path.format(region_id)
        self.file = upstream_path.format(region_id, 'nav', 'npz')
        self.paths, self.times, self.lengths, \
        self.map, self.alias_to_reach, self.reach_to_alias = self.load()
        self.reach_ids = set(self.reach_to_alias.keys())

    def load(self):
        assert os.path.isfile(self.file), "Upstream file {} not found".format(self.file)
        data = np.load(self.file, mmap_mode='r', allow_pickle=True)
        conversion_array = data['alias_index']
        reverse_conversion = dict(zip(conversion_array, np.arange(conversion_array.size)))
        return data['paths'], data['time'], data['length'], data['path_map'], conversion_array, reverse_conversion

    def upstream_watershed(self, reach_id, mode='reach', return_times=False, return_lengths=False, return_warning=False,
                           verbose=False):

        def unpack(array):
            first_row = [array[start_row][start_col:]]
            remaining_rows = list(array[start_row + 1:end_row])
            return np.concatenate(first_row + remaining_rows)

        # Look up reach ID and fetch address from pstream object
        reach = reach_id if mode == 'alias' else self.reach_to_alias.get(reach_id)
        reaches, adjusted_times, warning = np.array([]), np.array([]), None
        try:
            start_row, end_row, col = map(int, self.map[reach])
            start_col = list(self.paths[start_row]).index(reach)
        except TypeError:
            warning = "Reach {} not found in region".format(reach)
        except ValueError:
            warning = "{} not in upstream lookup".format(reach)
        else:
            # Fetch upstream reaches and times
            aliases = unpack(self.paths)
            reaches = aliases if mode == 'alias' else np.int32(self.alias_to_reach[aliases])

        # Determine which output to deliver
        output = [reaches]
        if return_times:
            times = unpack(self.times)
            adjusted_times = np.int32(times - self.times[start_row][start_col])
            output.append(adjusted_times)
        if return_lengths:
            lengths = unpack(self.lengths)
            output.append(lengths)
        if return_warning:
            output.append(warning)
        if verbose and warning is not None:
            report(warning, warn=1)
        return output[0] if len(output) == 1 else output

    def find_upstream(self, source_name, comid, join_table, direction='up'):
        try:
            upstream = np.array(self.upstream_watershed(int(comid), return_times=True))
        except NameError:
            return None
        upstream = pd.DataFrame(upstream.T, columns=['comid', 'days'])
        upstream_sites = upstream.merge(join_table, on='comid', how='inner')
        upstream_sites['direction'] = direction
        if direction == 'up':
            upstream_sites['station_id'] = source_name
            upstream_sites = upstream_sites.rename(columns={'site_id': 'intake_id'})
        elif direction == 'down':
            upstream_sites['intake_id'] = source_name
            upstream_sites = upstream_sites.rename(columns={'site_id': 'station_id'})
            upstream_sites['days'] = 0 - upstream_sites.days
        return upstream_sites

    def batch_upstream(self, reaches):
        all_upstream = {upstream for reach in reaches for upstream in self.upstream_watershed(reach)}
        return pd.Series(sorted(all_upstream), name='comid')


def collapse_array(paths, times, lengths):
    """
    Reduce the size of input arrays by truncating at the path length
    :param paths: Array with node IDs (np.array)
    :param times: Array with reach travel times (np.array)
    :param lengths: Array with reach lengths (np.array)
    :return:
    """
    out_paths = []
    out_times = []
    out_lengths = []
    path_starts = []
    for i, row in enumerate(paths):
        active_path = (row > 0)
        path_starts.append(np.argmax(active_path))
        out_paths.append(row[active_path])
        out_times.append(times[i][active_path])
        out_lengths.append(lengths[i][active_path])
    return map(np.array, (out_paths, out_times, out_lengths, path_starts))


def map_paths(paths):
    """
    Get the starting row and column for each path in the path array
    :param paths: Path array (np.array)
    :return:
    """

    column_numbers = np.tile(np.arange(paths.shape[1]) + 1, (paths.shape[0], 1)) * (paths > 0)
    path_begins = np.argmax(column_numbers > 0, axis=1)
    max_reach = np.max(paths)
    path_map = np.zeros((max_reach + 1, 3))
    n_paths = paths.shape[0]
    for i, path in enumerate(paths):
        for j, val in enumerate(path):
            if val:
                if i == n_paths:
                    end_row = 0
                else:
                    next_row = (path_begins[i + 1:] <= j)
                    if next_row.any():
                        end_row = np.argmax(next_row)
                    else:
                        end_row = n_paths - i - 1
                values = np.array([i, i + end_row + 1, j])
                path_map[val] = values

    return path_map


def process_nhd(nhd_table):
    nhd_table = process_divergence(nhd_table)
    nhd_table = identify_outlet_reaches(nhd_table)
    nhd_table = nhd_table[nhd_table.comid != 0]
    return nhd_table


def rapid_trace(nodes, outlets, times, dists, conversion, max_length=3000, max_paths=500000):
    """
    Trace upstream through the NHD Plus hydrography network and record paths,
    times, and lengths of traversals.
    :param nodes: Array of to-from node pairs (np.array)
    :param outlets: Array of outlet nodes (np.array)
    :param times: Array of travel times corresponding to nodes (np.array)
    :param dists: Array of flow lengths corresponding to nodes (np.array)
    :param conversion: Array to interpret node aliases (np.array)
    :param max_length: Maximum length of flow path (int)
    :param max_paths: Maximum number of flow paths (int)
    :return:
    """
    # Output arrays
    all_paths = np.zeros((max_paths, max_length), dtype=np.int32)
    all_times = np.zeros((max_paths, max_length), dtype=np.float32)
    all_dists = np.zeros((max_paths, max_length), dtype=np.float32)

    # Bounds
    path_cursor = 0
    longest_path = 0

    progress = 0  # Master counter, counts how many reaches have been processed
    already = set()  # This is diagnostic - the traversal shouldn't hit the same reach more than once

    # Iterate through each outlet
    for i in np.arange(outlets.size):
        start_node = outlets[i]

        # Reset everything except the master. Trace is done separately for each outlet
        queue = np.zeros((nodes.shape[0], 2), dtype=np.int32)
        active_reach = np.zeros(max_length, dtype=np.int32)
        active_times = np.zeros(max_length, dtype=np.float32)
        active_dists = np.zeros(max_length, dtype=np.float32)

        # Cursors
        start_cursor = 0
        queue_cursor = 0
        active_reach_cursor = 0
        active_node = start_node

        # Traverse upstream from the outlet.
        while True:
            # Report progress
            progress += 1
            if not progress % 10000:
                report(progress, 3)
            upstream = nodes[nodes[:, 0] == active_node]

            # Check to make sure active node hasn't already been passed
            l1 = len(already)
            already.add(conversion[active_node])
            if len(already) == l1:
                report("Loop at reach {}".format(conversion[active_node]))
                exit()

            # Add the active node and time to the active path arrays
            active_reach[active_reach_cursor] = active_node
            active_times[active_reach_cursor] = times[active_node]
            active_dists[active_reach_cursor] = dists[active_node]

            # Advance the cursor and determine if a longest path has been set
            active_reach_cursor += 1
            if active_reach_cursor > longest_path:
                longest_path = active_reach_cursor

            # If there is another reach upstream, continue to advance upstream
            if upstream.size:
                active_node = upstream[0][1]
                for j in range(1, upstream.shape[0]):
                    queue[queue_cursor] = upstream[j]
                    queue_cursor += 1

            # If not, write the active path arrays into the output matrices
            else:
                all_paths[path_cursor, start_cursor:] = active_reach[start_cursor:]
                all_times[path_cursor] = np.cumsum(active_times) * (all_paths[path_cursor] > 0)
                all_dists[path_cursor] = np.cumsum(active_dists) * (all_paths[path_cursor] > 0)
                queue_cursor -= 1
                path_cursor += 1
                last_node, active_node = queue[queue_cursor]
                if last_node == 0 and active_node == 0:
                    break
                for j in range(active_reach.size):
                    if active_reach[j] == last_node:
                        active_reach_cursor = j + 1
                        break
                start_cursor = active_reach_cursor
                active_reach[active_reach_cursor:] = 0.
                active_times[active_reach_cursor:] = 0.
                active_dists[active_reach_cursor:] = 0.

    return all_paths[:path_cursor, :longest_path], \
           all_times[:path_cursor, :longest_path], \
           all_dists[:path_cursor, :longest_path]


def unpack_nhd(nhd_table):
    """
    Extract nodes, times, distances, and outlets from NHD table
    :param nhd_table: Table of NHD Plus parameters (df)
    :return: Modified NHD table with selected fields
    """
    # Extract nodes and travel times
    nodes = nhd_table[['tocomid', 'comid']]
    times = nhd_table['travel_time'].values
    dists = nhd_table['lengthkm'].values * 1000.  # km -> m

    # Create an alias for nodes
    convert = pd.Series(np.arange(nhd_table.comid.size), index=nhd_table.comid.drop_duplicates())
    nodes = nodes.apply(lambda row: row.map(convert)).fillna(-1).astype(np.int32)

    # Extract outlets from aliased nodes
    outlets = nodes.comid[nhd_table.outlet == 1].values

    # Create a lookup key to convert aliases back to comids
    conversion_array = convert.sort_values().index.values

    # Return nodes, travel times, outlets, and conversion
    return nodes.values, times, dists, outlets, conversion_array


def build_navigator(region, nhd_table):
    """
    Initializes the creation of a Navigator object, which is used for rapid
    delineation of watersheds using NHD Plus catchments.
    :param nhd_table: Table of stream reach parameters from NHD Plus (df)
    """
    report("Processing NHD...", 2)
    nhd_table = process_nhd(nhd_table)

    report("Unpacking NHD...", 2)
    nodes, times, dists, outlets, conversion = unpack_nhd(nhd_table)

    report("Tracing upstream...", 2)
    # paths, times = self.upstream_trace(nodes, outlets, times)
    # TODO - add clean capability to cumulatively trace any attribute (e.g time, distance)
    paths, times, dists = rapid_trace(nodes, outlets, times, dists, conversion)

    report("Mapping paths...", 2)
    path_map = map_paths(paths)

    report("Collapsing array...", 2)
    paths, times, length, start_cols = collapse_array(paths, times, dists)

    write_nhd.navigator_file(region, paths, times, length, path_map, conversion)


def build_navigators():
    nhd_regions = ['07']
    overwrite = False
    for region in nhd_regions:
        nhd_path = condensed_nhd_path.format('nav', region, 'reach')
        if overwrite or not os.path.exists(nhd_path):
            reach_table, _ = \
                condensed = condense_nhd(region, navigator_map_path, 'internal_name')
            write_nhd.condensed_nhd('nav', region, reach_table)
        else:
            reach_table = read_nhd.condensed_nhd('nav', region, 'reach')

        build_navigator(region, reach_table)


if __name__ == '__main__':
    build_navigators()
