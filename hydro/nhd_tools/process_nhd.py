import os
import numpy as np
import pandas as pd

import read_nhd
import write_nhd
from params_nhd import vpus_nhd, fields_hydro as fields
from paths_nhd import nhd_region_dir
from efed_lib_hydro.read import dbf, report


def condense_nhd(region, field_map_path, rename_field='internal_name'):
    """
    Pull data from NHD Plus dbf files and consolidate in a .csv file that is smaller to store and easier to read.
    Tables and fields to be pulled are specified in an NHD map table. A template may be found in Tables/nhd_map.csv.
    :param region: NHD Hydroregion id (str)
    :param field_map_path: Path to the field map table (str)
    :param run_id: Identifier appended to the output (str)
    :param rename_field: Field in the field map which contains new field names to conver to (str)
    :return:
    """
    # TODO - too many rows from plusflow?
    # Read in the NHD map specifying which tables and fields to read
    field_map = read_nhd.nhd_map(field_map_path, rename_field=rename_field)

    # Check to see if the output files have already been created
    report(f"Condensing NHD Plus from map file {field_map_path}...")

    # Path to NHDPlus files for the given region
    region_path = nhd_region_dir.format(vpus_nhd[region], region)

    # Initialize separate tables for waterbodies and reaches
    reach_table = None
    lake_table = None

    # Loop through each NHD Plus table with fields selected in nhd_map.csv
    for (path, table_name, feature_type), subset in field_map.groupby(['path', 'table', 'feature_type']):
        print(f"Reading {table_name}...")

        # Read the table and select the fields in the field map
        table_path = os.path.join(region_path, path, table_name + ".dbf")
        table = dbf(table_path)[subset.field.values]

        # Rename fields if specified
        if rename_field is not None:
            rename_dict = dict(subset[~pd.isnull(subset[rename_field])][['field', rename_field]].values.tolist())
            table = table.rename(columns=rename_dict)

        # Append table to master
        table = table.drop_duplicates()
        if feature_type == 'reach':
            reach_table = reach_table.merge(table, on='comid', how='outer') if reach_table is not None else table
        elif feature_type == 'waterbody':
            table = table.rename(columns={'comid': 'wb_comid'})
            lake_table = lake_table.merge(table, on='wb_comid', how='outer') if lake_table is not None else table
        else:
            raise ValueError(f"Invalid feature type {feature_type}. Must be 'reach' or 'waterbody'")
    return reach_table, lake_table


def process_divergence(nhd_table):
    # Add the divergence and streamcalc of downstream reaches to each row
    downstream = nhd_table[['comid', 'divergence', 'stream_calc', 'fcode']]
    downstream.columns = ['tocomid'] + [f + "_ds" for f in downstream.columns.values[1:]]
    downstream = nhd_table[['comid', 'tocomid']].drop_duplicates().merge(
        downstream.drop_duplicates(), how='left', on='tocomid')

    # Where there is a divergence, select downstream reach with the highest streamcalc or lowest divergence
    downstream = downstream.sort_values('stream_calc_ds', ascending=False).sort_values('divergence_ds')
    downstream = downstream[~downstream.duplicated('comid')]
    nhd_table = nhd_table.merge(downstream, on=['comid', 'tocomid'], how='inner')

    return nhd_table


def calculate_surface_area(nhd_table):
    # Calculate surface area
    stream_channel_a = 4.28
    stream_channel_b = 0.55
    cross_section = nhd_table.q_ma / nhd_table.v_ma
    return stream_channel_a * np.power(cross_section, stream_channel_b)


def identify_outlet_reaches(nhd_table):
    # Indicate whether reaches are coastal
    nhd_table['coastal'] = np.int16(nhd_table.pop('fcode') == 56600)

    # Identify basin outlets
    nhd_table['outlet'] = 0

    # Identify all reaches that are a 'terminal path'. HydroSeq is used for Terminal Path ID in the NHD
    nhd_table.loc[nhd_table.hydroseq.isin(nhd_table.terminal_path), 'outlet'] = 1

    # Identify all reaches that empty into a reach outside the region
    nhd_table.loc[~nhd_table.tocomid.isin(nhd_table.comid) & (nhd_table.stream_calc > 0), 'outlet'] = 1

    # Designate coastal reaches as outlets. These don't need to be accumulated
    nhd_table.loc[nhd_table.coastal == 1, 'outlet'] = 1

    # Sever connection between outlet and downstream reaches
    nhd_table.loc[nhd_table.outlet == 1, 'tocomid'] = 0

    return nhd_table


def identify_waterbody_outlets(wb_table, reach_table):
    """
    Identifies stream outlets for each waterbody in the NHDPlus dataset. Fields to carry over are specified in the
    'lentic' column of fields_and_qc.csv
    :param wb_table: Condensed NHD waterbodies dataset for a region
    :param reach_table: Condensed NHD Plus reaches dataset for a region
    :return: Table with waterbody comids and associated reach outlet information
    """
    fields.refresh()
    erom_months = [str(m).zfill(2) for m in range(1, 13)] + ['ma']
    fields.expand('monthly', erom_months)

    # Get a table of all lentic reaches, with the COMID of the reach and waterbody
    lentic_table = reach_table[fields.fetch('lentic')].rename(columns={'q_ma': 'flow'})

    """ Identify the outlet reach corresponding to each reservoir """
    # Filter the reach table down to only outlet reaches by getting the minimum hydroseq for each wb_comid
    lentic_table = lentic_table.sort_values("hydroseq").groupby("wb_comid", as_index=False).first()
    del lentic_table['hydroseq']

    # Join the outlets to the waterbodies
    wb_table = wb_table.merge(lentic_table, how='left', on='wb_comid')

    return wb_table.rename(columns={'comid': 'outlet_comid'})
