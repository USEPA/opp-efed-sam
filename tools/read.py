import numpy as np
import pandas as pd
from dbfread import DBF, FieldParser
from tools.efed_lib import report


def dbf(dbf_file):
    """ Read the contents of a dbf file into a Pandas dataframe """

    class MyFieldParser(FieldParser):
        def parse(self, field, data):
            try:
                return FieldParser.parse(self, field, data)
            except ValueError as e:
                report(e)
                # raise e
                return None

    try:
        reader = DBF(dbf_file)
        table = pd.DataFrame(iter(reader))
    except ValueError:
        reader = DBF(dbf_file, parserclass=MyFieldParser)
        table = pd.DataFrame(iter(reader))

    table.rename(columns={column: column.lower() for column in table.columns}, inplace=True)

    return table


def gdb(gdb_file, select_table='all', input_fields=None):
    """ Reads the contents of a gdb table into a Pandas dataframe"""
    import ogr

    # Initialize file
    driver = ogr.GetDriverByName("OpenFileGDB")
    gdb_obj = driver.Open(gdb_file)

    # parsing layers by index
    tables = {gdb_obj.GetLayerByIndex(i).GetName(): i for i in range(gdb_obj.GetLayerCount())}
    table_names = sorted(tables.keys()) if select_table == 'all' else [select_table]
    for table_name in table_names:
        table = gdb_obj.GetLayer(tables[table_name])
        table_def = table.GetLayerDefn()
        table_fields = [table_def.GetFieldDefn(i).GetName() for i in range(table_def.GetFieldCount())]
        if input_fields is None:
            input_fields = table_fields
        else:
            missing_fields = set(input_fields) - set(table_fields)
            if any(missing_fields):
                report("Fields {} not found in table {}".format(", ".join(missing_fields), table_name))
                input_fields = [field for field in input_fields if field not in missing_fields]
        data = np.array([[row.GetField(f) for f in input_fields] for row in table])
        df = pd.DataFrame(data=data, columns=input_fields)
        if select_table != 'all':
            return df
        else:
            yield table_name, df
