import os
import re
import pandas as pd
import numpy as np
from collections import Iterable
from tempfile import mkstemp
import time

class MemoryMatrix(object):
    """ A wrapper for NumPy 'memmap' functionality which allows the storage and recall of arrays from disk """

    def __init__(self, dimensions, dtype=np.float32, path=None, existing=False, name='null', verbose=False,
                 index_col=0, persistent_read=False, persistent_write=False, max_calls=500):
        # TODO - right now, no capability to 'fetch' on any dimension other than the first
        self.dtype = dtype
        self.path = path
        self.existing = existing
        self.name = name
        self.index_col = index_col
        self.max_calls = max_calls

        self.shape = []
        self.labels = []
        self.index = None
        self.aliased = False
        self.index_cols = None
        for i, d in enumerate(dimensions):
            if isinstance(d, Iterable):
                if i == index_col:
                    self.aliased = True
                    self.index = dimensions[0]
                    self.index_cols = pd.DataFrame({'alias': self.index, 'i': np.arange(len(self.index))}).set_index(
                        'alias')
                self.shape.append(len(d))
            else:
                self.shape.append(d)
        self.shape = tuple(self.shape)
        self.initialize_array(verbose)
        self.n_calls = 0

        self.persistent_read = persistent_read
        self.persistent_write = persistent_write
        self._reader = None
        self._writer = None

        self.refresh()

    def alias_to_index(self, aliases, remove_missing=False, return_missing=False, verbose=True):
        singular = not isinstance(aliases, Iterable)
        if singular:
            aliases = [aliases]

        result = self.index_cols.reindex(aliases).reset_index()
        if any((remove_missing, return_missing)):
            found = ~pd.isnull(result.i)
            missing = result[~found].alias
            if verbose and not missing.empty:
                report(f"Missing {missing.shape[0]} values from {self.name} array")
            if remove_missing:
                result = result[found]
        if verbose and result.empty:
            report(f"No records found matching {aliases} in {self.name} array")
        if singular:
            result = result.astype(np.int32).iloc[0]
        if return_missing:
            return result, missing
        else:
            return result

    def read(self, index, pop=False, copy=False):
        # Read from the array
        reader = self.reader
        output = reader[index]
        if pop or copy:
            output = np.array(output)
            if pop:  # Set the selected rows to zero after extracting array
                reader[index] = 0.
        self.count_call()
        return output

    def count_call(self):
        self.n_calls += 1
        if self.n_calls >= self.max_calls:
            self.refresh()

    def get_index(self, index, iloc):
        # Convert aliased item(s) to indices if applicable
        if self.aliased and not iloc:
            alias_lookup, missing = self.alias_to_index(index, True, True, verbose=True)
            index = alias_lookup.i
        else:
            alias_lookup = None
        return index, alias_lookup

    def fetch(self, index, copy=False, verbose=False, iloc=False, pop=False, return_alias=False):
        index, alias_lookup = self.get_index(index, iloc)
        output = self.read(index, pop, copy)
        if alias_lookup is not None and return_alias:
            return output, alias_lookup
        else:
            return output

    def initialize_array(self, verbose=False):

        # Load from saved file if one is specified, else generate
        if self.path is None:
            self.existing = False
            self.path = mkstemp(suffix=".dat", dir=os.path.join("..", "bin", "temp"))[1]
        else:
            # Add suffix to path if one wasn't provided
            if not self.path.endswith("dat"):
                self.path += ".dat"
            if os.path.exists(self.path):
                self.existing = True
        if not self.existing:
            if verbose:
                report("Creating memory map {}...".format(self.path))
            try:
                os.makedirs(os.path.dirname(self.path))
            except FileExistsError:
                pass
            np.memmap(self.path, dtype=self.dtype, mode='w+', shape=tuple(self.shape))  # Allocate memory

    def write(self, index, values):
        array = self.writer
        array[index] = values
        self.count_call()

    def update(self, index, values, iloc=False, verbose=True):
        if self.aliased and not iloc:
            location, missing = self.alias_to_index(index, return_missing=True)
            if not missing.empty:
                if verbose:
                    report(f"Unable to update array '{self.name}': {index} not found in array index")
                return
            index = location.i
        self.write(index, values)

    def refresh(self):
        if self.persistent_read:
            del self._reader
            self._reader = np.memmap(self.path, dtype=self.dtype, mode='r+', shape=self.shape)

        if self.persistent_write:
            del self._writer
            mode = 'r+' if os.path.isfile(self.path) else 'w+'
            self._writer = np.memmap(self.path, dtype=self.dtype, mode=mode, shape=self.shape)
        self.n_calls = 0

    @property
    def reader(self):
        if self._reader is not None:
            return self._reader
        else:
            return np.memmap(self.path, dtype=self.dtype, mode='r+', shape=self.shape)

    @property
    def writer(self):
        if self._writer is not None:
            return self._writer
        else:
            mode = 'r+' if os.path.isfile(self.path) else 'w+'
            return np.memmap(self.path, dtype=self.dtype, mode=mode, shape=self.shape)


class DateManager(object):
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def adjust_dates(self, start_offset=0, end_offset=0):
        try:
            self.start_date += start_offset
            self.end_date += end_offset
        except TypeError:

            self.start_date += np.timedelta64(int(start_offset), 'D')
            self.end_date += np.timedelta64(int(end_offset), 'D')

    def date_offset(self, start_date, end_date, coerce=True, n_dates=None):
        """
        Find the overlap between the class' date range and the provided date range
        :param start_date: The beginning of the provided date range (datetime)
        :param end_date: The end of the provided date range (datetime)
        :param coerce: Adjust the class range to the overlap between ranges (bool)
        :return:
        """
        # A positive number indicates that the provided dates are inside the range, negative indicates outside
        start_offset = start_date - self.start_date
        end_offset = self.end_date - end_date
        if coerce:
            self.adjust_dates(start_offset, end_offset)
        if n_dates is not None and end_offset.astype('int') == 0:
            end_offset = np.timedelta64(-n_dates, 'D')
        return start_offset.astype('int'), end_offset.astype('int')

    @property
    def dates(self):
        return pd.date_range(self.start_date, self.end_date)

    @property
    def dates_julian(self):
        return (self.dates - self.dates[0]).days.astype(int)

    @property
    def mid_month(self):
        return self.months[:-1] + (self.months[1:] - self.months[:-1]) / 2

    @property
    def mid_month_julian(self):
        return np.int32((self.mid_month - self.dates[0]).days)

    @property
    def months(self):
        return pd.date_range(self.dates.min(), self.dates.max() + np.timedelta64(1, 'M'), freq='MS')

    @property
    def months_julian(self):
        return pd.date_range(self.dates.min(), self.dates.max() + np.timedelta64(1, 'M'), freq='MS').month

    @property
    def new_year(self):
        return np.int32([(np.datetime64("{}-01-01".format(year)) - self.start_date).astype(int)
                         for year in np.unique(self.dates.year)])

    @property
    def n_dates(self):
        return self.dates.size

    @property
    def month_index(self):
        return self.dates.month

    @property
    def year_index(self):
        return np.int16(self.dates.year - self.dates.year[0])

    @property
    def year_length(self):
        return np.unique(self.dates, return_counts=True)[1]


class FieldManager(object):
    """
    The Field Manager loads the table fields_and_qc.csv
    and uses that table to manage fields. Field management functions include: field name
    conversions from raw input data sources to internal field names, extending fields
    that are expansible (such as those linked to soil horizon or month), and performing
    QAQC by comparing the values in a table with the specified QC ranges in fields_and_qc.py
    This class is inherited by other classes which wrap pandas DataFrames.
    """

    def __init__(self, path, name_col='internal_name'):
        """ Initialize a FieldManager object. """
        self.path = path
        self.name_col = name_col
        self.matrix = None
        self.extended = []
        self.refresh()
        self._convert = None

        self.qc_fields = ['range_min', 'range_max', 'range_flag',
                          'general_min', 'general_max', 'general_flag',
                          'blank_flag', 'fill_value']

    def data_type(self, fetch=None, cols=None):
        """
        Give dtypes for fields in table
        :param fetch: Fetch a subset of keys, e.g. 'monthly' (str)
        :param how: 'internal' or 'external'
        :param cols: Only return certain columns (iter of str)
        :return: Dictionary with keys as field names and dtypes as values
        """
        if fetch:
            matrix = self.fetch(fetch, self.name_col, False)
        else:
            matrix = self.matrix
        data_types = matrix.set_index(self.name_col).data_type.to_dict()

        if cols is not None:
            data_types = {key: val for key, val in data_types.items() if key in cols}
        return {key: eval(val) for key, val in data_types.items()}

    def expand(self, select_field, numbers):
        """
        Certain fields are repeated during processing - for example, the streamflow (q) field becomes monthly
        flow (q_1, q_2...q_12), and soil parameters linked to soil horizon will have multiple values for a single
        scenario (e.g., sand_1, sand_2, sand_3). This function adds these extended fields to the FieldManager.
        :param mode: 'depth', 'horizon', or 'monthly'
        :param n_horizons: Optional parameter when specifying a number to expand to (int)
        """
        if type(numbers) == int:
            numbers = np.arange(numbers) + 1

        # Check to make sure it's only been extended once
        if not select_field in self.extended:
            condition = select_field + '_extended'
            # Find each row that applies, duplicate, and append to the matrix
            self.matrix[condition] = 0
            burn = self.matrix[condition].copy()
            new_rows = []
            for idx, row in self.matrix[self.matrix[select_field] == 1].iterrows():
                burn.iloc[idx] = 1
                for i in numbers:
                    new_row = row.copy()
                    new_row[self.name_col] = row[self.name_col] + "_" + str(i)
                    new_row[condition] = 1
                    new_rows.append(new_row)
            new_rows = pd.concat(new_rows, axis=1).T

            # Filter out the old rows and add new ones
            self.matrix = pd.concat([self.matrix[~(burn == 1)], new_rows], axis=0)

            # Record that the duplication has occurred
            self.extended.append(select_field)

    def fetch(self, source, dtypes=False, field_filter=None, index_field='internal_name'):
        """
        Subset the FieldManager matrix (fields_and_qc.csv) based on the values in a given column
        If the numbers are ordered, the returned list of fields will be in the same order. The names_only parameter
        can be turned off to return all other fields (e.g., QAQC fields) from fields_and_qc.csv for the same subset.
        :param source: The column in fields_and_qc.csv used to make the selection (str)
        :param dtypes: Return the data types for each column (bool)
        :param field_filter: Only return column names if they appear in the filter (iter)
        :return: Subset of the field matrix (df)
        """

        try:
            out_fields = self.matrix[self.matrix[source] > 0]
            if out_fields[source].max() > 0:
                out_fields = out_fields.sort_values(source)[index_field].values
            if field_filter is not None:
                out_fields = [f for f in out_fields if f in field_filter]
            data_type = self.data_type(cols=out_fields)
        except KeyError as e:
            raise e
            report("Unrecognized sub-table '{}'".format(source))
            out_fields, data_type = None, None

        if dtypes:
            return out_fields, data_type
        else:
            return out_fields

    @property
    def convert(self, from_col='external_name', to_col='internal_name'):
        """ Dictionary that can be used to convert 'external' variable names to 'internal' names """
        if self._convert is None:
            self._convert = {row[from_col]: row[to_col] for _, row in self.matrix.iterrows()}
        return self._convert

    def qc_table(self):
        """ Initializes an empty QAQC table with the QAQC fields from fields_and_qc_csv. """

        return self.matrix.set_index(self.name_col)[self.qc_fields] \
            .apply(pd.to_numeric, downcast='integer') \
            .dropna(subset=self.qc_fields, how='all')

    def perform_qc(self, other):
        """
        Check the value of all parameters in table against the prescribed QAQC ranges in fields_and_qc.csv.
        There are 3 checks performed: (1) missing data, (2) out-of-range data, and (3) 'general' ranges.
        The result of the check is a copy of the data table with the data replaced with flags. The flag values are
        set in fields_and_qc.csv - generally, a 1 is a warning and a 2 is considered invalid. The outfile parameter
        gives the option of writing the resulting table to a csv file if a path is provided.
        :param other: The table upon which to perform the QAQC check (df)
        :param outfile: Path to output QAQC file (str)
        :return: QAQC table (df)
        """
        # Confine QC table to fields in other table
        active_fields = {field for field in self.qc_table().index.values if field in other.columns.tolist()}
        qc_table = self.qc_table().loc[active_fields]

        # Flag missing data
        # Note - if this fails, check for fields with no flag or fill attributes
        # This can also raise an error if there are duplicate field names in fields_and_qc with qc parametersz
        flags = pd.isnull(other).astype(np.int8)
        duplicates = qc_table.index[qc_table.index.duplicated()]
        if not duplicates.empty:
            raise ValueError(f"Multiple QC ranges specified for {', '.join(duplicates.values)} in fields table")
        flags = flags.mask(flags > 0, qc_table.blank_flag, axis=1)

        # Flag out-of-range data
        for test in ('general', 'range'):
            ranges = qc_table[[test + "_min", test + "_max", test + "_flag"]].dropna()
            for param, (param_min, param_max, flag) in ranges.iterrows():
                if flag > 0:
                    out_of_range = ~other[param].between(param_min, param_max) * flag
                    flags[param] = np.maximum(flags[param], out_of_range).astype(np.int8)
        qc_table = pd.DataFrame(np.zeros(other.shape, dtype=np.int8), columns=other.columns)
        qc_table[flags.columns] = flags

        return qc_table

    def fill(self):
        """ Return the fill values for flagged data set in fields_and_qc.csv """
        return self.matrix.set_index(self.name_col).fill_value.dropna()

    def refresh(self):
        """ Reload fields_and_qc.csv, undoes 'extend' and other modifications """
        # Read the fields/QC matrix
        if self.path is not None:
            self.matrix = pd.read_csv(self.path)
        self.extended = []


def report(message, tabs=0):
    """ Display a message with a specified indentation """
    tabs = "\t" * tabs
    print(tabs + str(message))
