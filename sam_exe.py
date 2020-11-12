import numpy as np
import pandas as pd

from .utilities import fields
from .tools.efed_lib import report
from .pesticide_calculator import pesticide_calculator

from ..base.uber_model import UberModel, ModelSharedInputs


class SamInputs(ModelSharedInputs):
    """
    Input class for SAM.
    """

    def __init__(self):
        """Class representing the inputs for SAM"""
        super(SamInputs, self).__init__()


class SamOutputs(object):
    """
    Output class for SAM.
    """

    def __init__(self):
        """Class representing the outputs for SAM"""
        super(SamOutputs, self).__init__()


class InputDict(dict):
    """ Processes the input string from the front end into a form usable by tool """

    def __init__(self, pd_obj):

        # Unpack JSON string into dictionary
        # TODO - Taking a dict at the moment, will need json
        super(InputDict, self).__init__((k, v['0']) for k, v in pd_obj.items())

        self['applications'] = self.process_applications()
        self['endpoints'] = self.process_endpoints()
        self['sim_date_start'], self['sim_date_end'] = self.process_dates()
        self.coerce_data_type()
        self.check_fields()

    def check_fields(self):

        # Check if any required input data are missing or extraneous data are provided
        provided_fields = set(self.keys())
        required_fields = set(fields.fetch('input_param')) | {'applications', 'endpoints'}
        unknown_fields = provided_fields - required_fields
        missing_fields = required_fields - provided_fields
        if unknown_fields:
            report("Input field(s) \"{}\" not understood".format(", ".join(unknown_fields)))
        assert not missing_fields, "Required input field(s) \"{}\" not provided".format(", ".join(missing_fields))

    def coerce_data_type(self):
        _, data_types = fields.fetch('input_param', dtypes=True)
        for field, data_type in data_types.items():
            if data_type != object:
                self[field] = data_type(self[field])

    def process_applications(self):

        # Get fields and field types
        app_fields, data_types = fields.fetch('applications', dtypes=True)

        # Populate matrix
        matrix = []
        for app_num in range(1, int(self['napps']) + 1):
            crops = self[f"crop_{app_num}"].split(" ")
            for crop in crops:
                row = []
                for field in app_fields:
                    if field == 'crop':
                        val = crop
                    else:
                        val = self[f"{field}_{app_num}"]
                    dtype = data_types[field]
                    if dtype != object:
                        val = dtype(val)
                    row.append(val)
                matrix.append(row)
            for field in app_fields:
                del self[f"{field}_{app_num}"]

        return matrix

    def process_dates(self):
        date_format = lambda x: np.datetime64("{2}-{0}-{1}".format(*x.split("/")))
        return map(date_format, (self['sim_date_start'], self['sim_date_end']))

    def process_endpoints(self):
        from .utilities import endpoint_format

        endpoints = []
        for level in ('acute', 'chronic', 'overall'):
            endpoints.append([self.pop("{}_{}".format(level, species), np.nan) for species in endpoint_format.species])
        endpoints = np.array(endpoints)
        endpoints[endpoints == ''] = np.nan

        return np.float32(endpoints)


class Sam():
    def __init__(self, pd_obj, dummy_param=None):
        super(Sam, self).__init__()
        self.pd_obj = pd_obj
        self.pd_obj_out = pd.DataFrame(data=np.array([[0, 0], [0, 0]]), columns=["foo", "bar"])
        self.input_dict = InputDict(self.pd_obj)

    def execute_model(self):
        self.pd_obj_out = pesticide_calculator(self.input_dict)
