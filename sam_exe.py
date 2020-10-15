from __future__ import division
import numpy as np
import pandas as pd
from ..base.uber_model import UberModel, ModelSharedInputs

from .Code.utilities import fields


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
    def __init__(self, pd_obj):

        # Unpack JSON string into dictionary
        # JCH - Taking a dict at the moment, will need json
        super(InputDict, self).__init__((k, v['0']) for k, v in pd_obj.to_dict().items())

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
            report("Input field(s) \"{}\" not understood".format(", ".join(unknown_fields)), warn=1)
        assert not missing_fields, "Required input field(s) \"{}\" not provided".format(", ".join(missing_fields))

    def coerce_data_type(self):
        input_fields = fields.fetch('input_param')
        for field, data_type in zip(input_fields, fields.data_type(input_fields)):
            self[field] = data_type(self[field])

    def process_applications(self):

        input_indices = {'event': ['plant', 'harvest', 'emergence', 'bloom', 'maturity'],
                         'dist': ['ground', 'foliar'],
                         'method': ['uniform', 'step']}

        # Get fields and field types
        app_fields = fields.fetch_old('applications')
        app_fields.remove('crop')
        data_types = fields.data_type(app_fields, old_fields=True)

        # Populate matrix
        matrix = []
        for i in range(int(self['napps'])):
            crops = self.pop("crop_{}".format(i + 1)).split(" ")
            row_fields = ["{}_{}".format(field, i + 1) for field in app_fields]
            for crop in crops:
                row = [int(float(crop))]
                for field, field_type in zip(app_fields, data_types):
                    val = self["{}_{}".format(field, i + 1)]
                    replacement = input_indices.get(field)
                    if replacement:
                        val = replacement.index(val)
                    row.append(field_type(val))
                matrix.append(row)
            for field in row_fields:
                del self[field]

        return np.float32(matrix)

    def process_dates(self):
        date_format = lambda x: np.datetime64("{2}-{0}-{1}".format(*x.split("/")))
        return map(date_format, (self['sim_date_start'], self['sim_date_end']))

    def process_endpoints(self):
        from .Code.utilities import endpoint_format

        endpoints = []
        for level in ('acute', 'chronic', 'overall'):
            endpoints.append([self.pop("{}_{}".format(level, species), np.nan) for species in endpoint_format.species])
        endpoints = np.array(endpoints)
        endpoints[endpoints == ''] = np.nan

        return np.float32(endpoints)


class Sam(UberModel, SamInputs, SamOutputs):
    """
    Estimate chemical exposure from drinking water alone in birds and mammals.
    """

    def __init__(self, pd_obj, pd_obj_exp):
        """Class representing the Terrplant model and containing all its methods"""
        super(Sam, self).__init__()
        self.pd_obj = pd_obj
        self.pd_obj_exp = pd_obj_exp
        self.pd_obj_out = pd.DataFrame(data=np.array([[0, 0], [0, 0]]), columns=["foo", "bar"])
        self.input_dict = InputDict(self.pd_obj)

    def execute_model(self):
        from old import PesticideCalculator as pesticide_calculator
        pesticide_calculator(self.input_dict)
