import numpy as np
import pandas as pd
from ..base.uber_model import UberModel, ModelSharedInputs

from .pesticide_calculator import pesticide_calculator


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


class Sam(UberModel, SamInputs, SamOutputs):
    def __init__(self, pd_obj, dummy_param=None):
        super(Sam, self).__init__()
        self.pd_obj = pd_obj
        self.pd_obj_out = pd.DataFrame(data=np.array([[0, 0], [0, 0]]), columns=['foo', 'bar'])

    def execute_model(self):
        self.pd_obj_out = pesticide_calculator(self.pd_obj)
