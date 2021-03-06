import numpy as np
import pandas as pd

from .pesticide_calculator import pesticide_calculator


class Sam(object):
    def __init__(self, pd_obj, dummy_param=None):
        super(Sam, self).__init__()
        self.pd_obj = pd_obj
        self.pd_obj_out = pd.DataFrame(data=np.array([[0, 0], [0, 0]]), columns=['foo', 'bar'])

    def execute_model(self):
        self.pd_obj_out = pesticide_calculator(self.pd_obj)

def main(build=False):
    """ This is what gets run when running straight from Python """
    if build:
        from .dev.test_inputs import atrazine_json_mtb_build as input_dict
    else:
        from .dev.test_inputs import atrazine_json_mtb as input_dict
    print('Running pesticide calculator...')
    sam = Sam(input_dict)
    sam.execute_model()

