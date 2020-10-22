import re
import pandas as pd
import numpy as np
from hydro.nhd.params_nhd import nhd_states
from tools.efed_lib import FieldManager
from paths import types_path, fields_and_qc_path

# Initialize field matrix
fields = FieldManager(fields_and_qc_path)


class ParameterSet(object):
    def __init__(self, entries):
        self.__dict__.update(entries)


# First and last days of scenarios
scenario_start_date = np.datetime64('1980-01-01')
scenario_end_date = np.datetime64('2010-12-31')

# Unpacking scenario id (e.g., MO-CDo2r1a1l2-18071-1)
scenario_id_fmt = re.compile("([A-Z]{2})-(.+?)-(\d+?)-(\d{1,3})")  # state, soil id, weather grid, cdl
soil_id_fmt = re.compile("([A-D]{1,2})o(\d+)r(\d+)a(\d+)l(\d+)")  # hsg_letter, slope, orgC_5, sand_5, clay_5

stage_one_chunksize = 1000000
batch_size = 100
scenario_defaults = {'depletion_allowed': 0.1, 'leaching_frac': 0.5}

# Crop IDs in the input correspond to this field in the Stage 1 scenario file
crop_group_field = 'pwc_class'

""" These parameters can be adjusted during test runs """
output_params = {
    # Turn output datasets on and off
    "write_contributions": False,
    "write_exceedances": True,
    "write_time_series": True,
}

""" Parameters below are hardwired model parameters """

# Parameters related directly to pesticide degradation
# These inputs have defaults because pesticide-specific data are generally not available
# Future versions of SAM could accommodate alternate inputs, if available
plant_params = {
    "deg_foliar": 0.0,  # per day; assumes stability on foliage.
    "washoff_coeff": 0.1,  # Washoff coefficient; default value
}

# Parameters related to soils in the field
# Sources: PRZM5 Revision A Documentation (Young and Fry, 2016); PWC User Manual (Young, 2016)
# New version of PWC uses non-uniform runoff extraction extending below 2 cm - future update to SAM
# New version of PWC has alternate approach to PRBEN based on pesticide Kd - future update to SAM
# deplallw and leachfrac apply when irrigation triggered
soil_params = {
    "anetd": 0.08,  # 8 cm
    "cm_2": 0.75,  # Soil distribution, top 2 cm. Revised for 1 compartment - uniform extraction
    "runoff_effic": 0.266,  # Runoff efficiency, assuming uniform 2-cm layer, from PRZM User's Manual (PUM)
    "prben": 0.5,  # PRBEN factor - default PRZM5, MMF
    "erosion_effic": 0.266,  # Erosion effic. - subject to change, MMF, frac. of eroded soil interacting w/ pesticide
    "soil_depth": 0.1,  # soil depth in cm - subject to change, MMF; lowest depth erosion interacts w/ soil (PUM)
    "surface_increments": 1,  # number of increments in top 2-cm layer: 1 COMPARTMENT, UNIFORM EXTRACTION
    "n_increments": 20,  # number of increments in 2nd 100-cm layer (not used in extraction)
    "surface_dx": 0.02,  # TODO - cm?
    "layer_dx": 0.05,  # TODO - cm?
    "cn_min": 0.001,  # curve number to use if unavailable or <0,
    "sfac": 0.247  # snowmelt factor
}

# Time of Travel defaults
hydrology_params = {
    "flow_interpolation": 'quadratic',  # Must be None, 'linear', 'quadratic', or 'cubic'
    "gamma_convolve": False,
    "convolve_runoff": False,
    "minimum_residence_time": 1.5  # Minimum residence time in days for a reservoir to be treated as a reservoir
}

# Water Column Parameters - USEPA OPP defaults used in PWC, from VVWM documentation
# Sources: PRZM5 Revision A Documentation (Young and Fry, 2016); PWC User Manual (Young, 2016)
# corrections based on PWC/VVWM defaults
water_column_params = {
    "dfac": 1.19,  # default photolysis parameter from VVWM
    "sused": 30,  # water column suspended solid conc (mg/L); corrected to PWC/VVWM default
    "chloro": 0.005,  # water column chlorophyll conc (mg/L); corrected to PWC/VVWM default
    "froc": 0.04,  # water column organic carbon fraction on susp solids; corrected to PWC/VVWM default
    "doc": 5,  # water column dissolved organic carbon content (mg/L)
    "plmas": 0.4  # water column biomass conc (mg/L); corrected to PWC/VVWM default
}

# Benthic Parameters - USEPA OPP defaults from EXAMS/VVWM used in PWC
# Sources: PRZM5 Revision A Documentation (Young and Fry, 2016); PWC User Manual (Young, 2016)
# corrections based on PWC/VVWM defaults
benthic_params = {
    "depth": 0.05,  # benthic depth (m)
    "porosity": 0.50,  # benthic porosity (fraction); corrected to PWC/VVWM default
    "bulk_density": 1.35,  # bulk density, dry solid mass/total vol (g/cm3); corrected to PWC/VVWM default
    "froc": 0.04,  # benthic organic carbon fraction; corrected to PWC/VVWM default
    "doc": 5,  # benthic dissolved organic carbon content (mg/L)
    "bnmas": 0.006,  # benthic biomass intensity (g/m2); corrected to PWC/VVWM default
    "d_over_dx": 1e-8  # mass transfer coeff. for exch. betw. benthic, water column (m/s); corrected to PWC/VVWM default
}

# Create parameter sets
plant_params = ParameterSet(plant_params)
soil_params = ParameterSet(soil_params)
hydrology_params = ParameterSet(hydrology_params)
water_column_params = ParameterSet(water_column_params)
benthic_params = ParameterSet(benthic_params)
output_params = ParameterSet(output_params)

# Build a typical soil profile
depth_bins = np.array([5, 20])  # should match 'depth_bins' in aquatic_model_inputs/parameters.py
soil_params.delta_x = np.array([soil_params.surface_dx] + [soil_params.layer_dx] * (soil_params.n_increments - 1))
soil_params.depth = np.cumsum(soil_params.delta_x)
soil_params.bins = np.minimum(depth_bins.size - 1, np.digitize(soil_params.depth * 100., depth_bins))

"""
Values are from Table F1 of TR-55 (tr_55.csv), interpolated values are included to make arrays same size
type column is rainfall parameter (corresponds to IREG in PRZM5 manual) found in met_data.csv
rainfall is based on Figure 3.3 from PRZM5 manual (Young and Fry, 2016), digitized and joined with weather grid ID
Map source: Appendix B in USDA (1986). Urban Hydrology for Small Watersheds, USDA TR-55.
Used in the model to calculate time of concentration of peak flow for use in erosion estimation.
met_data.csv comes from Table 4.1 in the PRZM5 Manual (Young and Fry, 2016)
"""
types = pd.read_csv(types_path).set_index('type')

# All states
states = sorted(set().union(*nhd_states.values()))
