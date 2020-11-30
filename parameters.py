import numpy as np


class ParameterSet(object):
    def __init__(self, entries):
        self.__dict__.update(entries)


class ParameterManager(object):
    def __init__(self):
        # First and last days of scenarios
        # TODO - is this right?
        self.scenario_start_date = np.datetime64('1980-01-01')
        self.scenario_end_date = np.datetime64('2010-12-31')

        self.stage_one_chunksize = 1000000
        self.batch_size = 100
        self.scenario_defaults = {'depletion_allowed': 0.1,
                                  'leaching_frac': 0.5}

        # Crop IDs in the input correspond to this field in the Stage 1 scenario file
        self.crop_group_field = 'pwc_class'

        # Parameters related directly to pesticide degradation
        # These inputs have defaults because pesticide-specific data are generally not available
        # Future versions of SAM could accommodate alternate inputs, if available
        self.plant = ParameterSet({
            "deg_foliar": 0.0,  # per day; assumes stability on foliage.
            "washoff_coeff": 0.1})  # Washoff coefficient; default value

        # Parameters related to soils in the field
        # Sources: PRZM5 Revision A Documentation (Young and Fry, 2016); PWC User Manual (Young, 2016)
        # New version of PWC uses non-uniform runoff extraction extending below 2 cm - future update to SAM
        # New version of PWC has alternate approach to PRBEN based on pesticide Kd - future update to SAM
        # deplallw and leachfrac apply when irrigation triggered
        depth_bins = np.array([5, 20])  # should match 'depth_bins' in aquatic_model_inputs/parameters.py
        self.soil = ParameterSet({
            "anetd": 0.08,  # 8 cm
            "cm_2": 0.75,  # Soil distribution, top 2 cm. Revised for 1 compartment - uniform extraction
            "runoff_effic": 0.266,  # Runoff efficiency, assuming uniform 2-cm layer, from PRZM User's Manual (PUM)
            "prben": 0.5,  # PRBEN factor - default PRZM5, MMF
            "erosion_effic": 0.266,  # Erosion effic. frac. of eroded soil interacting w/ pesticide
            "soil_depth": 0.1,
            # soil depth in cm - subject to change, MMF; lowest depth erosion interacts w/ soil (PUM)
            "surface_increments": 1,  # number of increments in top 2-cm layer: 1 COMPARTMENT, UNIFORM EXTRACTION
            "n_increments": 20,  # number of increments in 2nd 100-cm layer (not used in extraction)
            "surface_dx": 0.02,  # TODO - cm?
            "layer_dx": 0.05,  # TODO - cm?
            "cn_min": 0.001,  # curve number to use if unavailable or <0,
            "sfac": 0.247})  # snowmelt factor

        # Build a soil profile
        self.soil.delta_x = np.array(
            [self.soil.surface_dx] + [self.soil.layer_dx] * (self.soil.n_increments - 1))
        self.soil.depth = np.cumsum(self.soil.delta_x)
        self.soil.bins = np.minimum(depth_bins.size - 1, np.digitize(self.soil.depth * 100., depth_bins))

        # Time of Travel defaults
        self.hydrology = ParameterSet(
            {"flow_interpolation": 'quadratic',  # Must be None, 'linear', 'quadratic', or 'cubic'
             "gamma_convolve": False,
             "convolve_runoff": False,
             "minimum_residence_time": 1.5})  # Minimum residence time in days to be treated as a reservoir)

        # Water Column Parameters - USEPA OPP defaults used in PWC, from VVWM documentation
        # Sources: PRZM5 Revision A Documentation (Young and Fry, 2016); PWC User Manual (Young, 2016)
        # corrections based on PWC/VVWM defaults
        self.water_column = \
            ParameterSet({"dfac": 1.19,  # default photolysis parameter from VVWM
                          "sused": 30,  # water column suspended solid conc (mg/L)
                          "chloro": 0.005,  # water column chlorophyll conc (mg/L)
                          "froc": 0.04,  # water column organic carbon fraction on susp solids
                          "doc": 5,  # water column dissolved organic carbon content (mg/L)
                          "plmas": 0.4})  # water column biomass conc (mg/L); corrected to PWC/VVWM default)

        # Benthic Parameters - USEPA OPP defaults from EXAMS/VVWM used in PWC
        # Sources: PRZM5 Revision A Documentation (Young and Fry, 2016); PWC User Manual (Young, 2016)
        # corrections based on PWC/VVWM defaults
        self.benthic = ParameterSet({
            "depth": 0.05,  # benthic depth (m)
            "porosity": 0.50,  # benthic porosity (fraction); corrected to PWC/VVWM default
            "bulk_density": 1.35,  # bulk density, dry solid mass/total vol (g/cm3); corrected to PWC/VVWM default
            "froc": 0.04,  # benthic organic carbon fraction; corrected to PWC/VVWM default
            "doc": 5,  # benthic dissolved organic carbon content (mg/L)
            "bnmas": 0.006,  # benthic biomass intensity (g/m2); corrected to PWC/VVWM default
            "d_over_dx": 1e-8})  # mass transfer coeff. for exch. betw. benthic, water column (m/s)

        self.output = ParameterSet({
            # Turn output datasets on and off
            "write_contributions": False,
            "write_exceedances": True,
            "write_time_series": True})
