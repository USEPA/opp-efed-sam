import dask
import numpy as np
from field import plant_growth, initialize_soil, process_erosion
from hydrology import surface_hydrology
from transport import pesticide_to_field, field_to_soil, soil_to_water
from parameters import soil_params, types, fields
from ast import literal_eval

@dask.delayed
def stage_one_to_two(precip, pet, temp, new_year,  # weather params
                     plant_date, emergence_date, maxcover_date, harvest_date,  # crop dates
                     max_root_depth, crop_intercept,  # crop properties
                     slope, slope_length,  # field properties
                     fc_5, wp_5, fc_20, wp_20,  # soil properties
                     cn_cov, cn_fallow, usle_k, usle_ls, usle_c_cov, usle_c_fal, usle_p,  # curve numbers and usle vars
                     irrigation_type, ireg, depletion_allowed, leaching_fraction):  # irrigation params

    # Model the growth of plant between emergence and maturity (defined as full canopy cover)
    plant_factor = plant_growth(precip.size, new_year, plant_date, emergence_date, maxcover_date, harvest_date)

    # Initialize soil properties for depth
    cn, field_capacity, wilting_point, usle_klscp = \
        initialize_soil(plant_factor, cn_cov, cn_fallow, usle_c_cov, usle_c_fal, fc_5, wp_5, fc_20,
                        wp_20, usle_k, usle_ls, usle_p, soil_params.cn_min, soil_params.delta_x, soil_params.bins)

    runoff, rain, effective_rain, soil_water, leaching = \
        surface_hydrology(field_capacity, wilting_point, plant_factor, cn, soil_params.depth,
                          irrigation_type, depletion_allowed, soil_params.anetd, max_root_depth, leaching_fraction,
                          crop_intercept, precip, temp, pet, soil_params.n_increments, soil_params.delta_x,
                          soil_params.sfac)

    # Calculate erosion loss
    type_matrix = types[types.index == ireg].values.astype(np.float32)  # ireg parameter
    erosion = process_erosion(slope, runoff, effective_rain, cn, usle_klscp, type_matrix, slope_length)

    # Output array order is specified in fields_and_qc.py
    arrays = []
    for field in fields.fetch('s2_arrays'):
        arrays.append(eval(field))
    return np.float32(arrays)


def stage_two_to_three(application_matrix, new_year, kd_flag, koc, deg_aqueous, leaching, runoff, erosion, soil_water,
                       rain, cm_2, delta_x_top_layer, erosion_effic, soil_depth, deg_foliar,
                       washoff_coeff, runoff_effic, plant_date, emergence_date, maxcover_date, harvest_date,
                       covmax, org_carbon, bulk_density, season):
    # Use Kd instead of Koc if flag is on. Kd = Koc * organic C in the top layer of soil
    # Reference: PRZM5 Manual(Young and Fry, 2016), Section 4.13
    if kd_flag:
        koc *= org_carbon

    # Calculate the application of pesticide to the landscape
    plant_dates = [plant_date, emergence_date, maxcover_date, harvest_date]
    application_mass = pesticide_to_field(application_matrix, new_year, plant_dates, rain)

    # Calculate plant factor (could have this info for s2 scenarios, but if it's quick then it saves space)
    plant_factor = plant_growth(runoff.size, new_year, plant_date, emergence_date, maxcover_date, harvest_date)

    # Calculate the daily mass of applied pesticide that reaches the soil (crop intercept, runoff)
    pesticide_mass_soil = field_to_soil(application_mass, rain, plant_factor, cm_2,
                                        deg_foliar, washoff_coeff, covmax)

    # Determine the loading of pesticide into runoff and eroded sediment
    # Also need to add deg_soil, deg_benthic here - NT 8/28/18
    aquatic_pesticide_mass = \
        soil_to_water(pesticide_mass_soil, runoff, erosion, leaching, bulk_density, soil_water, koc,
                      deg_aqueous, runoff_effic, delta_x_top_layer, erosion_effic, soil_depth)

    return aquatic_pesticide_mass
