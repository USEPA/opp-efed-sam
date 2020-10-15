import numpy as np
from numba import njit


@njit
def evapotranspiration_daily(plant_factor, available_soil_et, evaporation_node, root_max, anetd,
                             n_soil_increments, depth, soil_water, wilting_point, available_water, delta_x):

    soil_layer_loss = np.zeros(n_soil_increments, dtype=np.float32)
    et_factor = np.zeros(n_soil_increments, dtype=np.float32)

    # Set ET node and adjust et_depth by maximum root depth, scaled by plant growth factor
    et_node = evaporation_node
    if plant_factor > 0:
        et_depth = plant_factor * root_max
        if et_depth > anetd:
            et_node = find_node(depth, et_depth)

    # Reduce available soil ET if soil moisture < 0.6 of available water (PRZM5 Manual 4.8)
    total_soil_water = soil_water[:et_node].sum()
    total_available_water = available_water[:et_node].sum()
    if total_available_water > 0:
        frac = total_soil_water / total_available_water
        if frac < 0.6:
            available_soil_et *= frac

    # Calculate ET for each node, adjusted for depth
    for i in range(et_node):
        et_factor[i] = max(0., (depth[et_node] - depth[i] + delta_x[i]) * (soil_water[i] - wilting_point[i]))
    et_sum = et_factor.sum()
    if et_sum > 0:
        for i in range(et_node):
            soil_layer_loss[i] = available_soil_et * (et_factor[i] / et_sum)

    return soil_layer_loss, et_node


@njit
def find_node(depth, target_depth):
    return np.argmin(np.abs(depth - target_depth))


@njit
def initialize_irrigation(field_capacity, wilting_point, irrigation_type, depth, root_max,
                          irr_depletion, available_water):
    """
    Calculate total available water (field) capacity, irrigation trigger based on rooting depth
    Source: PRZM5 Manual, Section 4.4 (Young and Fry, 2016)
    """
    if irrigation_type > 0:
        irrigation_node = find_node(depth, root_max)
        target_dryness = 0
        for i in range(irrigation_node):
            target_dryness += available_water[i] * irr_depletion + wilting_point[i]
        total_fc = np.sum(field_capacity[:irrigation_node])
        return total_fc, target_dryness, irrigation_node
    else:
        return None, None, None


@njit
def leaching_daily(initial_input, n_soil_increments, field_capacity, soil_water, et_node, wilting_point,
                   soil_layer_loss):
    leaching = np.zeros(n_soil_increments, dtype=np.float32)
    remaining_leachate = initial_input
    for node in range(n_soil_increments):
        water_in_node = remaining_leachate - soil_layer_loss[node] + soil_water[node]
        leaching[node] = max(water_in_node - field_capacity[node], 0.)
        soil_water[node] = max(water_in_node - leaching[node], wilting_point[node])
        # TODO - keeping this for the moment, but it seems like this part should be taken care of in evapotranspiration
        if leaching[node] <= 0. and node > et_node:
            leaching[node:n_soil_increments] = 0.
            break
        remaining_leachate = leaching[node]
    return leaching, soil_water


@njit
def partition_precip_daily(precip, temp, snow_accumulation, sfac, irrigation_type, soil_water, irrigation_node, s,
                           target_dryness, total_fc, leaching_factor):
    # Calculate snow accumulation/melt (effective rain is rain + snow melt)
    if temp <= 0:

        snow_accumulation += precip
        rain = effective_rain = 0
    else:
        rain = precip
        snow_melt = min(snow_accumulation, (sfac / 100) * temp)
        effective_rain = rain + snow_melt
        snow_accumulation -= snow_melt

    # Process irrigation
    if effective_rain <= 0. and irrigation_type > 0:
        current_dryness = np.sum(soil_water[:irrigation_node])
        daily_max_irrigation = 0.2 * s
        if current_dryness < target_dryness:
            irrigation_required = (total_fc - current_dryness) * leaching_factor + 1.
            if irrigation_type == 3:  # overcanopy irrigation
                rain = effective_rain = min(irrigation_required, daily_max_irrigation)
            elif irrigation_type == 4:  # undercanopy irrigation
                effective_rain = min(irrigation_required, daily_max_irrigation)

    return rain, effective_rain, snow_accumulation


@njit
def runoff_and_interception_daily(rain, effective_rain, s, plant_factor, crop_intercept, canopy_water, pet):
    # Determine runoff by the Curve Number Method
    if effective_rain > (0.2 * s):
        runoff = max(0, (effective_rain - (0.2 * s)) ** 2 / (effective_rain + (0.8 * s)))
    else:
        runoff = 0

    # Determine canopy intercept
    if rain > 0.:
        # a_c_g is the % of RAIN not going to runoff
        available_canopy_gain = rain * (1. - runoff / effective_rain)
        interception = min(available_canopy_gain, (crop_intercept * plant_factor) - canopy_water)
        canopy_water += interception
    else:
        interception = 0
    canopy_water = max(0., canopy_water - pet)

    # Anything that doesn't runoff or get held up is leaching
    leaching = effective_rain - runoff - interception

    # Any PET not used to evaporate canopy water is still available
    excess_et = max(0., pet - canopy_water)
    return runoff, leaching, canopy_water, excess_et


@njit
def surface_hydrology(field_capacity, wilting_point, plant_factor, cn, depth,  # From other function output
                      irrigation_type, irr_depletion, anetd, root_max, leaching_factor, cintcp,  # From scenario
                      precip, temp, potential_et,  # From metfile
                      n_soil_increments, delta_x, sfac):  # From parameters
    """ Process hydrology parameters, returning daily runoff, soil water content, and leaching (velocity) """
    # Initialize arrays and running variables
    # Daily time series
    n_dates = plant_factor.size
    surface_velocity = np.zeros(n_dates, dtype=np.float32)
    surface_soil_water = np.zeros(n_dates, dtype=np.float32)
    daily_rain = np.zeros(n_dates, dtype=np.float32)
    daily_effective_rain = np.zeros(n_dates, dtype=np.float32)
    daily_runoff = np.zeros(n_dates, dtype=np.float32)

    # Soil profile arrays (by node)
    soil_water = field_capacity.copy()  # initialize at field capacity

    # Calculate these ahead of time for efficiency
    usle_s_factor = ((2540 / cn) - 25.4) / 100.  # cm -> m
    available_water = field_capacity - wilting_point

    total_fc, target_dryness, irrigation_node = \
        initialize_irrigation(field_capacity, wilting_point, irrigation_type, depth, root_max, irr_depletion,
                              available_water)

    # Set evaporation node
    evaporation_node = find_node(depth, anetd)

    # Running variables
    canopy_water = 0
    snow_accumulation = 0

    for day in range(precip.size):
        # 'rain' is water from above the canopy, 'effective rain' is above AND below canopy
        rain, effective_rain, snow_accumulation = \
            partition_precip_daily(precip[day], temp[day], snow_accumulation, sfac, irrigation_type, soil_water,
                                   irrigation_node, usle_s_factor[day], target_dryness, total_fc, leaching_factor)

        runoff, leaching, canopy_water, available_soil_et = \
            runoff_and_interception_daily(rain, effective_rain, usle_s_factor[day], plant_factor[day], cintcp, canopy_water,
                                          potential_et[day])

        soil_layer_loss, et_node = \
            evapotranspiration_daily(plant_factor[day], available_soil_et, evaporation_node, root_max, anetd,
                                     n_soil_increments, depth, soil_water, wilting_point, available_water, delta_x)

        velocity, soil_water = \
            leaching_daily(leaching, n_soil_increments, field_capacity, soil_water, et_node, wilting_point,
                           soil_layer_loss)

        surface_velocity[day] = velocity[0]
        surface_soil_water[day] = soil_water[0]
        daily_rain[day] = rain
        daily_effective_rain[day] = effective_rain
        daily_runoff[day] = runoff

    return daily_runoff, daily_rain, daily_effective_rain, surface_soil_water, surface_velocity
