import numpy as np
from numba import njit


@njit
def erosion_coefficients(daily_rain, daily_cn, type_matrix):
    # Calculate potential maximum retention after runoff begins (ia_over_p = S in NRCS curve number equa.)
    # PRZM5 Manual (Young and Fry, 2016), Sect 4.6; TR-55 (USDA NRCS, 1986), Chapter 2 """
    if daily_rain > 0:
        ia_over_p = .0254 * (200. / daily_cn - 2.) / daily_rain  # 0.2 * s, in inches
    else:
        ia_over_p = 0

    # lower and upper limit of applicability of NRCS Curve Number method according to TR-55
    # Source: PRZM5 Manual (Young and Fry, 2016), Sect 4.6; TR-55 (USDA NRCS, 1986), Chapter 2
    c = np.zeros(type_matrix.shape[1])
    if ia_over_p <= 0.1:
        c[:] = type_matrix[0]
    elif ia_over_p >= 0.5:
        c[:] = type_matrix[-1]
    else:  # interpolation of intermediate. clunky because numba
        lower = (20. * (ia_over_p - 0.05)) - 1
        delta = type_matrix[int(lower) + 1] - type_matrix[int(lower)]
        interp = (lower % 1) * delta
        c[:] = type_matrix[int(lower)] + interp

    return c


@njit
def initialize_soil(plant_factor, cn_crop, cn_fallow, usle_c_crop, usle_c_fallow, fc_5, wp_5, fc_20, wp_20,
                    usle_k, usle_ls, usle_p, cn_min, delta_x, soil_bins):
    # Interpolate curve number and c factor based on canopy coverage
    cn_daily = cn_fallow + (plant_factor * (cn_crop - cn_fallow))
    usle_c_daily = usle_c_fallow + (plant_factor * (usle_c_crop - usle_c_fallow))
    for i in range(cn_daily.size):
        if cn_daily[i] <= 0:
            cn_daily[i] = cn_min  # TODO - this shouldn't be necessary. Set cn bounds elsewhere

    # USLE K, LS, C, P factors are multiplied together to estimate erosion losses.
    # Source: PRZM5 Manual Section 4.10 (Young and Fry, 2016)
    usle_klscp_daily = usle_k * usle_ls * usle_c_daily * usle_p

    # Generalize soil properties with depth
    # multiply fc_5, wp_5, fc_20, wp_20 by the thickness (delta_x) to get total water retention capacity for layer
    n_increments = delta_x.size
    field_capacity = np.zeros(n_increments, dtype=np.float32)
    wilting_point = np.zeros(n_increments, dtype=np.float32)
    fc = [fc_5, fc_20]
    wp = [wp_5, wp_20]
    for i in range(n_increments):
        field_capacity[i] = fc[soil_bins[i]] * delta_x[i]
        wilting_point[i] = wp[soil_bins[i]] * delta_x[i]
    return cn_daily, field_capacity, wilting_point, usle_klscp_daily


def plant_growth(n_dates, new_year, plant_date, emergence_date, maxcover_date, harvest_date):
    # TODO - there may be a faster way to do this. Refer to my question on StackOverflow. numba?
    # TODO - error handling
    plant_factor = np.zeros(n_dates + 366)
    try:
        plant_date, emergence_date, maxcover_date, harvest_date = \
            map(int, (plant_date, emergence_date, maxcover_date, harvest_date))

        growth_period = (maxcover_date - emergence_date) + 1
        mature_period = (harvest_date - maxcover_date) + 1
        emergence_dates = (new_year + emergence_date).astype(np.int16)
        maxcover_dates = (new_year + maxcover_date).astype(np.int16)
        growth_dates = np.add.outer(emergence_dates, np.arange(growth_period))
        mature_dates = np.add.outer(maxcover_dates, np.arange(mature_period))
        plant_factor[growth_dates] = np.linspace(0, 1, growth_period)
        plant_factor[mature_dates] = 1
    except:
        pass
    return plant_factor[:n_dates]


@njit
def process_erosion(slope, runoff, rain, cn, usle_klscp, type_matrix, flow_length):
    """
    Estimate erosion using MUSLE
    Source: PRZM5 Manual (Young and Fry, 2016), Sect 4.10
    """
    erosion_loss = np.zeros(rain.size)
    for day in range(runoff.size):
        if runoff[day] > 0.:
            # Coefficients based on rainfall type, from tr_55.csv
            c = erosion_coefficients(rain[day], cn[day], type_matrix)

            # Time of Concentration by Watershed lag method from NEH-4 Chapter 4 """
            t_conc = (((flow_length * 3.28) ** 0.8) * ((((1000. / cn[day]) - 10.) + 1) ** 0.7)) / \
                     (1140. * (slope ** 0.5))

            # Calculation of peak storm runoff (qp) using Graphical Peak Discharge Method (qp=qu*A*Q*Fp)
            # 1 ft3 = 0.02832 m3; 1 m2 = 3.86102e-7 mi2; 1 cm = 0.3937 in; 1 m = 39.3701 in; 1 m3/s = 3.6e+6 mm/hr
            # qp = (((qu * (3.86102e-7 * a) * (39.3701 * r)) * 0.02832) / a) * 3.6e+6
            unit_peak_discharge = \
                10. ** (c[0] + c[1] * np.log10(t_conc) + c[2] * (np.log10(t_conc)) ** 2)
            qp = 1.54958679 * runoff[day] * unit_peak_discharge
            erosion_loss[day] = 1.586 * (runoff[day] * 1000. * qp) ** .56 * usle_klscp[day] * 1000.  # kg/d
    return erosion_loss
