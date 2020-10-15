import numpy as np
from numba import njit, guvectorize


def compute_concentration(transported_mass, runoff, n_dates, q):
    """
    Calculates pesticide concentration in water column from runoff inputs, accounting for time of travel
    Need to add references: VVWM (for basics), SAM write-up on time of travel
    """

    mean_runoff = runoff.mean()  # m3/d
    print(q.shape, mean_runoff.shape, n_dates)
    baseflow = np.subtract(q, mean_runoff, out=np.zeros(n_dates), where=(q > mean_runoff))
    total_flow = runoff + baseflow
    concentration = np.divide(transported_mass, total_flow, out=np.zeros(n_dates), where=(total_flow != 0))
    runoff_concentration = np.divide(transported_mass, runoff, out=np.zeros(n_dates), where=(runoff != 0))
    return total_flow, map(lambda x: x * 1000000., (concentration, runoff_concentration))  # kg/m3 -> ug/L


def partition_benthic(erosion, erosion_mass, surface_area):
    """ Compute concentration in the benthic layer based on mass of eroded sediment """

    from parameters import benthic_params

    soil_volume = benthic_params.depth * surface_area
    pore_water_volume = soil_volume * benthic_params.porosity
    benthic_mass = benthic_loop(erosion, erosion_mass, soil_volume)
    return benthic_mass / pore_water_volume


@njit
def benthic_loop(eroded_soil, erosion_mass, soil_volume):
    benthic_mass = np.zeros(erosion_mass.size, dtype=np.float32)
    benthic_mass[0] = erosion_mass[0]
    for i in range(1, erosion_mass.size):
        influx_ratio = eroded_soil[i] / (eroded_soil[i] + soil_volume)
        benthic_mass[i] = (benthic_mass[i - 1] * (1. - influx_ratio)) + (erosion_mass[i] * (1. - influx_ratio))
    return benthic_mass


@guvectorize(['void(float64[:], int16[:], int16[:], int16[:], float64[:])'], '(p),(o),(o),(p)->(o)')
def exceedance_probability(time_series, window_sizes, thresholds, years_since_start, res):
    # Count the number of times the concentration exceeds the test threshold in each year
    n_years = years_since_start.max()
    for test_number in range(window_sizes.size):
        window_size = window_sizes[test_number]
        threshold = thresholds[test_number]
        if threshold == 0:
            res[test_number] = -1
        else:
            window_sum = np.sum(time_series[:window_size])
            exceedances = np.zeros(n_years)
            for day in range(window_size, len(time_series)):
                year = years_since_start[day]
                window_sum += time_series[day] - time_series[day - window_size]
                avg = window_sum / window_size
                if avg > threshold:
                    exceedances[year] = 1
            res[test_number] = exceedances.sum() / n_years
