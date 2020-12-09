import numpy as np

durations = [1, 7, 21]


def exceedance_probability(time_series, durations, endpoints, years_since_start):
    # Count the number of times the concentration exceeds the test threshold in each year

    result = np.zeros(durations.shape)

    n_years = years_since_start.max()

    # Set up the test for each endpoints
    for test_number in range(durations.size):
        duration = durations[test_number]
        endpoint = endpoints[test_number]

        # If the duration or endpoint isn't set, set the value to 1
        if np.isnan(endpoint) or np.isnan(duration):
            result[test_number] = -1
        else:
            duration_total = np.sum(time_series[:duration])
            exceedances = np.zeros(n_years)
            for day in range(duration, len(time_series)):
                year = years_since_start[day]
                duration_total += time_series[day] - time_series[day - duration]
                avg = duration_total / duration
                if avg > endpoint:
                    exceedances[year] = 1
            result[test_number] = exceedances.sum() / n_years
    return result