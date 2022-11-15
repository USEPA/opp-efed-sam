import numpy as np
from numba import njit
import warnings

warnings.filterwarnings('error')


def pesticide_to_field(applications, new_years, event_dates, rain):
    # JCH - this can probably be streamlined if it turns out to be expensive
    application_mass = np.zeros((2, rain.size))  # canopy/ground
    for i in range(applications.shape[0]):  # n applications
        _, event, offset, canopy, step, window1, pct1, window2, pct2, effic, rate = applications[i]
        event_date = event_dates[int(event)]
        daily_mass_1 = rate * effic * (pct1 / 100.) / window1
        for year in range(new_years.size):  # n years
            new_year = new_years[year]
            for k in range(int(window1)):
                date = int(new_year + event_date + offset + k)
                application_mass[int(canopy), date] = daily_mass_1
            if step:
                daily_mass_2 = rate * effic * (pct2 / 100.) / window2
                for l in range(window2):
                    date = int(new_year + event_date + window1 + offset + l)
                    application_mass[int(canopy), date] = daily_mass_2
    return application_mass


def field_to_soil(application_mass, rain, plant_factor, soil_2cm, foliar_degradation, washoff_coeff, covmax):
    # Initialize output
    pesticide_mass_soil = np.zeros(rain.size)
    canopy_mass, last_application = 0, 0  # Running variables

    # Determine if any pesticide has been applied to canopy
    canopy_applications = application_mass[1].sum() > 0

    # Loop through each day
    for day in range(plant_factor.size):

        # Start with pesticide applied directly to soil
        pesticide_mass_soil[day] = application_mass[0, day] * soil_2cm

        # If pesticide has been applied to the canopy, simulate movement from canopy to soil
        if canopy_applications:
            if application_mass[1, day] > 0:  # Pesticide applied to canopy on this day

                # Amount of pesticide intercepted by canopy
                canopy_pesticide_additions = application_mass[1, day] * plant_factor[day] * (covmax / 100.)

                # Add non-intercepted pesticide to soil
                pesticide_mass_soil[day] += (application_mass[1, day] - canopy_pesticide_additions) * soil_2cm

                if pesticide_mass_soil[day] < 0:
                    print(1000001010101)
                    for var in ('application_mass[0, day]', 'application_mass[1, day]', 'canopy_pesticide_additions',
                                'plant_factor[day]', 'covmax', 'soil_2cm'):
                        val = eval(var)
                        print(var, val)
                canopy_mass = canopy_pesticide_additions + \
                              canopy_mass * np.exp((day - last_application) * foliar_degradation)
                last_application = day

            if rain[day] > 0:  # Simulate washoff
                canopy_mass *= np.exp((day - last_application) * foliar_degradation)
                pesticide_remaining = max(0, canopy_mass * np.exp(-rain[day] * washoff_coeff))
                pesticide_mass_soil[day] += canopy_mass - pesticide_remaining
                last_application = day  # JCH - sure?
    return pesticide_mass_soil


def soil_to_water(pesticide_mass_soil, runoff, erosion, leaching, bulk_density, soil_water, kd, deg_soil,
                  runoff_effic, erosion_effic, delta_x, soil_depth):
    # TODO - this is by far the most expensive transport function. See what we can do. njit at least

    # Initialize running variables
    runoff_mass = np.zeros(pesticide_mass_soil.size, dtype=np.float32)
    erosion_mass = np.zeros(pesticide_mass_soil.size, dtype=np.float32)
    total_mass, degradation_rate = 0, 0

    # Initialize erosion intensity
    erosion_intensity = erosion_effic / soil_depth

    # use degradation rate based on degradation in soil (deg_soil) - NT 8/28/18
    for day in range(pesticide_mass_soil.size):
        daily_runoff = runoff[day] * runoff_effic
        total_mass = total_mass * degradation_rate + pesticide_mass_soil[day]
        retardation = (soil_water[day] / delta_x) + (bulk_density * kd)
        deg_total = deg_soil + ((daily_runoff + leaching[day]) / (delta_x * retardation))
        if leaching[day] > 0:
            degradation_rate = np.exp(-deg_total)
        else:
            degradation_rate = np.exp(-deg_soil)
        average_conc = ((total_mass / retardation / delta_x) / deg_total) * (1 - degradation_rate)
        if runoff[day] > 0:
            runoff_mass[day] = average_conc * daily_runoff  # runoff
        elif runoff[day] < 0:
            for var in (
                    'daily_runoff', 'degradation_rate', 'deg_total', 'leaching[day]', 'retardation'):
                val = eval(var)
                if val < 0:
                    print(9999999, var, val)
        if erosion[day] > 0:
            enrich = np.exp(2.0 - (0.2 * np.log10(erosion[day])))
            enriched_eroded_mass = erosion[day] * enrich * kd * erosion_intensity * 0.1
            erosion_mass[day] = average_conc * enriched_eroded_mass
    return runoff_mass, erosion_mass
