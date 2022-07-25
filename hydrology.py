import pandas as pd
import numpy as np
import math
from numba import njit

from .hydro.navigator import Navigator
from .hydro.process_nhd import identify_waterbody_outlets, calculate_surface_area
from .tools.efed_lib import MemoryMatrix


class HydroRegion(Navigator):
    """
    Contains all datasets and functions related to the NHD Plus region, including all hydrological features and links
    """

    def __init__(self, region, sim):

        self.id = region
        self.sim = sim

        # Assign a watershed navigator to the class
        # TODO - a path should be provided here
        super(HydroRegion, self).__init__(sim.navigator_path.format(self.id))

        # Read hydrological input files
        self.reach_table = pd.read_csv(sim.condensed_nhd_path.format('sam', region, 'reach'))
        self.lake_table = pd.read_csv(sim.condensed_nhd_path.format('sam', region, 'waterbody'))
        self.huc_crosswalk = pd.read_csv(sim.nhd_wbd_xwalk_path, dtype=object)[['FEATUREID', 'HUC_12']] \
            .rename(columns={'FEATUREID': 'comid'})

        self.process_nhd()

        # Initialize the fields that will be used to pull flows based on month
        self.flow_fields = [f'q_{str(month).zfill(2)}' for month in sim.month_index]

        # Select which stream reaches will be fully processed, locally processed, or excluded
        self.active_reaches, self.output_reaches, self.reservoir_outlets, self.output_index = \
            self.sort_reaches()

        # Holder for reaches that have been processed
        self.burned_reaches = set()

    def cascade(self):
        # Tier the reaches by counting the number of outlets (lakes) upstream of each lake outlet
        reach_counts = []
        lake_outlets = set(self.reservoir_outlets.outlet_comid)
        for outlet in lake_outlets:
            upstream_lakes = len((set(self.upstream_watershed(outlet)) - {outlet}) & lake_outlets)
            reach_counts.append([outlet, upstream_lakes])
        reach_counts = pd.DataFrame(reach_counts, columns=['comid', 'n_upstream'])

        # Cascade downward through tiers
        upstream_outlets = set()  # outlets from previous tier
        for tier, lake_outlets in reach_counts.groupby('n_upstream')['comid']:
            lakes = self.lake_table[np.in1d(self.lake_table.outlet_comid, lake_outlets)]
            all_upstream = {reach for outlet in lake_outlets for reach in self.upstream_watershed(outlet)}
            reaches = (all_upstream - set(lake_outlets)) | upstream_outlets
            reaches &= set(self.active_reaches)
            reaches -= self.burned_reaches
            yield tier, reaches, lakes
            self.burned_reaches |= reaches
            upstream_outlets = set(lake_outlets)
        all_upstream = {reach for outlet in upstream_outlets for reach in self.upstream_watershed(outlet)}
        yield -1, all_upstream - self.burned_reaches, pd.DataFrame([])

    def confine(self, outlets):
        """ If running a series of intakes or reaches, confine analysis to upstream areas only """
        upstream_reaches = \
            sorted({upstream for outlet in outlets for upstream in self.upstream_watershed(outlet)})
        return pd.Series(upstream_reaches, name='comid')

    def daily_flows(self, reach_id):
        selected = self.reach_table.loc[reach_id]
        flows = selected.loc[self.flow_fields].values.astype(np.float32)
        surface_area = selected['surface_area']
        return flows, surface_area

    def process_nhd(self):
        self.lake_table = \
            identify_waterbody_outlets(self.lake_table, self.reach_table)

        # Add HUC ids to the reach table
        self.huc_crosswalk.comid = self.huc_crosswalk.comid.astype(np.int32)
        self.reach_table = self.reach_table.merge(self.huc_crosswalk, on='comid')
        self.reach_table['HUC_8'] = self.reach_table['HUC_12'].str.slice(0, 8)

        # Calculate average surface area of a reach segment
        self.reach_table['surface_area'] = calculate_surface_area(self.reach_table)

        # Calculate residence times of reservoirs
        self.lake_table = self.lake_table.merge(self.reach_table[['comid', 'q_ma']],
                                                left_on='outlet_comid', right_on='comid', how='left')
        self.lake_table['residence_time'] = self.lake_table.wb_volume / self.lake_table.q_ma

        # Remove reservoirs with residence times less than the minimum
        self.lake_table = self.lake_table[self.lake_table.residence_time > self.sim.minimum_residence_time]

        # Convert units
        self.reach_table['length'] = self.reach_table.pop('lengthkm') * 1000.  # km -> m
        for month in list(map(lambda x: str(x).zfill(2), range(1, 13))) + ['ma']:
            self.reach_table['q_{}'.format(month)] *= 2446.58  # cfs -> cmd
            self.reach_table['v_{}'.format(month)] *= 26334.7  # f/s -> md
        self.reach_table = self.reach_table.drop_duplicates().set_index('comid')

    def sort_reaches(self):
        """
        intakes - reaches corresponding to an intake
        local - all reaches upstream of an intake
        upstream - reaches for which contributions from upstream reaches are considered
        intakes_only - do we do the full monty for the intakes only, or all upstream?
        lake_outlets - reaches that correspond to the outlet of a lake
        """

        # All the reaches in the simulation. Confine the simulation geographically if confine reaches are provided
        if self.sim.confine_reaches is not None:
            active_reaches = self.confine(self.sim.confine_reaches)
        else:
            active_reaches = pd.Series(self.reach_table.index.drop_duplicates().values, name="comid")

        # Identify which reaches correspond to the outlet of a reservoir
        reservoir_outlets = \
            self.lake_table.loc[np.in1d(self.lake_table.outlet_comid, active_reaches)][['outlet_comid', 'wb_comid']]

        # If running in 'drinking water' mode, provide additional output for reaches containing a drinking water intake
        output_reaches = set()
        output_index = None
        if self.sim.sim_type == 'dwr':
            intakes_path = self.sim.dw_intakes_path.format(self.sim.tag)
            output_index = pd.read_csv(intakes_path)
            output_reaches = pd.Series(sorted(set(active_reaches.values) & set(output_index.comid)), name="comid")

        return active_reaches, output_reaches, reservoir_outlets, output_index


class ImpulseResponseMatrix(MemoryMatrix):
    """ A matrix designed to hold the results of an impulse response function for 50 day offsets """

    def __init__(self, n_dates, size=50):
        self.n_dates = n_dates
        self.size = size
        super(ImpulseResponseMatrix, self).__init__([size, n_dates], name='impulse response')
        for i in range(size):
            irf = self.generate(i, 1, self.n_dates)
            self.update(i, irf)

    def fetch(self, index):
        if index <= self.size:
            irf = super(ImpulseResponseMatrix, self).fetch(index, verbose=False)
        else:
            irf = self.generate(index, 1, self.n_dates)
        return irf

    @staticmethod
    def generate(alpha, beta, length):
        def gamma_distribution(t, a, b):
            a, b = map(float, (a, b))
            tau = a * b
            print(234567, a, b, tau)
            return ((t ** (a - 1)) / (((tau / a) ** a) * math.gamma(a))) * math.exp(-(a / tau) * t)

        return np.array([gamma_distribution(i, alpha, beta) for i in range(length)])


@njit
def benthic_concentration(erosion, erosion_mass, surface_area, benthic_depth, benthic_porosity):
    """ Compute concentration in the benthic layer based on mass of eroded sediment """

    soil_volume = benthic_depth * surface_area
    pore_water_volume = soil_volume * benthic_porosity
    benthic_mass = np.zeros(erosion_mass.size, dtype=np.float32)
    benthic_mass[0] = erosion_mass[0]
    for i in range(1, erosion_mass.size):
        influx_ratio = erosion[i] / (erosion[i] + soil_volume)
        benthic_mass[i] = (benthic_mass[i - 1] * (1. - influx_ratio)) + (erosion_mass[i] * (1. - influx_ratio))
    return benthic_mass / pore_water_volume


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
            runoff_and_interception_daily(rain, effective_rain, usle_s_factor[day], plant_factor[day], cintcp,
                                          canopy_water,
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


def water_column_concentration(runoff, transported_mass, n_dates, q):
    """
    Calculates pesticide concentration in water column from runoff inputs, accounting for time of travel
    Need to add references: VVWM (for basics), SAM write-up on time of travel
    """
    mean_runoff = runoff.mean()  # m3/d
    baseflow = q - np.subtract(q, mean_runoff, out=np.zeros(n_dates), where=(q > mean_runoff))
    total_flow = runoff + baseflow
    concentration = np.divide(transported_mass, total_flow, out=np.zeros(n_dates), where=(total_flow != 0))
    runoff_concentration = np.divide(transported_mass, runoff, out=np.zeros(n_dates), where=(runoff != 0))
    return total_flow, baseflow, map(lambda x: x * 1000000., (concentration, runoff_concentration))  # kg/m3 -> ug/L
