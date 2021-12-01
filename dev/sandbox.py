import os
import pandas as pd

# Input tables
combos_table = r"E:\opp-efed-data\scenarios\Intermediate\Combinations\07_{}.csv"  # year
crosswalk_table = r"A:\opp-efed\scenarios\Tables\met_params.csv"
scenarios_table = r"E:\opp-efed-data\sam\Inputs\SamScenarios\r07_1.csv"

crosswalk = pd.read_csv(crosswalk_table)[['stationID', 'state_met']].rename(columns={"stationID": "weather_grid"})
table = pd.read_csv(scenarios_table, usecols=['weather_grid', 'scenario_id'])
print(crosswalk)
print(table)
merged = table.merge(crosswalk, on='weather_grid').drop_duplicates().sort_values('scenario_id')
merged.to_csv("scenario_state.csv", index=None)



