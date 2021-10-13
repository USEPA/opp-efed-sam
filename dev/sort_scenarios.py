import pandas as pd

scenarios_path = r"E:\opp-efed-data\sam\Inputs\SamScenarios\r07_1.csv"
new_path = r"E:\opp-efed-data\sam\Inputs\SamScenarios\r07.csv"

s = pd.read_csv(scenarios_path)
s = s.sort_values("weather_grid")
s.to_csv(new_path, index=None)
