import pandas as pd

print("loading")

table = pd.read_csv(r"E:\opp-efed-data\global\NHD_HUC_Crosswalk\CrosswalkTable_NHDplus_HU12.csv")

print("loaded")
table = table.sort_values('FEATUREID')
table['HUC_12'] = table['HUC_12'].astype(str).str.zfill(12)
table.to_csv(r"E:\opp-efed-data\global\NHD_HUC_Crosswalk\CrosswalkTable_NHDplus_HU12.csv", index=None)