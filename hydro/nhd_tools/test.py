from process_nhd import condense_nhd
import pandas as pd
from navigator import build_navigator, Navigator
field_map_path = r"A:\opp-efed\sam\Tables\nhd_map_sam.csv"
reach_table, lake_table = condense_nhd('07', field_map_path)

reach_table.to_csv("r07_reach.csv", index=None)
lake_table.to_csv("r07_waterbody.csv", index=None)


