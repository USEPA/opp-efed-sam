import pandas as pd

selected = r"A:\sam_launcher\selected_test.csv"
full_table_path = r"E:/opp-efed-data/sam\Inputs\SamScenarios/r07.csv"

# The scenario index from the complete table
selected_rows = []
selected = pd.read_csv(selected)
for chunk in pd.read_csv(full_table_path, chunksize=100000):
    chunk = chunk.merge(selected, on='scenario_index', how='inner')
    if not chunk.empty:
        selected_rows.append(chunk)
subset = pd.concat(selected_rows, axis=0)
print(selected.shape)
print(subset.shape)
