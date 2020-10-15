import pandas as pd
from paths_nhd import condensed_nhd_path


def condensed_nhd(run_id=None, region=None, feature_type=None, path=None):
    path = condensed_nhd_path if path is None else path
    return pd.read_csv(path.format(run_id, region, feature_type))


def nhd_map(field_map_path, lower=True, all_cols=False, rename_field=None):
    data = pd.read_csv(field_map_path)
    if lower:
        data['field'] = [f.lower() for f in data.field]
    if not all_cols:
        out_fields = ['path', 'table', 'field', 'feature_type']
        if rename_field is not None:
            out_fields.append(rename_field)
        data = data[out_fields]
    return data
