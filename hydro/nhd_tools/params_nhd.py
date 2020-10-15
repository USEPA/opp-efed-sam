from efed_lib_hydro.efed_lib import FieldManager
from paths_nhd import fields_and_qc_path

nhd_states = {'01': {"ME", "NH", "VT", "MA", "CT", "RI", "NY"},
              '02': {"VT", "NY", "PA", "NJ", "MD", "DE", "WV", "DC", "VA"},
              '03N': {"VA", "NC", "SC", "GA"},
              '03S': {"FL", "GA"},
              '03W': {"FL", "GA", "TN", "AL", "MS"},
              '04': {"WI", "MN", "MI", "IL", "IN", "OH", "PA", "NY"},
              '05': {"IL", "IN", "OH", "PA", "WV", "VA", "KY", "TN"},
              '06': {"VA", "KY", "TN", "NC", "GA", "AL", "MS"},
              '07': {"MN", "WI", "SD", "IA", "IL", "MO", "IN"},
              '08': {"MO", "KY", "TN", "AR", "MS", "LA"},
              '09': {"ND", "MN", "SD"},
              '10U': {"MT", "ND", "WY", "SD", "MN", "NE", "IA"},
              '10L': {"CO", "WY", "MN", "NE", "IA", "KS", "MO"},
              '11': {"CO", "KS", "MO", "NM", "TX", "OK", "AR", "LA"},
              '12': {"NM", "TX", "LA"},
              '13': {"CO", "NM", "TX"},
              '14': {"WY", "UT", "CO", "AZ", "NM"},
              '15': {"NV", "UT", "AZ", "NM", "CA"},
              '16': {"CA", "OR", "ID", "WY", "NV", "UT"},
              '17': {"WA", "ID", "MT", "OR", "WY", "UT", "NV"},
              '18': {"OR", "NV", "CA"}}

vpus_nhd = {'01': 'NE', '02': 'MA', '03N': 'SA', '03S': 'SA', '03W': 'SA', '04': 'GL', '05': 'MS',
            '06': 'MS', '07': 'MS', '08': 'MS', '09': 'SR', '10L': 'MS', '10U': 'MS', '11': 'MS',
            '12': 'TX', '13': 'RG', '14': 'CO', '15': 'CO', '16': 'GB', '17': 'PN', '18': 'CA'}

nhd_regions = sorted(vpus_nhd.keys())

fields_hydro = FieldManager(fields_and_qc_path)
