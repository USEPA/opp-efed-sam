import pandas as pd
import json

test_json = \
    {"COMID": {"5038222": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038224": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038348": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038350": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038352": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038354": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038356": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038358": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038360": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038362": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038364": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038366": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038368": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038370": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038372": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038374": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0},
               "5038376": {"human_acute": 0.0, "human_chronic": 0.0, "human_overall": 0.0, "fw_fish_acute": 0.0,
                           "fw_fish_chronic": 0.0, "fw_fish_overall": 0.0, "fw_inv_acute": 0.0,
                           "fw_inv_chronic": 0.0,
                           "fw_inv_overall": 0.0, "em_fish_acute": 0.0, "em_fish_chronic": 0.0,
                           "em_fish_overall": 0.0,
                           "em_inv_acute": 0.0, "em_inv_chronic": 0.0, "em_inv_overall": 0.0,
                           "nonvasc_plant_acute": 0.0,
                           "nonvasc_plant_chronic": 0.0, "nonvasc_plant_overall": 0.0, "vasc_plant_acute": 0.0,
                           "vasc_plant_chronic": 0.0, "vasc_plant_overall": 0.0}}}

df = pd.DataFrame(test_json['COMID']).T.rename_axis("COMID").reset_index()
print(df)