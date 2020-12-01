import json
import datetime

a = {'_id': '14916fd3-9a49-4090-bb1c-b3389815f432', 'date': datetime.datetime(2020, 12, 1, 18, 23, 32, 631000),
     'data': '"{\\n    \\"COMID\\": {\\n        \\"4867727\\": {\\n            \\"em_fish_acute\\": 0.0,\\n            \\"em_fish_chronic\\": 0.0,\\n            \\"em_fish_overall\\": -1.0,\\n            \\"em_inv_acute\\": 0.0,\\n            \\"em_inv_chronic\\": 0.0,\\n            \\"em_inv_overall\\": -1.0,\\n            \\"fw_fish_acute\\": 0.0,\\n            \\"fw_fish_chronic\\": 0.0,\\n            \\"fw_fish_overall\\": -1.0,\\n            \\"fw_inv_acute\\": 0.0,\\n            \\"fw_inv_chronic\\": 0.0,\\n            \\"fw_inv_overall\\": -1.0,\\n            \\"human_acute\\": 0.0,\\n            \\"human_chronic\\": -1.0,\\n            \\"human_overall\\": -1.0,\\n            \\"nonvasc_plant_acute\\": 0.0,\\n            \\"nonvasc_plant_chronic\\": -1.0,\\n            \\"nonvasc_plant_overall\\": -1.0,\\n            \\"vasc_plant_acute\\": 0.0,\\n            \\"vasc_plant_chronic\\": -1.0,\\n            \\"vasc_plant_overall\\": -1.0\\n        },\\n        \\"5640266\\": {\\n            \\"em_fish_acute\\": 0.0,\\n            \\"em_fish_chronic\\": 0.0,\\n            \\"em_fish_overall\\": -1.0,\\n            \\"em_inv_acute\\": 0.0,\\n            \\"em_inv_chronic\\": 0.0,\\n            \\"em_inv_overall\\": -1.0,\\n            \\"fw_fish_acute\\": 0.0,\\n            \\"fw_fish_chronic\\": 0.0,\\n            \\"fw_fish_overall\\": -1.0,\\n            \\"fw_inv_acute\\": 0.0,\\n            \\"fw_inv_chronic\\": 0.0,\\n            \\"fw_inv_overall\\": -1.0,\\n            \\"human_acute\\": 0.0,\\n            \\"human_chronic\\": -1.0,\\n            \\"human_overall\\": -1.0,\\n            \\"nonvasc_plant_acute\\": 0.0,\\n            \\"nonvasc_plant_chronic\\": -1.0,\\n            \\"nonvasc_plant_overall\\": -1.0,\\n            \\"vasc_plant_acute\\": 0.0,\\n            \\"vasc_plant_chronic\\": -1.0,\\n            \\"vasc_plant_overall\\": -1.0\\n        },\\n        \\"5641032\\": {\\n            \\"em_fish_acute\\": 0.0,\\n            \\"em_fish_chronic\\": 0.0,\\n            \\"em_fish_overall\\": -1.0,\\n            \\"em_inv_acute\\": 0.0,\\n            \\"em_inv_chronic\\": 0.0,\\n            \\"em_inv_overall\\": -1.0,\\n            \\"fw_fish_acute\\": 0.0,\\n            \\"fw_fish_chronic\\": 0.0,\\n            \\"fw_fish_overall\\": -1.0,\\n            \\"fw_inv_acute\\": 0.0,\\n            \\"fw_inv_chronic\\": 0.0,\\n            \\"fw_inv_overall\\": -1.0,\\n            \\"human_acute\\": 0.0,\\n            \\"human_chronic\\": -1.0,\\n            \\"human_overall\\": -1.0,\\n            \\"nonvasc_plant_acute\\": 0.0,\\n            \\"nonvasc_plant_chronic\\": -1.0,\\n            \\"nonvasc_plant_overall\\": -1.0,\\n            \\"vasc_plant_acute\\": 0.0,\\n            \\"vasc_plant_chronic\\": -1.0,\\n            \\"vasc_plant_overall\\": -1.0\\n        },\\n        \\"5641062\\": {\\n            \\"em_fish_acute\\": 0.0,\\n            \\"em_fish_chronic\\": 0.0,\\n            \\"em_fish_overall\\": -1.0,\\n            \\"em_inv_acute\\": 0.0,\\n            \\"em_inv_chronic\\": 0.0,\\n            \\"em_inv_overall\\": -1.0,\\n            \\"fw_fish_acute\\": 0.0,\\n            \\"fw_fish_chronic\\": 0.0,\\n            \\"fw_fish_overall\\": -1.0,\\n            \\"fw_inv_acute\\": 0.0,\\n            \\"fw_inv_chronic\\": 0.0,\\n            \\"fw_inv_overall\\": -1.0,\\n            \\"human_acute\\": 0.0,\\n            \\"human_chronic\\": -1.0,\\n            \\"human_overall\\": -1.0,\\n            \\"nonvasc_plant_acute\\": 0.0,\\n            \\"nonvasc_plant_chronic\\": -1.0,\\n            \\"nonvasc_plant_overall\\": -1.0,\\n            \\"vasc_plant_acute\\": 0.0,\\n            \\"vasc_plant_chronic\\": -1.0,\\n            \\"vasc_plant_overall\\": -1.0\\n        },\\n        \\"5641108\\": {\\n            \\"em_fish_acute\\": 0.0,\\n            \\"em_fish_chronic\\": 0.0,\\n            \\"em_fish_overall\\": -1.0,\\n            \\"em_inv_acute\\": 0.0,\\n            \\"em_inv_chronic\\": 0.0,\\n            \\"em_inv_overall\\": -1.0,\\n            \\"fw_fish_acute\\": 0.0,\\n            \\"fw_fish_chronic\\": 0.0,\\n            \\"fw_fish_overall\\": -1.0,\\n            \\"fw_inv_acute\\": 0.0,\\n            \\"fw_inv_chronic\\": 0.0,\\n            \\"fw_inv_overall\\": -1.0,\\n            \\"human_acute\\": 0.0,\\n            \\"human_chronic\\": -1.0,\\n            \\"human_overall\\": -1.0,\\n            \\"nonvasc_plant_acute\\": 0.0,\\n            \\"nonvasc_plant_chronic\\": -1.0,\\n            \\"nonvasc_plant_overall\\": -1.0,\\n            \\"vasc_plant_acute\\": 0.0,\\n            \\"vasc_plant_chronic\\": -1.0,\\n            \\"vasc_plant_overall\\": -1.0\\n        }\\n    }\\n}"'}

b = a['data']

print(b)

c = json.loads(b)

print(c)

print(type(c))

d = {
    "COMID": {
        "4867727": {
            "em_fish_acute": 0.0,
            "em_fish_chronic": 0.0,
            "em_fish_overall": -1.0,
            "em_inv_acute": 0.0,
            "em_inv_chronic": 0.0,
            "em_inv_overall": -1.0,
            "fw_fish_acute": 0.0,
            "fw_fish_chronic": 0.0,
            "fw_fish_overall": -1.0,
            "fw_inv_acute": 0.0,
            "fw_inv_chronic": 0.0,
            "fw_inv_overall": -1.0,
            "human_acute": 0.0,
            "human_chronic": -1.0,
            "human_overall": -1.0,
            "nonvasc_plant_acute": 0.0,
            "nonvasc_plant_chronic": -1.0,
            "nonvasc_plant_overall": -1.0,
            "vasc_plant_acute": 0.0,
            "vasc_plant_chronic": -1.0,
            "vasc_plant_overall": -1.0
        },
        "5640266": {
            "em_fish_acute": 0.0,
            "em_fish_chronic": 0.0,
            "em_fish_overall": -1.0,
            "em_inv_acute": 0.0,
            "em_inv_chronic": 0.0,
            "em_inv_overall": -1.0,
            "fw_fish_acute": 0.0,
            "fw_fish_chronic": 0.0,
            "fw_fish_overall": -1.0,
            "fw_inv_acute": 0.0,
            "fw_inv_chronic": 0.0,
            "fw_inv_overall": -1.0,
            "human_acute": 0.0,
            "human_chronic": -1.0,
            "human_overall": -1.0,
            "nonvasc_plant_acute": 0.0,
            "nonvasc_plant_chronic": -1.0,
            "nonvasc_plant_overall": -1.0,
            "vasc_plant_acute": 0.0,
            "vasc_plant_chronic": -1.0,
            "vasc_plant_overall": -1.0
        },
        "5641032": {
            "em_fish_acute": 0.0,
            "em_fish_chronic": 0.0,
            "em_fish_overall": -1.0,
            "em_inv_acute": 0.0,
            "em_inv_chronic": 0.0,
            "em_inv_overall": -1.0,
            "fw_fish_acute": 0.0,
            "fw_fish_chronic": 0.0,
            "fw_fish_overall": -1.0,
            "fw_inv_acute": 0.0,
            "fw_inv_chronic": 0.0,
            "fw_inv_overall": -1.0,
            "human_acute": 0.0,
            "human_chronic": -1.0,
            "human_overall": -1.0,
            "nonvasc_plant_acute": 0.0,
            "nonvasc_plant_chronic": -1.0,
            "nonvasc_plant_overall": -1.0,
            "vasc_plant_acute": 0.0,
            "vasc_plant_chronic": -1.0,
            "vasc_plant_overall": -1.0
        },
        "5641062": {
            "em_fish_acute": 0.0,
            "em_fish_chronic": 0.0,
            "em_fish_overall": -1.0,
            "em_inv_acute": 0.0,
            "em_inv_chronic": 0.0,
            "em_inv_overall": -1.0,
            "fw_fish_acute": 0.0,
            "fw_fish_chronic": 0.0,
            "fw_fish_overall": -1.0,
            "fw_inv_acute": 0.0,
            "fw_inv_chronic": 0.0,
            "fw_inv_overall": -1.0,
            "human_acute": 0.0,
            "human_chronic": -1.0,
            "human_overall": -1.0,
            "nonvasc_plant_acute": 0.0,
            "nonvasc_plant_chronic": -1.0,
            "nonvasc_plant_overall": -1.0,
            "vasc_plant_acute": 0.0,
            "vasc_plant_chronic": -1.0,
            "vasc_plant_overall": -1.0
        },
        "5641108": {
            "em_fish_acute": 0.0,
            "em_fish_chronic": 0.0,
            "em_fish_overall": -1.0,
            "em_inv_acute": 0.0,
            "em_inv_chronic": 0.0,
            "em_inv_overall": -1.0,
            "fw_fish_acute": 0.0,
            "fw_fish_chronic": 0.0,
            "fw_fish_overall": -1.0,
            "fw_inv_acute": 0.0,
            "fw_inv_chronic": 0.0,
            "fw_inv_overall": -1.0,
            "human_acute": 0.0,
            "human_chronic": -1.0,
            "human_overall": -1.0,
            "nonvasc_plant_acute": 0.0,
            "nonvasc_plant_chronic": -1.0,
            "nonvasc_plant_overall": -1.0,
            "vasc_plant_acute": 0.0,
            "vasc_plant_chronic": -1.0,
            "vasc_plant_overall": -1.0
        }
    }
}

print(d)
