import pandas as pd
import numpy as np

atrazine_json_test_b = {'csrfmiddlewaretoken': {'0': '986432de-3493-4260-b026-d90e154e7ddc'},
                           'simulation_name': {'0': 'test'},
                           'chemical_name': {'0': 'atrazine_test'},
                           'kd_flag': {'0': '1.0'},
                           'koc': {'0': '75.0'},
                           'soil_hl': {'0': '139.0'},
                           'wc_metabolism_hl': {'0': '277.0'},
                           'ben_metabolism_hl': {'0': '277.0'},
                           'aq_photolysis_hl': {'0': '168.0'},
                           'hydrolysis_hl': {'0': '0.0'},
                           'napps': {'0': '1.0'},
                           'crop_1': {'0': '10 14 15 18'},
                           'event_1': {'0': 'plant'},
                           'offset_1': {'0': '0.0'},
                           'dist_1': {'0': 'ground'},
                           'window1_1': {'0': '7.0'},
                           'pct1_1': {'0': '100.0'},
                           'window2_1': {'0': '7.0'},
                           'pct2_1': {'0': '100.0'},
                           'method_1': {'0': 'uniform'},
                           'apprate_1': {'0': '1.0'},
                           'effic_1': {'0': '1.0'},
                           'region': {'0': '07'},
                           'sim_type': {'0': 'dwr'},
                           'sim_date_start': {'0': '01/01/2000'},
                           'sim_date_end': {'0': '12/31/2015'},
                           'acute_human': {'0': '3.4'},
                           'chronic_human': {'0': ''},
                           'overall_human': {'0': ''},
                           'acute_fw_fish': {'0': '2650.0'},
                           'chronic_fw_fish': {'0': '0.5'},
                           'acute_fw_inv': {'0': '360.0'},
                           'chronic_fw_inv': {'0': '60.0'},
                           'acute_em_fish': {'0': '1000.0'},
                           'chronic_em_fish': {'0': '0.5'},
                           'acute_em_inv': {'0': '24.0'},
                           'chronic_em_inv': {'0': '80.0'},
                           'acute_nonvasc_plant': {'0': '1.0'},
                           'chronic_nonvasc_plant': {'0': ''},
                           'acute_vasc_plant': {'0': '4.6'},
                           'chronic_vasc_plant': {'0': ''}}

atrazine_json_test_a = {'csrfmiddlewaretoken': {'0': '986432de-3493-4260-b026-d90e154e7ddc'},
                           'simulation_name': {'0': 'test&4867727&mtb'},
                           'chemical_name': {'0': 'atrazine_test'},
                           'kd_flag': {'0': '1.0'},
                           'koc': {'0': '75.0'},
                           'soil_hl': {'0': '139.0'},
                           'wc_metabolism_hl': {'0': '277.0'},
                           'ben_metabolism_hl': {'0': '277.0'},
                           'aq_photolysis_hl': {'0': '168.0'},
                           'hydrolysis_hl': {'0': '0.0'},
                           'napps': {'0': '1.0'},
                           'crop_1': {'0': '10 14 15 18'},
                           'event_1': {'0': 'plant'},
                           'offset_1': {'0': '0.0'},
                           'dist_1': {'0': 'ground'},
                           'window1_1': {'0': '7.0'},
                           'pct1_1': {'0': '100.0'},
                           'window2_1': {'0': '7.0'},
                           'pct2_1': {'0': '100.0'},
                           'method_1': {'0': 'uniform'},
                           'apprate_1': {'0': '1.0'},
                           'effic_1': {'0': '1.0'},
                           'region': {'0': '07'},
                           'sim_type': {'0': 'dwr'},
                           'sim_date_start': {'0': '01/01/2000'},
                           'sim_date_end': {'0': '12/31/2015'},
                           'acute_human': {'0': '3.4'},
                           'chronic_human': {'0': ''},
                           'overall_human': {'0': ''},
                           'acute_fw_fish': {'0': '2650.0'},
                           'chronic_fw_fish': {'0': '0.5'},
                           'acute_fw_inv': {'0': '360.0'},
                           'chronic_fw_inv': {'0': '60.0'},
                           'acute_em_fish': {'0': '1000.0'},
                           'chronic_em_fish': {'0': '0.5'},
                           'acute_em_inv': {'0': '24.0'},
                           'chronic_em_inv': {'0': '80.0'},
                           'acute_nonvasc_plant': {'0': '1.0'},
                           'chronic_nonvasc_plant': {'0': ''},
                           'acute_vasc_plant': {'0': '4.6'},
                           'chronic_vasc_plant': {'0': ''}}


atrazine_json_mtb_build = {'csrfmiddlewaretoken': {'0': '986432de-3493-4260-b026-d90e154e7ddc'},
                           'simulation_name': {'0': 'build&4867727&mtb'},
                           'chemical_name': {'0': 'BUILDS2'},
                           'kd_flag': {'0': '1.0'},
                           'koc': {'0': '75.0'},
                           'soil_hl': {'0': '139.0'},
                           'wc_metabolism_hl': {'0': '277.0'},
                           'ben_metabolism_hl': {'0': '277.0'},
                           'aq_photolysis_hl': {'0': '168.0'},
                           'hydrolysis_hl': {'0': '0.0'},
                           'napps': {'0': '1.0'},
                           'crop_1': {'0': '10 14 15 18'},
                           'event_1': {'0': 'plant'},
                           'offset_1': {'0': '0.0'},
                           'dist_1': {'0': 'ground'},
                           'window1_1': {'0': '7.0'},
                           'pct1_1': {'0': '100.0'},
                           'window2_1': {'0': '7.0'},
                           'pct2_1': {'0': '100.0'},
                           'method_1': {'0': 'uniform'},
                           'apprate_1': {'0': '1.0'},
                           'effic_1': {'0': '1.0'},
                           'region': {'0': '07'},
                           'sim_type': {'0': 'dwr'},
                           'sim_date_start': {'0': '01/01/2000'},
                           'sim_date_end': {'0': '12/31/2015'},
                           'acute_human': {'0': '3.4'},
                           'chronic_human': {'0': ''},
                           'overall_human': {'0': ''},
                           'acute_fw_fish': {'0': '2650.0'},
                           'chronic_fw_fish': {'0': '0.5'},
                           'acute_fw_inv': {'0': '360.0'},
                           'chronic_fw_inv': {'0': '60.0'},
                           'acute_em_fish': {'0': '1000.0'},
                           'chronic_em_fish': {'0': '0.5'},
                           'acute_em_inv': {'0': '24.0'},
                           'chronic_em_inv': {'0': '80.0'},
                           'acute_nonvasc_plant': {'0': '1.0'},
                           'chronic_nonvasc_plant': {'0': ''},
                           'acute_vasc_plant': {'0': '4.6'},
                           'chronic_vasc_plant': {'0': ''}}

atrazine_json_mtb = {'csrfmiddlewaretoken': {'0': '986432de-3493-4260-b026-d90e154e7ddc'},
                           'simulation_name': {'0': 'mtb'},
                           'chemical_name': {'0': 'BUILDS2'},
                           'kd_flag': {'0': '1.0'},
                           'koc': {'0': '75.0'},
                           'soil_hl': {'0': '139.0'},
                           'wc_metabolism_hl': {'0': '277.0'},
                           'ben_metabolism_hl': {'0': '277.0'},
                           'aq_photolysis_hl': {'0': '168.0'},
                           'hydrolysis_hl': {'0': '0.0'},
                           'napps': {'0': '1.0'},
                           'crop_1': {'0': '10 14 15 18'},
                           'event_1': {'0': 'plant'},
                           'offset_1': {'0': '0.0'},
                           'dist_1': {'0': 'ground'},
                           'window1_1': {'0': '7.0'},
                           'pct1_1': {'0': '100.0'},
                           'window2_1': {'0': '7.0'},
                           'pct2_1': {'0': '100.0'},
                           'method_1': {'0': 'uniform'},
                           'apprate_1': {'0': '1.0'},
                           'effic_1': {'0': '1.0'},
                           'region': {'0': 'Mark Twain Demo'},
                           'sim_type': {'0': 'eco'},
                           'sim_date_start': {'0': '01/01/2000'},
                           'sim_date_end': {'0': '12/31/2015'},
                           'acute_human': {'0': '3.4'},
                           'chronic_human': {'0': ''},
                           'overall_human': {'0': ''},
                           'acute_fw_fish': {'0': '2650.0'},
                           'chronic_fw_fish': {'0': '0.5'},
                           'acute_fw_inv': {'0': '360.0'},
                           'chronic_fw_inv': {'0': '60.0'},
                           'acute_em_fish': {'0': '1000.0'},
                           'chronic_em_fish': {'0': '0.5'},
                           'acute_em_inv': {'0': '24.0'},
                           'chronic_em_inv': {'0': '80.0'},
                           'acute_nonvasc_plant': {'0': '1.0'},
                           'chronic_nonvasc_plant': {'0': ''},
                           'acute_vasc_plant': {'0': '4.6'},
                           'chronic_vasc_plant': {'0': ''}}

bixafen_json = {'csrfmiddlewaretoken': {'0': 'bixafen_test_three'},
                'simulation_name': {'0': 'Bixafen Test June 2018'},
                'chemical_name': {'0': 'Bixafen'},
                'region': {'0': '07'},
                'sim_type': {'0': 'eco'},
                'sim_date_start': {'0': '01/01/2000'},
                'sim_date_end': {'0': '12/31/2015'},
                'kd_flag': {'0': '0'},
                'koc': {'0': '0.075'},
                'soil_hl': {'0': '1242.'},
                'wc_metabolism_hl': {'0': '5211.'},
                'ben_metabolism_hl': {'0': '0.'},
                'aq_photolysis_hl': {'0': '313'},
                'hydrolysis_hl': {'0': '0.'},
                'napps': {'0': '15'},
                'crop_1': {'0': '10.0'},
                'event_1': {'0': 'emergence'},
                'offset_1': {'0': '-7.0'},
                'method_1': {'0': 'uniform'},
                'dist_1': {'0': 'ground'},
                'window1_1': {'0': '21.0'},
                'pct1_1': {'0': '100.0'},
                'window2_1': {'0': '0.0'},
                'pct2_1': {'0': '0.0'},
                'effic_1': {'0': '0.0686'},
                'apprate_1': {'0': '0.99'},
                'crop_2': {'0': '10.0'},
                'event_2': {'0': 'emergence'},
                'offset_2': {'0': '17.0'},
                'method_2': {'0': 'uniform'},
                'dist_2': {'0': 'foliar'},
                'window1_2': {'0': '21.0'},
                'pct1_2': {'0': '100.0'},
                'window2_2': {'0': '0.0'},
                'pct2_2': {'0': '0.0'},
                'effic_2': {'0': '0.0686'},
                'apprate_2': {'0': '0.95'},
                'crop_3': {'0': '40.0'},
                'event_3': {'0': 'emergence'},
                'offset_3': {'0': '-7.0'},
                'method_3': {'0': 'uniform'},
                'dist_3': {'0': 'ground'},
                'window1_3': {'0': '21.0'},
                'pct1_3': {'0': '100.0'},
                'window2_3': {'0': '0.0'},
                'pct2_3': {'0': '0.0'},
                'effic_3': {'0': '0.0686'},
                'apprate_3': {'0': '0.99'},
                'crop_4': {'0': '40.0'},
                'event_4': {'0': 'emergence'},
                'offset_4': {'0': '20.0'},
                'method_4': {'0': 'uniform'},
                'dist_4': {'0': 'foliar'},
                'window1_4': {'0': '21.0'},
                'pct1_4': {'0': '100.0'},
                'window2_4': {'0': '0.0'},
                'pct2_4': {'0': '0.0'},
                'effic_4': {'0': '0.0686'},
                'apprate_4': {'0': '0.95'},
                'crop_5': {'0': '22.0'},
                'event_5': {'0': 'emergence'},
                'offset_5': {'0': '7.0'},
                'method_5': {'0': 'uniform'},
                'dist_5': {'0': 'foliar'},
                'window1_5': {'0': '59.0'},
                'pct1_5': {'0': '100.0'},
                'window2_5': {'0': '0.0'},
                'pct2_5': {'0': '0.0'},
                'effic_5': {'0': '0.0667'},
                'apprate_5': {'0': '0.95'},
                'crop_6': {'0': '22.0'},
                'event_6': {'0': 'emergence'},
                'offset_6': {'0': '36.0'},
                'method_6': {'0': 'uniform'},
                'dist_6': {'0': 'foliar'},
                'window1_6': {'0': '59.0'},
                'pct1_6': {'0': '100.0'},
                'window2_6': {'0': '0.0'},
                'pct2_6': {'0': '0.0'},
                'effic_6': {'0': '0.0667'},
                'apprate_6': {'0': '0.95'},
                'crop_7': {'0': '23.0'},
                'event_7': {'0': 'emergence'},
                'offset_7': {'0': '7.0'},
                'method_7': {'0': 'uniform'},
                'dist_7': {'0': 'foliar'},
                'window1_7': {'0': '59.0'},
                'pct1_7': {'0': '100.0'},
                'window2_7': {'0': '0.0'},
                'pct2_7': {'0': '0.0'},
                'effic_7': {'0': '0.0667'},
                'apprate_7': {'0': '0.95'},
                'crop_8': {'0': '23.0'},
                'event_8': {'0': 'emergence'},
                'offset_8': {'0': '36.0'},
                'method_8': {'0': 'uniform'},
                'dist_8': {'0': 'foliar'},
                'window1_8': {'0': '59.0'},
                'pct1_8': {'0': '100.0'},
                'window2_8': {'0': '0.0'},
                'pct2_8': {'0': '0.0'},
                'effic_8': {'0': '0.0667'},
                'apprate_8': {'0': '0.95'},
                'crop_9': {'0': '24.0'},
                'event_9': {'0': 'emergence'},
                'offset_9': {'0': '7.0'},
                'method_9': {'0': 'uniform'},
                'dist_9': {'0': 'foliar'},
                'window1_9': {'0': '59.0'},
                'pct1_9': {'0': '100.0'},
                'window2_9': {'0': '0.0'},
                'pct2_9': {'0': '0.0'},
                'effic_9': {'0': '0.0667'},
                'apprate_9': {'0': '0.95'},
                'crop_10': {'0': '24.0'},
                'event_10': {'0': 'emergence'},
                'offset_10': {'0': '36.0'},
                'method_10': {'0': 'uniform'},
                'dist_10': {'0': 'foliar'},
                'window1_10': {'0': '59.0'},
                'pct1_10': {'0': '100.0'},
                'window2_10': {'0': '0.0'},
                'pct2_10': {'0': '0.0'},
                'effic_10': {'0': '0.0667'},
                'apprate_10': {'0': '0.95'},
                'crop_11': {'0': '80.0'},
                'event_11': {'0': 'emergence'},
                'offset_11': {'0': '7.0'},
                'method_11': {'0': 'uniform'},
                'dist_11': {'0': 'foliar'},
                'window1_11': {'0': '59.0'},
                'pct1_11': {'0': '100.0'},
                'window2_11': {'0': '0.0'},
                'pct2_11': {'0': '0.0'},
                'effic_11': {'0': '0.0667'},
                'apprate_11': {'0': '0.95'},
                'crop_12': {'0': '80.0'},
                'event_12': {'0': 'emergence'},
                'offset_12': {'0': '36.0'},
                'method_12': {'0': 'uniform'},
                'dist_12': {'0': 'foliar'},
                'window1_12': {'0': '59.0'},
                'pct1_12': {'0': '100.0'},
                'window2_12': {'0': '0.0'},
                'pct2_12': {'0': '0.0'},
                'effic_12': {'0': '0.0667'},
                'apprate_12': {'0': '0.95'},
                'crop_13': {'0': '60.0'},
                'event_13': {'0': 'emergence'},
                'offset_13': {'0': '-7.0'},
                'method_13': {'0': 'uniform'},
                'dist_13': {'0': 'ground'},
                'window1_13': {'0': '21.0'},
                'pct1_13': {'0': '100.0'},
                'window2_13': {'0': '0.0'},
                'pct2_13': {'0': '0.0'},
                'effic_13': {'0': '0.0512'},
                'apprate_13': {'0': '0.99'},
                'crop_14': {'0': '60.0'},
                'event_14': {'0': 'emergence'},
                'offset_14': {'0': '17.0'},
                'method_14': {'0': 'uniform'},
                'dist_14': {'0': 'foliar'},
                'window1_14': {'0': '21.0'},
                'pct1_14': {'0': '100.0'},
                'window2_14': {'0': '0.0'},
                'pct2_14': {'0': '0.0'},
                'effic_14': {'0': '0.0512'},
                'apprate_14': {'0': '0.99'},
                'crop_15': {'0': '60.0'},
                'event_15': {'0': 'emergence'},
                'offset_15': {'0': '26.0'},
                'method_15': {'0': 'uniform'},
                'dist_15': {'0': 'foliar'},
                'window1_15': {'0': '21.0'},
                'pct1_15': {'0': '100.0'},
                'window2_15': {'0': '0.0'},
                'pct2_15': {'0': '0.0'},
                'effic_15': {'0': '0.0512'},
                'apprate_15': {'0': '0.99'},
                'acute_human': {'0': ''},
                'chronic_human': {'0': ''},
                'overall_human': {'0': ''},
                'acute_fw_fish': {'0': '37.0'},
                'chronic_fw_fish': {'0': '4.6'},
                'overall_fw_fish': {'0': ''},
                'acute_fw_inv': {'0': '550.0'},
                'chronic_fw_inv': {'0': '53.5'},
                'overall_fw_inv': {'0': ''},
                'acute_em_fish': {'0': '75.5'},
                'chronic_em_fish': {'0': '6.4'},
                'overall_em_fish': {'0': ''},
                'acute_em_inv': {'0': '121.5'},
                'chronic_em_inv': {'0': '88.6'},
                'overall_em_inv': {'0': ''},
                'acute_nonvasc_plant': {'0': '8.34'},
                'chronic_nonvasc_plant': {'0': ''},
                'overall_nonvasc_plant': {'0': ''},
                'acute_vasc_plant': {'0': '55.7'},
                'chronic_vasc_plant': {'0': ''},
                'overall_vasc_plant': {'0': ''},
                }

atrazine_demo = {'aq_photolysis_hl': 168.0,
                 'ben_metabolism_hl': 277.0,
                 'soil_hl': 139.0,
                 'wc_metabolism_hl': 277.0,
                 'hydrolysis_hl': 0.0,
                 'chemical_name': 'Atrazine',
                 'kd_flag': 1,
                 'koc': 0.075,
                 'napps': 1.0,
                 'region': '07',
                 'sim_date_end': np.datetime64('2015-12-31'),
                 'sim_date_start': np.datetime64('2000-01-01'),
                 'sim_type': 'manual',
                 'simulation_name': 'Mark Twain Demo',
                 'applications': [[1, 'plant', 0, 'uniform', 'ground', 7, 100.0, 7, 0., 1.0, 1.0],
                                  [5, 'plant', 0, 'uniform', 'ground', 7, 100.0, 7, 0., 1.0, 1.0],
                                  [21, 'plant', 0, 'uniform', 'ground', 7, 100.0, 7, 0., 1.0, 1.0],
                                  [31, 'plant', 0, 'uniform', 'ground', 7, 100.0, 7, 0., 1.0, 1.0]],
                 'endpoints': np.array([[3.40000010e+00, 2.65000000e+03, 3.60000000e+02, 1.00000000e+03, 2.40000000e+01,
                                         1.00000000e+00, 4.59999990e+00],
                                        [np.nan, 5.00000000e-01, 6.00000000e+01, 5.00000000e-01, 8.00000000e+01, np.nan,
                                         np.nan],
                                        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
                 }
