import numpy as np
import pandas as pd
from efed_lib.read import dbf

"""
ArbolateSu
NLCD_61 % of catchment area classified as Orchards/Vineyards/Other in NLCD
NLCD_71 % of catchment area classified as Grasslands/Herbaceous in NLCD
NLCD_81 % of catchment area classified as Pasture/Hay in NLCD
NLCD_82 % of catchment area classified as Row Crops in NLCD
NLCD_83 % of catchment area classified as Small Grains in NLCD

"""

# stream length, flowrates, catchment areas, and PCAs for the main HUC
vaa_path = r"E:\opp-efed-data\global\NHDPlusV21\NHDPlusMS\NHDPlus07\NHDPlusAttributes\PlusFlowlineVAA.dbf"
vaa_fields = ['comid', 'arbolatesu', 'totdasqkm']
erom_path = r"E:\opp-efed-data\global\NHDPlusV21\NHDPlusMS\NHDPlus07\EROMExtension\EROM_MA0001.DBF"
erom_fields = ['comid', 'q0001e']

nlcd_fields = ['ComID', 'NLCD81PC', 'NLCD82PC']
nlcd_path = r"E:\opp-efed-data\global\NHDPlusV21\NHDPlusMS\NHDPlus07\VPUAttributeExtension\CumDivNLCD2011.txt"

nlcd_table = pd.read_csv(nlcd_path)[nlcd_fields].rename(columns={'ComID': 'comid'})
vaa_table = dbf(vaa_path)[vaa_fields]
erom_table = dbf(erom_path)[erom_fields]

print(0, vaa_table.shape)
joined = vaa_table.merge(erom_table, on="comid")
print(1, joined.shape)
joined = joined.merge(nlcd_table, on="comid")
print(2, joined.shape)
joined.to_csv("joined.csv")
