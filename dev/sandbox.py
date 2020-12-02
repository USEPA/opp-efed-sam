hucLvl = 2
latitude = 39.7423
longitude = -92.4727
tempHuc = (12 - hucLvl * 2);
hucURL = f"https://watersgeo.epa.gov/arcgis/rest/services/NHDPlus_NP21/WBD_NP21_Simplified/MapServer/{hucLvl}" \
             f"/query?geometry=%7B%22x%22+%3A+{longitude}%2C+%22y%22+%3A+{latitude}" \
             f"%2C+%22spatialReference%22+%3A+%7B%22wkid%22+%3A+4326%7D%7D&geometryType=esriGeometryPoint&" + \
         f"inSR=%7B%22wkid%22+%3A+4326%7D&spatialRel=esriSpatialRelWithin&outFields=HUC_{tempHuc}" \
             f"%2C+STATES%2C+Shape%2C+AREA_SQKM%2C+HU_{tempHuc}_NAME&" + \
         f"returnGeometry=true&returnTrueCurves=false&outSR=%7B%22wkid%22+%3A+4326%7D&returnIdsOnly=false&" + \
         f"returnCountOnly=false&returnZ=false&returnM=false&returnDistinctValues=false&returnExtentsOnly=false&f=pjson"

print(hucURL)
