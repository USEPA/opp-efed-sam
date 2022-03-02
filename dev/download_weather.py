import os
from ftplib import FTP

# Run parameters
root_path = r"ftp2.psl.noaa.gov"
data_dir = r"Datasets/ncep.reanalysis.dailyavgs"
local_dir = r"E:\met_data"
var_dict = {"surface_gauss": ["air.2m", "tmax.2m", "tmin.2m", "uwnd.10m", "vwnd.10m"],
            "other_gauss": ["dswrf.ntat"]}
years = range(1970, 2023)

ftp = FTP(root_path)  # connect to host, default port
ftp.login()  # user anonymous, passwd anonymous@
ftp.cwd(data_dir)  # change into "debian" directory
for subdir, vars in var_dict.items():
    ftp.cwd(subdir)  # change into "debian" directory
    the_list = ftp.nlst()
    for var in vars:
        for year in years:
            filename = f"{var}.gauss.{year}.nc"
            local_file = os.path.join(local_dir, filename)
            print(f"Downloading {filename}...")
            try:
                with open(local_file, 'wb') as fp:
                    ftp.retrbinary(f'RETR {filename}', fp.write)
                print("\tSuccess.")
            except Exception as e:
                print(e)
            exit()
    ftp.cwd("..")
ftp.quit()