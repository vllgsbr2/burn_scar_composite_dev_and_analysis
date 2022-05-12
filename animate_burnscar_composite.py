import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import sys
from datetime import datetime
import time

#import python modules I made
from burn_scar_composites import get_burn_scar_RGB,\
                                 get_normalized_burn_ratio,\
                                 get_BRF_lat_lon,\
                                 get_BRF_RGB,\
                                 get_burn_scar_composite,\
                                 flip_arr
from read_VIIRS_raw_nc_files import get_VJ103_geo,\
                                   get_VJ102_ref,\
                                   get_CLDMSK

from regrid import regrid_latlon_source2target as regrid_data
from cloud_water_clear_data import mask_water_and_cloud

#calculate fields to plot
M_bands_BRF, lat, lon = get_BRF_lat_lon(geo_file, ref_file, which_bands)

R_M3, R_M4, R_M5, R_M7, R_M11 = \
                          M_bands_BRF[:,:,0], M_bands_BRF[:,:,1],\
                          M_bands_BRF[:,:,2], M_bands_BRF[:,:,3],\
                          M_bands_BRF[:,:,4]

BRF_RGB                 = get_BRF_RGB(R_M5,R_M4,R_M3)
NBR                     = get_normalized_burn_ratio(R_M7, R_M11)
burn_scar_RGB           = get_burn_scar_RGB(R_M11, R_M7, R_M5)
cldmsk, land_water_mask = get_CLDMSK(cldmsk_file)
burn_scar_composite     = get_burn_scar_composite(R_M7, R_M11)


NBR[np.isnan(NBR)]                         = -999
burn_scar_RGB[np.isnan(burn_scar_RGB)]     = -999
cldmsk[np.isnan(cldmsk)]                   = -999
land_water_mask[np.isnan(land_water_mask)] = -999
burn_scar_composite[np.isnan(burn_scar_composite)] = -999


burn_scar_composite_cloud_water_cleared = mask_water_and_cloud(burn_scar_composite, cldmsk, land_water_mask)

# regrid data #################################################################

# burn_scar_composite = regrid(data, lat_target,lon_target, lat_source, lon_source)
#need to choose a reference grid using an exemplary granule
#may 7th 2022 is a good centered granule
lat_source, lon_source = np.copy(), np.copy()

lat_target, lon_target = np.copy(lat), np.copy(lon)
# (source_lat, source_lon, target_lat, target_lon, source_data)
burn_scar_composite_cloud_water_cleared_r = regrid_data(source_lat, source_lon, target_lat, target_lon, burn_scar_composite_cloud_water_cleared)
cldmsk_r              = regrid_data(source_lat, source_lon, target_lat, target_lon, cldmsk)
burn_scar_RGB_r       = regrid_data(source_lat, source_lon, target_lat, target_lon, burn_scar_RGB)
NBR_r                 = regrid_data(source_lat, source_lon, target_lat, target_lon, NBR)




#plot ##########################################################################
f, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, \
                     figsize=(13,13))
#flip images over both axis for correct plotting with imshow
NBR             = flip_arr(NBR)
burn_scar_RGB   = flip_arr(burn_scar_RGB)
cldmsk          = flip_arr(cldmsk)
land_water_mask = flip_arr(land_water_mask)
burn_scar_composite_cloud_water_cleared = flip_arr(burn_scar_composite_cloud_water_cleared)

im1 = ax[0,0].imshow(NBR)
im2 = ax[0,1].imshow(cldmsk, cmap='Greys')
im3 = ax[1,0].imshow(burn_scar_composite_regridded, cmap='jet')
im4 = ax[1,1].imshow(1.2*burn_scar_RGB)

for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])

time_stamp_current = geo_file[-33:-21] #need to replace with filename
year     = time_stamp_current[:4]
DOY      = '{}'.format(time_stamp_current[4:7])
UTC_hr   = time_stamp_current[8:][:2]
UTC_min  = time_stamp_current[8:][2:]
date     = datetime.strptime(year + "-" + DOY, "%Y-%j").strftime("_%m.%d.%Y")

title = 'NOAA-20 VIIRS; {}; {:02d}:{:02d} UTC'.format(date, UTC_hr, UTC_min)
f.suptitle(title)
plt.tight_layout()
plt.show()
