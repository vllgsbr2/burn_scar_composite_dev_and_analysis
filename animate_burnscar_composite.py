import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import sys
from datetime import datetime
import time


#python modules I made
from burn_scar_composites import get_burn_scar_RGB,\
                                 get_normalized_burn_ratio,\
                                 get_BRF_lat_lon,\
                                 get_BRF_RGB,\
                                 get_burn_scar_composite,\
                                 flip_arr
from read_VIIRS_raw_nc_files import get_VJ103_geo,\
                                   get_VJ102_ref,\
                                   get_CLDMSK


def regrid(self, data, lat_source, lon_source):
    '''
    @param {nd.array} data
    @param {nd.array} lat_source
    @param {nd.array} lon_source

    all inputs are the same MxN but data can have depth K

    @return regridded @param {nd.array} data
    '''
    from regrid import regrid

    if data.ndim >= 3:
        data_temp = np.empty(np.shape(data))

        for i in range(data.ndim):
            data_temp[:,:,i] = regrid(data[:,:,i], self.lat_target,\
                                     self.lon_target, lat_source, lon_source)
        return data_temp
    else:
        return regrid(data, self.lat_target, self.lon_target, lat_source,\
                                                        lon_source)


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


BRF_RGB[np.isnan(BRF_RGB)]                 = -999
NBR[np.isnan(NBR)]                         = -999
burn_scar_RGB[np.isnan(burn_scar_RGB)]     = -999
cldmsk[np.isnan(cldmsk)]                   = -999
land_water_mask[np.isnan(land_water_mask)] = -999
burn_scar_composite[np.isnan(burn_scar_composite)] = -999

BRF_RGB         = flip_arr(BRF_RGB)
NBR             = flip_arr(NBR)
burn_scar_RGB   = flip_arr(burn_scar_RGB)
cldmsk          = flip_arr(cldmsk)
land_water_mask = flip_arr(land_water_mask)
burn_scar_composite = flip_arr(burn_scar_composite)

# group.create_dataset(observables[i], data=data[:,:,i], compression='gzip')
# group = hf_observables.create_group(time_stamp)

#write data to file
time_stamp_current = geo_file[-33:-21]
year     = time_stamp_current[:4]
DOY      = '{}'.format(time_stamp_current[4:7])
UTC_hr   = time_stamp_current[8:][:2]
UTC_min  = time_stamp_current[8:][2:]
date     = datetime.strptime(year + "-" + DOY, "%Y-%j").strftime("_%m.%d.%Y")


















#plot
f, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, \
                     figsize=(13,13))

im1 = ax[0,0].imshow(1.5*BRF_RGB)
im2 = ax[0,1].imshow(cldmsk, cmap='Greys')
im3 = ax[1,0].imshow(land_water_mask, cmap='jet')
im4 = ax[1,1].imshow(1.2*burn_scar_RGB)

for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])

title = 'NOAA-20 VIIRS; {}'.format(ts)
f.suptitle(ts)
plt.tight_layout()
plt.show()
