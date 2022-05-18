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
home                 = 'R:/satellite_data/viirs_data/noaa20/'
ref_filepath_home    = home + 'reflectance/'
geo_filepath_home    = home + 'geolocation/'
cldmsk_filepath_home = home + 'cldmsk/'

ref_filepaths    = np.sort([ref_filepath_home    + x for x in os.listdir(ref_filepath_home)])
geo_filepaths    = np.sort([geo_filepath_home    + x for x in os.listdir(geo_filepath_home)])
cldmsk_filepaths = np.sort([cldmsk_filepath_home + x for x in os.listdir(cldmsk_filepath_home)])

ref_file_timestamps    = [x[-33:-21] for x in ref_filepaths]
geo_file_timestamps    = [x[-33:-21] for x in geo_filepaths]
cldmsk_file_timestamps = [x[-33:-21] for x in cldmsk_filepaths]

boolean_intersection_ref_cldmsk = np.in1d(ref_file_timestamps, cldmsk_file_timestamps)
idx_last_not_match = np.where(boolean_intersection_ref_cldmsk==False)[0][-1]


ref_file_timestamps    = ref_file_timestamps[idx_last_not_match+1:]
geo_file_timestamps    = geo_file_timestamps[idx_last_not_match+1:]
ref_filepaths    = ref_filepaths[idx_last_not_match+1:]
geo_filepaths    = geo_filepaths[idx_last_not_match+1:]

#start with one file DOY 121/127 2022 May 1 or 7
# n=3
geo_files    = [x for x, y in zip(geo_filepaths, geo_file_timestamps) if ('2022121' in y or '2022127' in y)]
ref_files    = [x for x, y in zip(ref_filepaths, geo_file_timestamps) if ('2022121' in y or '2022127' in y)]
cldmsk_files = [x for x, y in zip(cldmsk_filepaths, geo_file_timestamps) if ('2022121' in y or '2022127' in y)]

which_bands  = [3,4,5,7,11] # [blue, green, red, veggie, burn]

#place holder reference lat lo grid for source regridding but would like to make
#custom grid soon
regrid_reference_geo_file    = ''
regrid_reference_ref_file    = ''
regrid_reference_cldmsk_file = ''
M_bands_BRF, lat, lon = get_BRF_lat_lon(regrid_reference_geo_file,\
                                        regrid_reference_ref_file,\
                                        regrid_reference_cldmsk_file)
lat_source, lon_source = np.copy(), np.copy()


for geo_file, ref_file, cldmsk_file in zip(geo_files, ref_files, cldmsk_files):

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
    burn_scar_composite[np.isnan(burn_scar_composite)] = -999


    # burn_scar_composite_cloud_water_cleared = mask_water_and_cloud(burn_scar_composite, cldmsk, land_water_mask)

    # # regrid data #################################################################
    #
    # # burn_scar_composite = regrid(data, lat_target,lon_target, lat_source, lon_source)
    # #need to choose a reference grid using an exemplary granule
    # #may 7th 2022 is a good centered granule
    # lat_source, lon_source = np.copy(), np.copy()
    #
    # lat_target, lon_target = np.copy(lat), np.copy(lon)
    # # (source_lat, source_lon, target_lat, target_lon, source_data)
    # burn_scar_composite_cloud_water_cleared_r = regrid_data(source_lat, source_lon, target_lat, target_lon, burn_scar_composite_cloud_water_cleared)
    # cldmsk_r              = regrid_data(source_lat, source_lon, target_lat, target_lon, cldmsk)
    # burn_scar_RGB_r       = regrid_data(source_lat, source_lon, target_lat, target_lon, burn_scar_RGB)
    # NBR_r                 = regrid_data(source_lat, source_lon, target_lat, target_lon, NBR)


    NBR[np.isnan(NBR)]                         = -999
    burn_scar_RGB[np.isnan(burn_scar_RGB)]     = -999
    cldmsk[np.isnan(cldmsk)]                   = -999
    burn_scar_composite[np.isnan(burn_scar_composite)] = -999

    #flip images over both axis for correct plotting with imshow
    NBR             = flip_arr(NBR)
    BRF_RGB             = flip_arr(BRF_RGB)
    burn_scar_RGB   = flip_arr(burn_scar_RGB)
    # cldmsk          = flip_arr(cldmsk)
    land_water_mask = flip_arr(land_water_mask)
    burn_scar_composite = flip_arr(burn_scar_composite)
    # burn_scar_composite_cloud_water_cleared = flip_arr(burn_scar_composite_cloud_water_cleared)

    #plot ##########################################################################
    f, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, \
                         figsize=(13,13))

    n_factor_1 = 1.5
    n_factor_2 = 1.5
    im1 = ax[0,0].imshow(NBR, cmap='jet', vmin=-0.5)
    im2 = ax[0,1].imshow(n_factor_1*BRF_RGB)
    im3 = ax[1,0].imshow(burn_scar_composite, cmap='jet', vmin=0)
    im4 = ax[1,1].imshow(n_factor_2*burn_scar_RGB)

    ax[0,0].set_title('NBR')
    ax[0,1].set_title('True Color RGB {}{}'.format('*', n_factor_1))
    ax[1,0].set_title('Burn Scar Composite')
    ax[1,1].set_title('Day Land Cloud Fire RGB {}{}'.format('*', n_factor_2))

    im1.cmap.set_under('k')
    im2.cmap.set_under('k')
    im3.cmap.set_under('k')
    im4.cmap.set_under('k')

    s = lat.shape
    x = np.arange(s[1])
    y = np.arange(s[0])
    X, Y = np.meshgrid(x, y)

    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
        lat_contours = a.contour(X, Y, lat, 20, colors='pink')
        a.clabel(lat_contours, inline=True, fontsize=8)
        lon_contours = a.contour(X, Y, lon, 20, colors='pink')
        a.clabel(lon_contours, inline=True, fontsize=8)

    time_stamp_current = geo_file[-33:-21] #need to replace with filename
    year     = time_stamp_current[:4]
    DOY      = '{}'.format(time_stamp_current[4:7])
    UTC_hr   = time_stamp_current[8:][:2]
    UTC_min  = time_stamp_current[8:][2:]
    date     = datetime.strptime(year + "-" + DOY, "%Y-%j").strftime("%m.%d.%Y")

    title = 'NOAA-20 VIIRS; {}; {}:{} UTC'.format(date, UTC_hr, UTC_min)
    f.suptitle(title)
    plt.tight_layout()
    plt.show()
