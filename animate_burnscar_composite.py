import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as matCol
from matplotlib.colors import ListedColormap
from netCDF4 import Dataset
import os
import sys
from datetime import datetime
import time
import h5py

import warnings
warnings.filterwarnings("ignore")

#import python modules I made
from burn_scar_composites import get_burn_scar_RGB,\
                                 get_normalized_burn_ratio,\
                                 get_BRF_lat_lon,\
                                 get_BRF_RGB,\
                                 get_burn_scar_composite,\
                                 flip_arr

from cloud_water_clear_data import mask_water_and_cloud

#calculate fields to plot
home = 'R:/satellite_data/viirs_data/noaa20/'

with h5py.File(home+'databases/VIIRS_burn_Scar_database.h5','r') as hf_database:
    timestamps = list(hf_database.keys())
    dataset_names = list(hf_database[timestamps[0]].keys())
    for timestamp in timestamps[0:]:
        dataset_dict = {}
        print('Grabbing datatsets')
        for dataset_name in dataset_names:
            dataset_dict[dataset_name] = hf_database[timestamp+'/'+dataset_name][:]
        print('Grabbing datatsets Successful')

        print('Regridding datatsets')
        r_row_idx    = dataset_dict['regrid_row_idx']
        r_col_idx    = dataset_dict['regrid_col_idx']
        fill_val_idx = dataset_dict['fill_val_idx']

        NBR           = dataset_dict['NBR'          ][r_row_idx, r_col_idx]
        BRF_RGB       = dataset_dict['BRF_RGB'      ][r_row_idx, r_col_idx]
        cldmsk        = dataset_dict['cldmsk'       ][r_row_idx, r_col_idx]
        burn_scar_RGB = dataset_dict['burn_scar_RGB'][r_row_idx, r_col_idx]
        lat_regrid    = dataset_dict['lat'][r_row_idx, r_col_idx]
        lon_regrid    = dataset_dict['lon'][r_row_idx, r_col_idx]

        print('Regridding datatsets Successful')

        print('Applying fill val to datasets')
        NBR[fill_val_idx[0], fill_val_idx[1]]           = np.nan
        BRF_RGB[fill_val_idx[0], fill_val_idx[1]]       = np.nan
        cldmsk[fill_val_idx[0], fill_val_idx[1]]        = np.nan
        burn_scar_RGB[fill_val_idx[0], fill_val_idx[1]] = np.nan
        lat_regrid[fill_val_idx[0], fill_val_idx[1]]    = np.nan
        lon_regrid[fill_val_idx[0], fill_val_idx[1]]    = np.nan
        print('Applying fill val to datasets Successful')


        # burn_scar_composite_cloud_water_cleared = mask_water_and_cloud(burn_scar_composite, cldmsk, land_water_mask)

        print('Plotting')
        #plot ##########################################################################
        plt.style.use('dark_background')
        f, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, \
                             figsize=(13,13))

        cmap_cldmsk = ListedColormap(['white', 'green', 'blue','black'])
        norm = matCol.BoundaryNorm(np.arange(0,5,1), cmap_cldmsk.N)

        n_factor_1 = 1.5
        n_factor_2 = 1.5
        im1 = ax[0,0].imshow(NBR, cmap='jet', vmin=-1)
        im2 = ax[0,1].imshow(n_factor_1*BRF_RGB)
        im3 = ax[1,0].imshow(cldmsk, cmap=cmap_cldmsk, norm=norm)
        im4 = ax[1,1].imshow(n_factor_2*burn_scar_RGB)

        vminlat, vmaxlat, vminlon, vmaxlon = 28,51,-127,-100
        ax[0,2].imshow(dataset_dict['lat'], cmap='jet', vmin=vminlat, vmax=vmaxlat)
        ax[0,3].imshow(dataset_dict['lon'], cmap='jet', vmin=vminlon, vmax=vmaxlon)

        ax[1,2].imshow(lat_regrid, cmap='jet', vmin=vminlat, vmax=vmaxlat)
        ax[1,3].imshow(lon_regrid, cmap='jet', vmin=vminlon, vmax=vmaxlon)

        ax[0,0].set_title('Normalized Burn Ratio')
        ax[0,1].set_title('True Color RGB {}{}'.format('*', n_factor_1))
        ax[1,0].set_title('Cloud Mask')
        ax[1,1].set_title('Day Land Cloud Fire RGB {}{}'.format('*', n_factor_2))

        ax[0,2].set_title('Latitude')
        ax[0,3].set_title('Longitude')

        ax[1,2].set_title('Latitude Regridded')
        ax[1,3].set_title('Longitude Regridded')

        im1.cmap.set_under('k')
        # im2.cmap.set_under('k')
        # im3.cmap.set_under('k')
        # im4.cmap.set_under('k')

        bad_color = 'k'
        im1.cmap.set_bad(color=bad_color)
        im2.cmap.set_bad(color=bad_color)
        im3.cmap.set_bad(color=bad_color)
        im4.cmap.set_bad(color=bad_color)

        # current_cmap.set_bad(color='red')

        # s = lat.shape
        # x = np.arange(s[1])
        # y = np.arange(s[0])
        # X, Y = np.meshgrid(x, y)

        for a in ax.flat:
            a.set_xticks([])
            a.set_yticks([])
            # lat_contours = a.contour(X, Y, lat, 20, colors='pink')
            # a.clabel(lat_contours, inline=True, fontsize=8)
            # lon_contours = a.contour(X, Y, lon, 20, colors='pink')
            # a.clabel(lon_contours, inline=True, fontsize=8)

        #colorbars
        cax  = f.add_axes([0.025, 0.08, 0.012, 0.32]) #[left, bottom, width, height]
        cbar = f.colorbar(im3, cax=cax, orientation='vertical')
        cbar.set_ticks([0.5,1.5,2.5,3.5])
        cbar.set_ticklabels(['CLD', 'UCLR', \
                             'PCLR', 'CLR'])

        time_stamp_current = timestamp[:12]
        year     = time_stamp_current[:4]
        DOY      = '{}'.format(time_stamp_current[4:7])
        UTC_hr   = time_stamp_current[8:][:2]
        UTC_min  = time_stamp_current[8:][2:]
        date     = datetime.strptime(year + "-" + DOY, "%Y-%j").strftime("%m.%d.%Y")

        title = 'NOAA-20 VIIRS; {}; {}:{} UTC'.format(date, UTC_hr, UTC_min)
        f.suptitle(title)
        plt.tight_layout()
        plt.show()


        # #plot ##########################################################################
        # f, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, \
        #                      figsize=(13,13))
        #
        # n_factor_1 = 1.5
        # n_factor_2 = 1.5
        # im1 = ax[0,0].imshow(NBR, cmap='jet', vmin=-0.5)
        # im2 = ax[0,1].imshow(n_factor_1*BRF_RGB)
        # im3 = ax[1,0].imshow(burn_scar_composite, cmap='jet', vmin=0)
        # im4 = ax[1,1].imshow(n_factor_2*burn_scar_RGB)
        #
        # ax[0,0].set_title('NBR')
        # ax[0,1].set_title('True Color RGB {}{}'.format('*', n_factor_1))
        # ax[1,0].set_title('Burn Scar Composite')
        # ax[1,1].set_title('Day Land Cloud Fire RGB {}{}'.format('*', n_factor_2))
        #
        # im1.cmap.set_under('k')
        # im2.cmap.set_under('k')
        # im3.cmap.set_under('k')
        # im4.cmap.set_under('k')
        #
        # s = lat.shape
        # x = np.arange(s[1])
        # y = np.arange(s[0])
        # X, Y = np.meshgrid(x, y)
        #
        # for a in ax.flat:
        #     a.set_xticks([])
        #     a.set_yticks([])
        #     lat_contours = a.contour(X, Y, lat, 20, colors='pink')
        #     a.clabel(lat_contours, inline=True, fontsize=8)
        #     lon_contours = a.contour(X, Y, lon, 20, colors='pink')
        #     a.clabel(lon_contours, inline=True, fontsize=8)
        #
        # time_stamp_current = geo_file[-33:-21] #need to replace with filename
        # year     = time_stamp_current[:4]
        # DOY      = '{}'.format(time_stamp_current[4:7])
        # UTC_hr   = time_stamp_current[8:][:2]
        # UTC_min  = time_stamp_current[8:][2:]
        # date     = datetime.strptime(year + "-" + DOY, "%Y-%j").strftime("%m.%d.%Y")
        #
        # title = 'NOAA-20 VIIRS; {}; {}:{} UTC'.format(date, UTC_hr, UTC_min)
        # f.suptitle(title)
        # plt.tight_layout()
        # plt.show()
