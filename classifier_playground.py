from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import h5py
import configparser
from sklearn.mixture import GaussianMixture
from burn_scar_composites import get_burn_scar_composite,\
                                 get_normalized_burn_ratio,\
                                 get_normalized_differenced_vegetation_index

#plots on correct screen and ctrl + c exits plot window
plt.switch_backend('qtAgg')

#open up database to access DLCF RGB composite
config = configparser.ConfigParser()
config.read('config_filepaths.txt')
home_dir = config['file paths mac']['database dir']
databse_name = 'daily_DLCF_composites_2021_west_CONUS_all_days.h5'

with h5py.File(home_dir + databse_name, 'r') as hf_database:
    X = hf_database['08.10.2021'][:]

rgb_OG = np.copy(X)
rgb_OG_copy = np.copy(X)

#ancillary composites to view
Rbrn, Rveg, Rvis = rgb_OG_copy[:,:,0], rgb_OG_copy[:,:,1], rgb_OG_copy[:,:,2]
# NDVI             = get_normalized_differenced_vegetation_index(Rvis, Rveg)
# NBR              = get_normalized_burn_ratio(Rveg, Rbrn)
burnscar_mask    = get_burn_scar_composite(Rveg, Rbrn)

#workflow to label and analyze burn scars in DLCF RGB composite
burnscar_semi_labeled_dataset_file = 'subsetted_burn_scar_coordinates.txt'
df_burnscar_semi_labeled = pd.read_csv(burnscar_semi_labeled_dataset_file,\
                                         header=0, delimiter=', ', skiprows=7)
# print(df_burnscar_semi_labeled)

#build boxes around burn scars then visualize on RGB
col1 = df_burnscar_semi_labeled['col1'].tolist()
col2 = df_burnscar_semi_labeled['col2'].tolist()
row1 = df_burnscar_semi_labeled['row1'].tolist()
row2 = df_burnscar_semi_labeled['row2'].tolist()

count = 0
for i in range(len(col1)):
    x=burnscar_mask[row1[i]:row2[i],col1[i]:col2[i]]
    idx_valid = np.where(np.isnan(x)==False)
    count += np.nansum(len(idx_valid[0]))
# print(count)
#only show NBR when burnscar_mask is not nan and index is within rectangles
burnscar_mask_manual_labels_combined = np.copy(burnscar_mask)
burnscar_mask_manual_labels_combined[:,:] = np.nan
manual_labels = np.copy(burnscar_mask)
manual_labels[:,:] = np.nan
for i in range(len(col1)):
    manual_labels[row1[i]:row2[i],col1[i]:col2[i]] = 1
# print(manual_labels)
# idx_valid = np.where(np.any(manual_labels==1) and np.any(np.isnan(burnscar_mask)==True))
burnscar_mask_manual_labels_combined[manual_labels==1] = burnscar_mask[manual_labels==1]
# print(idx_valid)

plt.rcParams.update({'font.size': 15})
plt.style.use('dark_background')
f, ax = plt.subplots(ncols=2, figsize=(35,20), sharex=True, sharey=True)
ax[0].imshow(burnscar_mask_manual_labels_combined, cmap='jet', vmax=0.25)
ax[1].imshow(1.5*rgb_OG)
im = ax[1].imshow(burnscar_mask, cmap='jet', vmax=0.25, alpha=0.75)
area_total = 0
for i in range(len(col1)):
    width, length = col2[i]-col1[i], row2[i]-row1[i]
    area_total += width*length
    rect = patches.Rectangle((col1[i], row1[i]), width, length, linewidth=1,\
                             edgecolor='r', facecolor='none')
    rect1 = patches.Rectangle((col1[i], row1[i]), width, length, linewidth=1,\
                             edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)
    ax[1].add_patch(rect1)

#[left, bottom, width, height]
cax  = f.add_axes([0.9, 0.02, 0.02, 0.78])
cbar = f.colorbar(im, cax=cax, orientation='vertical')
ticks = np.arange(-.350, 0.30, 0.050)
cbar.ax.set_yticks(ticks)
labels = [round(x, 2) for x in ticks]
cbar.ax.set_yticklabels(labels)

ax[0].set_title('Red rectangles contain burned areas, total pixels contained: '\
             +str(area_total), y=-0.1)
ax[0].set_title('Day Land Cloud Fire RGB\n[2.25, 0.86, 0.67]µm')
ax[1].set_title('Primitive Burn Scar Mask; NBR Shaded\nR_M7<0.2, R_M7>0.0281, R_M11>0.05, NBR>-0.35')
f.suptitle('NOAA-20 VIIRS Valid Sept 1, 2021\nComposited and Cloud-Cleared Over Previous 8 Days')

for a in ax:
    a.set_xticks([])
    a.set_yticks([])

plt.subplots_adjust(left=0.125,
                    bottom=0.,
                    right=0.9,
                    top=0.82,
                    wspace=0.2,
                    hspace=0.2)

plt.show()

# #open up netcdf4 file and save the RGB, primitive mask, lat/lon, timestamp
# grid_file_path = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/VIIRS_database/databases/Grids_West_CONUS_new.h5'
# save_path = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/VIIRS_database/databases/'
# from netCDF4 import Dataset
# import pyproj as proj
# sample_fname = 'burn_scar_mask_GIS_sample_v2.nc'
# with Dataset(save_path + sample_fname, 'w', format='NETCDF4_CLASSIC') as nc_burnscar,\
#      h5py.File(grid_file_path,'r') as h5_lat_lon:
#
#     # coordinates for the single burn scar that looks like australia
#     # r1, r2, c1, c2 = 1260, 1340, 360, 480
#     r1, r2, c1, c2 = 0, -1, 0, -1
#     pbsm_shape = np.shape(burnscar_mask[r1:r2,c1:c2])
#
#     # create dimensions of data
#     nc_burnscar.createDimension('lat', pbsm_shape[0])
#     nc_burnscar.createDimension('lon', pbsm_shape[1])
#     nc_burnscar.createDimension('time', None)
#     nc_burnscar.createDimension('channel', 3)
#
#     # define lat/lon and time variables
#     time           = nc_burnscar.createVariable('time', np.int8, ('time'))
#     time.units     = 'hours since 2021-10-14 20:54:00'
#     time.long_name = 'time'
#     time.calendar  = 'none'
#
#     lat           = nc_burnscar.createVariable('lat', np.float32, ('lat','lon'), fill_value=-999)
#     lat.units     = 'degrees_north'
#     lat.long_name = 'latitude'
#     lon           = nc_burnscar.createVariable('lon', np.float32, ('lat','lon'), fill_value=-999)
#     lon.units     = 'degrees_east'
#     lon.long_name = 'longitude'
#
#     #save data into created variables
#     lat[:,:] = h5_lat_lon['Geolocation/Latitude' ][r1:r2,c1:c2]
#     lon[:,:] = h5_lat_lon['Geolocation/Longitude'][r1:r2,c1:c2]*(-1)
#     #adjust lon to go from west to east
#     lon_adjust = lon * np.nan
#     for k in range(0, len(lon)):
#         lon_adjust[k] = sorted(lon[k])
#     lon[:,:] = np.copy(lon_adjust)
#
#     pbsm               = nc_burnscar.createVariable('pbsm',np.float64,('time','lat','lon'), fill_value=-999) # note: unlimited dimension is leftmost
#     pbsm.units         = 'unitless'
#     pbsm.standard_name = 'primitive burn scar mask'
#     pbsm[:,:]          = burnscar_mask[r1:r2,c1:c2].reshape((1,pbsm_shape[0], pbsm_shape[1]))
#
#     day_land_cloud_fire_RGB               = nc_burnscar.createVariable('day_land_cloud_fire_RGB',np.float64,('time','lat','lon','channel'), fill_value=-999) # note: unlimited dimension is leftmost
#     day_land_cloud_fire_RGB.units         = 'unitless'
#     day_land_cloud_fire_RGB.standard_name = 'day_land_cloud_fire_RGB'
#     day_land_cloud_fire_RGB[:,:,:]        = np.copy(X)[r1:r2,c1:c2,:].reshape((1,pbsm_shape[0], pbsm_shape[1], 3))
#
#     # define CRS for file
#     crs = nc_burnscar.createVariable('spatial_ref', 'i4')
#     crs.spatial_ref = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'


# with Dataset(save_path + sample_fname, 'r') as nc_burnscar:
#     print(nc_burnscar['pbsm'])
#     x=nc_burnscar['pbsm'][:]
#     y=nc_burnscar['day_land_cloud_fire_RGB'][:]
#
# plt.imshow(y[0])
# plt.imshow(x[0], cmap='jet', alpha=0.5, vmax=0.5, vmin=-0.3)
# plt.show()


# #make hists of RGB values and total values superimposed
# rgb_temp = rgb_OG[row1[i]:row2[i], col1[i]:col2[i], :]
# R_semi_labeled_pixels = rgb_temp[:,:,0].flatten()
# G_semi_labeled_pixels = rgb_temp[:,:,1].flatten()
# B_semi_labeled_pixels = rgb_temp[:,:,2].flatten()
# for i in range(1, len(col1)):
#     rgb_temp = rgb_OG[row1[i]:row2[i], col1[i]:col2[i], :]
#
#     R_semi_labeled_pixels = np.concatenate((R_semi_labeled_pixels, \
#                                        rgb_temp[:,:,0].flatten()))
#     G_semi_labeled_pixels = np.concatenate((R_semi_labeled_pixels, \
#                                        rgb_temp[:,:,1].flatten()))
#     B_semi_labeled_pixels = np.concatenate((R_semi_labeled_pixels, \
#                                        rgb_temp[:,:,2].flatten()))
#
# f1, ax1 = plt.subplots(ncols=3, figsize=(20,7))
# num_bins = 100
# min, max = 0, 0.5
# interval = (max-min)/(num_bins)
# bins = np.arange(min, max+interval, interval)
#
#
# #underlay hists of entire domain
# density=True
# rwidth=0.7
# ax1[0].hist(rgb_OG[:,:,0].flatten(), bins=bins, density=density, color='r', rwidth=rwidth)
# ax1[1].hist(rgb_OG[:,:,1].flatten(), bins=bins, density=density, color='r', rwidth=rwidth)
# ax1[2].hist(rgb_OG[:,:,2].flatten(), bins=bins, density=density, color='r', rwidth=rwidth)
#
# # overlay semi labeled hists
# ax1[0].hist(R_semi_labeled_pixels, bins=bins, density=density, alpha=0.5, color='b', rwidth=rwidth)
# ax1[1].hist(G_semi_labeled_pixels, bins=bins, density=density, alpha=0.5, color='b', rwidth=rwidth)
# ax1[2].hist(B_semi_labeled_pixels, bins=bins, density=density, alpha=0.5, color='b', rwidth=rwidth)
#
#
# for i, a in enumerate(ax1.flat):
#     a.set_xticks(np.arange(0,0.55, 0.05))
#     a.tick_params(axis='x', rotation=45)
#     a.set_yticks([])
#
# ax1[0].set_title('2.25 µm Ref')
# ax1[1].set_title('0.86 µm Ref')
# ax1[2].set_title('0.67 µm Ref')
#
# ax1[0].set_ylabel('Density / Frequency')


# plt.tight_layout()
# plt.show()

# #subset X to focus on ROI
# # r1,r2, c1,c2 = 1555,1841, 0  ,600 # general large case
# r1,r2, c1,c2 = 1260,1331, 370,460 # focused case on burn scar
# # r1,r2, c1,c2 = 1292,1304, 407,412 # only burn scar pixels case
# # r1,r2, c1,c2 = 1620,1740, 120,220 # burn scar next to dixie
#
# X = X[r1:r2, c1:c2, :]
#
# #take nan values out of rgb
# #(if at least 1 channel has a nan, kick all three channels in that x,y position)
# rgb_plot = np.copy(X)
# rgb = np.copy(X)
# shape_X             = X.shape
# X                   = X.reshape((shape_X[0]*shape_X[1], shape_X[2]))
# not_nan_idx         = np.where(~np.isnan(X))
# unique_not_nan_rows = np.unique(not_nan_idx[0])
# X                   = X[unique_not_nan_rows,:]
#
# #scatter plot for whole image; just for inspection, not for ML model
# X_OG = np.copy(rgb_OG)
# shape_X_OG             = X_OG.shape
# X_OG                   = X_OG.reshape((shape_X_OG[0]*shape_X_OG[1], shape_X_OG[2]))
# not_nan_idx_OG         = np.where(~np.isnan(X_OG))
# unique_not_nan_rows_OG = np.unique(not_nan_idx_OG[0])
# X_OG                   = X_OG[unique_not_nan_rows_OG,:]
#
# #ancillary composites to view
# Rbrn, Rveg, Rvis = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
# NDVI             = get_normalized_differenced_vegetation_index(Rvis, Rveg)
# NBR              = get_normalized_burn_ratio(Rveg, Rbrn)
# burnscar_mask    = get_burn_scar_composite(Rveg, Rbrn)
#
# # number of gaussian distributions to fit to the data in X
# n_componentsint = 3
#
# #the GaussianMixture().fit() will then create the distributions to match it
# X=X[:5000,:]
# model = GaussianMixture(n_components=n_componentsint).fit(X)
#
# print(model)
# f, ax = plt.subplots(ncols=4, figsize=(35,20))
#
# my_cmap = plt.cm.gist_ncar
# my_cmap.set_under('k',1)
# xmin, xmax, ymin, ymax = X_OG[:,0].min(), X_OG[:,0].max(),\
#                          X_OG[:,1].min(), X_OG[:,1].max()
#
# ax[0].sharex(ax[1])
# ax[0].sharey(ax[1])
#
# n=-1
# binWidth, binLength = 0.02, 0.02
# hist = ax[0].hist2d(X[:n,0], X[:n, 1],\
# bins = [np.arange(xmin, xmax, binWidth), np.arange(ymin, ymax, binLength)],\
# cmap=my_cmap, vmin=1, density=True)
# # print(hist[0].max())
#
# hist1 = ax[1].hist2d(X_OG[:n,0], X_OG[:n, 1],\
# bins = [np.arange(xmin, xmax, binWidth), np.arange(ymin, ymax, binLength)],\
# cmap=my_cmap, vmin=1, density=True)
#
#
# # rgb_OG[:,:,1:2] = rgb_OG[:,:,1:2]**2
# ax[2].imshow(rgb_plot)
# ax[3].imshow(rgb_OG)
#
# ax[0].set_xlabel('2.25 µm BRF')
# ax[0].set_ylabel('0.86 µm BRF')
# ax[0].set_title('2D Hist')
# ax[0].grid(linestyle='dashed', linewidth=2)
# ax[0].set_xlim([xmin, xmax])
# ax[0].set_ylim([ymin, ymax])
#
#
# ax[1].set_xlabel('2.25 µm BRF')
# ax[1].set_ylabel('0.86 µm BRF')
# ax[1].set_title('2D Hist Whole Domain')
# ax[1].grid(linestyle='dashed', linewidth=2)
# ax[1].set_xlim([xmin, xmax])
# ax[1].set_ylim([ymin, ymax])
#
# ax[2].set_title('DLCF RGB 2D Hist Domain')
# ax[3].set_title('DLCF RGB Entire Domain')

# burnscar_x = 0
# def on_xlims_change(event_ax):
#     new_row_coords = event_ax.get_xlim()
#     new_col_coords = event_ax.get_ylim()
#     # print("updated xlims: ", new_row_coords)
#     print('record this change? y/n')
#     stayinloop = True
#     while stayinloop==True:
#         answer = input()
#         if answer == 'y':
#             global burnscar_x
#             txt_burnscar_coords.writelines(str(burnscar_x))
#             txt_burnscar_coords.writelines(str(new_row_coords))
#             txt_burnscar_coords.writelines(str(new_col_coords))
#             burnscar_x = burnscar_x + 1
#             stayinloop = False
#         elif answer =='n':
#             stayinloop = False
#         else:
#             print('plz enter valid str, y or n for yes no respectively')
#
#
# # def on_ylims_change(event_ax):
# #     new_col_coords = event_ax.get_ylim()
# #     print("updated ylims: ", new_col_coords)
# #     print('record this change? y/n')
# #     stayinloop = True
# #     while stayinloop==True:
# #         answer = input()
# #         if answer == 'y':
# #             txt_burnscar_coords.writelines(str(new_col_coords))
# #             stayinloop = False
# #         elif answer =='n':
# #             stayinloop = False
# #         else:
# #             print('plz enter valid str, y or n for yes no respectively')
#
#     # print("updated ylims: ", event_ax.get_ylim())
#
# ax[2].callbacks.connect('xlim_changed', on_xlims_change)
# # ax[2].callbacks.connect('ylim_changed', on_ylims_change)
