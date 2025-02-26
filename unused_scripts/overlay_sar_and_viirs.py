import h5py
import numpy as np


commongrid_file = 'C:/Users/Javi/Documents/NOAA/Grids_West_CONUS_new.h5'

with h5py.File(commongrid_file, 'r') as hf_west_conus_grid:
    common_grid_lat = hf_west_conus_grid['Geolocation/Latitude'][:]
    common_grid_lon = hf_west_conus_grid['Geolocation/Longitude'][:]

common_grid_lon = np.flip(common_grid_lon, axis=1)*-1

with h5py.File('R:/satellite_data/viirs_data/noaa20/databases/VIIRS_burn_Scar_database.h5','r') as hf_database:

    timestamp = '2021167.2006_06.16.2021'
    dataset_names = list(hf_database[timestamp].keys())


    dataset_dict = {}
    for dataset_name in dataset_names:
        dataset_dict[dataset_name] = hf_database[timestamp+'/'+dataset_name][:]

    r_row_idx    = dataset_dict['regrid_row_idx']
    r_col_idx    = dataset_dict['regrid_col_idx']
    fill_val_idx = dataset_dict['fill_val_idx']

    # NBR           = dataset_dict['NBR'          ][r_row_idx, r_col_idx]
    # BRF_RGB       = dataset_dict['BRF_RGB'      ][r_row_idx, r_col_idx]
    cldmsk          = dataset_dict['cldmsk'       ]#[r_row_idx, r_col_idx]
    burn_scar_RGB   = dataset_dict['burn_scar_RGB']#[r_row_idx, r_col_idx]
    land_water_mask = dataset_dict['land_water_mask']#[r_row_idx, r_col_idx]
    lat_unprojected    = dataset_dict['lat']#[r_row_idx, r_col_idx]
    lon_unprojected    = dataset_dict['lon']#[r_row_idx, r_col_idx]
    # R_M7, R_M11 = burn_scar_RGB[:,:,1], burn_scar_RGB[:,:,0]
    # burn_scar_composite = get_burn_scar_composite(R_M7, R_M11, geotiff=False)

#     print(np.shape(cldmsk), np.shape(burn_scar_RGB), np.shape(fill_val_idx))
#     print(np.where(fill_val_idx[0]==3232))
    # NBR[fill_val_idx[0], fill_val_idx[1]]           = np.nan
    # BRF_RGB[fill_val_idx[0], fill_val_idx[1]]       = np.nan
#     cldmsk[fill_val_idx[0], fill_val_idx[1]]        = np.nan
#     burn_scar_RGB[fill_val_idx[0], fill_val_idx[1]] = np.nan





import scipy.misc
from scipy import ndimage

#burn_scar_RGB_mask
burn_scar_RGB_mask = np.copy(burn_scar_RGB)

thresh_upper_veggie = 0.1346
thresh_upper_idx_veggie = np.where(burn_scar_RGB[:,:,1]>thresh_upper_veggie)
thresh_lower_veggie = 0.0281
thresh_lower_idx_veggie = np.where(burn_scar_RGB[:,:,1]<thresh_lower_veggie)

for i in range(3):
    burn_scar_RGB_mask[:,:,i][thresh_upper_idx_veggie] = 0
    burn_scar_RGB_mask[:,:,i][thresh_lower_idx_veggie] = 0

#collopase RGB into one channel where non-zero positve values are valid
valid_idx_burn_scar_RGB_mask = np.where(burn_scar_RGB_mask !=0)

burn_scar_RGB_mask_collapsed = burn_scar_RGB_mask[:,:,0]

burn_scar_RGB_mask_collapsed = ndimage.gaussian_filter(burn_scar_RGB_mask_collapsed, sigma=3)
burn_scar_RGB_mask_collapsed[burn_scar_RGB_mask_collapsed<0.01]=np.nan


NBR_composite = (burn_scar_RGB[:,:,0] - burn_scar_RGB[:,:,1])/(burn_scar_RGB[:,:,0] + burn_scar_RGB[:,:,1])
NBR_composite[np.isnan(burn_scar_RGB_mask_collapsed)] = np.nan

#remove water and cloud
NBR_composite[cldmsk<2] = np.nan
NBR_composite[land_water_mask==0] = np.nan

#regrid
NBR_composite_regridded = NBR_composite[r_row_idx, r_col_idx]
NBR_composite_regridded[fill_val_idx[0], fill_val_idx[1]] = np.nan

#fill vals to -999
NBR_composite_regridded[np.isnan(NBR_composite_regridded)] = -999

#*******************************************************************************

# import os
# import xarray as xr
# import numpy as np
# import rasterio
#
#
# file_path = "C:/Users/Javi/Documents/NOAA/Roger_SAR_data/for_javier(2)/for_javier/"
# file_paths = [file_path + x for x in os.listdir(file_path)]
#
# f=file_paths[19]#14,15,16, 18, 19
# # 19 is view_descending_14th interferogram_10_13_2020-11_06_2020.tif
# # # Read the data
# da = xr.open_rasterio(f)
#
# # Compute the lon/lat coordinates with rasterio.warp.transform
# ny, nx = len(da['y']), len(da['x'])
# x, y = np.meshgrid(da['x'], da['y'])
#
# def get_geotiff_latlon_grids(geotiff_filepath):
#     '''
#     derived from:
#     https://xarray.pydata.org/en/v0.10.4/auto_gallery/plot_rasterio.html
#     '''
#     from rasterio.warp import transform
#     import xarray as xr
#     import numpy as np
#
#     # Read the data
#     da = xr.open_rasterio(geotiff_filepath)
#     # Compute the lon/lat coordinates with rasterio.warp.transform
#     #x is lon, y is lat
#     lon, lat = np.meshgrid(da['x'], da['y'])
#     nlon, nlat = da.sizes["x"], da.sizes["y"]
#
#     # Plot on a map
#     da.coords['lon'] = (('y', 'x'), lon)
#     da.coords['lat'] = (('y', 'x'), lat)
# #     print(da.coords['lon'])
#
#     da.close()
#
#     return lat, lon, nlat, nlon
#
#
# lat_SAR, lon_SAR, nlat, nlon = get_geotiff_latlon_grids(f)
# # print(nlon, nlat)
# # print(lat)
# with rasterio.open(f) as tif_data:
#     im_SAR = tif_data.read()
#     im_SAR = np.moveaxis(im_SAR, 0,2)
# im_SAR_shape = np.shape(im_SAR)
# im_SAR = np.reshape(im_SAR, (im_SAR_shape[0], im_SAR_shape[1]))
# # f_, ax = plt.subplots(figsize=(20,10))
# # # ax[0].imshow(lon_SAR, cmap='jet')
# # # ax[1].imshow(lat_SAR, cmap='jet')
# # ax.imshow(im_SAR, cmap='gist_ncar')
# # plt.show()

#*******************************************************************************


import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
# f, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
# im = ax[0].imshow(NBR_composite_regridded, cmap='gist_ncar', vmin=-1)
# im.cmap.set_under('k')
# ax[0].imshow(bsrgb)
# bsrgb = 1.5*burn_scar_RGB[r_row_idx, r_col_idx]
# print(np.shape(common_grid_lon), np.shape(common_grid_lat), np.shape(bsrgb))
ax_overlay = plt.axes(projection=ccrs.PlateCarree())

ax_overlay.set_extent([-130, -100, 30, 50], crs=ccrs.PlateCarree())

# f, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
# im = ax[0].imshow(NBR_composite_regridded, cmap='gist_ncar', vmin=-1)
# im.cmap.set_under('k')
# ax[0].imshow(bsrgb)
bsrgb = 1.5*burn_scar_RGB[r_row_idx, r_col_idx]
# print(np.shape(common_grid_lon), np.shape(common_grid_lat), np.shape(bsrgb))

# print(np.shape(im_SAR))
# r1,r2,c1,c2 = 2006,2245,1479,1703
# ax_overlay.pcolormesh(common_grid_lon[r1:r2,c1:c2],\
#                       common_grid_lat[r1:r2,c1:c2], bsrgb[r1:r2,c1:c2,1],\
#                       transform=ccrs.PlateCarree(), vmin=0, vmax=1, cmap='jet')

ax_overlay.pcolormesh(common_grid_lon,\
                      common_grid_lat, bsrgb[:,:,1],\
                      transform=ccrs.PlateCarree(), vmin=0, vmax=1, cmap='Greys_r')

NBR_composite_regridded[NBR_composite_regridded==-999] = np.nan
ax_overlay.pcolormesh(common_grid_lon,\
                      common_grid_lat, NBR_composite_regridded, cmap='gist_ncar', vmin=-1)

ax_overlay.coastlines()
ax_overlay.add_feature(cartopy.feature.STATES)

plt.show()

# ax_overlay.pcolormesh(common_grid_lon, common_grid_lat, bsrgb[:,:,1], transform=ccrs.PlateCarree())
# ax_overlay.pcolormesh(lon_SAR, lat_SAR, im_SAR, transform=ccrs.PlateCarree())

# ax.contourf(lon_unprojected, lat_unprojected, burn_scar_RGB[:,:,1], cmap='Greys_r', transform=ccrs.PlateCarree())

# plt.show()
