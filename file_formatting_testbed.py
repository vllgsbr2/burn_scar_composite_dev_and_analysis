from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from netCDF4 import Dataset
import h5py
import configparser
from burn_scar_composites import get_burn_scar_composite,\
                                 get_normalized_burn_ratio,\
                                 get_normalized_differenced_vegetation_index

#plots on correct screen and ctrl + c exits plot window
plt.switch_backend('qtAgg')

'''##########################################################################'''
'''################### Open up database to make GIS file ####################'''
#open up database to access DLCF RGB composite
config = configparser.ConfigParser()
config.read('config_filepaths.txt')
home_dir = config['file paths mac']['database dir']
databse_name = 'daily_DLCF_composites_2021_west_CONUS_all_days.h5'

with h5py.File(home_dir + databse_name, 'r') as hf_database:
    X = hf_database['08.10.2021'][:]

# rgb_OG = np.copy(X)
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

'''##########################################################################'''
'''############## save the RGB, primitive mask, lat/lon, timestamp ##########'''

grid_file_path = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/VIIRS_database/databases/Grids_West_CONUS_new.h5'
save_path = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/VIIRS_database/databases/'
sample_fname = 'burn_scar_mask_GIS_sample_version_1.1.nc'
with Dataset(save_path + sample_fname, 'w', format='NETCDF4_CLASSIC') as nc_burnscar,\
     h5py.File(grid_file_path,'r') as h5_lat_lon:

    # coordinates for the single burn scar that looks like australia
    # r1, r2, c1, c2 = 1260, 1340, 360, 480
    r1, r2, c1, c2 = 0, -1, 0, -1
    pbsm_shape = np.shape(burnscar_mask[r1:r2,c1:c2])

    # create dimensions of data
    nc_burnscar.createDimension('lat', pbsm_shape[0])
    nc_burnscar.createDimension('lon', pbsm_shape[1])
    nc_burnscar.createDimension('time', None)
    nc_burnscar.createDimension('channel', 3)

    # define lat/lon and time variables
    time           = nc_burnscar.createVariable('time', np.int8, ('time'))
    time.units     = 'hours since 2021-10-14 20:54:00'
    time.long_name = 'time'
    time.calendar  = 'none'

    lat           = nc_burnscar.createVariable('lat', np.float32, ('lat','lon'), fill_value=-999)
    lat.units     = 'degrees_north'
    lat.long_name = 'latitude'
    lon           = nc_burnscar.createVariable('lon', np.float32, ('lat','lon'), fill_value=-999)
    lon.units     = 'degrees_east'
    lon.long_name = 'longitude'

    #save data into created variables
    lat[:,:] = h5_lat_lon['Geolocation/Latitude' ][r1:r2,c1:c2]
    lon[:,:] = h5_lat_lon['Geolocation/Longitude'][r1:r2,c1:c2]*(-1)
    #adjust lon to go from west to east
    lon_adjust = lon * np.nan
    for k in range(0, len(lon)):
        lon_adjust[k] = sorted(lon[k])
    lon[:,:] = np.copy(lon_adjust)

    pbsm               = nc_burnscar.createVariable('pbsm',np.float64,('time','lat','lon'), fill_value=-999) # note: unlimited dimension is leftmost
    pbsm.units         = 'unitless'
    pbsm.standard_name = 'primitive burn scar mask'
    pbsm[:,:]          = burnscar_mask[r1:r2,c1:c2].reshape((1,pbsm_shape[0], pbsm_shape[1]))

    day_land_cloud_fire_RGB               = nc_burnscar.createVariable('day_land_cloud_fire_RGB',np.float64,('time','lat','lon','channel'), fill_value=-999) # note: unlimited dimension is leftmost
    day_land_cloud_fire_RGB.units         = 'unitless'
    day_land_cloud_fire_RGB.standard_name = 'day_land_cloud_fire_RGB'
    day_land_cloud_fire_RGB[:,:,:]        = np.copy(X)[r1:r2,c1:c2,:].reshape((1,pbsm_shape[0], pbsm_shape[1], 3))

    # define CRS for file
    crs = nc_burnscar.createVariable('spatial_ref', 'i4')
    crs.spatial_ref = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
























'''######################## Alex Mccombs code ref ###########################'''
'''##########################################################################'''
'''##########################################################################'''
'''##########################################################################'''
# openpath = '/Users/agmccombs/Documents/BurnScarDataset/'
# filename = 'burn_scar_mask_GIS_sample.nc'
# outputfilename = 'burn_scar_mask_sample_edited.nc'
# ##Import Libraries
# import netCDF4
# from netCDF4 import Dataset
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import numpy as np
#
# #Load file
# with netCDF4.Dataset(openpath+filename, 'r') as nc_burnscar:
#     print(nc_burnscar['pbsm'])
#     x=nc_burnscar['pbsm'][:]
#     y=nc_burnscar['day_land_cloud_fire_RGB'][:]
#     lat = nc_burnscar['lat'][:]
#     lon = nc_burnscar['lon'][:]
#
#     #create new NetCDF file that is georeferenced.
#     with Dataset(openpath+outputfilename,mode='w',format='NETCDF4_CLASSIC') as ds:
#
#         time = ds.createDimension('time', None)
#         lat2 = ds.createDimension('lat', len(lat))
#         lon2 = ds.createDimension('lon', len(lon[0,:]))
#         channel2 = ds.createDimension('channel', 3)
#
#         #create variables
#         times = ds.createVariable('time', np.float64, ('time',      ))
#         lats  = ds.createVariable('lat' , np.float64, ('lat' , 'lon'))
#         lons  = ds.createVariable('lon' , np.float64, ('lat' , 'lon'))
#         RGB   = ds.createVariable('day_land_cloud_fire_RGB', np.float64,\
#                                  ('time', 'lat', 'lon', 'channel'))
#         mask  = ds.createVariable('pbsm', np.float64, ('time', 'lat', 'lon',))
#
#         RGB.units  = 'unitless'
#         mask.units = 'unitless'
#         lat.units  = 'degrees_north'
#         lon.units  = 'degrees_east'
#
#         RGB.standard_name  = 'day_land_cloud_fire_RGB'
#         mask.standard_name = 'primitive burn scar mask'
#         times.long_name    = 'time'
#         times.calendar     = 'none'
#         lons.long_name     = 'longitude'
#         lats.long_name     = 'latitude'
#
#         crs = ds.createVariable('spatial_ref', 'i4')
#         crs.spatial_ref = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
#         #adjust lon to go from west to east
#         lon_adjust = lon * np.nan
#         for k in range(0, len(lon)):
#             lon_adjust[k] = sorted(lon[k])
#
#         #Assign values to netcdf
#         lats[:,:]       = np.copy(lat)
#         lons[:,:]       = np.copy(lon_adjust)
#         RGB[0, :, :, :] = np.copy(y)
#         mask[0, :, :]   = np.copy(x)
