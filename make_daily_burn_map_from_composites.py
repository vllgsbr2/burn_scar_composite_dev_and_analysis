from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
from netCDF4 import Dataset
import h5py
from datetime import datetime
from burn_scar_composites import get_burn_scar_composite,\
                                 get_normalized_burn_ratio,\
                                 get_normalized_differenced_vegetation_index

#plots on correct screen and ctrl + c exits plot window
plt.switch_backend('qtAgg')

'''##########################################################################'''
'''################### Open up database to make GIS file ####################'''
#open up database to access DLCF RGB composite
home_dir = "/scratch/zt1/project/vllgsbr2-prj/raw_data_burnscar/data/daily_dlcf_rgb_composites/operational_composite_files/"
current_date = datetime.today().strftime('%m_%d_%Y')
database_name = home_dir + f'operational_comsposite_{current_date}.h5'

with h5py.File(database_name, 'r') as hf_database:

    # for i, time_stamp in enumerate(list(hf_database.keys())):
    time_stamp = list(hf_database.keys())
    X = hf_database[time_stamp][:]

    # rgb_OG = np.copy(X)
    rgb_OG_copy = np.copy(X)

    #ancillary composites to view
    Rbrn, Rveg, Rvis = rgb_OG_copy[:,:,0], rgb_OG_copy[:,:,1], rgb_OG_copy[:,:,2]
    # NDVI             = get_normalized_differenced_vegetation_index(Rvis, Rveg)
    # NBR              = get_normalized_burn_ratio(Rveg, Rbrn)
    burnscar_mask    = get_burn_scar_composite(Rveg, Rbrn)

    # #workflow to label and analyze burn scars in DLCF RGB composite
    # burnscar_semi_labeled_dataset_file = 'subsetted_burn_scar_coordinates.txt'
    # df_burnscar_semi_labeled = pd.read_csv(burnscar_semi_labeled_dataset_file,\
    #                                          header=0, delimiter=', ', skiprows=7)
    # # print(df_burnscar_semi_labeled)
    #
    # #build boxes around burn scars then visualize on RGB
    # col1 = df_burnscar_semi_labeled['col1'].tolist()
    # col2 = df_burnscar_semi_labeled['col2'].tolist()
    # row1 = df_burnscar_semi_labeled['row1'].tolist()
    # row2 = df_burnscar_semi_labeled['row2'].tolist()
    #
    # count = 0
    # for i in range(len(col1)):
    #     x=burnscar_mask[row1[i]:row2[i],col1[i]:col2[i]]
    #     idx_valid = np.where(np.isnan(x)==False)
    #     count += np.nansum(len(idx_valid[0]))

    '''##########################################################################'''
    '''############## save the RGB, primitive mask, lat/lon, timestamp ##########'''

    home           = '/scratch/zt1/project/vllgsbr2-prj/'
    home_data      = home      + 'raw_data_burnscar/data/'
    grid_file_path = home_data + 'grids/Grids_West_CONUS_new.h5'
    save_path      = home_data + 'operational_data_feed/'
    sample_fname   = 'burned_area_map_GIS_valid_{time_stamp}.nc'
    with Dataset(save_path + sample_fname, 'w', format='NETCDF4_CLASSIC') as nc_burnscar,\
         h5py.File(grid_file_path,'r') as h5_lat_lon:

        r1,r2,c1,c2 = 0,-1,0,-1
        pbsm_shape = np.shape(burnscar_mask[r1:r2,c1:c2])

        # create dimensions of data
        nc_burnscar.createDimension('lat', pbsm_shape[0])
        nc_burnscar.createDimension('lon', pbsm_shape[1])
        nc_burnscar.createDimension('time', None)
        nc_burnscar.createDimension('channel', 3)

        # define lat/lon and time variables
        mon, day, year = time_stamp[:2], time_stamp[3:5], time_stamp[6:]
        time           = nc_burnscar.createVariable('time', np.int8, ('time'))
        time.units     = "hours since {}-{}-{}".format(mon, day, year)
        time.long_name = 'time'
        time.calendar  = 'none'
        time[:]        = 0

        lat           = nc_burnscar.createVariable('lat_grid', np.float32, ('lat','lon'), fill_value=-999)
        lat.units     = 'degrees_north'
        lat.long_name = 'latitude'
        lon           = nc_burnscar.createVariable('lon_grid', np.float32, ('lat','lon'), fill_value=-999)
        lon.units     = 'degrees_east'
        lon.long_name = 'longitude'

        #save data into created variables
        lat[:,:] = h5_lat_lon['Geolocation/Latitude' ][r1:r2,c1:c2]
        lon_temp = (h5_lat_lon['Geolocation/Longitude'][r1:r2,c1:c2]*(-1))
        lon[:,:] = np.flip(h5_lat_lon['Geolocation/Longitude'][:]*(-1), 1)[r1:r2,c1:c2]
        #adjust lon to go from west to east
        # lon_adjust = lon_temp * np.nan
        # for k in range(0, len(lon)):
        #     lon_adjust[k] = sorted(lon[k])
        # lon[:,:] = np.copy(lon_adjust)

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

