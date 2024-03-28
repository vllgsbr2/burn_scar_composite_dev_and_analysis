# from read_MODIS_02 import get_data, get_radiance_or_reflectance, prepare_data
# from read_MODIS_03 import get_lat, get_lon
# from read_MODIS_35 import *
# from pyhdf.SD import SD
# import pytaf#!pip install git+https://github.com/TerraFusion/pytaf.git
# #from multicore_processing import multi_core_crop








# code to try in gdal for gis ; https://gdal.org

# shape = data.shape
# outdata = driver.Create(filename, shape[1], shape[0], 1, gdal.GDT_int16)
# outdata.SetGeoTransform(geo_transform)
# outdata.SetProjection(projection)
# outdata.GetRasterBand(1).SetNoDataValue(no_data_value)
# outdata.GetRasterBand(1).WriteArray(data)
# outdata.FlushCache()
# data = numpy array representing the data
# from osgeo import gdal

#
import h5py
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

def flip_arr(arr):
    '''
    return: array flipped over each of the 1st 2 axes for proper display using
    ax.imshow(arr)
    '''
    arr=np.flip(arr, axis=0)
    arr=np.flip(arr, axis=1)
    return arr


def regrid_latlon_source2target(source_lat, source_lon, target_lat, target_lon, source_data,max_radius = 5556.):
    '''
    https://github.com/TerraFusion/pytaf
    Objective:
        take source geolocation and source data, and regrid it to a target
        geolocation using nearest nieghbor.
    Arguments:
        source_lat, source_lon {2D narrays} -- lat/lon of data to be regridded
        target_lat, target_lon {2D narrays} -- lat/lon of refrence grid
        source_data {2D narray} -- data that will be regridded
    Returns:
        target_data {2D narray} -- returns regridded source data, such that it
                                   matches up with the target lat/lon
    '''
    import pytaf
    #radius in meters to search around pixel for a neighbor

    target_data = pytaf.resample_n(source_lat, source_lon, target_lat,\
                                   target_lon, source_data, max_radius)
    return target_data

def regrid_latlon_source2target_new(source_lat, source_lon, target_lat, target_lon, max_radius = 5556.):
    '''
    https://github.com/TerraFusion/pytaf
    Objective:
        take source geolocation and source data, and regrid it to a target
        geolocation using nearest nieghbor.
    Arguments:
        source_lat, source_lon {2D narrays} -- lat/lon of data to be regridded
        target_lat, target_lon {2D narrays} -- lat/lon of refrence grid
    Returns:
        regrid_row_idx {2D narray} -- regridded row mesh which can be used
                                      to reindex all data associated with the
                                      source_lat/source_lon
        regrid_col_idx {2D narray} -- regridded col mesh which can be used
                                      to reindex all data associated with the
                                      source_lat/source_lon
        fill_val_idx {2D narray}   -- numpy where return of indicies of data
                                      that did not lie in the refrence grid
    '''
    import pytaf
    #radius in meters to search around pixel for a neighbor
    # max_radius = 5556.

    data_nx, data_ny             = np.shape(source_lat)
    data_rows                    = np.arange(data_nx).astype(np.float64)
    data_cols                    = np.arange(data_ny).astype(np.float64)
    data_col_mesh, data_row_mesh = np.meshgrid(data_cols, data_rows)

    source_lat = source_lat.astype(np.float64)
    source_lon = source_lon.astype(np.float64)

    target_lat = target_lat.astype(np.float64)
    target_lon = target_lon.astype(np.float64)

    import time
    start=time.time()

    print('regridding rows')
    regrid_row_idx = pytaf.resample_n(np.copy(source_lat),\
                                      np.copy(source_lon),\
                                      np.copy(target_lat),\
                                      np.copy(target_lon),\
                                      data_row_mesh.astype(np.float64),\
                                      max_radius).astype(int)
    print(-start+time.time())
    start=time.time()

    print('regridding cols')
    regrid_col_idx = pytaf.resample_n(np.copy(source_lat),\
                                      np.copy(source_lon),\
                                      np.copy(target_lat),\
                                      np.copy(target_lon),\
                                      data_col_mesh.astype(np.float64),\
                                      max_radius).astype(int)
    print(-start+time.time())

    #grab -999 fill values in regrid col/row idx
    #use these positions to write fill values when regridding the rest of the data
    fill_val     = -999
    fill_val_idx = np.where((regrid_row_idx == fill_val) | \
                            (regrid_col_idx == fill_val)   )

    return regrid_row_idx, regrid_col_idx, fill_val_idx

def make_custom_lat_lon_grid():
    '''
    Guangyu Zhao's MAIA lat lon grid code
    Use his sample CSV and reformat it to make any grid
    '''
    import pandas as pd
    import pyproj
    from pyresample import create_area_def, save_quicklook
    import h5py

    def get_lonlats(lat_p1,lat_p2,lat_c,lon_c,lat_lc,lon_lc,lat_rc,lon_rc,resolution):
        projname = '+proj=aea +lat_1='+str(lat_p1)+' +lat_2='+ str(lat_p2), \
                         ' +lat_0='+str(lat_c)+' +lon_0='+str(lon_c)+' +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '
        p_mt  = pyproj.Proj("".join(projname))
        xl,yl = p_mt(lon_lc,lat_lc)
        xr,yr = p_mt(lon_rc,lat_rc)
        # xl = -150000.000
        # yl = -200000.0
        # xr = 150000.000
        # yr = 200000.0
        #print(xl,yl)

        x, y = (xl-xr)/2, (yl-yr)/2
        xl = -1*np.abs(x)
        yl = -1*np.abs(y)
        xr = np.abs(x)
        yr = np.abs(y)

        proj_dict = {'proj': 'aea', 'lat_1': lat_p1, 'lat_2': lat_p2, 'lat_0': lat_c, 'lon_0': lon_c,'x_0': 0,'y_0':0, 'units': 'm'}
        area_extent = (xl, yl, xr, yr)
        # print((area_extent[2]-area_extent[0])/2, (area_extent[3]-area_extent[1])/2)
        area_def = create_area_def('pta_la', proj_dict, area_extent=area_extent,resolution=resolution)
        lons, lats = area_def.get_lonlats()
        #save_quicklook('check_grid.png', area_def, lats, label='Quick Check')
        return lons, lats

    home_dir = "/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/"
    excelfile = 'Master_Target_List_ancillary_file_prelim_v0.0.xlsx'
    with pd.ExcelFile(home_dir+excelfile) as reader:
        df = pd.read_excel(reader, sheet_name='Master_Target_List')

    #grab PTA name
    ptas = df.loc[df['TargetType'] == 'PTA']

    #fix loop to just grab one grid from custom csv
    #could even do static inputs into the function
    for index, las in ptas.iterrows():
        #just grab the last addition, which is the west cnous coordinates I made
        print(index)
        if index==41:
            print(1.5)
            ptaname = las['Target_Short_Name']
            print(ptaname)
            ptaname = ptaname.replace('-','_')

            #central coordinate of bounding box
            lat_c = las['Central_latitude_degrees']
            lon_c = las['Central_Longitude_degrees']

            #AEA = Albert's Equal Area
            #parallels are west and east bounds
            lat_p1 = las['AEA_1st_parallel']
            lat_p2 = las['AEA_2nd_parallel']

            #coordinates of bounding box corners
            lat_nw = las['L2_L4_BB_coord_NW_lat']
            lon_nw = las['L2_L4_BB_coord_NW_lon']

            lat_ne = las['L2_L4_BB_coord_NE_lat']
            lon_ne = las['L2_L4_BB_coord_NE_lon']

            lat_sw = las['L2_L4_BB_coord_SW_lat']
            lon_sw = las['L2_L4_BB_coord_SW_lon']

            lat_se = las['L2_L4_BB_coord_SE_lat']
            lon_se = las['L2_L4_BB_coord_SE_lon']

            #meters resolution of pixel
            resolution = (750,750)

            lat_lc = lat_sw
            lon_lc = lon_sw
            lat_rc = lat_ne
            lon_rc = lon_ne

            # print(lat_lc, lat_rc, lon_lc, lon_rc)

            lons,lats = get_lonlats(lat_p1,lat_p2,lat_c,lon_c,lat_lc,lon_lc,lat_rc,lon_rc,resolution)

            #write grid to h5 file
            grid_name = "{}Grids_{}_new.h5".format(home_dir, ptaname)
            with h5py.File(grid_name,'w') as f:
                grpm = f.create_group("Geolocation")
                h = grpm.create_dataset('Latitude', data=lats,dtype='float32')
                h = grpm.create_dataset('Longitude', data=lons,dtype='float32')
        else:
            pass

def get_lat_lon_grid_from_geotiff(tif_sar_f_name):
    # tif_sar_f_name = 'C:/Users/Javi/Documents/NOAA/Roger_SAR_data/for_javier(2)/for_javier/view_descending_5th interferogram_08_14_2020-09_19_2020.tif'

    # tif_sar_f_name = 'C:/Users/Javi/Desktop/AIRSAR_CVV_120304.tif'
    #https://gis.stackexchange.com/questions/129847/obtain-coordinates-and-corresponding-pixel-values-from-geotiff-using-python-gdal
    import rasterio
    import numpy as np
    from affine import Affine
    from pyproj import Proj, transform

    fname = tif_sar_f_name

    # Read raster
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    def rc2en(r, c):
        return (c, r) * T1
    # rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    lon, lat = np.vectorize(rc2en, otypes=[float, float])(rows, cols)

    # # Project all longitudes, latitudes
    # p2 = Proj(proj='latlong',datum='WGS84')
    # longs, lats = transform(p1, p2, eastings, northings)

    return lat, lon

def running_composite(viirs_database_file, num_days_2_composite=8):
    '''
    Inputs:
        viirs_database_file {str}: path to h5 database of viirs burnscar bands
        num_days_2_composite {int}: either 8 or 16 should be good
    regrids data to common grid, then lays down latest cloud free data starting
    with the 1st day until the last day. Such that the latest cloud free obs is
    visibile on the image.

    return the bands stacked, reprojected, and cloud free ready for further
    processing.
    '''

    #loop to regrid any subset of granules
    with h5py.File(viirs_database_file, 'r') as h5_viirs_f:
        timestamps = list(h5_viirs_f.keys())
        # timestamps_modifed = [x[:12] for x in timestamps]
        data_names = list(h5_viirs_f[timestamps[0]].keys())

        for timestamp in timestamps:
            data_dict = {}
            for data_name in data_names:
                data_dict[data_name] =  h5_viirs_f[timestamp+'/'+data_name][:]

            viirs_lat = data_dict['lat'][:]
            viirs_lon = data_dict['lon'][:]

            target_lat, target_lon = common_grid_lat, common_grid_lon
            source_lat, source_lon = viirs_lat, viirs_lon

            regrid_row_idx,\
            regrid_col_idx,\
            fill_val_idx   = regrid_latlon_source2target_new(source_lat,\
                                                             source_lon,\
                                                             target_lat,\
                                                             target_lon)

            viirs_DLCF_RGB_regridded = viirs_DLCF_RGB[regrid_row_idx, regrid_col_idx]
            source_lat_regridded     = source_lat[regrid_row_idx    , regrid_col_idx]
            source_lon_regridded     = source_lon[regrid_row_idx    , regrid_col_idx]

            viirs_DLCF_RGB_regridded[fill_val_idx] = np.nan
            source_lat_regridded[fill_val_idx]     = np.nan
            source_lon_regridded[fill_val_idx]     = np.nan


if __name__ == "__main__":
    import sys
    make_custom_lat_lon_grid()
    commongrid_file = 'C:/Users/Javi/Documents/NOAA/Grids_West_CONUS_new.h5'
    with h5py.File(commongrid_file, 'r') as hf_west_conus_grid:
        common_grid_lat = hf_west_conus_grid['Geolocation/Latitude'][:]
        common_grid_lon = hf_west_conus_grid['Geolocation/Longitude'][:]

    common_grid_lon = np.flip(common_grid_lon, axis=1)*-1
    # #put viirs and SAR data on same grid made by make_custom_lat_lon_grid()
    # #grab sar lat/lon grid
    # tif_sar_f_name   = 'C:/Users/Javi/Documents/NOAA/Roger_SAR_data/for_javier(2)/for_javier/view_descending_1th interferogram_07_21_2020-08_14_2020.tif'
    # sar_lat, sar_lon = get_lat_lon_grid_from_geotiff(tif_sar_f_name)
    #
    # def get_roger_SAR_data(tif_sar_f_name):
    #     import rasterio
    #     with rasterio.open(tif_sar_f_name) as src:
    #         image = src.read(1)
    #     return image
    # sar_data = get_roger_SAR_data(tif_sar_f_name)
    #
    # #grab VIIRS data from h5 database
    def get_VIIRS_database_composites(h5_viirs_name, timestamp):
        import h5py

        with h5py.File(h5_viirs_name, 'r') as h5_viirs_f:
            timestamps = list(h5_viirs_f.keys())
            timestamp  = [x for x in timestamps if x[:12]==timestamp]
            if len(timestamp) == 1:
                timestamp = timestamp[0]
                data_dict  = {}
                data_names = list(h5_viirs_f[timestamp].keys())
                for data_name in data_names:
                    data_dict[data_name] =  h5_viirs_f[timestamp+'/'+data_name][:]

                return data_dict
            else:
                print('timestamp not found, choose from')
                # print(timestamps)

    h5_viirs_name   = 'R:/satellite_data/viirs_data/noaa20/databases/VIIRS_burn_Scar_database.h5'
    timestamp       = '2021227.0954'#'2021225.1842'
    viirs_data_dict = get_VIIRS_database_composites(h5_viirs_name, timestamp)

    viirs_lat      = viirs_data_dict['lat']
    viirs_lon      = viirs_data_dict['lon']
    viirs_DLCF_RGB = viirs_data_dict['burn_scar_RGB']

    #put both on common grid using the pytaf resample_n wrapper function
    #just VIIRS for now, since the SAR is such a higher res, it won't snap
    #to the grid properly. Will need to upscale the VIIRS data and common grid
    #to match SAR, or downscale the SAR to the VIIRS data (might be better)

    target_lat = common_grid_lat
    target_lon = common_grid_lon
    source_lat, source_lon = viirs_lat, viirs_lon
    max_radius = 2000.

    # regrid_row_idx,\
    # regrid_col_idx,\
    # fill_val_idx   = regrid_latlon_source2target_new(source_lat,\
    #                                                  source_lon,\
    #                                                  target_lat,\
    #                                                  target_lon,\
    #                                                  max_radius)

    regrid_row_idx = viirs_data_dict['regrid_row_idx']
    regrid_col_idx = viirs_data_dict['regrid_col_idx']
    fill_val_idx   = viirs_data_dict['fill_val_idx']

    # print(regrid_row_idx-regrid_row_idx_database)
    # print(regrid_col_idx-regrid_col_idx_database)
    # print(fill_val_idx-fill_val_idx_database)
    #
    # sys.exit()
    viirs_DLCF_RGB_regridded = viirs_DLCF_RGB[regrid_row_idx, regrid_col_idx]
    source_lat_regridded     = source_lat[regrid_row_idx    , regrid_col_idx]
    source_lon_regridded     = source_lon[regrid_row_idx    , regrid_col_idx]

    viirs_DLCF_RGB_regridded[fill_val_idx[0], fill_val_idx[1]] = np.nan
    source_lat_regridded[fill_val_idx[0], fill_val_idx[1]]     = np.nan
    source_lon_regridded[fill_val_idx[0], fill_val_idx[1]]     = np.nan

    #plotting###################################################################
    plt.style.use('dark_background')
    f, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12,12))

    vminlat, vmaxlat, vminlon, vmaxlon = 28,51,-127,-100
    ax[0,0].imshow(2*viirs_DLCF_RGB)
    ax[0,1].imshow(source_lat, cmap='jet',vmin=vminlat,vmax=vmaxlat)
    ax[0,2].imshow(source_lon, cmap='jet',vmin=vminlon,vmax=vmaxlon)

    ax[1,0].imshow(2*viirs_DLCF_RGB_regridded)
    ax[1,1].imshow(source_lat_regridded, cmap='jet',vmin=vminlat,vmax=vmaxlat)
    ax[1,2].imshow(source_lon_regridded, cmap='jet',vmin=vminlon,vmax=vmaxlon)

    ax[2,1].imshow(common_grid_lat, cmap='jet',vmin=vminlat,vmax=vmaxlat)
    ax[2,2].imshow(common_grid_lon, cmap='jet',vmin=vminlon,vmax=vmaxlon)

    ax[0,0].set_title('viirs_DLCF_RGB')
    ax[0,1].set_title('source_lat')
    ax[0,2].set_title('source_lon')

    ax[1,0].set_title('viirs_DLCF_RGB regridded')
    ax[1,1].set_title('source_lat regridded')
    ax[1,2].set_title('source_lon regridded')

    ax[2,0].axis('off')
    ax[2,1].set_title('common_grid_lat')
    ax[2,2].set_title('common_grid_lon')

    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    plt.show()
