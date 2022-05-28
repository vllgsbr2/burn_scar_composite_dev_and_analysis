# from read_MODIS_02 import get_data, get_radiance_or_reflectance, prepare_data
# from read_MODIS_03 import get_lat, get_lon
# from read_MODIS_35 import *
# from pyhdf.SD import SD
# import pytaf#!pip install git+https://github.com/TerraFusion/pytaf.git
# #from multicore_processing import multi_core_crop
import h5py
import sys
import os
import time



def regrid_latlon_source2target(source_lat, source_lon, target_lat, target_lon, source_data):
    '''
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
    #radius in meters to search around pixel for a neighbor
    max_radius = 5556.
    target_data = pytaf.resample_n(source_lat, source_lon, target_lat,\
                                   target_lon, source_data, max_radius)
    return target_data

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
        xl = -150000.000
        yl = -200000.0
        xr = 150000.000
        yr = 200000.0
        #print(xl,yl)
        proj_dict = {'proj': 'aea', 'lat_1': lat_p1, 'lat_2': lat_p2, 'lat_0': lat_c, 'lon_0': lon_c,'x_0': 0,'y_0':0, 'units': 'm'}

        area_extent = (xl, yl, xr, yr)
        area_def = create_area_def('pta_la', proj_dict, area_extent=area_extent,resolution=resolution)
        lons, lats = area_def.get_lonlats()
        #save_quicklook('check_grid.png', area_def, lats, label='Quick Check')
        return lons, lats

    home_dir = "C:/Users/Javi/Documents/NOAA/"
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
            resolution = (75,75)

            lat_lc = lat_sw
            lon_lc = lon_sw
            lat_rc = lat_ne
            lon_rc = lon_ne

            lons,lats = get_lonlats(lat_p1,lat_p2,lat_c,lon_c,lat_lc,lon_lc,lat_rc,lon_rc,resolution)

            #write grid to h5 file
            grid_name = "{}Grids_{}.h5".format(home_dir, ptaname)
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

if __name__ == "__main__":
    print(0)
    make_custom_lat_lon_grid()
    print(1)
    #
    # import matplotlib.pyplot as plt
    # import h5py
    # import copy
    # #home       = '/data/keeling/a/vllgsbr2/c/LA_test_case_data/'
    # #file_mod03 = home + 'MOD03.A2017246.1855.061.2017257170030.hdf'
    # #file_mod02 = home + 'MOD021KM.A2017246.1855.061.2017258202757.hdf'
    # home       = '/data/keeling/a/vllgsbr2/c/MAIA_Threshold_Dev/LA_PTA_MODIS_Data'
    # file_mod03 = home + '/MOD_03/MOD03.A2002002.1920.061.2017179152611.hdf'
    # file_mod02 = home + '/MOD_02/MOD021KM.A2002002.1920.061.2017179161255.hdf'
    #
    # file_MAIA  = '/data/keeling/a/vllgsbr2/c/LA_PTA_MAIA.hdf5'
    # file_MAIA  = h5py.File(file_MAIA, 'r')
    #
    # #load in lat/lon from modis granule
    # s_lat = get_lat(file_mod03).astype(np.float64)
    # s_lon = get_lon(file_mod03).astype(np.float64)
    # t_lat = file_MAIA['lat'][()].astype(np.float64)
    # t_lon = file_MAIA['lon'][()].astype(np.float64)
    #
    # #load in data to be regridded from modis granule
    # red_band = prepare_data(file_mod02, 'EV_250_Aggr1km_RefSB', False)
    # s_data   = red_band[0,:,:].astype(np.float64)
    #
    #
    # #do something illegal
    # #make copies for regridding routine to modify
    # target_lat_row = np.copy(t_lat)
    # target_lon_row = np.copy(t_lon)
    # target_lat_col = np.copy(t_lat)
    # target_lon_col = np.copy(t_lon)
    # #new indices to regrid, use that as numpy where if you will
    # nx, ny = 2030, 1354
    # rows = np.arange(nx)
    # cols = np.arange(ny)
    # col_mesh, row_mesh = np.meshgrid(cols, rows)
    #
    # import time
    #
    # for i in range(5):
    #
    #     start=time.time()
    #
    #     regrid_row_idx =regrid_latlon_source2target(np.copy(s_lat), np.copy(s_lon), np.copy(t_lat), np.copy(t_lon),\
    #                                      np.copy(row_mesh).astype(np.float64)).astype(np.int)
    #
    #     regrid_col_idx = regrid_latlon_source2target(np.copy(s_lat), np.copy(s_lon), np.copy(t_lat), np.copy(t_lon),\
    #                                      np.copy(col_mesh).astype(np.float64)).astype(np.int)
    #
    #     print(-start+time.time())
    #     #print(regrid_row_idx.shape, regrid_col_idx.shape)
    #
    #     #get the regridded data
    #     target_data = s_data[regrid_row_idx, regrid_col_idx]#regrid_MODIS_2_MAIA(s_lat, s_lon, t_lat, t_lon, s_data)
    #     s_data[regrid_row_idx, regrid_col_idx] += 0.5
    #
    #     #plot the MAIA data next to modis data to see if result makes sense
    #     #print(np.shape(s_data), np.shape(target_data))
    #     f, ax = plt.subplots(ncols=2)
    #     #ax[0].imshow(regrid_col_idx)
    #     #ax[1].imshow(regrid_row_idx)
    #     ax[0].imshow(s_data)
    #     ax[1].imshow(target_data)
    #     #f.savefig('./test_pytaf.png', dpi=300)
    #     plt.show()
