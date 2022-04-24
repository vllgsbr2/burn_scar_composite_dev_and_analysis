import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os


def get_VJ103_geo(geo_file, get_all=True):
    '''
    input: VIIRS VJ103 .nc file
           get_all - set to true by default, can be set to return only lat, lon, SZA
    return: lat, lon, SZA, VZA, SAA, VAA
    '''
    from netCDF4 import Dataset

    with Dataset(geo_file, 'r') as nc_geo_file_obj:
        geolocation_ncObj = nc_geo_file_obj['geolocation_data']
        lat, lon = geolocation_ncObj['latitude'][:], geolocation_ncObj['longitude'][:]

        SZA = geolocation_ncObj['solar_zenith'][:]
        if get_all:
            VZA = geolocation_ncObj['sensor_zenith'][:]
            SAA = geolocation_ncObj['solar_azimuth'][:]
            VAA = geolocation_ncObj['sensor_azimuth'][:]

            return lat, lon, SZA, VZA, SAA, VAA
        else:
            return lat, lon, SZA


def get_VJ102_ref(ref_file, which_bands):
    '''
    input: VIIRS VJ103 .nc file
    return: lat, lon, SZA, VZA, SAA, VAA
    '''
    from netCDF4 import Dataset

    with Dataset(ref_file, 'r') as nc_ref_file_obj:
        observation_data_ncObj = nc_ref_file_obj['observation_data']

        num_bands=len(which_bands)
        # M_bands = np.zeros((3232,3200,len(which_bands)))
        M_bands = []
        for i, band_num in zip(range(num_bands), which_bands):
            M_bands.append(observation_data_ncObj['M{:02d}'.format(band_num)][:])

        M_bands = np.moveaxis(np.array(M_bands), 0,2)
        M_bands[M_bands >=65532 ] = np.nan

        return M_bands

def get_normalized_burn_ratio(R_M7, R_M11):
    # Grab VIIRS bands (0.86 microns I2 or 0.86 M7) and 2.25 M11
    # we can calculate (R225-R86)/(R86+R225)
    return (R_M11-R_M7)/(R_M11+R_M7)

def flip_arr(arr):
    '''
    return: array flipped over each of the 1st 2 axes for proper display using
    ax.imshow(arr)
    '''
    arr=np.flip(arr, axis=0)
    arr=np.flip(arr, axis=1)
    return arr

def get_BRF_lat_lon(geo_file, ref_file, which_bands):

    lat, lon, SZA = get_VJ103_geo(geo_file, get_all=False)
    time_stamp_current = geo_file[-33:-21]
    M_bands = get_VJ102_ref(ref_file, which_bands)

    cosSZA = np.cos(np.deg2rad(SZA))

    for i in range(len(which_bands)):
        try:
            M_bands[:,:,i] /=  cosSZA
        except:
            M_bands[:3216,:,i] /=  cosSZA[:3216,:]

    return M_bands, lat, lon

def get_BRF_RGB(R_M5,R_M4,R_M3):
    return np.dstack((R_M5,R_M4,R_M3))

def get_NDSI(R_M4, R_M10):
    #green and 1.6micron channels
    return (R_M4-R_M10)/(R_M4+R_M10)

def get_snow_RGB(R_M4,R_M10,R_M11):
    #m4 could be m3 but using green over blue will reduce atmospheric scattering
    #the only down side is the slightly increased signal from vegetation...
    #green, 1.6 micron, 2.25 micron
    return np.dstack((R_M4,R_M10,R_M11))

def get_burn_scar_RGB(R_M11, R_M7, R_M5):

    return np.dstack((R_M11, R_M7, R_M5))

if __name__=='__main__':


    import warnings
    import cartopy.crs as ccrs
    from datetime import datetime
    import h5py
    import time
    import numpy as np
    from netCDF4 import Dataset
    import matplotlib.pyplot as plt
    import os

    # VIIRS JPSS-1 naming convention
    # 'VJ102MOD.A2021025.1742.021.2021072143738.nc'
    # 'VJ103MOD.A2021025.1742.021.2021072125458.nc'
    # <product short name>.<aquisition date YYYYDDD>.<UTC HHMM>.<collection version>.<production date YYYYDDDHHMMSS>.nc

    # VJ102 => L1B calibrated radiances, 16 M bands, quality flag, 6 minute granules, 3232x3200 pixels
    # VJ103 => L1B terrain-corrected geolocation, lat/lon, SZA/VZA/VAA/SAA, land water mask,
    #          quality flag,



    ref_filepath_home = "R:/VIIRS_data/VJ102_ref/"
    geo_filepath_home = "R:/VIIRS_data/VJ103_geolocation/"

    ref_filepaths = np.array([ref_filepath_home + x for x in os.listdir(ref_filepath_home)])
    geo_filepaths = np.array([geo_filepath_home + x for x in os.listdir(geo_filepath_home)])

    # sort files by time stamp; after sort check time stamps match up between files
    # this will will allow looping between two file lists to access coincident geo and ref
    # data wihtout the overhead of check the timestamp match
    time_stamps_sorted_idx = np.argsort([x[-33:-21] for x in ref_filepaths])
    time_stamps = np.sort([x[-33:-21] for x in ref_filepaths])
    ref_filepaths = ref_filepaths[time_stamps_sorted_idx]
    geo_filepaths = geo_filepaths[time_stamps_sorted_idx]
    # print([x[-33:-21] for x in ref_filepaths] == [x[-33:-21] for x in geo_filepaths])


    # file_num = 100
    warnings.filterwarnings("ignore")

    with h5py.File('VIIRS_Composite_database.hdf5','w') as hf_database:
        for i, (geo_file, ref_file) in enumerate(zip(geo_filepaths, ref_filepaths)):
            start_time = time.time()
            # which_bands = [7,11]
            which_bands = [3,4,5,7,10,11]
            #             [0,1,2,3,4 ,5]
            M_bands_norm, lat, lon = get_BRF_lat_lon(geo_file, ref_file, which_bands)

            R_M3, R_M4, R_M5, R_M7, R_M10, R_M11 = \
                                      M_bands_norm[:,:,0], M_bands_norm[:,:,1],\
                                      M_bands_norm[:,:,2], M_bands_norm[:,:,3],\
                                      M_bands_norm[:,:,4], M_bands_norm[:,:,5]

            BRF_RGB       = get_BRF_RGB(R_M5,R_M4,R_M3)
            NBR           = get_normalized_burn_ratio(R_M7, R_M11)
            burn_scar_RGB = get_burn_scar_RGB(R_M11, R_M7, R_M5)
            snow_RGB      = get_snow_RGB(R_M4,R_M10,R_M11)
            NDSI          = get_NDSI(R_M4, R_M10)

            BRF_RGB[np.isnan(BRF_RGB)]             = -999
            NBR[np.isnan(NBR)]                     = -999
            burn_scar_RGB[np.isnan(burn_scar_RGB)] = -999
            snow_RGB[np.isnan(snow_RGB)]           = -999
            NDSI[np.isnan(NDSI)]                   = -999

            # group.create_dataset(observables[i], data=data[:,:,i], compression='gzip')
            # group = hf_observables.create_group(time_stamp)

            #write data to file
            time_stamp_current = geo_file[-33:-21]
            year     = time_stamp_current[:4]
            DOY      = '{}'.format(time_stamp_current[4:7])
            UTC_hr   = time_stamp_current[8:][:2]
            UTC_min  = time_stamp_current[8:][2:]
            date     = datetime.strptime(year + "-" + DOY, "%Y-%j").strftime("_%m.%d.%Y")

            group_timestamp = hf_database.create_group(time_stamp_current+date)
            group_timestamp.create_dataset('BRF_RGB'      , data=BRF_RGB)
            group_timestamp.create_dataset('NBR'          , data=NBR)
            group_timestamp.create_dataset('burn_scar_RGB', data=burn_scar_RGB)
            group_timestamp.create_dataset('snow_RGB'     , data=snow_RGB)
            group_timestamp.create_dataset('NDSI'         , data=NDSI)
            group_timestamp.create_dataset('lat'         , data=lat)
            group_timestamp.create_dataset('lon'         , data=lon)

            #print some diagnostics

            run_time = time.time() - start_time
            print('{:02d} VIIRS NOAA-20, {} ({}), run time: {:02.2f}'.format(i, date, time_stamp_current, run_time))
