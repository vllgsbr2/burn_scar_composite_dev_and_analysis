'''
Author: Javier Alfredo Villegas Bravo
Affiliation: NOAA/UMD-CISESS
Date modified: 04/27/2022
'''

if __name__=='__main__':

    import warnings
    from datetime import datetime
    import os
    import time
    import sys
    from netCDF4 import Dataset
    import h5py
    import numpy as np
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    #python modules I made
    from burn_scar_composites import get_burn_scar_RGB,\
                                     get_normalized_burn_ratio,\
                                     get_BRF_lat_lon,\
                                     get_BRF_RGB,\
                                     flip_arr
    from read_VIIRS_raw_nc_files import get_VJ103_geo,\
                                       get_VJ102_ref,\
                                       get_CLDMSK

    '''
    Background info on files and bands
    source: https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/s-npp-nasa-viirs-overview/

    VIIRS JPSS-1 naming convention
    'VJ102MOD.A2021025.1742.021.2021072143738.nc'
    'VJ103MOD.A2021025.1742.021.2021072125458.nc'
    <product short name>.<aquisition date YYYYDDD>.<UTC HHMM>.<collection version>.<production date YYYYDDDHHMMSS>.nc

    VJ102 => L1B calibrated radiances, 16 M bands, quality flag, 6 minute granules, 3232x3200 pixels
    VJ103 => L1B terrain-corrected geolocation, lat/lon, SZA/VZA/VAA/SAA, land water mask,
             quality flag,

    M1	0.402 - 0.422	Visible/Reflective
    M2	0.436 - 0.454	Visible/Reflective
    M3	0.478 - 0.488	Visible/Reflective
    M4	0.545 - 0.565	Visible/Reflective
    M5	0.662 - 0.682	Near Infrared
    M6	0.739 - 0.754	Near Infrared
    M7	0.846 - 0.885	Shortwave Infrared
    M8	1.23  - 1.25	Shortwave Infrared
    M9	1.371 - 1.386	Shortwave Infrared
    M10	1.58  - 1.64	Shortwave Infrared
    M11	2.23  - 2.28	Medium-wave Infrared
    M12	3.61  - 3.79	Medium-wave Infrared
    M13	3.97  - 4.13	Longwave Infrared
    M14	8.4   - 8.7	    Longwave Infrared
    M15	10.26 - 11.26	Longwave Infrared
    M16	11.54 - 12.49	Day/Night
    '''

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

    ref_filepaths    = ref_filepaths[idx_last_not_match+1:]
    geo_filepaths    = geo_filepaths[idx_last_not_match+1:]
    ref_file_timestamps    = ref_file_timestamps[idx_last_not_match+1:]
    geo_file_timestamps    = geo_file_timestamps[idx_last_not_match+1:]
    # cldmsk_filepaths =

    warnings.filterwarnings("ignore")
    start, end = 604, -1

    if True:
        with h5py.File(home+'databases/VIIRS_burn_Scar_database_regridded.h5','r+') as hf_database:
            for i, (geo_file, ref_file, cldmsk_file) in enumerate(zip(geo_filepaths[start:end],\
                        ref_filepaths[start:end], cldmsk_filepaths[start:end])):

                start_time = time.time()
                which_bands = [3,4,5,7,11] # [blue, green, red, veggie, burn]
                #             [0,1,2,3,4]
                M_bands_BRF, lat, lon = get_BRF_lat_lon(geo_file, ref_file, which_bands)

                R_M3, R_M4, R_M5, R_M7, R_M11 = \
                                          M_bands_BRF[:,:,0], M_bands_BRF[:,:,1],\
                                          M_bands_BRF[:,:,2], M_bands_BRF[:,:,3],\
                                          M_bands_BRF[:,:,4]

                BRF_RGB                 = get_BRF_RGB(R_M5,R_M4,R_M3)
                NBR                     = get_normalized_burn_ratio(R_M7, R_M11)
                burn_scar_RGB           = get_burn_scar_RGB(R_M11, R_M7, R_M5)
                cldmsk, land_water_mask = get_CLDMSK(cldmsk_file)


                BRF_RGB[np.isnan(BRF_RGB)]                 = -999
                NBR[np.isnan(NBR)]                         = -999
                burn_scar_RGB[np.isnan(burn_scar_RGB)]     = -999
                cldmsk[np.isnan(cldmsk)]                   = -999
                land_water_mask[np.isnan(land_water_mask)] = -999

                BRF_RGB         = flip_arr(BRF_RGB)
                NBR             = flip_arr(NBR)
                burn_scar_RGB   = flip_arr(burn_scar_RGB)
                cldmsk          = flip_arr(cldmsk)
                land_water_mask = flip_arr(land_water_mask)
                lat             = flip_arr(lat)
                lon             = flip_arr(lon)

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
                group_timestamp.create_dataset('BRF_RGB'        , data=BRF_RGB        , compression='gzip')
                group_timestamp.create_dataset('NBR'            , data=NBR            , compression='gzip')
                group_timestamp.create_dataset('burn_scar_RGB'  , data=burn_scar_RGB  , compression='gzip')
                group_timestamp.create_dataset('cldmsk'         , data=cldmsk         , compression='gzip')
                group_timestamp.create_dataset('land_water_mask', data=land_water_mask, compression='gzip')
                group_timestamp.create_dataset('lat'            , data=lat            , compression='gzip')
                group_timestamp.create_dataset('lon'            , data=lon            , compression='gzip')

                #print some diagnostics

                run_time = time.time() - start_time
                print('{:02d} VIIRS NOAA-20, {} ({}), run time: {:02.2f}'.format(i, date, time_stamp_current, run_time))


    # with h5py.File(home+'databases/VIIRS_burn_Scar_database.h5','r') as hf_database:
    #     k = list(hf_database.keys())
    #     k1 = '{}/{}'.format(k[0], 'BRF_RGB')
    #     plt.imshow(flip_arr(hf_database[k1][:]))
    #     plt.show()
