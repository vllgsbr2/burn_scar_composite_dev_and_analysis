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
    from netCDF4 import Dataset
    import h5py
    import numpy as np
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    #python modules I made
    from burn_scar_composites import get_burn_scar_RGB,\
                                     get_normalized_burn_ratio,\
                                     get_BRF_lat_lon
    from read_VIIRS_raw_nc_file import get_VJ103_geo,\
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

    home              = 'R:/satellite_data/viirs_data/noaa20/'
    ref_filepath_home = home + 'reflectance/'
    geo_filepath_home = home + 'geolocation/'

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

    with h5py.File(home+'databases/VIIRS_burn_Scar_database.h5','w') as hf_database:
        for i, (geo_file, ref_file) in enumerate(zip(geo_filepaths, ref_filepaths)):
            start_time = time.time()
            which_bands = [3,4,5,7,11] # [blue, green, red, veggie, burn]
            #             [0,1,2,3,4]
            M_bands_norm, lat, lon = get_BRF_lat_lon(geo_file, ref_file, which_bands)

            R_M3, R_M4, R_M5, R_M7, R_M11 = \
                                      M_bands_norm[:,:,0], M_bands_norm[:,:,1],\
                                      M_bands_norm[:,:,2], M_bands_norm[:,:,3],\
                                      M_bands_norm[:,:,4]

            BRF_RGB       = get_BRF_RGB(R_M5,R_M4,R_M3)
            NBR           = get_normalized_burn_ratio(R_M7, R_M11)
            burn_scar_RGB = get_burn_scar_RGB(R_M11, R_M7, R_M5)

            BRF_RGB[np.isnan(BRF_RGB)]             = -999
            NBR[np.isnan(NBR)]                     = -999
            burn_scar_RGB[np.isnan(burn_scar_RGB)] = -999

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

            if i==4:
                break
