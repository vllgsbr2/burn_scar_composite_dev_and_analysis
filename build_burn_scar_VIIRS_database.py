'''
Author       : Javier Alfredo Villegas Bravo
Affiliation  : University of Maryland College Park
               Cooperative Institute for Satellite Earth Systems Studies
Date modified: 02/24/2025
'''
def build_burn_scar_database(database_file = None, current_date = None, latest_date = None,\
                             database_type = None, VJX09_dir    = None, VJX03_dir   = None,\
                             start_date    = None, end_data     = None, save_path   = None):
    '''
    VJX09_dir, VJX03_dir: (str) directory paths to sfc ref and geolocation files
    start_date, end_data: (str) MM/DD/YYYY
    save_path           : (str) path to save h5 file into, i.e. user/path_to_dir
    '''

    print('Setting up environment')
    import warnings
    from datetime import datetime
    import os
    import time
    import sys
    #from netCDF4 import Dataset
    import h5py
    import numpy as np
    #import cartopy.crs as ccrs
    #import matplotlib.pyplot as plt

    #python modules I made
    from burn_scar_composites import get_burn_scar_RGB,\
                                     get_normalized_burn_ratio,\
                                     get_BRF_lat_lon,\
                                     get_BRF_RGB,\
                                     flip_arr
    from read_VIIRS_raw_nc_files import get_VJ103_geo,\
                                        get_VJ109_ref,\
                                        get_CLDMSK

    from regrid import regrid_latlon_source2target_new

    '''
    Background info on files and bands
    Sources:
    https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/s-npp-nasa-viirs-overview/
    https://www.star.nesdis.noaa.gov/jpss/VIIRS.php (Has band centers but not range, above link just range)

    VIIRS JPSS-1 naming convention
    'VJ102MOD.A2021025.1742.021.2021072143738.nc'
    'VJ103MOD.A2021025.1742.021.2021072125458.nc'
    <product short name>.<aquisition date YYYYDDD>.<UTC HHMM>.<collection version>.<production date YYYYDDDHHMMSS>.nc

    VJ102 => L1B calibrated radiances, 16 M bands, quality flag, 6 minute granules, 3232x3200 pixels
    VJ103 => L1B terrain-corrected geolocation, lat/lon, SZA/VZA/VAA/SAA, land water mask,
             quality flag,
    VJ109 => Surface Reflectance

    375 meter resolution
    I1 	0.6  - 0.68 µm	Visible/Reflective   center @ 640nm
    I2 	0.85 - 0.88 µm	Near Infrared        center @ 865nm
    I3 	1.58 - 1.64 µm	Shortwave Infrared   center @ 1610nm
    I4 	3.55 - 3.93 µm	Medium-wave Infrared center @ 3740nm
    I5 	10.5 - 12.4 µm	Longwave Infrared    center @ 11450nm

    750 meter resolution (µm unit for left wavelength column)
    M1	0.402 - 0.422 	Visible/Reflective
    M2	0.436 - 0.454	Visible/Reflective
    M3	0.478 - 0.488	Visible/Reflective   center @ 488nm
    M4	0.545 - 0.565	Visible/Reflective   center @ 555nm
    M5	0.662 - 0.682	Near Infrared        center @ 672nm
    M6	0.739 - 0.754	Near Infrared
    M7	0.846 - 0.885	Shortwave Infrared   center @ 865nm
    M8	1.23  - 1.25	Shortwave Infrared
    M9	1.371 - 1.386	Shortwave Infrared
    M10	1.58  - 1.64	Shortwave Infrared
    M11	2.23  - 2.28	Medium-wave Infrared center @ 2250nm
    M12	3.61  - 3.79	Medium-wave Infrared
    M13	3.97  - 4.13	Longwave Infrared
    M14	8.4   - 8.7     Longwave Infrared
    M15	10.26 - 11.26	Longwave Infrared
    M16	11.54 - 12.49	Day/Night
    '''

    if database_type=='OPERATIONAL':
        home                 = '/scratch/zt1/project/vllgsbr2-prj/'
        home_data            = home      + 'raw_data_burnscar/data/'
        operational_dir      = 'operational_data_feed'
        ref_filepath_home    = home_data + f'noaa_20_viirs/{operational_dir}/VJ109/'
        geo_filepath_home    = home_data + f'noaa_20_viirs/{operational_dir}/VJ103/'
        fire_filepath_home   = home_data + f'noaa_20_viirs/{operational_dir}/VJ114/'
        lai_filepath_home    = home_data + f'noaa_20_viirs/{operational_dir}/VJ115/'
        commongrid_file      = home_data + 'grids/Grids_West_CONUS_new.h5'
        database_file        = home_data + f'databases/operational_databse.h5'

    elif VJX09_dir==None: # if dirs and dates aren't provided, use these paths
        whole_year = False
        if whole_year: # for entire year of data

            analysis_year        = '2024'
            home                 = '/scratch/zt1/project/vllgsbr2-prj/'
            home_data            = home      + 'raw_data_burnscar/data/'
            ref_filepath_home    = home_data + f'noaa_20_viirs/{analysis_year}_fire_season/VJ109/'
            geo_filepath_home    = home_data + f'noaa_20_viirs/{analysis_year}_fire_season/VJ103/'
            fire_filepath_home   = home_data + f'noaa_20_viirs/{analysis_year}_fire_season/VJ114/'
            lai_filepath_home    = home_data + f'noaa_20_viirs/{analysis_year}_fire_season/VJ115/'
            commongrid_file      = home_data + 'grids/Grids_West_CONUS_new.h5'
            database_file        = home_data + f'databases/viirs_burnscar_database_{analysis_year}.h5'

        else:# else is for a single fire or event to be analyzed

            home                 = '/scratch/zt1/project/vllgsbr2-prj/'
            home_data            = home      + 'raw_data_burnscar/data/'
            ref_filepath_home    = home_data + f'noaa_20_viirs/LA_fires_Jan_2025/VJ109/'
            geo_filepath_home    = home_data + f'noaa_20_viirs/LA_fires_Jan_2025/VJ103/'
            fire_filepath_home   = home_data + f'noaa_20_viirs/LA_fires_Jan_2025/VJ114/'
            lai_filepath_home    = home_data + f'noaa_20_viirs/LA_fires_Jan_2025/VJ115/'
            commongrid_file      = home_data + 'grids/Grids_West_CONUS_new.h5'
            database_file        = home_data + f'databases/viirs_burnscar_database_LA_fires_Jan_2025.h5'
    else: # if dirs and dates are provided, use these paths instead
        #home                 = '/scratch/zt1/project/vllgsbr2-prj/'
        #home_data            = home      + 'raw_data_burnscar/data/'
        ref_filepath_home    = VJX09_dir #home_data + f'noaa_20_viirs/park_fire_07_24_2024/VJ109/'
        geo_filepath_home    = VJX03_dir #home_data + f'noaa_20_viirs/park_fire_07_24_2024/VJ103/'
        commongrid_file      = home_data + 'grids/Grids_West_CONUS_new.h5'
        database_file        = save_path + f'/viirs_burnscar_database_{start_date}_{end_date}.h5' #home_data + f'databases/viirs_burnscar_database_park_fire_07_24_2024.h5'

    ref_filepaths  = np.sort([ref_filepath_home  + x for x in os.listdir(ref_filepath_home)])
    geo_filepaths  = np.sort([geo_filepath_home  + x for x in os.listdir(geo_filepath_home)])
    fire_filepaths = np.sort([fire_filepath_home + x for x in os.listdir(fire_filepath_home)])
    lai_filepaths  = np.sort([lai_filepath_home  + x for x in os.listdir(lai_filepath_home)])
    # make the lai filepaths the same len as others. Product is every 7 days
    # timestamp of other files should be in the past no more than 7 days prior
    lai_filepaths_resized = []
    for f in lai_filepaths:
        for i in range(7):
            lai_filepaths_resized.append(f)
    #cldmsk_filepaths = np.sort([cldmsk_filepath_home + x for x in os.listdir(cldmsk_filepath_home)])
    if database_type != "OPERATIONAL":
        ref_file_timestamps    = [x[-34:-22] for x in ref_filepaths] # for TOA ref -33:-21
        geo_file_timestamps    = [x[-33:-21] for x in geo_filepaths]
        fire_file_timestamps   = [x[-33:-21] for x in fire_filepaths]
        lai_file_timestamps    = [x[-35:-28] for x in lai_filepaths]
    else:
        ref_file_timestamps    = [x[-20:-8] for x in ref_filepaths] # for TOA ref -33:-21
        geo_file_timestamps    = [x[-19:-7] for x in geo_filepaths]
        fire_file_timestamps   = [x[-33:-21] for x in fire_filepaths]
        lai_file_timestamps    = [x[-35:-28] for x in lai_filepaths]


    #cldmsk_file_timestamps = [x[-33:-21] for x in cldmsk_filepaths]

    #checks that files are exact sets of each other but I'm depricating this now
    #boolean_intersection_ref_geo    = np.in1d(ref_file_timestamps, geo_file_timestamps)
    #idx_last_not_match              = np.where(boolean_intersection_ref_geo==False)[0][-1]
    #ref_filepaths       = ref_filepaths[idx_last_not_match+1:]
    #geo_filepaths       = geo_filepaths[idx_last_not_match+1:]
    #ref_file_timestamps = ref_file_timestamps[idx_last_not_match+1:]
    #geo_file_timestamps = geo_file_timestamps[idx_last_not_match+1:]

    with h5py.File(commongrid_file, 'r') as hf_west_conus_grid:
        common_grid_lat = hf_west_conus_grid['Geolocation/Latitude'][:]
        common_grid_lon = hf_west_conus_grid['Geolocation/Longitude'][:]

    common_grid_lon = np.flip(common_grid_lon, axis=1)*-1

    warnings.filterwarnings("ignore")
    start, end = 0,-1

    print('Building Database')
    with h5py.File(database_file,'a') as hf_database:
        for i, (geo_file, ref_file) in enumerate(zip(geo_filepaths[start:end],\
                ref_filepaths[start:end])):
            #, cldmsk_file cldmsk_filepaths[start:end])):

            # define time stamp to write to file
            if database_type != "OPERATIONAL":
                time_stamp_current = geo_file[-33:-21]
            else:
                time_stamp_current = geo_file[-19:-7]
            year               = time_stamp_current[:4]
            DOY                = '{}'.format(time_stamp_current[4:7])
            UTC_hr             = time_stamp_current[8:][:2]
            UTC_min            = time_stamp_current[8:][2:]
            date               = datetime.strptime(year + "-" + DOY, "%Y-%j")\
                                 .strftime("_%m.%d.%Y")

            # fill_val_idx is last to get populated. If it exists assume dataset is complete
            # and continue to next time stamp, otherwise work on current time step
            if (f"{time_stamp_current+date}/fill_val_idx" in hf_database) == True:
                print(f"{time_stamp_current+date} already exists, continuing to next file")
                continue

            try:
                # check if timestamps from geo and ref files match
                if geo_file_timestamps[i] != ref_file_timestamps[i]:
                    print('Error: geo and ref file time stamps do not match. Must sort')
                    sys.exit()

                # grab what we need from the raw files
                start_time = time.time()
                which_bands = [5,7,11] # [red, veggie, burn]
                #             [0,1,2]
                M_bands_BRF, lat, lon, land_water_mask = get_BRF_lat_lon(geo_file, ref_file, which_bands)
                # make sure this retains meaning to VJ109 file
                R_M5, R_M7, R_M11 = M_bands_BRF[0,:,:],\
                                    M_bands_BRF[1,:,:],\
                                    M_bands_BRF[2,:,:]

                burn_scar_RGB   = get_burn_scar_RGB(R_M11, R_M7, R_M5)

                #****************************************************************
                # modify to work with
                cldmsk         , cldmsk_quality,\
                snow_ice_mask  ,\
                cld_shadow_mask = get_VJ109_ref(ref_file, cld_shadow_snowice=True)

                #cldmsk_return   = get_CLDMSK(cldmsk_file)
                #cldmsk          = cldmsk_return[0]
                #snow_ice_flag   = cldmsk_return[4] # cloud mask includes this flag

                geofile_dict    = get_VJ103_geo(geo_file, include_latlon=False, include_SZA=True,\
                                                include_VZA=True, include_SAAVAA=True, include_lwm=True)

                # land_water_mask	Land/Water mask	ubyte(number_of_lines, number_of_pixels)
                # long_name	string	"Land/Water mask at pixel locations"
                # _FillValue	ubyte		255
                # flag_values	ubyte[8]	0, 1, 2, 3, 4, 5, 6, 7
                # flag_meanings	string
                # "Shallow_Ocean Land Coastline Shallow_Inland Ephemeral Deep_Inland Continental Deep_Ocean"
                land_water_mask = geofile_dict['land_water_mask']
                #****************************************************************

                VZA             = geofile_dict['VZA']
                SZA             = geofile_dict['SZA']
                RAA             = geofile_dict['SAA'] - geofile_dict['VAA']

                # force nan values to -999 if any
                burn_scar_RGB[np.isnan(burn_scar_RGB)]     = -999
                cldmsk[np.isnan(cldmsk)]                   = -999
                cld_shadow_mask[np.isnan(cld_shadow_mask)] = -999
                snow_ice_mask[np.isnan(snow_ice_mask)]     = -999
                land_water_mask[np.isnan(land_water_mask)] = -999
                VZA[np.isnan(VZA)]                         = -999
                SZA[np.isnan(SZA)]                         = -999
                RAA[np.isnan(RAA)]                         = -999

                # flip arrays so the top is North and left is West
                burn_scar_RGB   = flip_arr(burn_scar_RGB)

                cldmsk          = flip_arr(cldmsk)
                cld_shadow_mask = flip_arr(cld_shadow_mask)
                snow_ice_mask   = flip_arr(snow_ice_mask)
                land_water_mask = flip_arr(land_water_mask)
                lat             = flip_arr(lat)
                lon             = flip_arr(lon)
                VZA             = flip_arr(VZA)
                SZA             = flip_arr(SZA)
                RAA             = flip_arr(RAA)

                #******************************************************************
                # get VJ114 therm anomaly and VJ115 leaf area index

                #fire_mask       = get_VJ114_thermal_anomalies(VJ114_file)
                #leaf_area_index = get_VJ115_leaf_area_index(VJ115_file)
                #******************************************************************

                # regrid the data to fixed grid defined by common grid file
                #group_timestamp_check = time_stamp_current+date
                #if group_timestamp_check in hf_database:

                target_lat = common_grid_lat
                target_lon = common_grid_lon
                source_lat = lat
                source_lon = lon

                # most deggraded pixel size according to
                # https://agupubs.onlinelibrary.wiley.com/doi/10.1002/jgrd.50873
                # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JD021170
                # 1.5km at swath edge for 750m res only x2 degredation vs MODIS is like x5
                # due to pixel aggregation technique described in
                # [Imagery Algorithm Theoretical Basis Document (ATBD), 2011].
                # so lets only look in a 1500m radius to improve speed, vs 6000m radius for MODIS
                max_radius = 1500.

                # returns idx values so we can use them to regrid each dataset
                # this allows regridding to only be done once saving runtime
                # This is possible cause all data from same timesatmp is already
                # on the same grid. i.e. lat = lat[regrid_row_idx, regrid_col_idx]
                # would put the data that intersects the common grid on the common grid
                regrid_row_idx,\
                regrid_col_idx,\
                fill_val_idx   = regrid_latlon_source2target_new(source_lat,\
                                                                 source_lon,\
                                                                 target_lat,\
                                                                 target_lon,\
                                                                 max_radius)
                #else:
                #    print(group_timestamp_check, 'in dataset, not reprocessed')
                #    continue


                group_timestamp = hf_database.create_group(time_stamp_current+date)
                group_timestamp.create_dataset('burn_scar_RGB'  , data=burn_scar_RGB  , compression='gzip')
                group_timestamp.create_dataset('cldmsk'         , data=cldmsk         , compression='gzip')
                group_timestamp.create_dataset('land_water_mask', data=land_water_mask, compression='gzip')
                group_timestamp.create_dataset('snow_ice_mask'  , data=snow_ice_mask  , compression='gzip')
                group_timestamp.create_dataset('cld_shadow_mask', data=cld_shadow_mask, compression='gzip')
                #group_timestamp.create_dataset('lat'            , data=lat            , compression='gzip')
                #group_timestamp.create_dataset('lon'            , data=lon            , compression='gzip')
                group_timestamp.create_dataset('VZA'            , data=VZA            , compression='gzip')
                group_timestamp.create_dataset('SZA'            , data=SZA            , compression='gzip')
                group_timestamp.create_dataset('RAA'            , data=RAA            , compression='gzip')
                group_timestamp.create_dataset('regrid_row_idx' , data=regrid_row_idx , compression='gzip')
                group_timestamp.create_dataset('regrid_col_idx' , data=regrid_col_idx , compression='gzip')
                group_timestamp.create_dataset('fill_val_idx'   , data=fill_val_idx   , compression='gzip')

                '''
                # Just keeping this here incase want to change data in existing dataset
                try:
                    hf_database[time_stamp_current+date+'/burn_scar_RGB'][:] = burn_scar_RGB
                except:
                    try:
                        group_timestamp.create_dataset('burn_scar_RGB', data=burn_scar_RGB, compression='gzip')
                    except:
                        print("broken")

                '''
                # print some diagnostics
                run_time = time.time() - start_time
                print('{:02d} VIIRS NOAA-20, {} ({}), run time: {:02.2f}'\
                        .format(i, date[1:], time_stamp_current, run_time))
            except Exception as e:
                run_time = time.time() - start_time
                print(f'Failed **** {i} {ref_file} {e}')


def plot_hdf_for_inspection(database_file, savefig_path):
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    with h5py.File(database_file, 'r') as hf_database:
        for group_name in hf_database.keys():
            group = hf_database[group_name]

            regrid_col_idx = group['regrid_col_idx'][:]
            regrid_row_idx = group['regrid_row_idx'][:]
            regrid_col_idx[regrid_col_idx==-999] = 0
            regrid_row_idx[regrid_row_idx==-999] = 0

            fill_val_idx = group['fill_val_idx'  ][:]

            for dataset_name in group.keys():

                if not ('regrid' in dataset_name or 'fill' in dataset_name):
                    print(dataset_name)

                    dataset = group[dataset_name][:]
                    dataset = dataset.astype(dtype=np.float64)
                    dataset[dataset==-999] = np.nan

                    dataset_list = [dataset]

                    dataset_regrid = dataset[regrid_row_idx, regrid_col_idx]
                    dataset_regrid[fill_val_idx[0], fill_val_idx[1]] = np.nan

                    dataset_list.append(dataset_regrid)

                    # Plot the dataset

                    #plt.figure()

                    if 'burn_scar_RGB' == dataset_name:
                        f, ax = plt.subplots(ncols=2)
                        im = ax[0].imshow(dataset_list[0])
                        ax[1].imshow(dataset_list[1])
                    elif 'cldmsk' == dataset_name:
                        f, ax = plt.subplots(ncols=2)
                        im = ax[0].imshow(dataset_list[0], cmap='bone_r', vmin=0, vmax=3)
                        ax[1].imshow(dataset_list[1], cmap='bone_r', vmin=0, vmax=3)
                    elif 'RAA' in dataset_name or \
                         'SZA' in dataset_name or \
                         'VZA' in dataset_name:
                        f, ax = plt.subplots(ncols=2)
                        im = ax[0].imshow(dataset_list[0], cmap='jet',\
                                     vmin=dataset_list[0].min() ,\
                                     vmax=dataset_list[0].max() )
                        ax[1].imshow(dataset_list[1], cmap='jet',\
                                     vmin=dataset_list[0].min() ,\
                                     vmax=dataset_list[0].max() )
                    else:
                        f, ax = plt.subplots(ncols=2)
                        im = ax[0].imshow(dataset_list[0], cmap='bone',\
                                     vmin=dataset_list[0].min()  ,\
                                     vmax=dataset_list[0].max()  )
                        ax[1].imshow(dataset_list[1], cmap='bone',\
                                     vmin=dataset_list[0].min()  ,\
                                     vmax=dataset_list[0].max()  )

                    ax[0].set_title(f'{group_name}\n{dataset_name}')
                    ax[1].set_title('regrid')

                    f.colorbar(im, ax=ax[0])

                    plt.tight_layout()

                    # Save the plot to a PDF file at 300dpi resolution
                    pdf_file_path = f'{savefig_path}/{group_name}_{dataset_name}.pdf'
                    plt.savefig(pdf_file_path, dpi=300)
                    plt.close()

        '''
        for group_name in file.keys():
        group = file[group_name]
        num_datasets = len(group.keys())

        # Create subplots based on the number of datasets in the group
        fig, axs = plt.subplots(1, num_datasets, figsize=(num_datasets*5, 5))

        for i, dataset_name in enumerate(group.keys()):
            dataset = group[dataset_name]

            # Plot the dataset in the corresponding subplot
            axs[i].imshow(dataset, cmap='viridis')
            axs[i].set_title(dataset_name)

        # Save the subplots to a PDF file at 300dpi resolution
        pdf_file_path = f'{group_name}_plots.pdf'
        plt.savefig(pdf_file_path, dpi=300)
        plt.close()
        '''

if __name__=='__main__':
    import sys
    '''
    analysis_year        = '2020'
    home                 = '/scratch/zt1/project/vllgsbr2-prj/'
    home_data            = home      + 'raw_data_burnscar/data/'
    ref_filepath_home    = home_data + f'noaa_20_viirs/{analysis_year}_fire_season/VJ109/'
    geo_filepath_home    = home_data + f'noaa_20_viirs/{analysis_year}_fire_season/VJ103/'
    #cldmsk_filepath_home = home_data + 'cldmsk/'
    commongrid_file      = home_data + 'grids/Grids_West_CONUS_new.h5'
    database_file        = home_data + f'databases/viirs_burnscar_database_{analysis_year}.h5'

    savefig_path         = home_data + 'database_inspect_figures_no_ref_QC'

    '''
    print(sys.argv[1])
    if sys.argv[1] != 'OPERATIONAL':
        build_burn_scar_database()
    elif sys.argv[1] == 'OPERATIONAL':
        # write database if not yet created
        # if day already exist, skip
        build_burn_scar_database(database_file = None, current_date = None, latest_date = None,\
                                 database_type = 'OPERATIONAL', VJX09_dir    = None, VJX03_dir   = None,\
                                 start_date    = None, end_data     = None, save_path   = None)

    #plot_hdf_for_inspection(database_file, savefig_path)

