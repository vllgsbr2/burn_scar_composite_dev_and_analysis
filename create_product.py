def get_daily_composites(database_file       = None, composite_file = None, current_date        = None,\
                         latest_date         = None, composite_type = None, viirs_database_file = None,\
                         daily_composite_dir = None):
    '''
    INPUTS:

    viirs_database_file: full path to h5 file containing each regridded granule of viirs data
    daily_composite_dir: directory to place the daily composite file

    RETURNS:

    Saves daily composited cloud cleared sfc refs for what ever data is in the viirs database file
    '''

    print('Setting up environment')

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import sys
    import h5py
    import time
    from burn_scar_composites import flip_arr
    import warnings
    from datetime import datetime, timedelta
    warnings.filterwarnings("ignore")
    #plt.switch_backend('qtAgg')

    # define paths for lat lon common grid file
    # and database file
    if composite_type=="OPERATIONAL":
            home                 = '/scratch/zt1/project/vllgsbr2-prj/'
            home_data            = home      + 'raw_data_burnscar/data/'
            composite_file       = composite_file
            database_file        = database_file

    elif viirs_database_file==None:
        whole_year = False
        if whole_year:

            analysis_year        = '2024'
            home                 = '/scratch/zt1/project/vllgsbr2-prj/'
            home_data            = home      + 'raw_data_burnscar/data/'
            composite_file       = home_data + f'daily_dlcf_rgb_composites/{analysis_year}_daily_DLCF_RGB_composites_no_ref_QC_no_cldshadow_QC_keep_landwatermask_1_3_4_cldmsk_keep_probclear_clear_fix_bowtie_del_age_quality_flag_VZA_RAA.h5'
            database_file        = home_data + f'databases/viirs_burnscar_database_{analysis_year}.h5'

        else:

            home                 = '/scratch/zt1/project/vllgsbr2-prj/'
            home_data            = home      + 'raw_data_burnscar/data/'
            composite_file       = home_data + f'daily_dlcf_rgb_composites/LA_fires_Jan_2025_daily_DLCF_RGB_composites_no_ref_QC_no_cldshadow_QC_keep_landwatermask_1_3_4_cldmsk_keep_probclear_clear_fix_bowtie_del_age_quality_flag.h5'
            database_file        = home_data + f'databases/viirs_burnscar_database_LA_fires_Jan_2025.h5'
    else:
        database_file  = viirs_database_file
        composite_file = f'{daily_composite_dir}/daily_composites_{viirs_database_file[-21:]}'
    '''
    import mpi4py.MPI as MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for r in range(size):
        if rank==r:
    '''

    print('Calculating daily burn scar composites')
    # open databse file and write composite file
    with h5py.File(database_file,'r') as hf_database,\
         h5py.File(composite_file,'w') as hf_daily_composites:

        timestamps    = list(hf_database.keys())
        dataset_names = list(hf_database[timestamps[0]].keys())

        # grab last 9 days of times stamps to use for operational feed from
        # operational database
        if composite_type=="OPERATIONAL":
            new_timestamps = []
            seconds_in_9_days = 9*24*60*60
            current_date = datetime.today()
            for t in timestamps:
                t_ = datetime.strptime(t[-10:], '%m.%d.%Y')
                if (t_ - current_date).total_seconds() < seconds_in_9_days:
                    new_timestamps.append(t)
            timestamps = new_timestamps

        # stack composites here, where latest pixel overwrites all others.
        # The oldest pixel should be no more than n days old (start w/8 days)
        # note a granbule does not eqaul a day, i.e. there is over lap, so use
        # time and day to decide which granule is newer in order to overwrite.
        # empty/ non valid pixels should be auto overwritten, i.e. np.nan or x<=-997
        # work backwords from newest
        # once it works try again but cloud and water clear with -998, -997 repectively

        base_image_shape           = (3604, 2354, 3)
        composite_burn_scar_RGB    = np.empty(base_image_shape)
        composite_burn_scar_RGB[:] = np.nan

        # make composites for VZA, RAA as well
        composite_VZA = np.empty((base_image_shape[0], base_image_shape[1]))
        composite_VZA[:] = np.nan

        composite_RAA = np.empty((base_image_shape[0], base_image_shape[1]))
        composite_RAA[:] = np.nan


        start, end = 0,-1#*n_days
        days_composited = 0

        # pixel age quality flag array
        # 2D array keep track age of each pixel in days
        # when pixel exceeds 8 days, it is set to nan
        # until a new 1 day old pixel becomes available
        pixel_age_quality_flag = np.ones((base_image_shape[0], base_image_shape[1]))

        for i, timestamp in enumerate(timestamps[start:end]):
            start_time = time.time()

            i_modified = i+start
            # print(i_modified)
            # populate dataset_dict with all useful data to make composites of
            dataset_dict = {}
            for dataset_name in dataset_names:
                dataset_dict[dataset_name] = hf_database[timestamp+'/'+dataset_name][:]

            # grab data from dataset_dict and slice with regrid to get domain
            VZA          = dataset_dict['VZA']
            RAA          = dataset_dict['RAA']
            r_row_idx    = dataset_dict['regrid_row_idx']
            r_col_idx    = dataset_dict['regrid_col_idx']
            fill_val_idx = dataset_dict['fill_val_idx'  ]

            cldmsk          = flip_arr(dataset_dict['cldmsk'], flip_y_axis=False)[r_row_idx, r_col_idx]
            burn_scar_RGB   = dataset_dict['burn_scar_RGB'  ][r_row_idx, r_col_idx]
            land_water_mask = dataset_dict['land_water_mask'][r_row_idx, r_col_idx]
            snow_ice_mask   = dataset_dict['snow_ice_mask'  ][r_row_idx, r_col_idx]
            cld_shadow_mask = dataset_dict['cld_shadow_mask'][r_row_idx, r_col_idx]
            VZA             = dataset_dict['VZA'            ][r_row_idx, r_col_idx]
            RAA             = dataset_dict['RAA'            ][r_row_idx, r_col_idx]

            find_bowtie_deletion_idx = np.where(burn_scar_RGB<0)

            cldmsk[find_bowtie_deletion_idx[0],find_bowtie_deletion_idx[1]]        = np.nan
            burn_scar_RGB[find_bowtie_deletion_idx[0],find_bowtie_deletion_idx[1],:] = np.nan

            '''
            # calculate primitive burn scar mask (PBSM)
            R_M7, R_M11 = burn_scar_RGB[:,:,1], burn_scar_RGB[:,:,0]
            burn_scar_composite = get_burn_scar_composite(R_M7, R_M11, geotiff=False)
            '''

            cldmsk[fill_val_idx[0], fill_val_idx[1]]          = np.nan
            burn_scar_RGB[fill_val_idx[0], fill_val_idx[1],:] = np.nan
            VZA[fill_val_idx[0], fill_val_idx[1]]             = np.nan
            RAA[fill_val_idx[0], fill_val_idx[1]]             = np.nan

            cldmsk[cldmsk==-999]               = np.nan
            burn_scar_RGB[burn_scar_RGB==-999] = np.nan
            VZA[VZA==-999]                     = np.nan
            RAA[RAA==-999]                     = np.nan

            '''
            fig, axs = plt.subplots(ncols=2,nrows=2,sharex=True, sharey=True)

            axs[0,0].imshow(burn_scar_RGB)
            axs[0,0].set_title('burn_scar_RGB no filter\n'+timestamp)

            # Plot the second array on the second subplot
            axs[0,1].imshow(cldmsk, cmap='binary')
            axs[0,1].set_title('cldmsk no filter')
            #plt.show()
            '''

            # 0 Shallow_Ocean
            # 1 Land
            # 2 Coastline
            # 3 Shallow_Inland
            # 4 Ephemeral
            # 5 Deep_Inland
            # 6 Continental
            # 7 Deep_Ocean

            # cloud and water clear
            # [0,1,2,3]
            # [cloudy, prob cloud, prob clear, clear]
            burn_scar_RGB[cldmsk <= 1] = np.nan #noise from probably clear, may mask it out
            # the probably cloudy is important to keep other wise lose alot to smoke maybe
            #burn_scar_RGB[land_water_mask != 1] = np.nan #just take out all water types
            # take out 0,2,5,6,7
            # keep     1,3,4
            burn_scar_RGB[land_water_mask == 0] = np.nan
            burn_scar_RGB[land_water_mask >= 5] = np.nan
            burn_scar_RGB[land_water_mask == 2] = np.nan
            # clear snow ice and cloud shadow pixels
            burn_scar_RGB[snow_ice_mask == 1] = np.nan
            #burn_scar_RGB[cld_shadow_mask == 1] = np.nan

            '''
            # Plot the first array on the first subplot
            axs[1,0].imshow(burn_scar_RGB)
            axs[1,0].set_title('burn_scar_RGB\n'+timestamp)

            # Plot the second array on the second subplot
            axs[1,1].imshow(cldmsk, cmap='binary')
            axs[1,1].set_title('cldmsk')

            for a in axs.flat:
                a.axis('off')
            plt.tight_layout()
            plt.show()
            #continue

            import sys
            sys.exit()
            '''

            # combine data by populating nan with cloud free water free data from latest granule
            # where it's empty on composite grid

            # unpopulated_current_idx is where the current granule is nan
            unpopulated_current_idx = np.where(np.isnan(burn_scar_RGB))

            # where the new granule has data, reset the time to 1
            populated_current_idx   = np.where(~np.isnan(burn_scar_RGB))
            pixel_age_quality_flag[populated_current_idx[0], populated_current_idx[1]] = 1

            # temp composite holds the last composite_burn_scar_RGB
            temp_composite          = np.copy(composite_burn_scar_RGB)
            temp_composite_VZA      = np.copy(composite_VZA)
            temp_composite_RAA      = np.copy(composite_RAA)
            # composite_burn_scar_RGB then takes on the current granules burn_scar_RGB
            composite_burn_scar_RGB = np.copy(burn_scar_RGB)
            composite_VZA           = np.copy(VZA)
            composite_RAA           = np.copy(RAA)
            # where ever the current burn_scar_RGB is nan, which is what
            # composite_burn_scar_RGB is pointing to now, put the old comosite values
            # stored in temp_composite, which only has vlaues from the last
            # composite where the current granule has invalid values
            composite_burn_scar_RGB[unpopulated_current_idx[0],\
                                    unpopulated_current_idx[1], :] = \
                                        temp_composite[unpopulated_current_idx[0], \
                                                       unpopulated_current_idx[1],:]
            composite_VZA[unpopulated_current_idx[0],\
                unpopulated_current_idx[1]] = temp_composite_VZA[unpopulated_current_idx[0], \
                                                                 unpopulated_current_idx[1]]
            composite_RAA[unpopulated_current_idx[0],\
                unpopulated_current_idx[1]] = temp_composite_RAA[unpopulated_current_idx[0], \
                                                                 unpopulated_current_idx[1]]
            # now take the composite_burn_scar_RGB and turn into composite pbsm
            # then save that into the file along with the RGB

            print('{:02d}/{} {} composite updated'.format(i_modified+1, len(timestamps), timestamp))

            # save composite when next granule is from next day
            # granules are in chronilogical order
            if i_modified<len(timestamps)-1:
                if int(timestamps[i_modified+1][4:7]) > int(timestamps[i_modified][4:7]) or\
                   int(timestamps[i_modified+1][:4] ) > int(timestamps[i_modified][:4]):
                    # print(timestamps[i+1][4:7], int(timestamps[i][4:7]))

                    '''
                    # add day to composite counter
                    days_composited += 1
                    '''

                    # where there is no replacement for previous days pixel, add 1 day
                    #pixel_age_quality_flag[unpopulated_current_idx[0],unpopulated_current_idx[1]] += 1
                    composite_burn_scar_RGB[pixel_age_quality_flag >=9] = np.nan
                    composite_VZA[pixel_age_quality_flag >=9]           = np.nan
                    composite_RAA[pixel_age_quality_flag >=9]           = np.nan
                    print('avg age of pixels in days: ',np.mean(pixel_age_quality_flag))
                    #print(composite_burn_scar_RGB[2110, 219], composite_burn_scar_RGB[2110, 218])
                    #print(len(np.where(composite_burn_scar_RGB<-999)[0])/(3604*2503))
                    # make group with day of compsite
                    current_group_day = hf_daily_composites.create_group(f"{timestamps[i_modified][-10:]}")
                    current_group_day.create_dataset("composite_burn_scar_RGB",\
                                                     data=composite_burn_scar_RGB,\
                                                     compression='gzip')
                    current_group_day.create_dataset("composite_VZA",\
                                                     data=composite_VZA,\
                                                     compression='gzip')
                    current_group_day.create_dataset("composite_RAA",\
                                                     data=composite_RAA,\
                                                     compression='gzip')
                    # add 1 day to pixels not populated
                    pixel_age_quality_flag[unpopulated_current_idx[0],unpopulated_current_idx[1]] += 1

                    run_time = time.time() - start_time
                    print('saved daily composite {} ***** run time: {:02.2f}'\
                        .format(timestamps[i_modified][-10:], run_time))

                    '''
                    plt.figure(figsize=(10,10))
                    plt.imshow(composite_burn_scar_RGB)
                    plt.title(timestamps[i_modified][-10:])
                    plt.show()
                    '''
                    # print('plotting {:03d}/10 {}'.format(i_modified, timestamp))

                    '''
                    # delete everything after 8 days are composited
                    if days_composited==8:
                        composite_burn_scar_RGB[:] = np.nan
                        days_composited = 0
                    '''

            elif i_modified==len(timestamps)-1:
                # print(timestamps[i+1][4:7], int(timestamps[i][4:7]))
                #make group with day of compsite
                composite_burn_scar_RGB = composite_burn_scar_RGB
                composite_VZA           = composite_VZA
                composite_RAA           = composite_RAA
                current_group_day = hf_daily_composites.create_group(f"{timestamps[i_modified][-10:]}")
                current_group_day.create_dataset("composite_burn_scar_RGB",\
                                                 data=composite_burn_scar_RGB,\
                                                 compression='gzip')
                current_group_day.create_dataset("composite_VZA",\
                                                 data=composite_VZA,\
                                                 compression='gzip')
                current_group_day.create_dataset("composite_RAA",\
                                                 data=composite_RAA,\
                                                 compression='gzip')
                print('saved daily composite {} *********************************'\
                      .format(timestamps[i_modified][-10:]))


                #print('plotting {:03d}/10 {}'.format(i_modified, timestamp))

if __name__=='__main__':
    from datetime import datetime

    current_date = datetime.today().strftime('%m_%d_%Y')

    if sys.argv != ['']:
        get_daily_composites()
    elif sys.argv == ['OPERATIONAL']:
        # write database if not yet created
        # if day already exist, skip
        home           = "/scratch/zt1/project/vllgsbr2-prj/raw_data_burnscar/data/"
        database_file  = home + "databases/operational_databse.h5"
        composite_file = home + f"daily_dlcf_rgb_composites/operational_composite_files/operational_comsposite_{current_date}.h5"
        get_daily_composites(database_file       = database_file, composite_file = None, current_date        = None,\
                             latest_date         = None, composite_type = "OPERATIONAL", viirs_database_file = None,\
                             daily_composite_dir = None)

