import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import time
from burn_scar_composites import flip_arr
import warnings
warnings.filterwarnings("ignore")
plt.switch_backend('qtAgg')
#calculate fields to plot
home = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/VIIRS_database/'



database_path = home+'databases/VIIRS_burn_Scar_database_test_cases_Danielle.h5'
composite_path = home+'databases/daily_DLCF_composites_test_cases_Danielle.h5'
with h5py.File(database_path,'r') as hf_database,\
     h5py.File(composite_path,'r+') as hf_daily_composites:

    timestamps    = list(hf_database.keys())
    dataset_names = list(hf_database[timestamps[0]].keys())

    #stack composites here, where latest pixel overwrites all other
    #the oldest pixel should be no more than n days old (start w/8 days)
    #note a granbule does not eqaul a day, i.e. there is over lap, so use
    #time and day to decide which granule is newer in order to overwrite.
    #empty/ non valid pixels should be auto overwritten, i.e. np.nan or x<=-997
    #work backwords from newest
    #once it works try again but cloud and water clear with -998, -997 repectively
    base_image_shape           = (3604, 2354, 3)
    composite_burn_scar_RGB    = np.empty(base_image_shape)
    composite_burn_scar_RGB[:] = np.nan

    start, end = 60,66#*n_days
    for i, timestamp in enumerate(timestamps[start:end]):
        i_modified = i+start
        # print(i_modified)
        # populate dataset_dict with all useful data to make composites of
        dataset_dict = {}
        for dataset_name in dataset_names:
            dataset_dict[dataset_name] = hf_database[timestamp+'/'+dataset_name][:]

        # grab data from dataset_dict
        r_row_idx    = dataset_dict['regrid_row_idx']
        r_col_idx    = dataset_dict['regrid_col_idx']
        fill_val_idx = dataset_dict['fill_val_idx']

        cldmsk          = dataset_dict['cldmsk'       ][r_row_idx, r_col_idx]
        burn_scar_RGB   = dataset_dict['burn_scar_RGB'][r_row_idx, r_col_idx]
        land_water_mask = dataset_dict['land_water_mask'][r_row_idx, r_col_idx]

        # NBR           = dataset_dict['NBR'          ][r_row_idx, r_col_idx]
        # BRF_RGB       = dataset_dict['BRF_RGB'      ][r_row_idx, r_col_idx]
        # lat_regrid    = dataset_dict['lat'][r_row_idx, r_col_idx]
        # lon_regrid    = dataset_dict['lon'][r_row_idx, r_col_idx]
        # R_M7, R_M11 = burn_scar_RGB[:,:,1], burn_scar_RGB[:,:,0]
        # burn_scar_composite = get_burn_scar_composite(R_M7, R_M11, geotiff=False)
        #
        #
        cldmsk[fill_val_idx[0], fill_val_idx[1]]          = np.nan
        burn_scar_RGB[fill_val_idx[0], fill_val_idx[1],:] = np.nan
        # NBR[fill_val_idx[0], fill_val_idx[1]]           = np.nan
        # BRF_RGB[fill_val_idx[0], fill_val_idx[1]]       = np.nan
        # lat_regrid[fill_val_idx[0], fill_val_idx[1]]    = np.nan
        # lon_regrid[fill_val_idx[0], fill_val_idx[1]]    = np.nan
        #
        cldmsk[cldmsk<=-997]               = np.nan
        burn_scar_RGB[burn_scar_RGB<=-997] = np.nan
        # NBR[NBR<=-997]                = np.nan
        # BRF_RGB[BRF_RGB<=-997]        = np.nan
        #
        # #cloud and water clear
        burn_scar_RGB[cldmsk < 1] = np.nan #noise from probably clear, may mask it out
        # #the probably cloudy is important to keep other wise lose alot to smoke maybe
        burn_scar_RGB[land_water_mask != 1] = np.nan #just take out all water types



        #combine data by populating nan with cloud free water free data from latest granule
        #where it's empty on composite grid

        # unpopulated_current_idx is where the current granule is nan
        unpopulated_current_idx = np.where(np.isnan(burn_scar_RGB))

        # temp composite holds the last composite_burn_scar_RGB
        temp_composite          = np.copy(composite_burn_scar_RGB)
        # composite_burn_scar_RGB then takes on the current granules burn_scar_RGB
        composite_burn_scar_RGB = np.copy(burn_scar_RGB)

        # where ever the current burn_scar_RGB is nan, which is what
        # composite_burn_scar_RGB is pointing to now, put the old comosite values
        # stored in temp_composite, which only has vlaues from the last
        # composite where the current granule has invalid values
        composite_burn_scar_RGB[unpopulated_current_idx[0],\
                                unpopulated_current_idx[1], :] = \
                                    temp_composite[unpopulated_current_idx[0], \
                                                   unpopulated_current_idx[1],:]

        # now take the composite_burn_scar_RGB and turn into composite pbsm
        # then save that into the file along with the RGB

        print('{:02d}/{} {} composite updated'.format(i_modified+1, len(timestamps), timestamp))

        #save composite when next granule is from next day (granules are in chronilogical order)
        if i_modified<len(timestamps)-1:
           if int(timestamps[i_modified+1][4:7]) > int(timestamps[i_modified][4:7]) or\
              int(timestamps[i_modified+1][:4] ) > int(timestamps[i_modified][:4]):
              # print(timestamps[i+1][4:7], int(timestamps[i][4:7]))
              #make group with day of compsite
              composite_burn_scar_RGB = flip_arr(composite_burn_scar_RGB)
              hf_daily_composites.create_dataset(timestamps[i_modified][-10:],\
                                                 data=composite_burn_scar_RGB,\
                                                 compression='gzip')
              print('saved daily composite {} *********************************'\
                    .format(timestamps[i_modified][-10:]))


              # plt.figure(figsize=(10,10))
              # plt.imshow(composite_burn_scar_RGB)
              # plt.show()
              # print('plotting {:03d}/10 {}'.format(i_modified, timestamp))

              # delete everything after each day is composited
              composite_burn_scar_RGB[:] = np.nan
        elif i_modified==len(timestamps)-1:
            # print(timestamps[i+1][4:7], int(timestamps[i][4:7]))
            #make group with day of compsite
            composite_burn_scar_RGB = flip_arr(composite_burn_scar_RGB)
            hf_daily_composites.create_dataset(timestamps[i_modified][-10:],\
                                               data=composite_burn_scar_RGB,\
                                               compression='gzip')
            print('saved daily composite {} *********************************'\
                  .format(timestamps[i_modified][-10:]))


            # plt.figure(figsize=(10,10))
            # plt.imshow(composite_burn_scar_RGB)
            # plt.show()
            # print('plotting {:03d}/10 {}'.format(i_modified, timestamp))





'''**************************************************************************'''
'''************************ convert to functions ****************************'''
'''**************************************************************************'''
#for non consecutive test cases put True, otherwise false
single_granule_composite = True

def single_granule_cloud_clear_data(data, cloud_mask, land_water_mask, r_row_idx, r_col_idx, fill_val_idx):

    '''
    given only 1 granules of data, replace the cloudy, water and invalid
    pixels with np.nan
    '''
    # regrid the data
    cloud_mask      = cloud_mask[r_row_idx, r_col_idx]
    data            = data[r_row_idx, r_col_idx]
    land_water_mask = land_water_mask[r_row_idx, r_col_idx]

    # set bad data to nan
    cloud_mask[fill_val_idx[0], fill_val_idx[1]] = np.nan
    data[fill_val_idx[0], fill_val_idx[1],:]     = np.nan

    # set any -999 data to nan
    data[data==-999] = np.nan

    #cloud and water clear
    data[cloud_mask < 1] = np.nan #noise from probably clear, may mask it out
    #the probably cloudy is important to keep other wise lose alot to smoke maybe
    data[land_water_mask != 1] = np.nan #just take out all water types

    return data
#cloud and water clear, regrid and clean bad/missing data in current granule
# burn_scar_RGB = single_granule_cloud_clear_data(burn_scar_RGB,\
#         cldmsk, land_water_mask, r_row_idx, r_col_idx, fill_val_idx)

# if single_granule_composite == True:
#     hf_daily_composites.create_dataset(timestamps[i][-10:],\
#                                        data=burn_scar_RGB,\
#                                        compression='gzip')
#     print('saved daily composite {} *********************************'\
#                                            .format(timestamps[i][-10:]))
#     continue
# else:

# # condition to skip processing if granule only has a small over lap
# # domain, to save compute resources
# if np.count_nonzero(r_row_idx == -999)/len(r_row_idx.flatten()) < 0.85:
#     pass
# else:
#     if not(int(timestamps[i+1][4:7]) > int(timestamps[i][4:7]) or \
#        int(timestamps[i+1][:4] ) > int(timestamps[i][:4])):
#
#         print(i, 'pass')
#         continue


# def multi_granule_cloud_clear_data(data, cloud_mask, land_water_mask,\
#                                    r_row_idx, r_col_idx, fill_val_idx):
#     '''
#     given at least 2 granules of data, combine them into 1 array which
#     has no clouds in it
#     '''
#     composites = []
#
#
#     single_granule_cloud_clear_data(data, cloud_mask, land_water_mask,\
#                                     r_row_idx, r_col_idx, fill_val_idx)
#
#     #combine data by populating nan with cloud free water free data from latest granule
#     #where it's empty on composite grid
#     unpopulated_composite_idx = np.where(np.isnan(composite_burn_scar_RGB))
#     #where the grid is empty on the current working granule
#     unpopulated_current_idx   = np.where(np.isnan(burn_scar_RGB))
#
#     #make a copy of the composite to restore the original where current working granule had no data
#     temp_composite            = np.copy(composite_burn_scar_RGB)
#     composite_burn_scar_RGB   = np.copy(burn_scar_RGB)
#
#     #restore composite pixels overwritten by nan on the current working granule
#     composite_burn_scar_RGB[unpopulated_current_idx[0],\
#                             unpopulated_current_idx[1], :] = \
#                             temp_composite[unpopulated_current_idx[0],\
#                             unpopulated_current_idx[1], :]
#
#
#     return cleaned_data
