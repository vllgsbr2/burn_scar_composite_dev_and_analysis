import matplotlib.pyplot as plt
import h5py

# Open the HDF5 file
analysis_year        = '2020'
home                 = '/scratch/zt1/project/vllgsbr2-prj/'
home_data            = home      + 'raw_data_burnscar/data/'
composite_file       = home_data + f'daily_dlcf_rgb_composites/{analysis_year}_daily_DLCF_RGB_composites.h5'
database_file        = home_data + f'databases/viirs_burnscar_database_{analysis_year}.h5'

database_or_daily_composites = True #False

if database_or_daily_composites:
    with h5py.File(composite_file, 'r') as hdf5_file:
        # Loop through each dataset
        start, end =10,-1
        hdf5_file_keys = list(hdf5_file.keys())[start:end]
        for dataset_name in hdf5_file_keys:
            dataset = hdf5_file[dataset_name]
            
            # Assuming the dataset contains RGB data
            rgb_data = dataset[:]
            
            # Plot the RGB data
            plt.imshow(rgb_data)
            plt.title(dataset_name)
            plt.show()
else:
    with h5py.File(database_file, 'r') as hdf5_file:
        # Loop through each dataset
        for granule in hdf5_file.keys():
            rgb_data = hdf5_file[granule+'/burn_scar_RGB'][:]

            # Assuming the dataset contains RGB data
            #rgb_data = dataset[:]

            # Plot the RGB data
            plt.imshow(rgb_data)
            plt.title(granule)
            plt.show()
