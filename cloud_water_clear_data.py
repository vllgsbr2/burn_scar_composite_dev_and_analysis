    # with h5py.File(database_filepath,'r') as hf_database:
    #     time_stamps = [x[:13] for x in list(hf_database.keys())]
    #     for i in hf_database[time_stamps]:


def mask_water_and_cloud(data, cldmsk, land_water_mask):
    '''
    inputs: data, cldmsk, land_water_mask 2D numpy arrays from VIIRS database
    return: masked data
    '''
    data[cldmsk>=1] = -998
    water   = 0
    coastal = 1
    desert  = 3
    land    = 4

    data[land_water_mask >= coastal] = -997

    return data
