import numpy as np
from netCDF4 import Dataset


def get_VJ103_geo(geo_file, include_latlon=False, include_SZA=False,
                  include_VZA=False, include_SAAVAA=False):
    '''
    input: VIIRS VJ103 (or VNP103) .nc file
    return: conditionally; lat, lon, SZA, VZA, SAA, VAA
    '''

    with Dataset(geo_file, 'r') as nc_geo_file_obj:
        geolocation_ncObj = nc_geo_file_obj['geolocation_data']

        return_dict = {}
        if include_latlon:
            return_dict['lat'] = geolocation_ncObj['latitude'][:]
            return_dict['lon'] = geolocation_ncObj['longitude'][:]
        if include_SZA:
            return_dict['SZA'] = geolocation_ncObj['solar_zenith'][:]
        if include_VZA:
            return_dict['VZA'] = geolocation_ncObj['sensor_zenith'][:]
        if include_SAAVAA:
            return_dict['SAA'] = geolocation_ncObj['solar_azimuth'][:]
            return_dict['VAA'] =  geolocation_ncObj['sensor_azimuth'][:]

    return return_dict

def get_VJ102_ref(ref_file, which_bands=None):
    '''
    input: VIIRS VJ102 (or VNP102) .nc file
    return: lat, lon, SZA, VZA, SAA, VAA
    '''

    with Dataset(ref_file, 'r') as nc_ref_file_obj:
        observation_data_ncObj = nc_ref_file_obj['observation_data']

        M_bands                  = np.zeros((3248,3200,16))
        M_band_solar_irradiances = np.ones(16)
        if which_bands == None:
            which_bands == range(16)
        else:
            which_bands -= 1


        for i in which_bands:
            #band names 1 indexed
            band_num = i+1
            M_bands_temp = observation_data_ncObj['M{:02d}'.format(band_idx)][:]
            M_band_shape = np.shape(M_bands_temp)
            #fill empty data with nans
            M_bands[:M_band_shape[0],:M_band_shape[1],i] = M_bands_temp
            M_bands[M_band_shape[0]:,M_band_shape[1]:,i] = np.nan

            #get rid of dat with Digital Number of bad quality
            M_bands[M_bands >=65532 ] = np.nan

        return M_bands


def get_CLDMSK(cldmsk_file):

##########################################
    def get_bits(data_SD, N, cMask_or_QualityAssur=True):
        '''
        INPUT:
              data_SD               - 3D numpy array  - cloud mask SD from HDF
              N                     - int             - byte to work on
              cMask_or_QualityAssur - boolean         - True for mask, False for QA
        RETURNS:
              numpy.bytes array of byte stack of shape 2030x1354
        '''
        shape = np.shape(data_SD)

        #convert MODIS 35 signed ints to unsigned ints
        if cMask_or_QualityAssur:
            data_unsigned = np.bitwise_and(data_SD[N, :, :], 0xff)
        else:
            data_unsigned = np.bitwise_and(data_SD[:, :, N], 0xff)



        #type is int16, but unpackbits taks int8, so cast array
        data_unsigned = data_unsigned.astype(np.uint8)#data_unsigned.view('uint8')

        #return numpy array of length 8 lists for every element of data_SD
        data_bits = np.unpackbits(data_unsigned)

        if cMask_or_QualityAssur:
            data_bits = np.reshape(data_bits, (shape[1], shape[2], 8))
        else:
            data_bits = np.reshape(data_bits, (shape[0], shape[1], 8))

        return data_bits

    def decode_byte_1(decoded_mod35_hdf):
    '''
    INPUT:
          decoded_mod35_hdf: - numpy array (2030, 1354, 8) - bit representation
                               of MOD_35
    RETURN:
          Cloud_Mask_Flag,
          new_Unobstructed_FOV_Quality_Flag,
          Day_Night_Flag,
          Sun_glint_Flag,
          Snow_Ice_Background_Flag,
          new_Land_Water_Flag
                           : - numpy array (6, 2030, 1354) - first 6 MOD_35
                                                             products from byte1
    '''
    data = decoded_mod35_hdf
    shape = np.shape(data)

    #create empty arrays to fill later
    #binary 1 or 0 fill
    Cloud_Mask_Flag           = data[:,:, 7]
    Day_Night_Flag            = data[:,:, 4]
    Sun_glint_Flag            = data[:,:, 3]
    Snow_Ice_Background_Flag  = data[:,:, 2]

    #0,1,2,or 3 fill
    #cloudy, uncertain clear, probably clear, confident clear
    Unobstructed_FOV_Quality_Flag = data[:,:, 5:7]
    #find index of each cloud possibility
    #new list to stuff new laues into and still perform a good search
    new_Unobstructed_FOV_Quality_Flag = np.empty((shape[0], shape[1]))

    cloudy_index          = np.where((Unobstructed_FOV_Quality_Flag[:,:, 0]==0)\
                                   & (Unobstructed_FOV_Quality_Flag[:,:, 1]==0))
    uncertain_clear_index = np.where((Unobstructed_FOV_Quality_Flag[:,:, 0]==0)\
                                   & (Unobstructed_FOV_Quality_Flag[:,:, 1]==1))
    probably_clear_index  = np.where((Unobstructed_FOV_Quality_Flag[:,:, 0]==1)\
                                   & (Unobstructed_FOV_Quality_Flag[:,:, 1]==0))
    confident_clear_index = np.where((Unobstructed_FOV_Quality_Flag[:,:, 0]==1)\
                                   & (Unobstructed_FOV_Quality_Flag[:,:, 1]==1))

    new_Unobstructed_FOV_Quality_Flag[cloudy_index]          = 0
    new_Unobstructed_FOV_Quality_Flag[uncertain_clear_index] = 1
    new_Unobstructed_FOV_Quality_Flag[probably_clear_index]  = 2
    new_Unobstructed_FOV_Quality_Flag[confident_clear_index] = 3

    #water, coastal, desert, land
    Land_Water_Flag = data[:,:, 0:2]
    #find index of each land type possibility
    new_Land_Water_Flag = np.empty((shape[0], shape[1]))

    water_index   = np.where((Land_Water_Flag[:,:, 0]==0) & \
                             (Land_Water_Flag[:,:, 1]==0))
    coastal_index = np.where((Land_Water_Flag[:,:, 0]==0) & \
                             (Land_Water_Flag[:,:, 1]==1))
    desert_index  = np.where((Land_Water_Flag[:,:, 0]==1) & \
                             (Land_Water_Flag[:,:, 1]==0))
    land_index    = np.where((Land_Water_Flag[:,:, 0]==1) & \
                             (Land_Water_Flag[:,:, 1]==1))

    new_Land_Water_Flag[water_index]   = 0
    new_Land_Water_Flag[coastal_index] = 1
    new_Land_Water_Flag[desert_index]  = 2
    new_Land_Water_Flag[land_index]    = 3

    return Cloud_Mask_Flag,\
           new_Unobstructed_FOV_Quality_Flag,\
           Day_Night_Flag,\
           Sun_glint_Flag,\
           Snow_Ice_Background_Flag,\
           new_Land_Water_Flag

    with Dataset(ref_file, 'r') as nc_ref_file_obj:
        observation_data_ncObj = nc_ref_file_obj['']
    data_SD   = get_data(filename_MOD_35, 'Cloud_Mask', 2)

    fast_bits = get_bits(data_SD, 0)
    fast_bits = decode_byte_1(fast_bits)

#######################

    return cldmsk
