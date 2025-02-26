import numpy as np
from netCDF4 import Dataset
import h5py

def get_VJ114_thermal_anomalies(VJ114_file):
    # netcdf4 - VJ114.A2024231.2236.002.2024232050547.nc
    '''
    "fire_mask" - Pixel Class Definition
    0 Not processed
    1 Bow-tie deletion
    2 Sun glint
    3 Water
    4 Cloud
    5 Land
    6 Unclassified
    7 Low confidence fire pixel
    8 Nominal confidence fire pixel
    9 High confidence fire pixel
    '''
    
    with Dataset(VJ114_file, 'r') as nc_VJ114:
        return nc_VJ114['fire_mask'][:]

def get_VJ115_leaf_area_index(VJ115_file):
    # h5 files - VJ115A2H.A2024177.h07v05.002.2024185091418.h5
    #scale factor of 0.1
    with h5py.File(VJ115_file, 'r') as hf_VJ115:
        return np.array(hf_VJ115['HDFEOS/GRIDS/Data_Fields/Lai'][:], dtype=np.int8)*0.1

def get_VJ103_geo(geo_file, include_latlon=False, include_SZA=False,
                  include_VZA=False, include_SAAVAA=False, include_lwm=False):
    '''
    input: VIIRS VJ103 (or VNP103) .nc file
    return: conditionally; lat, lon, SZA, VZA, SAA, VAA, land_water_mask
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
        if include_lwm:
            # 0 Shallow_Ocean
            # 1 Land
            # 2 Coastline
            # 3 Shallow_Inland
            # 4 Ephemeral
            # 5 Deep_Inland
            # 6 Continental
            # 7 Deep_Ocean

            return_dict['land_water_mask'] = geolocation_ncObj['land_water_mask'][:]

    return return_dict

def get_VJ102_ref(ref_file, which_bands=None):
    '''
    input: VIIRS VJ102 (or VNP102) .nc file
    return: lat, lon, SZA, VZA, SAA, VAA
    '''

    with Dataset(ref_file, 'r') as nc_ref_file_obj:
        observation_data_ncObj = nc_ref_file_obj['observation_data']

        n = len(which_bands)

        # M_bands                  = np.zeros((3248,3200,n))
        M_bands = []

        for i, band_num in enumerate(which_bands):
            #band names 1 indexed
            # print(band_num)
            M_bands_temp = observation_data_ncObj['M{:02d}'.format(band_num)][:]
            M_band_shape = np.shape(M_bands_temp)
            #fill empty data with nans


            # M_bands[:M_band_shape[0],:M_band_shape[1], i] = M_bands_temp
            # M_bands[M_band_shape[0]:,M_band_shape[1]:, i] = np.nan

            M_bands.append(M_bands_temp)

        #get rid of dat with Digital Number of bad quality
        M_bands = np.moveaxis(M_bands, 0,2)
        M_bands[M_bands >=65532 ] = np.nan

        return M_bands
def get_bits(data_SD, N, cMask_or_QualityAssur=True):
    '''
    INPUT:
          data_SD               - 3D numpy array  - cloud mask SD from HDF
          N                     - int             - byte to work on
          cMask_or_QualityAssur - boolean         - True for mask, False for QA
    RETURNS:
          numpy.bytes array of byte stack of shape 2030x1354
          expands integer byte into 8 bits to read individually
    '''
    shape = np.shape(data_SD)

    #convert MODIS 35 signed ints to unsigned ints
    if cMask_or_QualityAssur:
        if np.ndim(data_SD)==3:
            data_unsigned = np.bitwise_and(data_SD[N, :, :], 0xff)
        elif np.ndim(data_SD)==2:
            data_unsigned = np.bitwise_and(data_SD, 0xff)
    else:
        data_unsigned = np.bitwise_and(data_SD[:, :, N], 0xff)



    #type is int16, but unpackbits taks int8, so cast array
    data_unsigned = data_unsigned.astype(np.uint8)#data_unsigned.view('uint8')

    #return numpy array of length 8 lists for every element of data_SD
    data_bits = np.unpackbits(data_unsigned)

    if cMask_or_QualityAssur:
        if np.ndim(data_SD)==3:
            data_bits = np.reshape(data_bits, (shape[1], shape[2], 8))
        elif np.ndim(data_SD)==2:
            data_bits = np.reshape(data_bits, (shape[0], shape[1], 8))
    else:
        data_bits = np.reshape(data_bits, (shape[0], shape[1], 8))

    return data_bits


def decode_QFX_bytes_VJ109(qfx_byte, x):
    '''
    VJ109 user guide source for QFX tables:
    https://viirsland.gsfc.nasa.gov/PDF/VIIRS_Surf_Refl_UserGuide_v2.0.pdf

    INPUT:
          qfx_byte: - numpy array (y,x,8) - contains 8 bits from the byte for y by x array
          x       : - int - tells which QFX, i.e. QF1 from VJ109
                    QF1 - bits 0 & 1 => cloud mask quality 
                          - 00 - Poor
                          - 01 - Low
                          - 10 - Medium
                          - 11 - High
                        - bits 2 & 3 => cloud mask confidence
                          - 00 - Confident Clear
                          - 01 - Probably Clear
                          - 10 - Probably Cloudy
                          - 11 - Confident Cloudy
                    QF2 - bit 3 => Shadow Mask (Cloud Shadows)
                          - 0 - No Cloud Shadow
                          - 1 - Shadow
                          bit 5 => Snow/Ice
                          - 0 - No  Snow/Ice
                          - 1 - Yes Snow/Ice
                    QF5 - bit 6 - Overall Quality M5  red  Surface Reflcetance
                          - 0 - Good
                          - 1 - Bad
                        - bit 7 - Overall Quality M7  burn Surface Reflcetance
                          - 0 - Good
                          - 1 - Bad
                    QF6 - bit 2 - Overall Quality M11 veg  Surface Reflcetance
                          - 0 - Good
                          - 1 - Bad
                        - bit 3 - Overall Quality I1  red  Surface Reflcetance
                          - 0 - Good
                          - 1 - Bad
                        - bit 4 - Overall Quality I2  veg  Surface Reflcetance
                          - 0 - Good
                          - 1 - Bad
    RETURN:
          cldmsk_qual,
          cldmsk,
          shadow_mask,
          snow_ice_mask,
          m5_qual_mask,
          m7_qual_mask,
          m11_qual_mask,
          i1_qual_mask,
          i2_qual_mask
    '''
    qfx_byte_mask_shape = np.shape(qfx_byte)

    if x==1:
        '''
        QF1 - bits 0 & 1 => cloud mask quality
              - 00 - Poor
              - 01 - Low
              - 10 - Medium
              - 11 - High
            - bits 2 & 3 => cloud mask confidence
              - 00 - Confident Clear
              - 01 - Probably Clear
              - 10 - Probably Cloudy
              - 11 - Confident Cloudy
        '''
        cldmsk_qual = qfx_byte[:,:,6:8][::-1]
        cldmsk      = qfx_byte[:,:,4:6][::-1]
    
        #convert bits to integer mask (cloud mask)
        shape = cldmsk.shape
        new_cldmsk = np.empty((shape[0], shape[1]))
        confident_clear_idx  = np.where((cldmsk[:,:, 0]==0) &\
                                        (cldmsk[:,:, 1]==0))
        probably_clear_idx   = np.where((cldmsk[:,:, 0]==0)& \
                                        (cldmsk[:,:, 1]==1))
        probably_cloudy_idx  = np.where((cldmsk[:,:, 0]==1) &\
                                        (cldmsk[:,:, 1]==0))
        confident_cloudy_idx = np.where((cldmsk[:,:, 0]==1) &\
                                        (cldmsk[:,:, 1]==1))

        new_cldmsk[confident_cloudy_idx] = 0
        new_cldmsk[probably_cloudy_idx]  = 1
        new_cldmsk[probably_clear_idx]   = 2
        new_cldmsk[confident_clear_idx]  = 3

        #convert bits to integer mask (cloud mask quality)
        new_cldmsk_qual = np.empty((shape[0], shape[1]))
        poor_qual_idx   = np.where((cldmsk_qual[:,:, 0]==0) &\
                                   (cldmsk_qual[:,:, 1]==0))
        low_qual_idx    = np.where((cldmsk_qual[:,:, 0]==0)& \
                                   (cldmsk_qual[:,:, 1]==1))
        med_qual_idx    = np.where((cldmsk_qual[:,:, 0]==1) &\
                                   (cldmsk_qual[:,:, 1]==0))
        high_qual_idx   = np.where((cldmsk_qual[:,:, 0]==1) &\
                                   (cldmsk_qual[:,:, 1]==1))

        new_cldmsk_qual[poor_qual_idx] = 0
        new_cldmsk_qual[low_qual_idx]  = 1
        new_cldmsk_qual[med_qual_idx]  = 2
        new_cldmsk_qual[high_qual_idx] = 3

        return new_cldmsk, new_cldmsk_qual

    elif x==2:
        '''
        QF2 - bit 3 => Shadow Mask (Cloud Shadows)
              - 0 - No Cloud Shadow
              - 1 - Shadow
            - bit 5 => Snow/Ice
              - 0 - No  Snow/Ice
              - 1 - Yes Snow/Ice
        '''
        shadow_mask   = qfx_byte[:,:,4]
        snow_ice_mask = qfx_byte[:,:,2]
        
        return shadow_mask, snow_ice_mask

    elif x==5:
        '''
        QF5 - bit 6 - Overall Quality M5  red  Surface Reflcetance
              - 0 - Good
              - 1 - Bad
            - bit 7 - Overall Quality M7  burn Surface Reflcetance
              - 0 - Good
              - 1 - Bad
        '''

        m5_qual_mask = qfx_byte[:,:,1]
        m7_qual_mask = qfx_byte[:,:,0]

        return m5_qual_mask, m7_qual_mask

    elif x==6:
        '''
        QF6 - bit 2 - Overall Quality M11 veg  Surface Reflcetance
              - 0 - Good
              - 1 - Bad
            - bit 3 - Overall Quality I1  red  Surface Reflcetance
              - 0 - Good
              - 1 - Bad
            - bit 4 - Overall Quality I2  veg  Surface Reflcetance
              - 0 - Good
              - 1 - Bad
        '''

        m11_qual_mask = qfx_byte[:,:,5]
        i1_qual_mask = qfx_byte[:,:,4]
        i2_qual_mask = qfx_byte[:,:,3]

        return m11_qual_mask, i1_qual_mask, i2_qual_mask

    else:
        print('not supported bit')


def get_VJ109_ref(ref_file, which_bands=[5,7,11], cld_shadow_snowice=True):
    '''
    input: VIIRS VJ109 (or VNP109) .nc file
    return: lat, lon, SZA, VZA, SAA, VAA
    '''

    with Dataset(ref_file, 'r') as nc_ref_file_obj:
        '''
        print(nc_ref_file_obj.variables.keys())
        import sys
        sys.exit()
        
        dict_keys(['375m Surface Reflectance Band I1', '375m Surface Reflectance Band I2', '375m Surface Reflectance Band I3', '750m Surface Reflectance Band M1', '750m Surface Reflectance Band M2', '750m Surface Reflectance Band M3', '750m Surface Reflectance Band M4', '750m Surface Reflectance Band M5', '750m Surface Reflectance Band M7', '750m Surface Reflectance Band M8', '750m Surface Reflectance Band M10', '750m Surface Reflectance Band M11', 'QF1 Surface Reflectance', 'QF2 Surface Reflectance', 'QF3 Surface Reflectance', 'QF4 Surface Reflectance', 'QF5 Surface Reflectance', 'QF6 Surface Reflectance', 'QF7 Surface Reflectance', 'land_water_mask'])
        
        '''
        #observation_data_ncObj = nc_ref_file_obj['SurfReflect_VNP/Data_Fields/']
        if not cld_shadow_snowice:
            n = len(which_bands)
            M_bands = []

        # qaulity control flag set to -999
        # https://viirsland.gsfc.nasa.gov/PDF/VIIRS_Surf_Refl_UserGuide_v1.3.pdf
        # overall sfc ref quality (bits zero indexed)
        # QF5 (bit4 M3, bit5 M4, bit6 M5, bit7 M7)
        # QF6 (bit2, M11, bit3 I1, bit4 I2) 
        # M3 blue,4 green,5 red,7 veggie,11 burn I1 640,I2 865
        if cld_shadow_snowice:
            QF1 = decode_QFX_bytes_VJ109(get_bits(nc_ref_file_obj.variables['QF1 Surface Reflectance'][:], 0),1)
            QF2 = decode_QFX_bytes_VJ109(get_bits(nc_ref_file_obj.variables['QF2 Surface Reflectance'][:], 0),2)
        else:
            QF5 = decode_QFX_bytes_VJ109(get_bits(nc_ref_file_obj.variables['QF5 Surface Reflectance'][:], 0),5)
            QF6 = decode_QFX_bytes_VJ109(get_bits(nc_ref_file_obj.variables['QF6 Surface Reflectance'][:], 0),6)
        
        # unpack returns from QF<X>
        if cld_shadow_snowice:
            new_cldmsk   , new_cldmsk_qual            = QF1
            shadow_mask  , snow_ice_mask              = QF2
        else:
            m5_qual_mask , m7_qual_mask               = QF5
            m11_qual_mask, i1_qual_mask, i2_qual_mask = QF6

            for i, band_num in enumerate(which_bands):
                #band names 1 indexed
                # print(band_num)
                M_bands_temp = nc_ref_file_obj.variables['750m Surface Reflectance Band M{}'\
                                                      .format(band_num)][:]
               
                # this shit removes bad NDVI vals... what?!? Just return QC
                # and save into dataset. Can investigate another time. 
                # Didn't know I had enemies until today. ATBD/User Guide useless
                '''
                # now use masks from QF<X> to quality control bands
                if band_num==5:
                    M_bands_temp[m5_qual_mask==1]=-999
                elif band_num==7:
                    M_bands_temp[m7_qual_mask==1]=-999
                elif band_num==11:
                    M_bands_temp[m11_qual_mask==1]=-999
                else:
                    print('band not yet supported')

                # add filtered band to band list for return
                '''
                M_bands.append(M_bands_temp)
               
        if cld_shadow_snowice:
            # if cloud mask is poor quality, set to -999
            # becuase burn scars dont exhibit spectral
            # properties similar to clouds, this is ok
            new_cldmsk[new_cldmsk_qual==0]=-999

    if cld_shadow_snowice:
        return new_cldmsk,new_cldmsk_qual,\
               snow_ice_mask, shadow_mask
    else:
        return np.array(M_bands)


def get_CLDMSK(cldmsk_file):

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

    with Dataset(cldmsk_file, 'r') as nc_cldmsk_file_obj:
        cldmsk_ints    = nc_cldmsk_file_obj['geophysical_data']['Cloud_Mask'][:]
        # integer_cldmsk = nc_cldmsk_file_obj['geophysical_data']['Integer_Cloud_Mask'][:]
        clsmsk_bits            = get_bits(cldmsk_ints, 0)
        cldmsk_and_misc_fields = decode_byte_1(clsmsk_bits)
        integer_cldmsk         = cldmsk_and_misc_fields[1]
        land_water_mask        = cldmsk_and_misc_fields[-1]

        return integer_cldmsk, land_water_mask

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


    home = 'R:/satellite_data/viirs_data/noaa20/geolocation/'
    geo_file = home + 'VJ103MOD.A2021154.2048.021.2021155015136.nc'
    # print(land_water_mask)
    def flip_arr(arr):
        '''
        return: array flipped over each of the 1st 2 axes for proper display using
        ax.imshow(arr)
        '''
        arr=np.flip(arr, axis=0)
        arr=np.flip(arr, axis=1)
        return arr
    land_water_mask = flip_arr(get_VJ103_geo(geo_file))

    plt.imshow(land_water_mask)
    plt.show()

