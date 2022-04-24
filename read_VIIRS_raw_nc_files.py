import numpy as np
from netCDF4 import Dataset


def get_VJ103_geo(geo_file):
    '''
    input: VIIRS VJ103 .nc file
    return: lat, lon, SZA, VZA, SAA, VAA
    '''

    with Dataset(geo_file, 'r') as nc_geo_file_obj:
        geolocation_ncObj = nc_geo_file_obj['geolocation_data']
        lat, lon = geolocation_ncObj['latitude'][:], geolocation_ncObj['longitude'][:]
        SZA = geolocation_ncObj['solar_zenith'][:]
        VZA = geolocation_ncObj['sensor_zenith'][:]
        SAA = geolocation_ncObj['solar_azimuth'][:]
        VAA = geolocation_ncObj['sensor_azimuth'][:]

    return lat, lon, SZA, VZA, SAA, VAA

def get_VJ102_ref(ref_file):
    '''
    input: VIIRS VJ103 .nc file
    return: lat, lon, SZA, VZA, SAA, VAA
    '''

    with Dataset(ref_file, 'r') as nc_ref_file_obj:
        observation_data_ncObj = nc_ref_file_obj['observation_data']

        M_bands                    = np.zeros((3248,3200,16))
        M_band_solar_irradiances = np.ones(16)
#         M_band_quality_flags  = np.zeros((3232,3200,16))
        for i in range(16):
            i_ = i+1
#             M_band_quality_flags[:,:,i] = observation_data_ncObj['M{:02d}_quality_flags'.format(i_)][:]
            M_bands_temp = observation_data_ncObj['M{:02d}'.format(i_)][:]
            M_band_shape = np.shape(M_bands_temp)
            M_bands[:M_band_shape[0],:,i] = M_bands_temp
            M_bands[M_band_shape[0]:,:,i] = np.nan
#             try:
#                 M_bands[:,:,i]     = M_bands_temp
#             except:
#                 shape
#                 M_bands[:3216,:,i] = M_bands_temp
#                 M_bands[3216:,:,i] = np.nan

#                 (3248,3200) into shape (3232,3200)
            try:
                M_bands_rad_scale_factor      = observation_data_ncObj['M{:02d}'.format(i_)].radiance_scale_factor
                M_bands_ref_scale_factor      = observation_data_ncObj['M{:02d}'.format(i_)].scale_factor
                M_band_solar_irradiances[i]   = M_bands_rad_scale_factor/M_bands_ref_scale_factor
#                 print(M_band_solar_irradiances[i])
                M_bands[M_bands >=65532 ]     = np.nan
#                 M_bands[:,:,i]                *= M_bands_ref_scale_factor

            except:
                M_bands[M_bands >=65532 ] = np.nan

        return M_bands, M_band_solar_irradiances


def get_CLDMSK(cldmsk_file):

    with Dataset(ref_file, 'r') as nc_ref_file_obj:
        observation_data_ncObj = nc_ref_file_obj['']

    return cldmsk
