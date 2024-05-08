# Grab VIIRS bands (0.86 microns I2 or 0.86 M7) and 2.25 M11
# we can calculate (R225-R86)/(R86+R225)
# then make a threshold cut off for the burn scar area
import numpy as np


def get_normalized_burn_ratio(R_M7, R_M11):

    return (R_M11-R_M7)/(R_M11+R_M7)

# burn scar RGB composite <2.1, 0.86, 0.64> = <M11, I2, I1>
# can also do a threshold for burn scars

def get_burn_scar_RGB(R_M11, R_M7, R_M5):

    return np.dstack((R_M11, R_M7, R_M5))

def get_normalized_differenced_vegetation_index(R_M1, R_M7):

    return (R_M7 - R_M1)/(R_M7 + R_M1)

def get_BRF_lat_lon_TOA_VJ102(geo_file, ref_file, which_bands):
    #this is for get_VJ102 TOA reflectance; use other func for VJ109
    from read_VIIRS_raw_nc_files import get_VJ103_geo, get_VJ102_ref
    geolocation_dict = get_VJ103_geo(geo_file, include_latlon=True, include_SZA=True, include_lwm=True)
    lat, lon, SZA, LWM = geolocation_dict['lat'], geolocation_dict['lon'], geolocation_dict['SZA'], geolocation_dict['land_water_mask']
    time_stamp_current = geo_file[-33:-21]
    M_bands = get_VJ102_ref(ref_file, which_bands)

    cosSZA = np.cos(np.deg2rad(SZA))

    for i in range(len(which_bands)):
        # try:
        M_bands[:,:,i] /=  cosSZA
        # except:
        #
        #     M_bands[:3216,:,i] /=  cosSZA[:3216,:]

    return M_bands, lat, lon, LWM

def get_BRF_lat_lon(geo_file, ref_file, which_bands):
    #this is for get_VJ109 surface  reflectance; use other func for VJ102
    from read_VIIRS_raw_nc_files import get_VJ103_geo, get_VJ109_ref
    geolocation_dict = get_VJ103_geo(geo_file, include_latlon=True, include_SZA=True, include_lwm=True)
    lat, lon, SZA, LWM = geolocation_dict['lat'], geolocation_dict['lon'], geolocation_dict['SZA'], geolocation_dict['land_water_mask']
    time_stamp_current = geo_file[-33:-21]
    M_bands = get_VJ109_ref(ref_file, which_bands, cld_shadow_snowice=False)

    cosSZA = np.cos(np.deg2rad(SZA))

    for i in range(len(which_bands)):
        # try:
        '''
        print(type(M_bands))
        print(np.shape(M_bands), np.shape(cosSZA))
        import sys
        sys.exit()
        '''
        M_bands[i,:,:] /=  cosSZA
        # except:
        #
        #     M_bands[:3216,:,i] /=  cosSZA[:3216,:]

    return M_bands, lat, lon, LWM



def get_BRF_RGB(R_M5,R_M4,R_M3):
    return np.dstack((R_M5,R_M4,R_M3))

def flip_arr(arr):
    '''
    return: array flipped over each of the 1st 2 axes for proper display using
    ax.imshow(arr)
    '''
    arr=np.flip(arr, axis=0)
    arr=np.flip(arr, axis=1)
    return arr

def get_burn_scar_composite(R_M7, R_M11, geotiff=False, landwater_mask=None):
    from scipy import ndimage

    if not geotiff:
        # R_M11[R_M7  > 0.12] = np.nan #for clear no smoke only
        # R_M11[R_M7 < 0.0281] = np.nan
        # R_M11[R_M11 < 0.01] = np.nan

        # R_M11[R_M7  > 0.1346] = np.nan #for clear no smoke only
        # R_M11[R_M11 < 0.0281]

        # R_M11 = ndimage.gaussian_filter(R_M11, sigma=1)
        # R_M7 = ndimage.gaussian_filter(R_M7, sigma=1)

        R_M11[R_M7  > 0.2]   = np.nan #for clear no smoke only
        R_M11[R_M7  < 0.0281] = np.nan
        R_M11[R_M11 < 0.05]   = np.nan

        NBR = get_normalized_burn_ratio(R_M7, R_M11)
        NBR[NBR<-0.35] = np.nan
        burn_scar_mask = NBR
        if landwater_mask == None:
            return burn_scar_mask
        else:#need to add some flags from cloud mask
            burn_scar_mask[landwater_mask==desert] = 0
            return burn_scar_mask
    else:
        R_M11[R_M7  > 55] = 0
        R_M11[R_M11 < 45] = 0
        return ndimage.gaussian_filter(R_M11, sigma=2)
