# Grab VIIRS bands (0.86 microns I2 or 0.86 M7) and 2.25 M11
# we can calculate (R225-R86)/(R86+R225)
# then make a threshold cut off for the burn scar area

def get_normalized_burn_ratio(R_M7, R_M11):

    return (R_M11-R_M7)/(R_M11+R_M7)

# burn scar RGB composite <2.1, 0.86, 0.64> = <M11, I2, I1>
# can also do a threshold for burn scars

def get_burn_scar_RGB(R_M11, R_M7, R_M5):

    return np.dstack((R_M11, R_M7, R_M5))

def get_burn_scar_RGB_composite(R_M7, R_M5):

    return (R_M7-R_M5)/(R_M7+R_M5)

def get_BRF_lat_lon(geo_file, ref_file, which_bands):

    lat, lon, SZA = get_VJ103_geo(geo_file, get_all=False)
    time_stamp_current = geo_file[-33:-21]
    M_bands = get_VJ102_ref(ref_file, which_bands)

    cosSZA = np.cos(np.deg2rad(SZA))

    for i in range(len(which_bands)):
        try:
            M_bands[:,:,i] /=  cosSZA
        except:
            M_bands[:3216,:,i] /=  cosSZA[:3216,:]

    return M_bands, lat, lon

def get_BRF_RGB(R_M5,R_M4,R_M3):
    return np.dstack((R_M5,R_M4,R_M3))
