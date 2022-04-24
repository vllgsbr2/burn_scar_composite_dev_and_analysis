# Grab VIIRS bands (0.86 microns I2 or 0.86 M7) and 2.25 M11
# we can calculate (R225-R86)/(R86+R225)
# then make a threshold cut off for the burn scar area

def normalized_burn_ratio(R_M7, R_M11):

    return (R_M11-R_M7)/(R_M11+R_M7)

# burn scar RGB composite <2.1, 0.86, 0.64> = <M11, I2, I1>
# can also do a threshold for burn scars

def burn_scar_RGB_composite(R_M11, R_M7, R_M5):

    return np.dstack((R_M11, R_M7, R_M5))

def burn_scar_RGB_composite(R_M7, R_M5):

    return (R_M7-R_M5)/(R_M7+R_M5)
