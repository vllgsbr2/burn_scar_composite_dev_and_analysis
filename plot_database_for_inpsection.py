import h5py
import numpy as np
import matplotlib.pyplot as plt
database_file= 'R:/satellite_data/viirs_data/noaa20/databases/VIIRS_burn_Scar_database.h5'
with h5py.File(database_file, 'r') as hf_database:
    time_stamps   = list(hf_database.keys())
    dataset_names = ['BRF_RGB', 'NBR', 'burn_scar_RGB', 'cldmsk',\
                     'land_water_mask', 'lat', 'lon']
    start, end = 0,-1
    for ts in time_stamps[start:end]:

        BRF_RGB         = hf_database['{}/{}'.format(ts, dataset_names[0])][:]
        burn_scar_RGB   = hf_database['{}/{}'.format(ts, dataset_names[2])][:]
        cldmsk          = hf_database['{}/{}'.format(ts, dataset_names[3])][:]
        land_water_mask = hf_database['{}/{}'.format(ts, dataset_names[4])][:]

        BRF_RGB[BRF_RGB==-999] = np.nan
        burn_scar_RGB[burn_scar_RGB==-999] = np.nan
        cldmsk[cldmsk==-999] = np.nan
        land_water_mask[land_water_mask==-999] = np.nan

        f, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, \
                             figsize=(13,13))

        im1 = ax[0,0].imshow(1.5*BRF_RGB)
        im2 = ax[0,1].imshow(cldmsk, cmap='Greys')
        im3 = ax[1,0].imshow(land_water_mask, cmap='jet')
        im4 = ax[1,1].imshow(1.2*burn_scar_RGB)

        for a in ax.flat:
            a.set_xticks([])
            a.set_yticks([])

        title = 'NOAA-20 VIIRS; {}'.format(ts)
        f.suptitle(ts)
        plt.tight_layout()
        plt.show()


# cases
# 6/13/2021 clear shot over cali
# 2021165.2042 6/14/21
# 21021166.1842 midwest/east case very clear
# 2021166.2212 6/15/21 alaska case
# 21021168.2130 6/17 west coast case lots of scars
# 169.2112 6/18 west north coast cali to washington
# 170.2054 6/19 NW washingotn case with a 1 pixel burning fire slight smoke
# 171.2030 6/20 clear shot of western US
# 172.2012 6/21 clear shot of central west/plains
# 175.2054 6/24 nice shot of cali and west
# lost it but its before 1180.0754 6/29 shows fire and smoke
# 181.1906 6/30 smoke marked as confident cloud
# 181.2042 6/30 smoke marked as probably clear and 181.2048
# 183.1824 smoke over great lakes, falsley marked as cloud
# 188.1836 7/7 canadian fires with smoke
# 188.2154 cali fire in bow tie region
# 188.2200 cali fire in bow tie region
# 191.1918 7/10 canadian fires peppered every where
# 191.2054 7/10  cali fires with smoke
# 192.1900 7/11 canadian fires


# general notes
# snow/ice on mtns shows as sky blue on bs rgb
# high ice clouds are a little blueer than low cumulus
# cloud shadows are rlly visible from bs RGB (much darker than bs)
# land water mask and cloud msk from cldmsk file keep bowtie
# some clouds are red, 2.25 is red microphysics band
# desert lake beds are cyan in bs rgb
# desert shows as a week redish=brown signal similar to bs edges
# fire smoke shows as a dark translucent cyan in bs rgb, in rgb its thick and brown
# smoke is pretty translucentin bs rgb, cldmsk marks it as confident cloud
# clouds are generally on top of the smoke
# smoke plume and underlying smoked area, rough and smooth texture respectively
