import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
from burn_scar_composites import get_burn_scar_composite

home = 'C:/Users/Javi/Documents/NOAA/Roger_SAR_data/for_javier(1)/for_javier/'
roger_files         = [home+x for x in os.listdir(home)]
landsat_8_files     = [x for x in roger_files if 'L1TP' in x]
interferogram_files = [x for x in roger_files if 'inter' in x]
time_step_files     = [x for x in roger_files if 'time_step' in x]

#take land sat bands 1 and 2 to make burn scar composite
#save back into geotiff
#shouldn't need to touch anything else, Sam can just put into Real Earth
# for ls_f in landsat_8_files[3:]:
#     f,ax= plt.subplots(nrows=1, ncols=2, figsize=(20,10), sharex=True, sharey=True)
#     with rasterio.open(ls_f) as src_ls_8:
#         data = np.moveaxis(src_ls_8.read(), 0,2)
#         ax[1].imshow(data)
#         composite = get_burn_scar_composite(data[:,:,1], data[:,:,0], geotiff=True)
#         im1 = ax[0].imshow(composite, cmap='jet', vmin=1, vmax=80)
#         im1.cmap.set_under('k')
#         for a in ax.flat:
#             a.set_xticks([])
#             a.set_yticks([])
#
#         plt.show()
#         plt.tight_layout()


for in_f in time_step_files[3:]:
    plt.figure(figsize=(10,10))
    with rasterio.open(in_f) as src_in_f:
        data = src_in_f.read()[0,:,:]
        plt.imshow(data)
        plt.xticks([])
        plt.yticks([])

        plt.tight_layout()
        plt.show()
