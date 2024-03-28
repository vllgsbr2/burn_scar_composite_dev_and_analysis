from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as matCol
import os
import pandas as pd
import numpy as np
import h5py
import configparser
from burn_scar_composites import get_burn_scar_composite,\
                                 get_normalized_burn_ratio,\
                                 get_normalized_differenced_vegetation_index

#plots on correct screen and ctrl + c exits plot window
plt.switch_backend('qtAgg')

#open up database to access DLCF RGB composite
config = configparser.ConfigParser()
config.read('config_filepaths.txt')
home_dir = config['file paths mac']['database dir']
databse_name = 'daily_DLCF_composites_2021_west_CONUS_all_days.h5'

with h5py.File(home_dir + databse_name, 'r') as hf_database:
    X = hf_database['08.10.2021'][:]

rgb_OG = np.copy(X)
rgb_OG_copy = np.copy(X)

#ancillary composites to view
Rbrn, Rveg, Rvis = rgb_OG_copy[:,:,0], rgb_OG_copy[:,:,1], rgb_OG_copy[:,:,2]
# NDVI             = get_normalized_differenced_vegetation_index(Rvis, Rveg)
# NBR              = get_normalized_burn_ratio(Rveg, Rbrn)
burnscar_mask    = get_burn_scar_composite(Rveg, Rbrn)

#workflow to label and analyze burn scars in DLCF RGB composite
burnscar_semi_labeled_dataset_file = 'subsetted_burn_scar_coordinates.txt'
df_burnscar_semi_labeled = pd.read_csv(burnscar_semi_labeled_dataset_file,\
                                         header=0, delimiter=', ', skiprows=7)
# print(df_burnscar_semi_labeled)

#build boxes around burn scars then visualize on RGB
col1 = df_burnscar_semi_labeled['col1'].tolist()
col2 = df_burnscar_semi_labeled['col2'].tolist()
row1 = df_burnscar_semi_labeled['row1'].tolist()
row2 = df_burnscar_semi_labeled['row2'].tolist()

#only show NBR when burnscar_mask is not nan and index is within rectangles
burnscar_mask_manual_labels_combined = np.copy(burnscar_mask)
burnscar_mask_manual_labels_combined[:,:] = np.nan
manual_labels = np.copy(burnscar_mask)
manual_labels[:,:] = np.nan
for i in range(len(col1)):
    manual_labels[row1[i]:row2[i],col1[i]:col2[i]] = 1

burnscar_mask_manual_labels_combined[manual_labels==1] = burnscar_mask[manual_labels==1]


'''****************** make masked array to label data ***********************'''
'''****************** make masked array to label data ***********************'''
'''****************** make masked array to label data ***********************'''
manual_labels_burned_area = np.copy(burnscar_mask)
manual_labels_burned_area[:,:] = 0
manual_labels_burned_area[~np.isnan(burnscar_mask_manual_labels_combined)] = 1


'''********************************** plot **********************************'''
'''********************************** plot **********************************'''
'''********************************** plot **********************************'''
plot=True
if plot==True:
    plt.rcParams.update({'font.size': 35})
    plt.style.use('dark_background')
    f, ax = plt.subplots(ncols=5, figsize=(35,20), sharex=True, sharey=True)
    ax[2].imshow(burnscar_mask_manual_labels_combined, cmap='jet', vmax=0.25)
    ax[1].imshow(1.5*rgb_OG)
    ax[0].imshow(1.5*rgb_OG)
    im = ax[1].imshow(burnscar_mask, cmap='jet', vmax=0.25, alpha=0.75)
    area_total = 0
    for i in range(len(col1)):
        width, length = col2[i]-col1[i], row2[i]-row1[i]
        area_total += width*length
        rect = patches.Rectangle((col1[i], row1[i]), width, length, linewidth=2,\
                                 edgecolor='r', facecolor='none')
        rect1 = patches.Rectangle((col1[i], row1[i]), width, length, linewidth=2,\
                                 edgecolor='r', facecolor='none')
        rect2 = patches.Rectangle((col1[i], row1[i]), width, length, linewidth=2,\
                                 edgecolor='r', facecolor='none')
        rect3 = patches.Rectangle((col1[i], row1[i]), width, length, linewidth=2,\
                                 edgecolor='r', facecolor='none')
        rect4 = patches.Rectangle((col1[i], row1[i]), width, length, linewidth=2,\
                                 edgecolor='r', facecolor='none')

        ax[0].add_patch(rect)
        ax[1].add_patch(rect1)
        ax[2].add_patch(rect2)
        ax[3].add_patch(rect3)
        ax[4].add_patch(rect4)

    ax[2].set_title('Intersection of Primitive Burn Scar Mask\nand Manual Labels')
    ax[1].set_title('Day Land Cloud Fire RGB\nManual Labels\nPrimitive Burn Scar Mask')
    ax[0].set_title('Day Land Cloud Fire RGB\n[2.25, 0.86, 0.67]µm\nManual Labels (Red Squares) ')
    f.suptitle('NOAA-20 VIIRS Valid 08.10.2021 Composited & Cloud-Cleared Over Previous 8 Days')

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    plt.subplots_adjust(left  =0.000,
                        bottom=0.011,
                        right =1.000,
                        top   =0.911,
                        wspace=0.000,
                        hspace=0.200)
    # cmap=plt.cm.viridis
    # norm = matCol.BoundaryNorm(np.arange(n_clusters+1), cmap.N)
    im2 = ax[3].imshow(manual_labels_burned_area)
    ax[3].set_title('binary burned area map truth')


    #plot the DLCF RGB but zero out the blue channel with 0.67 to see w/o smoke
    rgb_no_b = 1.5*rgb_OG
    rgb_no_b[:,:,2] = 0
    ax[4].imshow(rgb_no_b)
    ax[4].set_title('DLCF RGB but zero out\n blue channel (smokeless)')

    plt.tight_layout()
    plt.show()











    # f, ax = plt.subplots(ncols=1, figsize=(35,20))
    #
    # ax.imshow(1.5*rgb_OG)
    # im = ax.imshow(burnscar_mask, cmap='jet', vmax=0.25, alpha=0.75)
    # area_total = 0
    # for i in range(len(col1)):
    #     width, length = col2[i]-col1[i], row2[i]-row1[i]
    #     area_total += width*length
    #     rect = patches.Rectangle((col1[i], row1[i]), width, length, linewidth=2,\
    #                              edgecolor='r', facecolor='none')
    #
    #     ax.add_patch(rect)
    #
    # ticks=np.around(np.arange(-0.4,1.1, 0.1), decimals=1)
    # cbar2 = plt.colorbar(im, ax=ax, ticks=ticks, orientation='horizontal')
    # tick_labels = [str(x) for x in ticks]
    # cbar2.set_ticklabels(tick_labels)
    #
    # ax.set_title('Day Land Cloud Fire RGB\n[2.25, 0.86, 0.67]µm\nManual Labels (Red Squares) ')
    # f.suptitle('NOAA-20 VIIRS Valid 08.10.2021 Composited & Cloud-Cleared Over Previous 8 Days')
    #
    #
    # ax.set_xticks([])
    # ax.set_yticks([])
    #
    # # ax[3].imshow(manual_labels_burned_area)
    #
    # plt.tight_layout()
    # plt.show()
