from matplotlib import pyplot as plt
plt.switch_backend('qtAgg') #plots on correct screen and ctrl + c exits plot window
import numpy as np
from sklearn.mixture import GaussianMixture
from burn_scar_composites import get_burn_scar_composite,\
                                 get_normalized_burn_ratio,\
                                 get_normalized_differenced_vegetation_index
import h5py
import configparser

config = configparser.ConfigParser()
config.read('config_filepaths.txt')
home_dir = config['file paths mac']['database dir']
databse_name = 'daily_DLCF_composites_2021_west_CONUS_all_days.h5'

with h5py.File(home_dir + databse_name, 'r') as hf_database:
    X = hf_database['08.10.2021'][:]

rgb_OG = np.copy(X)

#subset X to focus on ROI
# r1,r2, c1,c2 = 1555,1841, 0  ,600 # general large case
r1,r2, c1,c2 = 1260,1331, 370,460 # focused case on burn scar
# r1,r2, c1,c2 = 1292,1304, 407,412 # only burn scar pixels case
# r1,r2, c1,c2 = 1620,1740, 120,220 # burn scar next to dixie

X = X[r1:r2, c1:c2, :]

#take nan values out of rgb
#(if at least 1 channel has a nan, kick all three channels in that x,y position)
rgb_plot = np.copy(X)
rgb = np.copy(X)
shape_X             = X.shape
X                   = X.reshape((shape_X[0]*shape_X[1], shape_X[2]))
not_nan_idx         = np.where(~np.isnan(X))
unique_not_nan_rows = np.unique(not_nan_idx[0])
X                   = X[unique_not_nan_rows,:]

#scatter plot for whole image; just for inspection, not for ML model
X_OG = np.copy(rgb_OG)
shape_X_OG             = X_OG.shape
X_OG                   = X_OG.reshape((shape_X_OG[0]*shape_X_OG[1], shape_X_OG[2]))
not_nan_idx_OG         = np.where(~np.isnan(X_OG))
unique_not_nan_rows_OG = np.unique(not_nan_idx_OG[0])
X_OG                   = X_OG[unique_not_nan_rows_OG,:]

#ancillary composites to view
Rbrn, Rveg, Rvis = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
NDVI             = get_normalized_differenced_vegetation_index(Rvis, Rveg)
NBR              = get_normalized_burn_ratio(Rveg, Rbrn)
burnscar_mask    = get_burn_scar_composite(Rveg, Rbrn)

# number of gaussian distributions to fit to the data in X
n_componentsint = 3

# #the GaussianMixture().fit() will then create the distributions to match it
# X=X[:5000,:]
# model = GaussianMixture(n_components=n_componentsint).fit(X)
#
# print(model)

# with open('./burn_scar_location_coordinates.txt', 'w') as txt_burnscar_coords:
plt.rcParams.update({'font.size': 22})
plt.style.use('dark_background')
plt.figure(figsize=(35,20))
# rgb_OG[:,:,2]=0
# rgb_OG[:,:,1]*=1.5
composite = np.prod(rgb_OG, axis=2)
plt.imshow(rgb_OG, cmap='gist_ncar')
plt.tight_layout()

# f, ax = plt.subplots(ncols=4, figsize=(35,20))
#
# my_cmap = plt.cm.gist_ncar
# my_cmap.set_under('k',1)
# xmin, xmax, ymin, ymax = X_OG[:,0].min(), X_OG[:,0].max(),\
#                          X_OG[:,1].min(), X_OG[:,1].max()
#
# ax[0].sharex(ax[1])
# ax[0].sharey(ax[1])
#
# n=-1
# binWidth, binLength = 0.02, 0.02
# hist = ax[0].hist2d(X[:n,0], X[:n, 1],\
# bins = [np.arange(xmin, xmax, binWidth), np.arange(ymin, ymax, binLength)],\
# cmap=my_cmap, vmin=1, density=True)
# # print(hist[0].max())
#
# hist1 = ax[1].hist2d(X_OG[:n,0], X_OG[:n, 1],\
# bins = [np.arange(xmin, xmax, binWidth), np.arange(ymin, ymax, binLength)],\
# cmap=my_cmap, vmin=1, density=True)
#
#
# # rgb_OG[:,:,1:2] = rgb_OG[:,:,1:2]**2
# ax[2].imshow(rgb_plot)
# ax[3].imshow(rgb_OG)
#
# ax[0].set_xlabel('2.25 µm BRF')
# ax[0].set_ylabel('0.86 µm BRF')
# ax[0].set_title('2D Hist')
# ax[0].grid(linestyle='dashed', linewidth=2)
# ax[0].set_xlim([xmin, xmax])
# ax[0].set_ylim([ymin, ymax])
#
#
# ax[1].set_xlabel('2.25 µm BRF')
# ax[1].set_ylabel('0.86 µm BRF')
# ax[1].set_title('2D Hist Whole Domain')
# ax[1].grid(linestyle='dashed', linewidth=2)
# ax[1].set_xlim([xmin, xmax])
# ax[1].set_ylim([ymin, ymax])
#
# ax[2].set_title('DLCF RGB 2D Hist Domain')
# ax[3].set_title('DLCF RGB Entire Domain')

# burnscar_x = 0
# def on_xlims_change(event_ax):
#     new_row_coords = event_ax.get_xlim()
#     new_col_coords = event_ax.get_ylim()
#     # print("updated xlims: ", new_row_coords)
#     print('record this change? y/n')
#     stayinloop = True
#     while stayinloop==True:
#         answer = input()
#         if answer == 'y':
#             global burnscar_x
#             txt_burnscar_coords.writelines(str(burnscar_x))
#             txt_burnscar_coords.writelines(str(new_row_coords))
#             txt_burnscar_coords.writelines(str(new_col_coords))
#             burnscar_x = burnscar_x + 1
#             stayinloop = False
#         elif answer =='n':
#             stayinloop = False
#         else:
#             print('plz enter valid str, y or n for yes no respectively')
#
#
# # def on_ylims_change(event_ax):
# #     new_col_coords = event_ax.get_ylim()
# #     print("updated ylims: ", new_col_coords)
# #     print('record this change? y/n')
# #     stayinloop = True
# #     while stayinloop==True:
# #         answer = input()
# #         if answer == 'y':
# #             txt_burnscar_coords.writelines(str(new_col_coords))
# #             stayinloop = False
# #         elif answer =='n':
# #             stayinloop = False
# #         else:
# #             print('plz enter valid str, y or n for yes no respectively')
#
#     # print("updated ylims: ", event_ax.get_ylim())
#
# ax[2].callbacks.connect('xlim_changed', on_xlims_change)
# # ax[2].callbacks.connect('ylim_changed', on_ylims_change)



plt.show()
plt.close()
