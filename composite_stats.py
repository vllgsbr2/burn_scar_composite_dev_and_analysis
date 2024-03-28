fontsize=22
plt.rcParams['font.size'] = fontsize
plt.style.use('dark_background')
fig_scat, ax_scat = plt.subplots(nrows=2, ncols=2)
r1,r2, c1,c2 = 378,563, 1527,1710
r1,r2, c1,c2 = 240,746, 1235,1836
# r1,r2, c1,c2 = 424,440, 1586,1598
NBR_flat = NBR[r1:r2,c1:c2].flatten()
bins=100
cmap='jet'
vmin, vmax = 0,500
burn_scar_RGB_unscaled = burn_scar_RGB/2.5
ax_scat[0,0].hist2d(NBR_flat, NDVI[r1:r2,c1:c2].flatten(), bins=bins, cmap=cmap, vmin=vmin, vmax=vmax)
ax_scat[0,1].hist2d(NBR_flat, burn_scar_RGB_unscaled[r1:r2,c1:c2,0].flatten(), bins=bins, cmap=cmap, vmin=vmin, vmax=vmax)
ax_scat[1,0].hist2d(NBR_flat, burn_scar_RGB_unscaled[r1:r2,c1:c2,1].flatten(), bins=bins, cmap=cmap, vmin=vmin, vmax=vmax)
im = ax_scat[1,1].hist2d(NBR_flat, burn_scar_RGB_unscaled[r1:r2,c1:c2,2].flatten(), bins=bins, cmap=cmap, vmin=vmin, vmax=vmax)

ylabels = ['NDVI', '2.25 micron', '0.86 micron', '0.67 microns']
for i, a in enumerate(ax_scat.flat):
    a.set_xlabel("NBR", fontsize=fontsize)
    a.set_ylabel(ylabels[i], fontsize=fontsize)

cb_ax = fig_scat.add_axes([0.15, 0.05, 0.7, 0.03]) #[left, bottom, width, height]
cbar = fig_scat.colorbar(im[3], cax=cb_ax, ticks = np.arange(vmin, vmax, 20),orientation="horizontal")

plt.tight_layout()
plt.show()


################################################################################

from matplotlib.widgets import RangeSlider
burn_scar_RGB_unscaled = burn_scar_RGB/2.5
img = burn_scar_RGB_unscaled[r1:r2,c1:c2,:]
plt.style.use('dark_background')
fontsize=16
plt.rcParams['font.size'] = fontsize
fig, axs = plt.subplots(2, 4, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)


img_full = axs[0,0].imshow(2.5*img)
axs[1,0].axis('off')
for i, a in enumerate(axs.flat):
    a.set_xticks([])
    a.set_yticks([])
    if i==3:
        break

img_225 = img[:,:,0]
img_086 = img[:,:,1]
img_067 = img[:,:,2]

im_225 = axs[0,1].imshow(img[:,:,0])
axs[1,1].hist(img_225.flatten(), bins='auto')
axs[1,1].set_title('2.25 microns')
im_086 = axs[0,2].imshow(img[:,:,1])
axs[1,2].hist(img_086.flatten(), bins='auto')
axs[1,2].set_title('0.86 microns')
im_067 = axs[0,3].imshow(img[:,:,2])
axs[1,3].hist(img_067.flatten(), bins='auto')
axs[1,3].set_title('0.67 microns')

im_225.cmap.set_under('r')
im_225.cmap.set_over('r')
im_086.cmap.set_under('r')
im_086.cmap.set_over('r')
im_067.cmap.set_under('r')
im_067.cmap.set_over('r')

# Create the RangeSlider
slider_ax_225 = plt.axes([0.20, 0.03, 0.60, 0.03])#[left, bottom, width, height]
slider_225 = RangeSlider(slider_ax_225, "2.25 micron thresh", img_225.min(), img_225.max())
slider_ax_086 = plt.axes([0.20, 0.06, 0.60, 0.03])
slider_086 = RangeSlider(slider_ax_086, "0.86 micron thresh", img_086.min(), img_086.max())
slider_ax_067 = plt.axes([0.20, 0.09, 0.60, 0.03])
slider_067 = RangeSlider(slider_ax_067, "0.67 micron thresh", img_067.min(), img_067.max())

# Create the Vertical lines on the histogram
lower_limit_line_225 = axs[1,1].axvline(slider_225.val[0], color='r')
upper_limit_line_225 = axs[1,1].axvline(slider_225.val[1], color='r')
lower_limit_line_086 = axs[1,2].axvline(slider_086.val[0], color='r')
upper_limit_line_086 = axs[1,2].axvline(slider_086.val[1], color='r')
lower_limit_line_067 = axs[1,3].axvline(slider_067.val[0], color='r')
upper_limit_line_067 = axs[1,3].axvline(slider_067.val[1], color='r')




def update_225(val):
    # The val passed to a callback by the RangeSlider will
    # be a tuple of (min, max)

    # Update the image's colormap
    im_225.norm.vmin = val[0]
    im_225.norm.vmax = val[1]


    # Update the position of the vertical lines
    lower_limit_line_225.set_xdata([val[0], val[0]])
    upper_limit_line_225.set_xdata([val[1], val[1]])

    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()
    bs_RGB = np.copy(img)
    bs_RGB[bs_RGB<val[0] or bs_RGB>val[1]]=0

    return val

def update_086(val):
    # Update the image's colormap
    im_086.norm.vmin = val[0]
    im_086.norm.vmax = val[1]

    # Update the position of the vertical lines
    lower_limit_line_086.set_xdata([val[0], val[0]])
    upper_limit_line_086.set_xdata([val[1], val[1]])

    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()
    bs_RGB = np.copy(img)
    bs_RGB[bs_RGB<val[0] or bs_RGB>val[1]]=0

    return val

def update_067(val):
     # Update the image's colormap
    im_067.norm.vmin = val[0]
    im_067.norm.vmax = val[1]

    # Update the position of the vertical lines
    lower_limit_line_067.set_xdata([val[0], val[0]])
    upper_limit_line_067.set_xdata([val[1], val[1]])

    # Redraw the figure to ensure it updates
    fig.canvas.draw_idle()
    bs_RGB = np.copy(img)
    bs_RGB[bs_RGB<val[0] or bs_RGB>val[1]]=0

    return val

val_225 = slider_225.on_changed(update_225)
val_086 = slider_086.on_changed(update_086)
val_067 = slider_067.on_changed(update_067)
plt.show()













































################################################################################

import h5py
import numpy as np

with h5py.File(database_file, 'r') as hf_database:
    time_stamps   = list(hf_database.keys())
    dataset_names = ['BRF_RGB', 'NBR', 'burn_scar_RGB', 'cldmsk',\
                     'land_water_mask', 'lat', 'lon']

    burn_scar_RGB = hf_database['{}/{}'.format(time_stamps[0], dataset_names[2])]
    BRF_2250nm    = burn_scar_RGB[:,:,0]
    BRF_0865nm    = burn_scar_RGB[:,:,1]

    #remove -999 missing data (leave clouds, take out water)
    BRF_2250nm    = BRF_2250nm[BRF_2250nm != -999]
    BRF_0865nm    = BRF_0865nm[BRF_0865nm != -999]

    num_bins = 128
    hist_BRF_2250nm = np.histogram(BRF_2250nm, bins=num_bins, density=False)[0]
    hist_BRF_0865nm = np.histogram(BRF_0865nm, bins=num_bins, density=False)[0]

    hist_2D_BRF2250nm_VS_BRF0865nm = histogram2d(BRF_2250nm, BRF_0865nm,\
                                            bins=num_bins, density=False)[0]

    for ts in time_stamps[1:]:

        burn_scar_RGB = hf_database['{}/{}'.format(ts, dataset_names[2])]
        BRF_2250nm    = burn_scar_RGB[:,:,0]
        BRF_0865nm    = burn_scar_RGB[:,:,1]

        #remove -999 missing data (leave clouds, take out water)
        BRF_2250nm    = BRF_2250nm[BRF_2250nm != -999]
        BRF_0865nm    = BRF_0865nm[BRF_0865nm != -999]

        num_bins = 128
        hist_BRF_2250nm = np.histogram(BRF_2250nm, bins=num_bins, density=False)[0]
        hist_BRF_0865nm = np.histogram(BRF_0865nm, bins=num_bins, density=False)[0]

        hist_2D_BRF2250nm_VS_BRF0865nm = histogram2d(BRF_2250nm, BRF_0865nm,\
                                                bins=num_bins, density=False)[0]





























        #
