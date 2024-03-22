import h5py
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
import warnings
from burn_scar_composites import get_burn_scar_RGB,\
                                 get_normalized_burn_ratio,\
                                 get_BRF_lat_lon,\
                                 get_BRF_RGB,\
                                 get_burn_scar_composite,\
                                 flip_arr

home = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/VIIRS_database/databases/daily_composite_files/'

composite_path = home+'daily_DLCF_composites_2021_west_CONUS_all_days.h5'
dlcfs = []
with h5py.File(composite_path,'r') as hf_daily_composites:
    timestamps = list(hf_daily_composites.keys())[-5:]
    print('Collecting composites...')
    for time in timestamps:
        dlcfs.append(1.5*hf_daily_composites[time][:])
    print('Done collecting composites')

pbsms = []
print('Calculating burn scar masks...')
for dlcf in dlcfs:
    pbsms.append(get_burn_scar_composite(np.copy(dlcf[:,:,1]), np.copy(dlcf[:,:,0])))
print('Done calculating burn scar masks')

'''animate'''
print('Animating into GIF...')
# Generate 50 random 2D arrays
# Create a figure and axis
fig, ax = plt.subplots()
# Initialize the plot
ax.set_axis_off()

im0 = ax.imshow(dlcfs[0])
im = ax.imshow(pbsms[0], cmap='jet', vmin=-0.4, vmax=1)
# Set the initial title
title = ax.set_title(timestamps[0])
# Update function for each frame
def update(frame):
    im0.set_array(dlcfs[frame])
    im.set_array(pbsms[frame])  # Update the plot with new array
    title.set_text(f"{timestamps[frame]}")  # Update the title with frame number
    if frame%20 or frame==(len(pbsms)-1):
        print('{}/{}...'.format(frame, len(pbsms)))
    return im0, im, title
# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(pbsms), interval=500)

# Save the animation as an mp4 file
ani.save(home+'daily_DLCF_composites_2021_west_CONUS_all_days.gif', dpi=300)
print('Done!')
# Show the animation (optional)
# plt.show()
