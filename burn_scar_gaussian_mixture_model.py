from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import matplotlib.colors as matCol
import numpy as np
import h5py
import sys
import os
import configparser
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from burn_scar_composites import get_burn_scar_composite,\
                                 get_normalized_burn_ratio,\
                                 get_normalized_differenced_vegetation_index

def dict_to_string(my_dict):
    # Convert the dictionary to a string with no spaces, quotes, colons, commas, or curly braces, separate the key-value pairs with underscores, and put periods between each key and value
    my_string = ".".join([f"{k}_{v}" for k, v in my_dict.items()])

    # my_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    # my_string = dict_to_string(my_dict)
    # print(my_string)  # Output: key1_value1.key2_value2.key3_value3

    return my_string

def train_GMM(n_clusters = 5):

    print('grabbing data to train on')
    #plots on correct screen and ctrl + c exits plot window
    plt.switch_backend('qtAgg')

    #open up database to access DLCF RGB composite
    config = configparser.ConfigParser()
    config.read('config_filepaths.txt')
    home_dir = config['file paths mac']['database dir']
    databse_name = 'daily_DLCF_composites_2021_west_CONUS_all_days.h5'

    with h5py.File(home_dir + databse_name, 'r') as hf_database:
        rgb = hf_database['08.10.2021'][:]

    rgb_OG_copy = np.copy(rgb)

    #ancillary composites to view
    Ref_2250nm, Ref_0860nm, Ref_0670nm = rgb_OG_copy[:,:,0].flatten(),\
                                         rgb_OG_copy[:,:,1].flatten(),\
                                         rgb_OG_copy[:,:,2].flatten()
    # NDVI             = get_normalized_differenced_vegetation_index(Rvis, Rveg)
    # NBR              = get_normalized_burn_ratio(Rveg, Rbrn)
    # burnscar_mask    = get_burn_scar_composite(Rveg, Rbrn)

    print('training gaussian mixture model')

    Ref_2250nm = Ref_2250nm[np.isnan(Ref_2250nm)==False]
    Ref_0860nm = Ref_0860nm[np.isnan(Ref_0860nm)==False]
    Ref_0670nm = Ref_0670nm[np.isnan(Ref_0670nm)==False]
    X = np.concatenate((Ref_2250nm.reshape(-1,1),\
                        Ref_0860nm.reshape(-1,1),\
                        Ref_0670nm.reshape(-1,1)), axis=1)

    # Fit Gaussian Mixture Model

    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(X)

    print('using model to assign clusters')
    # Predict the cluster labels
    labels = gmm.predict(X)

    # Print the cluster centers
    print("Cluster Centers:")
    print(gmm.means_)

    # Print the cluster labels
    print("Cluster Labels:")
    print(labels)

    print('writing model to disk')
    #save results to file

    # Create an HDF5 file
    inputs = {'R_2250': 1, 'R_0860': 1, 'R_0670': 1, 'VZA': 0,'cosSZA': 0,\
              'RAA'   : 0, 'NDVI'  : 0, 'NBR'   : 0, 'pbsm': 0}
    inputs = dict_to_string(inputs)
    params = {'nClusters': '{:02d}'.format(n_clusters)}
    params = dict_to_string(params)
    results_filename = 'burn_scar_gmm_results.{}.{}.h5'.format(params, inputs)

    home_path = '/Users/javiervillegasbravo/Documents/NOAA/'
    home_path +='burn_scar_proj/VIIRS_database/databases/GMM_results/'
    with h5py.File(home_path + results_filename, 'w') as file:
        gmm_params = file.create_group('gmm_params')
        gmm_params.create_dataset('means'      , data=gmm.means_)
        gmm_params.create_dataset('covariances', data=gmm.covariances_)
        gmm_params.create_dataset('weights'    , data=gmm.weights_)

        # Save cluster labels
        file.create_dataset('labels', data=labels)
        # Save data points
        file.create_dataset('data', data=X)
        #save the RGB into the file for inspection
        file.create_dataset('DLCF_RGB', data=rgb_OG_copy)


def plot_GMM_clusters(X, means, covariances, weights):
    # plot the results
    print('plotting results')
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Initialize an empty list to store the contour plot artists

    # Generate histogram of data points colored by cluster labels
    hist, xedges, yedges, im = ax.hist2d(X[:, 0], X[:, 1], bins=500, norm=mpl.colors.LogNorm(), cmap='jet')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Counts')

    # Plot cluster centers
    ax.scatter(means[:, 0], means[:, 1], marker='x', color='pink', s=100,
                label='Cluster Centers')

    # plot the gaussian curves as contours over the plot to show the extent and uncertainty
    # of the cluster centers

    # Generate x and y values for the grid
    x = np.linspace(-10, 10, 1000)
    y = np.linspace(-10, 10, 1000)
    X, Y = np.meshgrid(x, y)

    # Function to update the plot for each frame
    def update(frame):
        # ax.cla()

        # Calculate the probability density function for each Gaussian component
        Z = np.zeros_like(X)
        n_clusters = 5
        # for i in range(5):
        i = frame
        mean   = means[i]
        cov    = covariances[i]
        weight = weights[i]
        Z += weight * multivariate_normal.pdf(np.dstack((X, Y)), mean=mean, cov=cov)

        # Plot the contours
        CS = ax.contour(X, Y, Z, levels=5, colors='black', alpha=0.5)
        # ax.clabel(CS, CS.levels, inline=True, fontsize=10)

        ax.set_xlabel('Ref 2.25µm')
        ax.set_ylabel('Ref 0.86µm')
        ax.set_title('Gaussian Mixture Model w/ {} Clusters & {} Data Points'.format(5, len(labels)))
        plt.legend()


    # Create the animation
    animation = FuncAnimation(fig, update, frames=5, interval=1000, repeat=True)

    plt.show()

def plot_segmented_classified_images_for_analysis(DLCF_RGB, labels, n_clusters=5, fname=None):
    # plot the RGB and next to it plot the categories from the GMM to compare what
    # features got assigned to what, hopefully we see burn scars delineated
    # Create a figure and axes for the 2-panel plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16,9))
    mpl.rcParams.update({'font.size': 14})

    im1 = ax[0].imshow(1.5*DLCF_RGB)
    ax[0].set_title('DLCF RGB 08/10/2021')

    # Plot data2 in the second panel
    DLCF_RGB_shape = DLCF_RGB.shape
    labeled_DLCF_RGB = np.zeros((DLCF_RGB_shape[0], DLCF_RGB_shape[1]))
    labeled_DLCF_RGB[np.where(np.isnan(DLCF_RGB[:,:,0])==False)] = labels

    values = np.arange(n_clusters+1)
    cmap=plt.cm.viridis
    norm = matCol.BoundaryNorm(np.arange(n_clusters+1), cmap.N)
    im2 = ax[1].imshow(labeled_DLCF_RGB, cmap=cmap, norm=norm)
    ax[1].set_title('Segmented by {} GMM Clusters'.format(n_clusters))

    cax            = fig.add_axes([0.92, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    colorbar       = plt.colorbar(im2, cax=cax)
    tick_locations = np.arange(0.5, n_clusters+.5, 1)  # Custom tick locations
    colorbar.set_ticks(tick_locations)
    tick_labels    = np.arange(1,n_clusters+1)  # Custom tick labels
    colorbar.set_ticklabels(tick_labels)

    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])

    if fname==None:
        # Show the plot
        plt.show()
    else:
        plt.savefig('{}.png'.format(fname))

def open_GMM_results_file(results_file):

    with h5py.File(results_file, 'r') as file:
        # Read GMM parameters
        gmm_params  = file['gmm_params']
        means       = gmm_params['means'][:]
        covariances = gmm_params['covariances'][:]
        weights     = gmm_params['weights'][:]
        labels      = file['labels'][:]
        X           = file['data'][:]
        DLCF_RGB    = file['DLCF_RGB'][:]

        return gmm_params  ,\
               means       ,\
               covariances ,\
               weights     ,\
               labels      ,\
               X           ,\
               DLCF_RGB


if __name__ == "__main__":

    # read in file
    home_path = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/VIIRS_database/'
    results_files  = np.sort(os.listdir(home_path+'databases/GMM_results/'))
    results_file   = [home_path + 'databases/GMM_results/' + x for x in results_files]
    fig_file_names = [home_path + 'figures/GMM_results/'   + x[:-3] for x in results_files]

    for i in range(2,15):
        # print('\n******** Training GMM w/ {} clusters *******************\n'.format(i))
        # train_GMM(n_clusters=i)
        # plot_GMM_clusters(X, means, covariances, weights)
        print('\n******** Making Figures GMM w/ {} clusters *******************\n'.format(i))
        gmm_params  ,\
        means       ,\
        covariances ,\
        weights     ,\
        labels      ,\
        X           ,\
        DLCF_RGB    = open_GMM_results_file(results_file[i-2])
        plot_segmented_classified_images_for_analysis(DLCF_RGB, labels,\
                                                      n_clusters=i    ,\
                                                      fname=fig_file_names[i-2])
