'''
Author: Javier Alfredo Villegas Bravo
Affiliation: UMD-CISESS / NOAA NWS OPG
Date modified: 08/07/2024

Purpose: Automate the generation of the burn scar mask from download to final product file ready for end users in GIS and AWIPS-2
'''


def download_data(start_date, end_date, max_lat, min_lat, max_lon, min_lon, download_dir, which_satellite="j01"):
    from download_viirs_from_ssec_repo import download_viirs_granules

    # will need to make a new download code for the NOAA site that Batzli and Losos use
    download_viirs_granules(start_date, end_date, download_dir, which_satellite="j01")

def build_database(VJX09_dir, VJX03_dir, start_date, end_data, save_path):
    from build_burn_scar_VIIRS_database import build_burn_scar_database
    build_burn_scar_database(VJX09_dir, VJX03_dir, start_date, end_data, save_path)

def create_product(database):
    from create_product import get_daily_composites

    get_daily_composites(viirs_database_file, daily_composite_dir)

def get_pbsm(daily_composite_file):

    from burn_scar_composites import get_burn_scar_composite
    import h5py

    daily_pbsm = {}
    with h5py.File(daily_composite_file, 'r') as hf_composites:
        day_composites = list(hf_composites.keys())

        for day_comp in day_composites:
            R_M7  = hf_composites[day_comp][:,:,1]
            R_M11 = hf_composites[day_comp][:,:,0]
            daily_pbsms[f'{day_comp}_pbsm_daily_composite'] = \
                                get_burn_scar_composite(R_M7, R_M11)

    return daily_pbsms

def unet_cnn_model(daily_composite_file):
    from unet_cnn_model import get_unet_cnn_model

    '''
    this code comes from Ainsley's saved model to run new predictions through

    in the future we can also set up a way to added more training data to a previously
    trained model
    '''

def gis_file_formatting(daily_compiste_file, final_product_save_path):
    from file_formatting import make_gis_rdy_file

    make_gis_rdy_file(daily_compiste_file, final_product_save_path)

if __name__=="__main__":

    # define data to download and environment variables
    start_date, end_date = '07/24/2024', '08/01/2024'
    max_lat, min_lat     = 40.521, 39.685
    max_lon, min_lon     = -121.319, -122.220 
    home_dir             = '/scratch/zt1/project/vllgsbr2-prj/raw_data_burnscar/data'
    download_dir         = f'{home_dir}/noaa_20_viirs/park_fire_07_24_2024'
    VJX09_dir, VJX03_dir = f'{download_dir}/VJ109', f'{download_dir}/VJ103'
    database_file        = f'{home_dir}/'
    daily_composite_file = f'{home_dir}/'
    burnscar_mask_file   = f'{home_dir}/'
    lat_lon_file         = f'{home_dir}/'

    download_data(start_date, end_date, max_lat, min_lat, max_lon, min_lon, download_dir, which_satellite="NOAA-20")

    build_database(VJX09_dir, VJX03_dir, start_date, end_data)

    create_product(database)

    unet_cnn_model(daily_composite_file)

    gis_file_formatting(burn_scar_mask, lat, lon)
