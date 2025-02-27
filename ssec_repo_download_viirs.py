def download_viirs_granules(start_date, end_date, download_dir, which_satellite="j01"):
    
    ssec_viirs_repo_url = 'https://bin.ssec.wisc.edu/pub/eosdb/'

    sat_list = ['npp', 'j01', 'j02']
    instrument_list = ['viirs']
    dates_avail = # probably can url request whats in here
                  # format is YYYY_MM_DD_DOY_HHMM as directory

    # then we want the edr data
    # then check if this file is avail
    # SurfRefl_v1r2_j01_s202408061605528_e202408061607173_c202408061656290.nc

    # full path built to files
    # https://bin.ssec.wisc.edu/pub/eosdb/j01/viirs/2024_08_06_219_1605/edr/SurfRefl_v1r2_j01_s202408061605528_e202408061607173_c202408061656290.nc


    '''
    to choose the correct file to download we need the local time of the Pacific
    and mountain timezones. Using python's datetime we can automatically
    get day light savings. That should cover the western US from -125 to -100.
    To get the 30 to 49 deg lat we need to look +- 6.44 minutes from the 
    over pass time of 1:30pm local time. If we assume 7km/sec speed of the sat
    then 6.44 min should cover 2703 km north and south of the center of the 1:30pm
    domain. That should cover all data within -125W,-100W, 30N, 49N. May need to
    mess with that however for each npp, j01 and j02. Maybe don't bother with npp
    since it's on its way out.
    '''
