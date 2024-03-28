'''
this class has a grid object that data is fit to using a reference lat/lon
grid and the data's lat lon grid. the class will have a function to then create
running composites of the gridded data in time.

The first composite will be a VIIRS burn scar product on a 16 day running cycle
(maybe 8 depending cloudiness). The composite should reveal information of the
existance of a burn scar and the age of the burn scar which is related to that
ground's permiability in a flooding context.
'''

class CommonGrid:

    def __init__(self, lat_target, lon_target):
        '''
        lat/lon reference is the reference grid
        '''
        #<160MB to store in memory
        lat_target, lon_target = lat_target, lon_target
        #<1.5GB to store in memory
        burn_scar_RGB_regridded_and_stacked_for_time_range = []

    def regrid(self, data, lat_source, lon_source):
        '''
        @param {nd.array} data
        @param {nd.array} lat_source
        @param {nd.array} lon_source

        all inputs are the same MxN but data can have depth K

        @return regridded @param {nd.array} data
        '''
        from regrid import regrid

        if data.ndim >= 3:
            data_temp = np.empty(np.shape(data))

            for i in range(data.ndim):
                data_temp[:,:,i] = regrid(data[:,:,i], self.lat_target,\
                                         self.lon_target, lat_source, lon_source)
            return data_temp
        else:
            return regrid(data, self.lat_target, self.lon_target, lat_source,\
                                                            lon_source)

    def get_burn_scar_RGB(self, VIIRS_database, start_DOY, start_year, composite_time_range):
        '''
        @param {str} VIIRS_database: holds pre processed granules
        @param {int} start_DOY     : Julian DOY , 1<=start_DOY<=365
        @param {int} start_year    : year of the data (range depends on dataset)
        @param {int} composite_time_range : Number of days composited after start_DOY

        @return {void} data on new grid gets saved into object
        '''
        import h5py

        with h5py.File(VIIRS_database, 'r') as hf_VIIRS_database:
            time_stamps = list(hf_VIIRS_database.keys())

            #grab days that fall within the composite range
            #take care of edge case when composite goes into the next/previous year
            DOY_end = DOY_start + composite_time_range
            if DOY_start < DOY_end:
                subsettedByDOY_time_stamps = [x for x in time_stamps \
                                     if x[4:7]>=DOY_start & x[4:7]<=DOY_end]
            elif DOY_start > DOY_end:
                subsettedByDOY_time_stamps = \
                [x for x in time_stamps if x[4:7]>=DOY_start & x[4:7]<=365 & x[:4]] +\
                [x for x in time_stamps if x[4:7]>=0 & x[4:7]<=(365-DOY_end)]

            else DOY_start == DOY_end:
                raise ValueError('DOY.')


            for time_stamp in subsettedByDOY_time_stamps:
                burn_scar_RGB = hf_VIIRS_database[time_stamp+'/burn_scar_RGB']
                lat_source    = hf_VIIRS_database[time_stamp+'/lat']
                lon_source    = hf_VIIRS_database[time_stamp+'/lon']
                burn_scar_RGB = regrid(self.lat_target, self.lon_target,\
                                       burn_scar_RGB, lat_source, lon_source)

                self.burn_scar_RGB_regridded_and_stacked_for_time_range.append(burn_scar_RGB)

                #populate with latest burn scar RGB and then remove clouds
                #then fill in removed clouds with most recent confident clear pixels
                #stop when pixels are 99% filled or ran out of data in date range

                composite = self.burn_scar_RGB_regridded_and_stacked_for_time_range[0]
                cloudy = ''#cloud mask
                composite[composite==cloudy]= -999
