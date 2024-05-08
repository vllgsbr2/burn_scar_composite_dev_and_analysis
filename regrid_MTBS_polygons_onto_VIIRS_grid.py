import sys
#import shapely
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gp
import numpy as np
import h5py
from regrid import regrid_latlon_source2target
from geopy.distance import geodesic
import pyresample

import time
start_time = time.time()


# open lat/lon VIIRS file (might just use the databsefile...)
home            = '/scratch/zt1/project/vllgsbr2-prj/'
commongrid_file = home + 'raw_data_burnscar/data/Grids_West_CONUS_new.h5'
with h5py.File(commongrid_file, 'r') as hf_west_conus_grid:
    common_grid_lat = hf_west_conus_grid['Geolocation/Latitude'][:].astype(dtype=np.double)
    common_grid_lon = hf_west_conus_grid['Geolocation/Longitude'][:].astype(dtype=np.double)

common_grid_lon = np.flip(common_grid_lon, axis=1)*-1

#open MTBS polygons and load in geopandas polygons
mtbs_file = home + 'raw_data_burnscar/data/mtbs_perimeter_data/mtbs_perims_DD.shp'
shp_mtbs = gp.read_file(mtbs_file)
# shp_mtbs_geometry = shp_mtbs['geometry']
# print(shp_mtbs.columns)
# print(shp_mtbs['geometry'].head())
#turn the polygon GeoSeries into a list of lats and lons

# sort by dates to get 2021 fires
shp_mtbs['Ig_Date'] = pd.to_datetime(shp_mtbs['Ig_Date'], format='%Y-%m-%d')
shp_mtbs_sorted = shp_mtbs.sort_values('Ig_Date')
shp_mtbs_2021 = shp_mtbs_sorted[shp_mtbs_sorted['Ig_Date'].dt.year == 2021]

# # sort by num vertices (not sure y, but I like, so regrid gets faster w/run time)
# shp_mtbs_2021.sort_values('geometry', ascending=True)
#
# def coord_lister(geom):
#     multi = geom.type.startswith("Multi")
#     coords = ()
#     if not multi:
#         pass#coords = list(geom.exterior.coords)
#     elif multi:
#         coords = ()
#         for poly in list(geom.geoms):
#             coords += tuple(poly.exterior.coords)
#
#     return (coords)
#
# shp_mtbs_2021_coordinates = shp_mtbs_2021.geometry.apply(coord_lister)
# print(shp_mtbs_2021_coordinates.values)


# this line explodes the geomtries into single polygons while still retaining the
# information about where the polygon is from by repeating rows, but by changing the
# polygon
shp_mtbs_2021_exploded = shp_mtbs_2021.explode(index_parts=True)
polygons = []
n=5
for i, row in shp_mtbs_2021_exploded.head(n=n).iterrows():
        polygons.append(list(row.geometry.exterior.coords))
# print(len(polygons))
# print(polygons)

# now put the lats in a 2d list where each row is a single polygons that matches
# the row in shp_mtbs_2021_exploded and do the same for the lons
# then we can resample them to VIIRS grid using pytaf regrid
lats_polygons = []
lons_polygons = []
for poly in polygons:
    lats_polygons_temp = []
    lons_polygons_temp = []
    for coord in poly:
        lats_polygons_temp.append(coord[1])
        lons_polygons_temp.append(coord[0])
    lats_polygons.append(np.array(lats_polygons_temp, dtype=np.double))
    lons_polygons.append(np.array(lons_polygons_temp, dtype=np.double))

#print(lats_polygons[0])
#print(np.shape(lats_polygons[0]))
#sys.exit()

# Pass in two 1D source arrays per MTBS polygon to regrid, a list of lats & list
# of lons where the lat and lon match up so index 0,1 lat is associated with
# 0,1 lon coordinates
# => polygon((y1,x1),(y2,x2),...) > lats[y1,y2, ...], lon[x1,x2, ...]

#most deggraded pixel size according to
#https://agupubs.onlinelibrary.wiley.com/doi/10.1002/jgrd.50873
#https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013JD021170
# 1.5km at swath edge for 750m res only x2 degredation vs MODIS is like x5
# due to pixel aggregation technique described in
# [Imagery Algorithm Theoretical Basis Document (ATBD), 2011].
# so lets only look in a 1500m radius to improve speed, vs 6000m radius for MODIS
max_radius = 750. #grid is at 750 meter res so thi sis min dist to find closest pixel
polygon_lats_regridded = []
polygon_lons_regridded = []

'''
define function to find closest index in VIIRS grid
to given lat/lon
'''
'''
def find_closest_index(latitudes, longitudes, new_lat, new_lon):
    def distance_func(lat, lon):
        return geodesic((lat, lon), (new_lat, new_lon)).meters

    distance_vectorized = np.vectorize(distance_func)
    distances           = distance_vectorized(latitudes, longitudes)

    valid_distances    = np.where((distances <= 1500) &\
                                  (distances != 0), distances, np.inf)
    min_distance_index = np.unravel_index(np.argmin(valid_distances),\
                                              valid_distances.shape)

    return min_distance_index

# Example usage
latitudes  = np.copy(common_grid_lat)# np.array([[40.7128, 34.0522], [37.7749, 41.8781]])
longitudes = np.copy(common_grid_lon)# np.array([[-74.0060, -118.2437], [-122.4194, -87.6298]])
new_lat = 39.7509875
new_lon = -123.4098709870897

closest_index = find_closest_index(latitudes, longitudes, new_lat, new_lon)
print(closest_index)
sys.exit()
'''

# Define lat-lon grid
#lon = np.linspace(30, 40, 100)
#lat = np.linspace(10, 20, 100)
#lon_grid, lat_grid = np.meshgrid(lon, lat)

lon_grid = np.copy(common_grid_lon)
lat_grid = np.copy(common_grid_lat)

#print(lon_grid[:5,:5])
#print(lat_grid[:5,:5])


grid = pyresample.geometry.GridDefinition(lats=lat_grid, lons=lon_grid)

# Generate some random data on the grid
data_grid = np.ones(lon_grid.shape)*0

# Define some sample points *************************************************
#****************************************************************************
# replace sample points with polygon lat/lons
for i in range(n):
    my_lats = lats_polygons[i] #np.array([48.634563565, 30.335463565, 35.35463565])
    my_lons = lons_polygons[i] #np.array([-122.0, -115.2352350, -120.0345])

    start_time = time.time()
    swath = pyresample.geometry.SwathDefinition(lons=my_lons, lats=my_lats)

    # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
    _, _, index_array, distance_array = pyresample.kd_tree.get_neighbour_info(
        source_geo_def=grid, target_geo_def=swath, radius_of_influence=1500,
        neighbours=1)

    # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute
    # the 2D grid indices:
    index_array_2d = np.unravel_index(index_array, grid.shape)

    print( "Indices of nearest neighbours:", index_array_2d)
    print( "Longitude of nearest neighbours:", lon_grid[index_array_2d])
    print( "Latitude of nearest neighbours:", lat_grid[index_array_2d])
    print( "Great Circle Distance:", distance_array)

    end_time = time.time()
    execution_time = end_time - start_time
    print(i, "Execution time:",execution_time)
sys.exit()



for i in range(shp_mtbs_2021_exploded.head(n=n).shape[0]):
    target_lat = np.copy(common_grid_lat)
    target_lon = np.copy(common_grid_lon)
    source_lat = np.copy(lats_polygons[i])
    source_lon = np.copy(lons_polygons[i])
    
    # instead of regridding lat lon we could
    # try regridding the burn severity...?
    source_data_lat = np.copy(lats_polygons[i])
    source_data_lon = np.copy(lons_polygons[i])
    # print(source_data_lat)
    # sys.exit()

    polygon_lats_regridded.append(regrid_latlon_source2target(source_lat, source_lon,\
                                target_lat, target_lon, source_data_lat, max_radius=max_radius))

    polygon_lons_regridded.append(regrid_latlon_source2target(source_lat, source_lon,\
                                target_lat, target_lon, source_data_lon, max_radius=max_radius))

    print(polygon_lats_regridded[i][polygon_lats_regridded[i]!=-999])
    print(np.shape(polygon_lats_regridded))
    mtbs_file = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/'
    # sys.exit()

f, ax = plt.subplots(nrows=1, ncols=2)

for i in range(len(polygon_lats_regridded)):
    polygon_lons_regridded[i][polygon_lons_regridded[i]==-999] = np.nan
    polygon_lats_regridded[i][polygon_lats_regridded[i]==-999] = np.nan

    ax[0].imshow(polygon_lats_regridded[i])
    ax[1].imshow(polygon_lons_regridded[i])
plt.show()
