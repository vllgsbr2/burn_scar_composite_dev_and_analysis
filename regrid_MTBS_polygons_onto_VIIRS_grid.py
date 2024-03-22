import sys
import shapely
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gp
import numpy as np
import h5py
from regrid import regrid_latlon_source2target

# open lat/lon VIIRS file (might just use the databsefile...)
commongrid_file = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/'
commongrid_file+= 'VIIRS_database/databases/Grids_West_CONUS_new.h5'
with h5py.File(commongrid_file, 'r') as hf_west_conus_grid:
    common_grid_lat = hf_west_conus_grid['Geolocation/Latitude'][:].astype(dtype=np.double)
    common_grid_lon = hf_west_conus_grid['Geolocation/Longitude'][:].astype(dtype=np.double)

common_grid_lon = np.flip(common_grid_lon, axis=1)*-1

#open MTBS polygons and load in geopandas polygons
mtbs_file = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/'
mtbs_file+= 'mtbs_perimeter_data/mtbs_perims_DD.shp'
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
# information about where the polygon is grom by rpeating rows, but changing the
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

# sys.exit()



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
max_radius = 100. #maybe 100m becuase polygons are at 30m res
polygon_lats_regridded = []
polygon_lons_regridded = []

for i in range(shp_mtbs_2021_exploded.head(n=n).shape[0]):
    target_lat = np.copy(common_grid_lat)
    target_lon = np.copy(common_grid_lon)
    source_lat = np.copy(lats_polygons[i])
    source_lon = np.copy(lons_polygons[i])

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
    # sys.exit()

f, ax = plt.subplots(nrows=1, ncols=2)

for i in range(len(polygon_lats_regridded)):
    polygon_lons_regridded[i][polygon_lons_regridded[i]==-999] = np.nan
    polygon_lats_regridded[i][polygon_lats_regridded[i]==-999] = np.nan

    ax[0].imshow(polygon_lats_regridded[i])
    ax[1].imshow(polygon_lons_regridded[i])
plt.show()
