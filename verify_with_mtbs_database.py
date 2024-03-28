import geopandas as gp
from netCDF4 import Dataset
import sys
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as matCol
import os
import pandas as pd
import numpy as np
import h5py
import pyproj
from burn_scar_composites import get_burn_scar_composite,\
                                 get_normalized_burn_ratio,\
                                 get_normalized_differenced_vegetation_index

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
# plot mtbs data with western us burnscar domain
# plt.switch_backend('qtAgg')

#open burnscar GIS file and then overlay with mtbs
burnscar_file = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/VIIRS_database/databases/danielle_test_cases_GIS_files/burn_scar_mask_GIS_sample_whole_domain_valid_08.10.2021.nc'
with Dataset(burnscar_file,'r') as nc_burnscar_mask:
    # print(nc_burnscar_mask)
    burnscar_mask = nc_burnscar_mask.variables['pbsm'][:]
    spatial_ref   = nc_burnscar_mask.variables['spatial_ref'][:]
    # print(spatial_ref)
    dlcf_rbg      = nc_burnscar_mask.variables['day_land_cloud_fire_RGB'][:]
    viirs_lat_grid = nc_burnscar_mask.variables['lat_grid'][:]
    viirs_lon_grid = nc_burnscar_mask.variables['lon_grid'][:]
    # print(np.shape(dlcf_rbg))

mtbs_file = '/Users/javiervillegasbravo/Documents/NOAA/burn_scar_proj/mtbs_perimeter_data/mtbs_perims_DD.shp'
# with gp.read_file(mtbs_file,'r') as shp_mtbs:
#     print(shp_mtbs)

shp_mtbs = gp.read_file(mtbs_file)
# # print(shp_mtbs.columns)
# # print(shp_mtbs['geometry'].head())
# # print(len(shp_mtbs.iloc[0]['geometry']))
# # print first five rows for inspection, over 30k rows/shapes total tho
# # print(shp_mtbs.head())
# # print(shp_mtbs.columns)
#
# # print(shp_mtbs.loc[0,'Ig_Date'])
#
#
# # shp_mtbs.plot()
# # plt.show()
#
# sort by dates to get 2021 fires
shp_mtbs['Ig_Date'] = pd.to_datetime(shp_mtbs['Ig_Date'], format='%Y-%m-%d')
shp_mtbs_sorted = shp_mtbs.sort_values('Ig_Date')
shp_mtbs_2021 = shp_mtbs_sorted[shp_mtbs_sorted['Ig_Date'].dt.year == 2021]
print(len(shp_mtbs_2021))

# # sort by num vertices
# shp_mtbs_2021 = shp_mtbs_2021.sort_values('geometry')

# convert CRS to equal-area projection
# the length unit is now `meter`
shp_mtbs_2021 = shp_mtbs_2021.to_crs(epsg=6933)
print(shp_mtbs_2021.crs.axis_info)
sys.exit()


#print number of vertices in each polygon
df = shp_mtbs_2021
num_vertices = 0
num_polygons = 0
num_polygons_with_less_than_133_vertices = 0
temp_vertex_counter = 0

for i, row in df.iterrows():
    # It's better to check if multigeometry
    multi = row.geometry.type.startswith("Multi")
    num_polygons+=1
    if multi:
        num_polygons-=1
        temp_vertex_counter = 0
        # iterate over all parts of multigeometry
        for part in row.geometry:
            num_polygons+=1
            temp_vertex_counter += len(part.exterior.coords)
    else: # if single geometry like point, linestring or polygon
        temp_vertex_counter = len(row.geometry.exterior.coords)
    num_vertices+=temp_vertex_counter
    if temp_vertex_counter <133:
        num_polygons_with_less_than_133_vertices+=1
    temp_polygon_area = row['geometry'].area
    print('index: ', i,' area: ', temp_polygon_area/10e6, 'num vertices: ', temp_vertex_counter)

avg_num_vertices = num_vertices/num_polygons
print('avg # of vertices: ', avg_num_vertices)
print('num_polygons_with_less_than_133_vertices: ',num_polygons_with_less_than_133_vertices)
sys.exit()
# viirs_proj_str = "+proj=aea +lat_0=39.5 +lon_0=114 +lat_1=30 +lat_2=49 +x_0=0 "
# viirs_proj_str+= "+y_0=0 +datum=NAD83 +units=m +no_defs +ellps=GRS80 +towgs84=0,0,0"
# viirs_proj_CRS = pyproj.CRS.from_string(viirs_proj_str)
# viirs_proj_obj = pyproj.Proj.from_string(viirs_proj_str)
# print(viirs_proj_CRS)
# MTBS_proj_CRS = shp_mtbs.crs
# print(MTBS_proj_CRS)
# # convert MTBS to VIIRS CRS
# MTBS_2_viirs_transformer = pyproj.Transformer.from_crs(viirs_proj_CRS, MTBS_proj_CRS)
# print(MTBS_2_viirs_transformer)
# MTBS_lat_lon_polygons = shp_mtbs['geometries'].iloc[0]
# MTBS_2_viirs_transformer.transform()

# plt.figure(figsize=(10,15))
proj_cartopy = ccrs.AlbersEqualArea()
fig, ax = plt.subplots(subplot_kw={'projection': proj_cartopy})
# ax._autoscaleXon = False
# ax._autoscaleYon = False

lon_min, lon_max, lat_min, lat_max = np.nanmin(viirs_lon_grid),\
                                     np.nanmax(viirs_lon_grid),\
                                     np.nanmin(viirs_lat_grid),\
                                     np.nanmax(viirs_lat_grid)
extent = [lon_min, lon_max, lat_min, lat_max]
print(extent)
ax.set_extent(extent)

plt.pcolormesh(viirs_lon_grid[0], viirs_lat_grid[:,0], dlcf_rbg[0,:,:,2],
             transform=ccrs.AlbersEqualArea(), cmap="Greys_r")
ax.coastlines(resolution='10m', color='red', linestyle='-', alpha=1, zorder=10)
plt.show()

# fig, ax = plt.subplots(figsize=[5, 5], subplot_kw={'projection': ccrs.PlateCarree()})
# fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
# ax.set_extent([-120, -60, 15, 75])
# ax.stock_img()
# ax.coastlines()
#
# plt.show()

# #grab the lat lon pairs for each vertex
# def lonlat_to_xy_coords(lon, lat):
#     return viirs_proj_obj(lon, lat)
# def xy_coords_to_lonlat(x,y):
#     return viirs_proj_obj(x, y, inverse=True)
#
# shp_mtbs["geometry"].apply(lambda p: list(p.exterior.coords)).explode().apply(pd.Series).rename(columns=({0:"x", 1:"y"}))


# g = [i for i in df.geometry]
#
# all_coords = []
# for b in g[0].boundary: # for first feature/row
#     coords = np.dstack(b.coords.xy).tolist()
#     all_coords.append(*coords)

# print(type(viirs_proj_CRS))
# shp_mtbs_2021.plot()
# plt.show()
# print(shp_mtbs_2021['geometry'].head())
# print(shp_mtbs['geometry'].head())








# # crs = pyproj.CRS(spatial_ref)
# # print(crs)
#
#
# #convert proj str def from VIIRS data to a CRS definition
# from pyproj import Proj
# import pyproj
# import cartopy
# import cartopy.crs as ccrs
# proj_str = "proj=aea lat_0=39.5 lon_0=114 lat_1=30 lat_2=49 x_0=0 y_0=0 datum=NAD83 units=m no_defs ellps=GRS80 towgs84=0,0,0"
# P = pyproj.Proj(proj_str)
# proj_crs = pyproj.CRS(proj_str)
# # crs_epsg = proj_crs.to_epsg()
# # print(crs_epsg)
# # print(proj_crs)
# # print('\n\n\n\n\n\n')
# #now transform the CRS of the burn scar MTBS data to the VIIRS CRS
# outproj = 9822
# proj = pyproj.Transformer.from_crs(proj_crs, outproj, always_xy=True)
# # print(proj)
#
# cartopy_crs = cartopy.crs.epsg(outproj)
# # print('\n\n\n\n\n\n')
# # print(cartopy_crs)
#
# '''plot'''
# # f, ax = plt.subplots()
# # tract_data.to_crs(house_data.crs).plot(ax=ax)
# # house_data.plot(ax=ax)
#
# fig = plt.figure()
# projection = cartopy_crs
# ax = fig.add_subplot(111, projection=projection)
# ax.imshow(dlcf_rbg[0])
# import cartopy.feature as cfeature
# min_lon_x, min_lat_y = P(-129, 30)
# max_lon_x, max_lat_y = P(-100, 49)
# print(min_lon_x, max_lon_x, min_lat_y, max_lat_y)
# 8444012.676171554 6679119.40092884 5232130.094338377 7935000.410265504
# x_limits=(1360148.4364888417, 2254632.6827739254), y_limits=(1131427.4887314436, 1339754.1403672944)
#
# import sys
# sys.exit()
# extent = (min_lon_x, max_lon_x, min_lat_y, max_lat_y)
# ax.set_extent(extent, crs=projection)
# ax.add_feature(cfeature.COASTLINE, edgecolor='yellow',zorder=10, linewidth=1)
# ax.add_feature(cfeature.BORDERS  , edgecolor='yellow',zorder=10, linewidth=1)
# ax.add_feature(cfeature.STATES   , edgecolor='yellow',zorder=10, linewidth=1)
# plt.show()
#
# # print(ccrs.PlateCarree())
#
#
#
#
# #crs
# # spatial_ref: GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,
# # AUTHORITY["EPSG","7030"]],
# # AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,
# # AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,
# # AUTHORITY["EPSG","9122"]],
# # AUTHORITY["EPSG","4326"]]
# #pyproj definition of projection
#
# # proj_str = "proj=aea lat_0=39.5 lon_0=114 lat_1=30 lat_2=49 x_0=0 y_0=0 datum=NAD83 units=m no_defs ellps=GRS80 towgs84=0,0,0"
# # {'proj': 'aea', 'lat_1': 30.0, 'lat_2': 49.0, 'lat_0': 39.5, 'lon_0': 114.0, 'x_0': 0, 'y_0': 0, 'units': 'm'}
# #above 2 lines make the dict below with paramters below dict
# # Projection  = {'datum': 'WGS84', 'lat_0': '39.5', 'lat_1': '30', 'lat_2': '49',
# #                'lon_0': '114', 'no_defs': 'None', 'proj': 'aea', 'type': 'crs',
# #                'units': 'm', 'x_0': '0', 'y_0': '0'}
# # Number of columns: 2354
# # Number of rows: 3604
# # Area extent: (-882446.6376, -1351435.158, 882446.6376, 1351435.158)
#
#
#
#
# # import matplotlib.pyplot as plt
# # import cartopy.crs as ccrs
# # import cartopy.feature as cfeature
# # import pyproj
# #
# # # Define the PyProj CRS
# # pyproj_str = '+proj=aea +lat_1=30.0 +lat_2=49.0 +lat_0=39.5 +lon_0=114.0 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs'
# # proj_crs = pyproj.CRS.from_string(pyproj_str)
# #
# # # Create a Cartopy projection using the PyProj CRS
# # projection = ccrs.Projection(proj_crs)
# #
# # # Create a figure and axis with the Cartopy projection
# # fig, ax = plt.subplots(subplot_kw={'projection': projection})
# #
# # # # Plot your data here
# # # # Example: plotting a point at (x, y)
# # # x = 500000
# # # y = 500000
# # # ax.plot(x, y, 'ro', transform=projection)
# #
# # # Add map features (optional)
# # ax.add_feature(cfeature.LAND)
# # ax.add_feature(cfeature.OCEAN)
# # ax.add_feature(cfeature.COASTLINE)
# #
# # # Set the extent of the plot (optional)
# # ax.set_extent([400000, 600000, 400000, 600000], crs=projection)
# #
# # # Show the plot
# # plt.show()
