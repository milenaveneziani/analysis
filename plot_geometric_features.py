from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import xarray as xr
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
from matplotlib.pyplot import cm
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import BoundaryNorm
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cmocean
import shapely.geometry

from geometric_features.feature_collection import read_feature_collection

def _add_land_lakes_coastline(ax):
    land_50m = cfeature.NaturalEarthFeature(
            'physical', 'land', '50m', edgecolor='black',
            facecolor='#cdc0b0', linewidth=0.5)
    lakes_50m = cfeature.NaturalEarthFeature(
            'physical', 'lakes', '50m', edgecolor='black',
            facecolor='aliceblue', linewidth=0.5)
    coast_50m = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '50m', edgecolor='face',
            facecolor='None', linewidth=0.5)
    ax.add_feature(land_50m, zorder=3)
    ax.add_feature(lakes_50m, zorder=4)
    ax.add_feature(coast_50m, zorder=5)

def _add_bbox(ax, bbox):
    # This plots parallel and meridian arcs using the 'bbox' information 
    # effectively making that the map boundary
    myProj = ccrs.PlateCarree()
    [ax_hdl] = ax.plot([bbox[0], bbox[1], bbox[1], bbox[0], bbox[0]],
                       [bbox[2], bbox[2], bbox[3], bbox[3], bbox[2]],
                       color='black', linewidth=0.5, marker='none', 
                       transform=myProj)
    # Get the `Path` of the plot
    tx_path = ax_hdl._get_transformed_path()
    path_in_data_coords, _ = tx_path.get_transformed_path_and_affine()
    # Use the path's vertices to create a polygon
    polygon = mpath.Path(path_in_data_coords.vertices)
    # Different method that doesn't seem to work:
    #npoints = 20
    #polygon = mpath.Path(
    #    list(zip(np.linspace(bbox[0], bbox[1], npoints), np.full(npoints, bbox[3]))) + \
    #    list(zip(np.full(npoints, bbox[1]), np.linspace(bbox[3], bbox[2], npoints))) + \
    #    list(zip(np.linspace(bbox[1], bbox[0], npoints), np.full(npoints, bbox[2]))) + \
    #    list(zip(np.full(npoints, bbox[0]), np.linspace(bbox[2], bbox[3], npoints)))
    #)
    ax.set_boundary(polygon)

bathyfile = '/lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean/bathymetry_database//BedMachineAntarctica_v3_and_GEBCO_2023_0.0125_degree_20230831.nc'
featurefile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/arcticRegions.geojson'

figdir = './general'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figsize = [20, 20]
figdpi = 200

clevels = [10, 40, 80, 120, 160, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
#clevels = [50, 100, 200, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
colormap = plt.get_cmap(cmocean.cm.deep_r)
colorIndices = np.int_(np.linspace(0, 255, num=len(clevels)+1, endpoint=True))
underColor = colormap(colorIndices[0])
overColor = colormap(colorIndices[-1])
colorIndices = colorIndices[1:-1]
colormap = cols.ListedColormap(colormap(colorIndices))
colormap.set_under(underColor)
colormap.set_over(overColor)
cnorm = mpl.colors.BoundaryNorm(clevels, colormap.N)

# Read in bathymetry and reduce its resolution by 'subsample' times
ds = xr.open_dataset(bathyfile)
subsample = 4
new_lon = np.linspace(ds.lon[0], ds.lon[-1], np.int_(ds.dims["lon"] / subsample), endpoint=True)
new_lat = np.linspace(ds.lat[0], ds.lat[-1], np.int_(ds.dims["lat"] / subsample), endpoint=True)
bathymetry = -ds.bathymetry.interp(lon=new_lon, lat=new_lat)

# Read in features
features = read_feature_collection(featurefile)
#print(dir(features))
#featureNames = [fc['properties']['name'] for fc in features.features]
#print(featureNames)
featuresToPlot = ['Barents Sea', 'Kara Sea', 'Laptev Sea', 'East Siberian Sea', 'Chukchi Sea', 'Canada Basin', 'Canadian Archipelago']
#featuresToPlot = ['Barents Sea', 'Kara Sea', 'Laptev Sea', 'East Siberian Sea', 'Chukchi Sea', 'Beaufort Gyre', 'Beaufort Gyre Shelf', 'Canadian Archipelago']
colors = ['#2166ac', '#d6604d', '#01665e', '#92c5de', '#b2182b', '#80cdc1', '#a6dba0', '#9970ab', '#ffffbf']
color_names = ['dark blue', 'light red', 'dark green', 'light blue', 'dark red', 'aqua', 'light green', 'purple', 'lemon']

# The lat-lon proj
noProj = ccrs.PlateCarree()

## The projection of the map
#mapProj = ccrs.NorthPolarStereo(central_longitude=0)
#
#extent = [-180, 180, 63, 90]
#
#plt.figure(figsize=figsize, dpi=figdpi)
#ax = plt.axes(projection=mapProj)
#_add_land_lakes_coastline(ax)
#ax.set_extent(extent, crs=noProj)
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, color='k', linestyle=':', zorder=5)
#fcIndex = 0
#for fcname in featuresToPlot:
#    for fc in features.features:
#        if fc['properties']['name']==fcname:
#            color = colors[fcIndex]
#            geomType = fc['geometry']['type']
#            shape = shapely.geometry.shape(fc['geometry'])
#            props = {'linewidth': 2.0, 'edgecolor': color, 'alpha': 0.5, 'facecolor': color}
#            ax.add_geometries((shape,), crs=noProj, **props)
#            fcIndex = fcIndex+1
#plt.savefig(f'{figdir}/arcticRegions.png', bbox_inches='tight')
#plt.close()

featuresToPlot = ['Barents Sea', 'Kara Sea', 'Laptev Sea']
colors = ['#2166ac', '#d6604d', '#01665e']
extent = [10, 145, 63, 85]
dlon = extent[1]-extent[0]
dlat = extent[3]-extent[2]

# The projection of the map
mapProj = ccrs.NorthPolarStereo(central_longitude=extent[0]+dlon/2)
mapProj._threshold = mapProj._threshold/20.  # Set for higher precision of the projection

plt.figure(figsize=figsize, dpi=figdpi)
ax = plt.axes(projection=mapProj)
ax.set_extent(extent, crs=noProj)

cf = ax.pcolormesh(new_lon, new_lat, bathymetry, cmap=colormap, norm=cnorm, transform=noProj)
cbar = plt.colorbar(cf, ticks=clevels, boundaries=clevels, extend='both', shrink=0.4, pad=0.03, orientation='vertical')
cbar.ax.tick_params(labelsize=14, labelcolor='black')
cbar.set_label('[m]', fontsize=16)
#ax.contour(new_lon, new_lat, bathymetry, clevels, colors=['grey'], transform=noProj)
#ax.contour(new_lon, new_lat, bathymetry, [300], colors=['black'], transform=noProj)
#ax.contour(new_lon, new_lat, bathymetry, [1000], colors=['blue'], transform=noProj)
#ax.contour(new_lon, new_lat, bathymetry, [4000], colors=['#9900CC'], transform=noProj)
fcIndex = 0
for fcname in featuresToPlot:
    for fc in features.features:
        if fc['properties']['name']==fcname:
            color = colors[fcIndex]
            geomType = fc['geometry']['type']
            shape = shapely.geometry.shape(fc['geometry'])
            props = {'linewidth': 2.5, 'edgecolor': 'white', 'facecolor': 'None'}
            #props = {'linewidth': 2.5, 'edgecolor': color, 'facecolor': 'None'}
#            This goes well with ax.contour above:
#            props = {'linewidth': 2.0, 'edgecolor': color, 'alpha': 0.5, 'facecolor': color}
            ax.add_geometries((shape,), crs=noProj, **props)
            fcIndex = fcIndex+1

_add_bbox(ax, extent)
ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, color='k', linestyle=':', zorder=6)
_add_land_lakes_coastline(ax)
ax.set_title('Barents, Kara, and Laptev Sea regions', y=1.04, fontsize=22)
plt.savefig(f'{figdir}/barentsKaraLaptevRegions.png', bbox_inches='tight')
plt.close()

featuresToPlot = ['East Siberian Sea', 'Chukchi Sea']
colors = ['#92c5de', '#b2182b']
extent = [140, 210, 63, 80]
dlon = extent[1]-extent[0]
dlat = extent[3]-extent[2]

# The projection of the map
mapProj = ccrs.NorthPolarStereo(central_longitude=extent[0]+dlon/2)
mapProj._threshold = mapProj._threshold/20.  # Set for higher precision of the projection

plt.figure(figsize=figsize, dpi=figdpi)
ax = plt.axes(projection=mapProj)
ax.set_extent(extent, crs=noProj)

cf = ax.pcolormesh(new_lon, new_lat, bathymetry, cmap=colormap, norm=cnorm, transform=noProj)
cbar = plt.colorbar(cf, ticks=clevels, boundaries=clevels, extend='both', shrink=0.4, pad=0.03, orientation='vertical')
cbar.ax.tick_params(labelsize=14, labelcolor='black')
cbar.set_label('[m]', fontsize=16)
#ax.contour(new_lon, new_lat, bathymetry, clevels, colors=['grey'], transform=noProj)
#ax.contour(new_lon, new_lat, bathymetry, [300], colors=['black'], transform=noProj)
#ax.contour(new_lon, new_lat, bathymetry, [1000], colors=['blue'], transform=noProj)
#ax.contour(new_lon, new_lat, bathymetry, [4000], colors=['#9900CC'], transform=noProj)
fcIndex = 0
for fcname in featuresToPlot:
    for fc in features.features:
        if fc['properties']['name']==fcname:
            color = colors[fcIndex]
            geomType = fc['geometry']['type']
            shape = shapely.geometry.shape(fc['geometry'])
            props = {'linewidth': 2.5, 'edgecolor': 'white', 'facecolor': 'None'}
            #props = {'linewidth': 2.5, 'edgecolor': color, 'facecolor': 'None'}
#            props = {'linewidth': 2.0, 'edgecolor': color, 'alpha': 0.5, 'facecolor': color}
            ax.add_geometries((shape,), crs=noProj, **props)
            fcIndex = fcIndex+1

_add_bbox(ax, extent)
ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, color='k', linestyle=':', zorder=6)
_add_land_lakes_coastline(ax)
ax.set_title('East Siberian and Chukchi Sea regions', y=1.04, fontsize=22)
plt.savefig(f'{figdir}/eastSiberianChukchiRegions.png', bbox_inches='tight')
plt.close()

featuresToPlot = ['Chukchi Sea', 'Canada Basin']
colors = ['#b2182b', '#80cdc1']
extent = [178, 260, 65, 82]
dlon = extent[1]-extent[0]
dlat = extent[3]-extent[2]

# The projection of the map
mapProj = ccrs.NorthPolarStereo(central_longitude=extent[0]+dlon/2)
mapProj._threshold = mapProj._threshold/20.  # Set for higher precision of the projection

plt.figure(figsize=figsize, dpi=figdpi)
ax = plt.axes(projection=mapProj)
ax.set_extent(extent, crs=noProj)

cf = ax.pcolormesh(new_lon, new_lat, bathymetry, cmap=colormap, norm=cnorm, transform=noProj)
cbar = plt.colorbar(cf, ticks=clevels, boundaries=clevels, extend='both', shrink=0.4, pad=0.03, orientation='vertical')
cbar.ax.tick_params(labelsize=14, labelcolor='black')
cbar.set_label('[m]', fontsize=16)
#ax.contour(new_lon, new_lat, bathymetry, clevels, colors=['grey'], transform=noProj)
#ax.contour(new_lon, new_lat, bathymetry, [300], colors=['black'], transform=noProj)
#ax.contour(new_lon, new_lat, bathymetry, [1000], colors=['blue'], transform=noProj)
#ax.contour(new_lon, new_lat, bathymetry, [4000], colors=['#9900CC'], transform=noProj)
fcIndex = 0
for fcname in featuresToPlot:
    for fc in features.features:
        if fc['properties']['name']==fcname:
            color = colors[fcIndex]
            geomType = fc['geometry']['type']
            shape = shapely.geometry.shape(fc['geometry'])
            props = {'linewidth': 2.5, 'edgecolor': 'white', 'facecolor': 'None'}
            #props = {'linewidth': 2.5, 'edgecolor': color, 'facecolor': 'None'}
#            props = {'linewidth': 2.0, 'edgecolor': color, 'alpha': 0.5, 'facecolor': color}
            ax.add_geometries((shape,), crs=noProj, **props)
            fcIndex = fcIndex+1

_add_bbox(ax, extent)
ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, color='k', linestyle=':', zorder=6)
_add_land_lakes_coastline(ax)
ax.set_title('Chukchi Sea and Canada Basin regions', y=1.04, fontsize=22)
plt.savefig(f'{figdir}/chukchiCanadaBasinRegions.png', bbox_inches='tight')
plt.close()
