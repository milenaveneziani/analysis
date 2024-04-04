from __future__ import absolute_import, division, print_function, \
    unicode_literals
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
from matplotlib.colors import BoundaryNorm
import cartopy
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.ticker as mticker
import cmocean

from common_functions import add_land_lakes_coastline


def make_scatter_plot(lon, lat, fld, dotSize, cmap, clevels, cindices, cbarLabel, projectionName, figTitle, figFile, lon0=-180, lon1=180, dlon=40, lat0=-90, lat1=90, dlat=20):
    
    figdpi = 150
    figsize = [20, 20]

    # Make colormap
    colormap0 = cmap
    colorIndices0 = cindices
    if len(clevels)+1 == len(colorIndices0):
        # we have 2 extra values for the under/over so make the colormap
        # without these values
        underColor = colormap0(colorIndices0[0])
        overColor = colormap0(colorIndices0[-1])
        colorIndices = colorIndices0[1:-1]
    else:
        colorIndices = colorIndices0
        underColor = None
        overColor = None
    colormap = cols.ListedColormap(colormap0(colorIndices))
    if underColor is not None:
        colormap.set_under(underColor)
    if overColor is not None:
        colormap.set_over(overColor)
    cnorm = mpl.colors.BoundaryNorm(clevels, colormap.N)

    data_crs = ccrs.PlateCarree()

    plt.figure(figsize=figsize, dpi=figdpi)

    if projectionName=='NorthPolarStereo':
        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
    elif projectionName=='SouthPolarStereo':
        ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=0))
    else:
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    ax.set_extent([lon0, lon1, lat0, lat1], crs=data_crs)
    gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6, draw_labels=True)
    gl.xlocator = mticker.FixedLocator(np.arange(lon0, lon1+dlon, dlon))
    gl.ylocator = mticker.FixedLocator(np.arange(lat0+dlat/2, lat1-dlat/2, dlat))
    gl.n_steps = 100
    gl.right_labels = False
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl.rotate_labels = False

    add_land_lakes_coastline(ax)

    sc = ax.scatter(lon, lat, s=dotSize, c=fld, cmap=colormap, norm=cnorm, marker='o', transform=data_crs)
    cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5, extend='both')
    cbar.ax.tick_params(labelsize=16, labelcolor='black')
    cbar.set_label(cbarLabel, fontsize=14)

    ax.set_title(figTitle, y=1.04, fontsize=16)
    plt.savefig(figFile, bbox_inches='tight')
    plt.close()

def make_streamline_plot(lon, lat, u, v, speed, density, cmap, clevels, cindices, cbarLabel, projectionName, figTitle, figFile, lon0=-180, lon1=180, dlon=40, lat0=-90, lat1=90, dlat=20):
    
    figdpi = 150
    figsize = [20, 20]

    # Make colormap
    colormap0 = cmap
    colorIndices0 = cindices
    if len(clevels)+1 == len(colorIndices0):
        # we have 2 extra values for the under/over so make the colormap
        # without these values
        underColor = colormap0(colorIndices0[0])
        overColor = colormap0(colorIndices0[-1])
        colorIndices = colorIndices0[1:-1]
    else:
        colorIndices = colorIndices0
        underColor = None
        overColor = None
    colormap = cols.ListedColormap(colormap0(colorIndices))
    if underColor is not None:
        colormap.set_under(underColor)
    if overColor is not None:
        colormap.set_over(overColor)
    cnorm = mpl.colors.BoundaryNorm(clevels, colormap.N)

    data_crs = ccrs.PlateCarree()

    plt.figure(figsize=figsize, dpi=figdpi)

    if projectionName=='NorthPolarStereo':
        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
    elif projectionName=='SouthPolarStereo':
        ax = plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=0))
    else:
        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
    ax.set_extent([lon0, lon1, lat0, lat1], crs=data_crs)
    gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6, draw_labels=True)
    gl.xlocator = mticker.FixedLocator(np.arange(lon0, lon1+dlon, dlon))
    gl.ylocator = mticker.FixedLocator(np.arange(lat0-dlat/2, lat1+dlat/2, dlat))
    gl.n_steps = 100
    gl.right_labels = False
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl.rotate_labels = False

    add_land_lakes_coastline(ax)

    ucyclic, loncyclic = add_cyclic_point(u, lon)
    vcyclic, loncyclic = add_cyclic_point(v, lon)
    speedcyclic, loncyclic = add_cyclic_point(speed, lon)
    loncyclic = np.where(loncyclic>=180., loncyclic-360., loncyclic)
    sl = ax.streamplot(loncyclic, lat, ucyclic, vcyclic, density=density, broken_streamlines=False,
                       color=speedcyclic, cmap=colormap, norm=cnorm, linewidth=1.5, transform=data_crs)
    #sl = ax.streamplot(lon, lat, u, v, density=density, broken_streamlines=False,
    #                   color=speed, cmap=colormap, linewidth=1.5, transform=data_crs)
    cbar = plt.colorbar(sl.lines, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5, extend='both')
    cbar.ax.tick_params(labelsize=16, labelcolor='black')
    cbar.set_label(cbarLabel, fontsize=14)

    ax.set_title(figTitle, y=1.04, fontsize=16)
    plt.savefig(figFile, bbox_inches='tight')
    plt.close()
