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
import copy

from common_functions import add_land_lakes_coastline


def make_scatter_plot(lon, lat, dotSize, figTitle, figFile, projectionName='Robinson', lon0=-180, lon1=180, dlon=40, lat0=-90, lat1=90, dlat=20, fld=None, cmap=None, clevels=None, cindices=None, cbarLabel=None, contourfld=None, contourLevels=None, contourColors=None):
    
    figdpi = 150
    figsize = [20, 20]

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

    if fld is not None:
        [colormap, cnorm] = _make_discrete_colormap(cmap, cindices, clevels)
        sc = ax.scatter(lon, lat, s=dotSize, c=fld, cmap=colormap, norm=cnorm, marker='o', transform=data_crs)
        if cindices is not None:
            cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5, extend='both')
        else:
            cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5)
        cbar.ax.tick_params(labelsize=16, labelcolor='black')
        cbar.set_label(cbarLabel, fontsize=14)
    else:
        sc = ax.scatter(lon, lat, s=dotSize, c='k', marker='D', transform=data_crs)


    if contourfld is not None:
        if contourLevels is None:
            raise ValueError('contourLevels needs to be defined if contourfld is')
        if contourColors is not None:
            ax.tricontour(lon, lat, contourfld, levels=contourLevels, colors=contourColors, transform=data_crs)
        else:
            ax.tricontour(lon, lat, contourfld, levels=contourLevels, colors='k', transform=data_crs)

    add_land_lakes_coastline(ax)

    ax.set_title(figTitle, y=1.04, fontsize=16)
    plt.savefig(figFile, bbox_inches='tight')
    plt.close()

def make_streamline_plot(lon, lat, u, v, speed, density, cmap, clevels, cindices, cbarLabel, projectionName, figTitle, figFile, lon0=-180, lon1=180, dlon=40, lat0=-90, lat1=90, dlat=20):
    
    figdpi = 150
    figsize = [20, 20]

    [colormap, cnorm] = _make_discrete_colormap(cmap, cindices, clevels)

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


def make_contourf_plot(lon, lat, fld, cmap, clevels, cindices, cbarLabel, figTitle, figFile, contourFld=None, contourValues=None, projectionName='Robinson', lon0=-180, lon1=180, dlon=40, lat0=-90, lat1=90, dlat=20):
    
    figdpi = 150
    figsize = [20, 20]

    [colormap, cnorm] = _make_discrete_colormap(cmap, cindices, clevels)

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

    fldcyclic, loncyclic = add_cyclic_point(fld, lon)
    [lon, lat] = np.meshgrid(loncyclic, lat)

    cf = ax.contourf(lon, lat, fldcyclic, cmap=colormap, norm=cnorm, levels=clevels, extend='both', transform=data_crs)
    cbar = plt.colorbar(cf, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5, extend='both')
    cbar.ax.tick_params(labelsize=14, labelcolor='black')
    cbar.set_label(cbarLabel, fontsize=12, fontweight='bold')
    if (contourFld is not None) and (contourValues is not None):
        cs = ax.contour(lon, lat, contourFld, contourValues, colors='k', linewidths=1.5)
        cb = plt.clabel(cs, levels=contourValues, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)

    ax.set_title(figTitle, y=1.04, fontsize=16)
    plt.savefig(figFile, bbox_inches='tight')
    plt.close()
    

def make_pcolormesh_plot(lon, lat, fld, cmap, clevels, cindices, cbarLabel, figTitle, figFile, contourFld=None, contourValues=None, projectionName='Robinson', lon0=-180, lon1=180, dlon=40, lat0=-90, lat1=90, dlat=20):
    
    figdpi = 150
    figsize = [20, 20]

    [colormap, cnorm] = _make_discrete_colormap(cmap, cindices, clevels)

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

    fldcyclic, loncyclic = add_cyclic_point(fld, lon)
    [lon, lat] = np.meshgrid(loncyclic, lat)

    cf = ax.pcolormesh(lon, lat, fldcyclic, cmap=colormap, norm=cnorm, transform=data_crs)
    cbar = plt.colorbar(cf, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5, extend='both')
    cbar.ax.tick_params(labelsize=14, labelcolor='black')
    cbar.set_label(cbarLabel, fontsize=12, fontweight='bold')
    if (contourFld is not None) and (contourValues is not None):
        cs = ax.contour(lon, lat, contourFld, contourValues, colors='k', linewidths=1.5)
        cb = plt.clabel(cs, levels=contourValues, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)

    ax.set_title(figTitle, y=1.04, fontsize=16)
    plt.savefig(figFile, bbox_inches='tight')
    plt.close()
    

def _make_discrete_colormap(colormap, colorindices, colorlevels):
    if colorindices is not None:
        colorindices0 = colorindices
        if len(colorlevels)+1 == len(colorindices0):
            # we have 2 extra values for the under/over so make the colormap
            # without these values
            colorindices = colorindices0[1:-1]
            underColor = colormap(colorindices0[0])
            overColor = colormap(colorindices0[-1])
        else:
            colorindices = colorindices0
            underColor = None
            overColor = None
        colormap = cols.ListedColormap(colormap(colorindices))
        if underColor is not None:
            colormap.set_under(underColor)
        if overColor is not None:
            colormap.set_over(overColor)
    colornorm = mpl.colors.BoundaryNorm(colorlevels, colormap.N)
    return [colormap, colornorm]
