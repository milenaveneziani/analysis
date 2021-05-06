from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import glob
import xarray as xr
import numpy as np
import gsw
import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
from matplotlib.pyplot import cm
from matplotlib.colors import BoundaryNorm
mpl.use('Agg')

from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf
from mpas_analysis.shared.io.utility import decode_strings
from mpas_analysis.ocean.utility import compute_zmid

from geometric_features import FeatureCollection, read_feature_collection

from common_functions import timeseries_analysis_plot, add_inset

# Settings for anvil/chrysalis
#regionMaskDir = '/lcrc/group/e3sm/ac.milena/mpas-region_masks'
#meshName = 'EC30to60E2r2'
#restartFile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/{}/ocean.EC30to60E2r2.200908.nc'.format(meshName)
#runName = '20210416_JRA_tidalMixingEnG_EC30to60E2r2'
#runNameShort = 'JRA_tidalMixingEnG'
#climodir = '/lcrc/group/e3sm/ac.milena/E3SM_simulations/{}/clim/mpas/avg/unmasked_{}'.format(runName, meshName)
#runName = '20210424.v2rc1b.piControl.ne30pg2_EC30to60E2r2'
#runNameShort = 'v2rc1b'
#climodir = '/lcrc/group/e3sm/ac.milena/E3SM_simulations/{}/clim/mpas/avg/unmasked_{}'.format(runName, meshName)
#climodir = '/lcrc/group/e3sm/ac.maltrud/E3SM_simulations/20210424.v2rc1b.piControl.ne30pg2_EC30to60E2r2.chrysalis/post/analysis/mpas_analysis/ts_0001-0200_climo_0151-0200/clim/mpas/avg/unmasked_{}'.format(meshName)
#runName = '20210422.v2rc1a.piControl.ne30pg2_EC30to60E2r2'
#runNameShort = 'v2rc1a'
#climodir = '/lcrc/group/e3sm/ac.milena/E3SM_simulations/{}/clim/mpas/avg/unmasked_{}'.format(runName, meshName)
#climodir = '/lcrc/group/e3sm/ac.golaz/E3SM_simulations/20210422.v2rc1a.piControl.ne30pg2_EC30to60E2r2.chrysalis/post/analysis/mpas_analysis/ts_0001-0200_climo_0151-0200/clim/mpas/avg/unmasked_{}'.format(meshName)

# Settings for cori
regionMaskDir = '/global/project/projectdirs/m1199/milena/mpas-region_masks'
meshName = 'ARRM60to10'
restartFile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oARRM60to10/ocean.ARRM60to10.180715.nc'
runName = 'ARRM60to10_JRA_GM_ramp'
runNameShort = 'E3SM-Arctic-OSI'
climodir = '/global/project/projectdirs/m1199/milena/analysis/mpas/ARRM60to10_new/clim/mpas/avg/unmasked_{}'.format(meshName)

#climoYear1 = 151
#climoYear2 = 200
climoYear1 = 166
climoYear2 = 177

seasons = ['ANN', 'JFM', 'JAS']
#seasons = ['ANN']

regionMaskFile = '{}/{}_oceanSubBasins20210315.nc'.format(regionMaskDir, meshName)
if os.path.exists(regionMaskFile):
     dsRegionMask = xr.open_dataset(regionMaskFile)
     regionNames = decode_strings(dsRegionMask.regionNames)
     regionsToPlot = ['Southern Ocean Atlantic Sector',
                      'South Atlantic Basin', 'North Atlantic Basin',
                      'Arctic Ocean Basin']
else:
     raise IOError('No regional mask file found')

figdir = './verticalSections/{}'.format(runNameShort)
if not os.path.isdir(figdir):
    os.makedirs(figdir)
figsize = (32, 5)
figdpi = 300
colorIndices0 = [0, 10, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]
clevelsT = [-1.0, -0.5, 0.0, 0.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8., 10., 12.]
clevelsS = [32.0, 33.0, 34.0, 34.2, 34.4, 34.6, 34.7, 34.8, 34.9, 35.0, 35.2, 35.5, 36.0]
colormapT = plt.get_cmap('RdBu_r')
colormapS = cmocean.cm.haline
#
underColor = colormapT(colorIndices0[0])
overColor = colormapT(colorIndices0[-1])
if len(clevelsT) + 1 == len(colorIndices0):
    # we have 2 extra values for the under/over so make the colormap
    # without these values
    colorIndices = colorIndices0[1:-1]
elif len(clevelsT) - 1 != len(colorIndices0):
    # indices list must be either one element shorter
    # or one element longer than colorbarLevels list
    raise ValueError('length mismatch between indices and '
                     'T colorbarLevels')
colormapT = cols.ListedColormap(colormapT(colorIndices))
colormapT.set_under(underColor)
colormapT.set_over(overColor)
#
underColor = colormapS(colorIndices0[0])
overColor = colormapS(colorIndices0[-1])
if len(clevelsS) + 1 == len(colorIndices0):
    # we have 2 extra values for the under/over so make the colormap
    # without these values
    colorIndices = colorIndices0[1:-1]
elif len(clevelsS) - 1 != len(colorIndices0):
    # indices list must be either one element shorter
    # or one element longer than colorbarLevels list
    raise ValueError('length mismatch between indices and '
                     'S colorbarLevels')
colormapS = cols.ListedColormap(colormapS(colorIndices))
colormapS.set_under(underColor)
colormapS.set_over(overColor)
#
cnormT = mpl.colors.BoundaryNorm(clevelsT, colormapT.N)
cnormS = mpl.colors.BoundaryNorm(clevelsS, colormapS.N)
#
sigma2contours = [30, 35, 36, 36.5, 36.8, 37.1, 37.2]
sigma2contoursCessi = [32, 37, 37.25, 37.44, 37.52, 37.6]
#sigma2contours = None
#sigma0contours = np.arange(26.0, 28.0, 0.2)
sigma0contours = None

if os.path.exists(restartFile):
    dsRestart = xr.open_dataset(restartFile)
    dsRestart = dsRestart.isel(Time=0)
else:
    raise IOError('No MPAS restart/mesh file found')
if 'landIceMask' in dsRestart:
    # only the region outside of ice-shelf cavities
    openOceanMask = dsRestart.landIceMask == 0
else:
    openOceanMask = None
latCell = dsRestart.latCell
areaCell = dsRestart.areaCell
refBottomDepth = dsRestart.refBottomDepth
maxLevelCell = dsRestart.maxLevelCell
nVertLevels = dsRestart.sizes['nVertLevels']
vertIndex = xr.DataArray.from_dict(
    {'dims': ('nVertLevels',), 'data': np.arange(nVertLevels)})
depthMask = (vertIndex < maxLevelCell).transpose('nCells', 'nVertLevels')

for season in seasons:
    print('\nPlotting Atlantic section for season {}...'.format(season))
    # Read in global data from climofile
    climofile = glob.glob('{}/mpaso_{}_{:04d}??_{:04d}??_climo.nc'.format(
                    climodir, season, climoYear1, climoYear2))[0]
    if not os.path.exists(climofile):
        raise IOError('Climatology file: {} not found'.format(climofile))
    ds = xr.open_dataset(climofile)
    ds = ds.isel(Time=0, drop=True)

    # Global depth-masked layer thickness and layer volume
    layerThickness = ds.timeMonthly_avg_layerThickness
    layerThickness = layerThickness.where(depthMask, drop=False)
    volCell = areaCell*layerThickness

    temp = ds['timeMonthly_avg_activeTracers_temperature']
    salt = ds['timeMonthly_avg_activeTracers_salinity']
    # Apply depthMask
    temp = temp.where(depthMask, drop=False)
    salt = salt.where(depthMask, drop=False)

    # Compute regional quantities and plot them
    fig1, ax1 = plt.subplots(1, len(regionsToPlot), figsize=figsize) # Salinity
    fig2, ax2 = plt.subplots(1, len(regionsToPlot), figsize=figsize) # Temperature
    for iregion in range(len(regionsToPlot)):

        regionName = regionsToPlot[iregion]
        regionIndex = regionNames.index(regionName)
        print('  Compute regional quantities for region: {}'.format(regionName))

        # Select regional data
        dsMask = dsRegionMask.isel(nRegions=regionIndex)
        cellMask = dsMask.regionCellMasks == 1
        if openOceanMask is not None:
            cellMask = np.logical_and(cellMask, openOceanMask)
        latRegion = 180./np.pi*latCell.where(cellMask, drop=True)
        volRegion = volCell.where(cellMask, drop=True)
        tempRegion = temp.where(cellMask, drop=True)
        saltRegion = salt.where(cellMask, drop=True)

        # Create 0.5deg-wide latitude bins
        latmin = np.min(latRegion.values)
        latmax = np.max(latRegion.values)
        dlat = 0.5
        latbins = np.arange(latmin, latmax+dlat, dlat)
        latbincenters = 0.5*(latbins[:-1] + latbins[1:])

        # Create output dataset for each region
        dsOut = xr.Dataset(
                {
                       'zonalAvgTemp': (['nLatbins', 'nVertLevels'], 
                           np.nan*np.ones([len(latbincenters), nVertLevels])),
                       'zonalAvgSalt': (['nLatbins', 'nVertLevels'],
                           np.nan*np.ones([len(latbincenters), nVertLevels])),
                },
                coords={
                       'latbins': (['nLatbins'],
                           latbincenters),
                       'depth': (['nVertLevels'],
                           -refBottomDepth),
                },
        )
        # Compute averages for each latitude bin
        for ilat in range(1, len(latbins)):
            binMask = np.logical_and(latRegion >= latbins[ilat-1], latRegion < latbins[ilat])
            binVol = volRegion.where(binMask, drop=True)
            binTemp = tempRegion.where(binMask, drop=True)
            binTemp = (binTemp*binVol).sum(dim='nCells') / binVol.sum(dim='nCells')
            binSalt = saltRegion.where(binMask, drop=True)
            binSalt = (binSalt*binVol).sum(dim='nCells') / binVol.sum(dim='nCells')
            
            dsOut['zonalAvgTemp'][ilat-1, :] = binTemp
            dsOut['zonalAvgTemp'].attrs['units'] = '$^\circ$C'
            dsOut['zonalAvgTemp'].attrs['description'] = 'Zonal average of regional potential temperature'
            dsOut['zonalAvgSalt'][ilat-1, :] = binSalt
            dsOut['zonalAvgSalt'].attrs['units'] = 'psu'
            dsOut['zonalAvgSalt'].attrs['description'] = 'Zonal average of regional salinity'

        x = latbincenters
        y = -refBottomDepth
        fldtemp = dsOut.zonalAvgTemp.values.T
        fldsalt = dsOut.zonalAvgSalt.values.T
        # Compute sigma0 and sigma2
        [lat, z] = np.meshgrid(x, y)
        pressure = gsw.p_from_z(z, lat)
        SA = gsw.SA_from_SP(fldsalt, pressure, 0., lat)
        CT = gsw.CT_from_pt(SA, fldtemp)
        sigma2 = gsw.density.sigma2(SA, CT)
        sigma0 = gsw.density.sigma0(SA, CT)

        ax1[iregion].set_facecolor('darkgrey')
        cfS = ax1[iregion].contourf(x, y, fldsalt, cmap=colormapS, norm=cnormS, levels=clevelsS, extend='both')
        if sigma2contours is not None:
            cs1 = ax1[iregion].contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
            cs2 = ax1[iregion].contour(x, y, sigma2, sigma2contoursCessi, colors='k', linewidths=2.5)
            plt.clabel(cs1, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
            plt.clabel(cs2, levels=sigma2contoursCessi, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
        if sigma0contours is not None:
            cs = ax1[iregion].contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
            plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
        if iregion > 0:
            ax1[iregion].set_yticklabels([])
        ax1[iregion].set_title(regionName, fontsize=28, fontweight='bold')

        ax2[iregion].set_facecolor('darkgrey')
        cfT = ax2[iregion].contourf(x, y, fldtemp, cmap=colormapT, norm=cnormT, levels=clevelsT, extend='both')
        if sigma2contours is not None:
            cs1 = ax2[iregion].contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
            cs2 = ax2[iregion].contour(x, y, sigma2, sigma2contoursCessi, colors='k', linewidths=2.5)
            plt.clabel(cs1, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
            plt.clabel(cs2, levels=sigma2contoursCessi, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
        if sigma0contours is not None:
            cs = ax2[iregion].contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
            plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
        if iregion > 0:
            ax2[iregion].set_yticklabels([])
        ax2[iregion].set_title(regionName, fontsize=28, fontweight='bold')
 
    ax1[0].set_ylabel('Depth (m)', fontsize=24, fontweight='bold')
    fig1.tight_layout(pad=0.5)
    fig1.suptitle('{} (years={}-{})'.format(runNameShort, climoYear1, climoYear2), fontsize=28, fontweight='bold', y=1.1)
    cax, kw = mpl.colorbar.make_axes(ax1[-1], location='right', pad=0.05, shrink=0.9)
    cbar = fig1.colorbar(cfS, cax=cax, ticks=clevelsS, **kw)
    cbar.ax.tick_params(labelsize=16, labelcolor='black')
    cbar.set_label('psu', fontsize=16, fontweight='bold')
    figname = '{}/zonalAvgAtlanticSection_salt_{}_years{:04d}-{:04d}.png'.format(figdir, season, climoYear1, climoYear2)
    fig1.savefig(figname, dpi=figdpi, bbox_inches='tight')
 
    ax2[0].set_ylabel('Depth (m)', fontsize=24, fontweight='bold')
    fig2.tight_layout(pad=0.5)
    fig2.suptitle('{} (years={}-{})'.format(runNameShort, climoYear1, climoYear2), fontsize=28, fontweight='bold', y=1.1)
    cax, kw = mpl.colorbar.make_axes(ax2[-1], location='right', pad=0.05, shrink=0.9)
    cbar = fig2.colorbar(cfT, cax=cax, ticks=clevelsT, **kw)
    cbar.ax.tick_params(labelsize=16, labelcolor='black')
    cbar.set_label('C$^\circ$', fontsize=16, fontweight='bold')
    figname = '{}/zonalAvgAtlanticSection_temp_{}_years{:04d}-{:04d}.png'.format(figdir, season, climoYear1, climoYear2)
    fig2.savefig(figname, dpi=figdpi, bbox_inches='tight')
