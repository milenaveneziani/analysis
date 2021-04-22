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
regionMaskDir = '/lcrc/group/e3sm/ac.milena/mpas-region_masks'
meshName = 'EC30to60E2r2'
restartFile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/{}/ocean.EC30to60E2r2.200908.nc'.format(meshName)
runName = '20210416_JRA_tidalMixingEnG_EC30to60E2r2'
runNameShort = 'JRA_tidalMixingEnG'
climodir = '/lcrc/group/e3sm/ac.milena/E3SM_simulations/{}/clim/mpas/avg/unmasked_{}'.format(runName, meshName)

# Settings for cori
#regionMaskDir = '/global/cscratch1/sd/milena/mpas-region_masks'
#meshName = 'ARRM60to10'
#restartFile = '/global/cscratch1/sd/milena/E3SM_simulations/ARRM60to10_JRA_GM_ramp/run/mpaso.rst.0004-01-01_00000.nc'
#runName = 'ARRM60to10_JRA_GM_ramp'
#runNameShort = 'E3SM-Arctic-OSI60to10'
#climodir = '/global/cscratch1/sd/milena/E3SM_simulations/ARRM60to10_JRA_GM_ramp/run'

climoYear1 = 6
climoYear2 = 10

seasons = ['ANN', 'JFM', 'JAS']

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
clevelsS = [10.0, 15.0, 20.0, 25.0, 28.0, 30.0, 31.0, 32.0, 33.0, 33.5, 34.0, 35.0, 36.0]
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
sigma2contours = [35, 36, 36.5, 36.8, 37, 37.1, 37.2]
sigma2contoursCessi = [30, 32, 35, 36, 36.5, 36.8, 37, 37.25, 37.44, 37.52, 37.6]
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
    temp = ds['timeMonthly_avg_activeTracers_temperature']
    salt = ds['timeMonthly_avg_activeTracers_salinity']
    # Apply depthMask
    temp = temp.where(depthMask, drop=False)
    salt = salt.where(depthMask, drop=False)
    # Apply NaN mask (for some reason the MPAS-Analysis generated
    # climatologies have unexplained values of -1e+34, even after
    # applying the depthMask
    temp = temp.where(temp > -20., drop=False)
    salt = salt.where(salt >   0., drop=False)

    # Compute regional quantities and plot them
    fig, ax = plt.subplots(1, len(regionsToPlot), figsize=figsize)
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
        areaRegion = areaCell.where(cellMask, drop=True)
        tempRegion = temp.where(cellMask, drop=True)
        saltRegion = salt.where(cellMask, drop=True)

        # Create 0.5deg wide latitude bins
        latbins = np.arange(np.min(latRegion.values), np.max(latRegion.values), 0.5)

        # Create output dataset for each region
        dsOut = xr.Dataset(
                {
                       'zonalAvgTemp': (['nLatbins', 'nVertLevels'], 
                           np.nan*np.ones([len(latbins)-1, nVertLevels])),
                       'zonalAvgSalt': (['nLatbins', 'nVertLevels'],
                           np.nan*np.ones([len(latbins)-1, nVertLevels])),
                },
                coords={
                       'latbins': (['nLatbins'],
                           latbins[:-1]),
                       'depth': (['nVertLevels'],
                           -refBottomDepth),
                },
        )
        # Compute averages for each latitude bin
        for ilat in range(1, len(latbins)):
            binMask = np.logical_and(latRegion >= latbins[ilat-1], latRegion <= latbins[ilat])
            binArea = areaRegion.where(binMask, drop=True)
            binTemp = tempRegion.where(binMask, drop=True)
            binTemp = (binTemp*binArea).sum(dim='nCells') / binArea.sum(dim='nCells')
            binSalt = saltRegion.where(binMask, drop=True)
            binSalt = (binSalt*binArea).sum(dim='nCells') / binArea.sum(dim='nCells')
            
            dsOut['zonalAvgTemp'][ilat-1, :] = binTemp
            dsOut['zonalAvgTemp'].attrs['units'] = '$^\circ$C'
            dsOut['zonalAvgTemp'].attrs['description'] = 'Zonal average of regional potential temperature'
            dsOut['zonalAvgSalt'][ilat-1, :] = binSalt
            dsOut['zonalAvgSalt'].attrs['units'] = 'psu'
            dsOut['zonalAvgSalt'].attrs['description'] = 'Zonal average of regional salinity'

        x = latbins[:-1]
        y = -refBottomDepth
        fldtemp = dsOut.zonalAvgTemp.values.T
        fldsalt = dsOut.zonalAvgSalt.values.T
        print(np.min(fldsalt), np.max(fldsalt))
        # Compute sigma0 and sigma2
        [lat, z] = np.meshgrid(x, y)
        pressure = gsw.p_from_z(z, lat)
        print(np.shape(pressure))
        SA = gsw.SA_from_SP(fldsalt, pressure, 0., lat)
        CT = gsw.CT_from_pt(SA, fldtemp)
        sigma2 = gsw.density.sigma2(SA, CT)
        sigma0 = gsw.density.sigma0(SA, CT)
        print(np.min(sigma2), np.max(sigma2))
        print(np.min(sigma0), np.max(sigma0))

        ax[iregion].set_facecolor('darkgrey')
        cf = ax[iregion].contourf(x, y, fldsalt, cmap=colormapS, norm=cnormS, levels=clevelsS, extend='both')
        if sigma2contours is not None:
            cs1 = ax[iregion].contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
            cs2 = ax[iregion].contour(x, y, sigma2, sigma2contoursCessi, colors='k', linewidths=2.5)
            plt.clabel(cs1, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
            plt.clabel(cs2, levels=sigma2contoursCessi, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
        if sigma0contours is not None:
            cs = ax.contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
            plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)

        ax[iregion].set_xlabel('Latitude', fontsize=24, fontweight='bold')
        ax[iregion].set_title(regionName, fontsize=28, fontweight='bold')
                    
    ax[0].set_ylabel('Depth (m)', fontsize=24, fontweight='bold')
    fig.tight_layout(pad=0.5)
    #plt.colorbar(cf, format='%1.1f')
    cax, kw = mpl.colorbar.make_axes(ax[-1], location='right', pad=0.05, shrink=0.9)
    cbar = plt.colorbar(cf, cax=cax, ticks=clevelsS, boundaries=clevelsS, **kw)
    cbar.ax.tick_params(labelsize=16, labelcolor='black')
    cbar.set_label('psu', fontsize=16, fontweight='bold')

    figname = '{}/zonalAvgAtlanticSection_salt_{}_years{:04d}-{:04d}.png'.format(figdir, season, climoYear1, climoYear2)
    plt.savefig(figname, dpi=figdpi, bbox_inches='tight')
    plt.close()
