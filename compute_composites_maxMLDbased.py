from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import subprocess
from subprocess import call
import xarray as xr
import numpy as np
import netCDF4
from netCDF4 import Dataset as netcdf_dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
from matplotlib.pyplot import cm
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cmocean

from common_functions import add_land_lakes_coastline


#startSimYear = 1950
#startYear = 1950
#endYear = 2014
startSimYear = 1
startYear = 1
endYear = 140
#startYear = 245
#endYear = 386
years = np.arange(startYear, endYear + 1)
calendar = 'gregorian'
referenceDate = '0001-01-01'

# Settings for nersc
meshFile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runName = 'E3SM-Arcticv2.1_historical0151'
runName = 'E3SMv2.1B60to10rA02'
rundir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runName}'
# Note: the following two variables cannot be both True
isShortTermArchive = True # if True '{modelComp}/hist' will be affixed to rundir later on
isSingleVarFiles = False # if True '{modelComp}/singleVarFiles' will be affixed to rundir later on
 
maxMLDdir = f'./timeseries_data/{runName}/maxMLD'
outdir = f'./composites_maxMLDbased_data/{runName}/Years{startYear}-{endYear}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
figdir = f'./composites_maxMLDbased/{runName}/Years{startYear}-{endYear}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

regionGroup = 'Arctic Regions'
groupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')
regions = ['Greenland Sea', 'Norwegian Sea']

monthsToPlot = [1, 2, 3, 4] # JFMA
titleMonthsToPlot = 'JFMA'

figsize = [20, 20]
figdpi = 150
colorIndices0 = [0, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 250, 255]

# Choose either variables in timeSeriesStatsMonthly
# or variables in timeSeriesStatsMonthlyMax (2d only) or
# ice variables (2d only)
#
#   Ocean variables
modelComp = 'ocn'
modelName = 'mpaso'
#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'title': 'Maximum MLD',
#              'units': 'm',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthlyMax_max_dThreshMLD'}
#            ]
#
mpasFile = 'timeSeriesStatsMonthly'
variables = [
             {'name': 'dThreshMLD',
              'title': 'Mean MLD',
              'units': 'm',
              'factor': 1,
              'isvar3d': False,
              'mpas': 'timeMonthly_avg_dThreshMLD',
              'clevels': [10, 20, 50, 80, 100, 120, 150, 180, 250, 300, 400, 500, 800],
              'colormap': plt.get_cmap('viridis')},
#             {'name': 'sensibleHeatFlux',
#              'title': 'Sensible Heat Flux',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_sensibleHeatFlux',
#              'clevels': [-250, -200, -150, -120, -100, -80, -60, -40, -20, -10, 0, 10, 20],
#              'colormap': cmocean.cm.thermal},
             {'name': 'activeTracers_temperature',
              'title': 'Potential Temperature',
              'units': 'degC',
              'factor': 1,
              'isvar3d': True,
              'mpas': 'timeMonthly_avg_activeTracers_temperature',
              'clevels': [-1.0, -0.5, 0.0, 0.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8., 10., 12.],
              'colormap': cmocean.cm.thermal},
             {'name': 'activeTracers_salinity',
              'title': 'Salinity',
              'units': 'psu',
              'factor': 1,
              'isvar3d': True,
              'mpas': 'timeMonthly_avg_activeTracers_salinity',
              'clevels': [31.0, 33.0, 34.2,  34.4,  34.6, 34.7,  34.8,  34.87, 34.9, 34.95, 35.0, 35.3, 35.5],
              'colormap': cmocean.cm.haline}
             ]
             #{'name': 'surfaceBuoyancyForcing',
             # 'title': 'Surface buoyancy flux',
             # 'units': 'm$^2$ s$^{-3}$',
             # 'factor': 1,
             # 'isvar3d': False,
             # 'mpas': 'timeMonthly_avg_surfaceBuoyancyForcing'}
             #{'name': 'latentHeatFlux',
             # 'title': 'Latent Heat Flux',
             # 'units': 'W/m$^2$',
             # 'factor': 1,
             # 'isvar3d': False,
             # 'mpas': 'timeMonthly_avg_latentHeatFlux'}
#   Sea ice variables
#modelComp = 'ice'
#modelName = 'mpassi'
#mpasFile = 'timeSeriesStatsMonthly'
#variables = [
#             {'name': 'iceArea',
#              'title': 'Sea Ice Concentration',
#              'units': '%',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_iceAreaCell',
#              'clevels': [0.15, 0.3, 0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 1.0],
#              'colormap': cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
#                                               (0.169, 0.435, 0.223), (0.455, 0.478, 0.196), (0.757, 0.474, 0.435),\
#                                               (0.827, 0.561, 0.772), (0.761, 0.757, 0.949), (0.808, 0.921, 0.937)])},
#             {'name': 'iceVolume',
#              'title': 'Sea Ice Thickness',
#              'units': 'm',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_iceVolumeCell',
#              'clevels': [0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
#              'colormap': plt.get_cmap('YlGnBu_r')}
#            ]
#   Atmosphere variables
#modelComp = 'atm'
#modelName = 'eam'

if isShortTermArchive:
    rundir = f'{rundir}/{modelComp}/hist'
    #rundir = f'{rundir}/archive/{modelComp}/hist'
if isSingleVarFiles:
    rundir = f'{rundir}/{modelComp}/singleVarFiles'

# z levels [m] (relevant for 3d variables)
dlevels = [0.]

# Info about MPAS mesh
dsMesh = xr.open_dataset(meshFile)
lonCell = dsMesh.lonCell.values
latCell = dsMesh.latCell.values
z = dsMesh.refBottomDepth.values
lonCell = 180/np.pi*lonCell
latCell = 180/np.pi*latCell
# Find model levels for each depth level
zlevels = np.zeros(np.shape(dlevels), dtype=np.int64)
for id in range(len(dlevels)):
    dz = np.abs(z-dlevels[id])
    zlevels[id] = np.argmin(dz)

# First, identify high-convection and low-convection years
# based on previously computed regional averages of JFMA
# maxMLD fields
timeSeriesFiles = []
for year in years:
    timeSeriesFiles.append(f'{maxMLDdir}/{groupName}_max_year{year:04d}.nc')
dsIn = xr.open_mfdataset(timeSeriesFiles, combine='nested',
                         concat_dim='Time', decode_times=False)
regionNames = dsIn.regionNames[0].values

datetimes = netCDF4.num2date(dsIn.Time, f'days since {referenceDate}', calendar=calendar)
timeyears = []
for date in datetimes.flat:
    timeyears.append(date.year)

for regionName in regions:
    print(f'    region: {regionName}')
    regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
    regionIndex = np.where(regionNames==regionName)[0]

    maxMLD = np.squeeze(dsIn.maxMLD.isel(nRegions=regionIndex).values)
    maxMLD_seasonal = np.zeros(len(years))
    for iy, year in enumerate(years):
        yearmask = [i for i, x in enumerate(timeyears) if x==year]
        dsIn_yearly = dsIn.isel(Time=yearmask)
        datetimes = netCDF4.num2date(dsIn_yearly.Time, f'days since {referenceDate}', calendar=calendar)
        timemonths = []
        for date in datetimes.flat:
            timemonths.append(date.month)
        monthmask = [i for i, x in enumerate(timemonths) if x in set(monthsToPlot)]
        maxMLD_seasonal[iy] = dsIn_yearly.maxMLD.isel(Time=monthmask, nRegions=regionIndex).mean().values

    ax = plt.subplot(3, 1, 1)
    n, bins, patches = plt.hist(maxMLD_seasonal, bins=8, color='#607c8e', alpha=0.7, rwidth=0.9)
    ax.set_xticks(bins)
    ax.set_xticklabels(np.int16(bins))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(f'{titleMonthsToPlot}-avg maxMLD')
    plt.ylabel('# of years')
    plt.title(f'{regionName}')
    ax = plt.subplot(3, 1, 2)
    n, bins, patches = plt.hist(maxMLD, bins=10, color='#607c8e', alpha=0.7, rwidth=0.9)
    ax.set_xticks(bins)
    ax.set_xticklabels(np.int16(bins))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(f'{titleMonthsToPlot} (monthly) maxMLD')
    plt.ylabel('# of months')
    ax = plt.subplot(3, 1, 3)
    plt.plot(years, maxMLD_seasonal, linewidth=2)
    plt.xlabel('years')
    plt.ylabel(f'{titleMonthsToPlot}-avg maxMLD')
    plt.savefig(f'{figdir}/maxMLDhist_{regionNameShort}.png', dpi='figure', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    maxMLDstd = np.std(maxMLD_seasonal)
    mld1 = np.min(maxMLD_seasonal) + maxMLDstd
    mld2 = np.max(maxMLD_seasonal) - maxMLDstd
    conditionLow  = np.where(maxMLD_seasonal<mld1)
    conditionHigh = np.where(maxMLD_seasonal>=mld2)
    conditionMed  = np.logical_and(maxMLD_seasonal>=mld1, maxMLD_seasonal<mld2)
    #dbinRange = bins[-1]-bins[0]
    #conditionLow  = np.where(maxMLD_seasonal<bins[0]+0.2*dbinRange)
    #conditionHigh = np.where(maxMLD_seasonal>=bins[-1]-0.2*dbinRange)
    #conditionMed  = np.logical_and(maxMLD_seasonal>=bins[0]+0.2*dbinRange, maxMLD_seasonal<bins[-1]-0.2*dbinRange)

    years_low  = years[conditionLow]
    years_high = years[conditionHigh]
    years_med  = years[conditionMed]
    # Save this information somewhere
    #print(bins)
    print(years_low)
    print(years_high)

    # Now compute monthly climatologies associated with these composites and plot them
    for im in range(1, 13):
        for var in variables:
            varname = var['name']
            varfactor = var['factor']
            varunits = var['units']
            vartitle = var['title']
            if modelName == 'mpaso' or modelName == 'mpassi':
                varmpasname = var['mpas']
                #variableList = [varmpasname]
                #if varname=='fwc':
                #    variableList = variableList + ['timeMonthly_avg_layerThickness']

            colormap = var['colormap']
            clevels = var['clevels']
            if varname != 'iceArea':
                underColor = colormap(colorIndices0[0])
                overColor = colormap(colorIndices0[-1])
                if len(clevels) + 1 == len(colorIndices0):
                    # we have 2 extra values for the under/over so make the colormap
                    # without these values
                    colorIndices = colorIndices0[1:-1]
                elif len(clevels) - 1 != len(colorIndices0):
                    # indices list must be either one element shorter
                    # or one element longer than colorbarLevels list
                    raise ValueError('length mismatch between indices and colorbarLevels')
                colormap = cols.ListedColormap(colormap(colorIndices))
                colormap.set_under(underColor)
                colormap.set_over(overColor)
            cnorm = mpl.colors.BoundaryNorm(clevels, colormap.N)

            outfileLow  = f'{outdir}/{varname}_maxMLDlow_{titleMonthsToPlot}_{regionNameShort}_M{im:02d}.nc'
            outfileHigh = f'{outdir}/{varname}_maxMLDhigh_{titleMonthsToPlot}_{regionNameShort}_M{im:02d}.nc'
            #outfileMed = f'{outdir}/{varname}_maxMLDmed_{titleMonthsToPlot}_{regionNameShort}_M{im:02d}.nc'
            if not os.path.isfile(outfileLow):
                print(f'Composite file {outfileLow} does not exist. Creating it with ncea...')
                infiles = []
                for k in range(len(years_low)):
                    iy = years_low[k]
                    if im > np.max(monthsToPlot) and iy != startSimYear:
                        iy = iy-1  # pick months *preceding* the monthsToPlot period of each year
                    if modelComp == 'atm':
                        if isSingleVarFiles:
                            datafile = f'{rundir}/{varname}.{runName}.{modelName}.h0.{int(iy):04d}-{int(im):02d}.nc'
                        else:
                            datafile = f'{rundir}/{runName}.{modelName}.h0.{int(iy):04d}-{int(im):02d}.nc'
                    else:
                        if isSingleVarFiles:
                            datafile = f'{rundir}/{varname}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                        else:
                            datafile = f'{rundir}/{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                    # Check if file exists
                    if not os.path.isfile(datafile):
                        raise SystemExit(f'File {datafile} not found. Exiting...\n')
                    infiles.append(datafile)
                args = ['ncea', '-O', '-v', varmpasname]
                args.extend(infiles)
                args.append(outfileLow)
                subprocess.check_call(args)
            if not os.path.isfile(outfileHigh):
                print(f'Composite file {outfileHigh} does not exist. Creating it with ncea...')
                infiles = []
                for k in range(len(years_high)):
                    iy = years_high[k]
                    if im > np.max(monthsToPlot) and iy != startSimYear:
                        iy = iy-1  # pick months *preceding* the monthsToPlot period of each year
                    if modelComp == 'atm':
                        if isSingleVarFiles:
                            datafile = f'{rundir}/{varname}.{runName}.{modelName}.h0.{int(iy):04d}-{int(im):02d}.nc'
                        else:
                            datafile = f'{rundir}/{runName}.{modelName}.h0.{int(iy):04d}-{int(im):02d}.nc'
                    else:
                        if isSingleVarFiles:
                            datafile = f'{rundir}/{varname}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                        else:
                            datafile = f'{rundir}/{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                    # Check if file exists
                    if not os.path.isfile(datafile):
                        raise SystemExit(f'File {datafile} not found. Exiting...\n')
                    infiles.append(datafile)
                args = ['ncea', '-O', '-v', varmpasname]
                args.extend(infiles)
                args.append(outfileHigh)
                subprocess.check_call(args)

            # Read in composites

            # Plot
            if var['isvar3d']:
                for iz in range(len(dlevels)):
                    figtitleLow   = f'Composite for low maxMLD ({regionName})\nmonth={im}, {vartitle}, z={z[zlevels[iz]]:5.1f} m'
                    figtitleHigh  = f'Composite for high maxMLD ({regionName})\nmonth={im}, {vartitle}, z={z[zlevels[iz]]:5.1f} m'
                    figfileLow  = f'{figdir}/{varname}_depth{int(dlevels[iz]):04d}_maxMLDlow_{titleMonthsToPlot}_{regionNameShort}_M{im:02d}.png'
                    figfileHigh = f'{figdir}/{varname}_depth{int(dlevels[iz]):04d}_maxMLDhigh_{titleMonthsToPlot}_{regionNameShort}_M{im:02d}.png'

                    dsFieldLow  = xr.open_dataset(outfileLow).isel(Time=0, nVertLevels=zlevels[iz])
                    dsFieldHigh = xr.open_dataset(outfileHigh).isel(Time=0, nVertLevels=zlevels[iz])
                    fldLow  = dsFieldLow[varmpasname].values
                    fldHigh = dsFieldHigh[varmpasname].values

                    plt.figure(figsize=figsize, dpi=figdpi)
                    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
                    add_land_lakes_coastline(ax)

                    data_crs = ccrs.PlateCarree()
                    ax.set_extent([-50, 50, 60, 80], crs=data_crs)
                    gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6)
                    # This will work with cartopy 0.18:
                    #gl.xlocator = mticker.FixedLocator(np.arange(-180., 181., 40.))
                    #gl.ylocator = mticker.FixedLocator(np.arange(-80., 81., 20.))

                    sc = ax.scatter(lonCell, latCell, s=1.2, c=fldLow, cmap=colormap, norm=cnorm,
                                    marker='o', transform=data_crs)
                    cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5, extend='both')
                    cbar.ax.tick_params(labelsize=16, labelcolor='black')
                    cbar.set_label(var['units'], fontsize=14)

                    ax.set_title(figtitleLow, y=1.04, fontsize=16)
                    plt.savefig(figfileLow, bbox_inches='tight')
                    plt.close()

                    plt.figure(figsize=figsize, dpi=figdpi)
                    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
                    add_land_lakes_coastline(ax)

                    data_crs = ccrs.PlateCarree()
                    ax.set_extent([-50, 50, 60, 80], crs=data_crs)
                    gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6)

                    sc = ax.scatter(lonCell, latCell, s=1.2, c=fldHigh, cmap=colormap, norm=cnorm,
                                    marker='o', transform=data_crs)
                    cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5, extend='both')
                    cbar.ax.tick_params(labelsize=16, labelcolor='black')
                    cbar.set_label(var['units'], fontsize=14)

                    ax.set_title(figtitleHigh, y=1.04, fontsize=16)
                    plt.savefig(figfileHigh, bbox_inches='tight')
                    plt.close()
            else:
                figtitleLow   = f'Composite for low maxMLD ({regionName})\nmonth={im}, {vartitle}'
                figtitleHigh  = f'Composite for high maxMLD ({regionName})\nmonth={im}, {vartitle}'
                figfileLow  = f'{figdir}/{varname}_maxMLDlow_{titleMonthsToPlot}_{regionNameShort}_M{im:02d}.png'
                figfileHigh = f'{figdir}/{varname}_maxMLDhigh_{titleMonthsToPlot}_{regionNameShort}_M{im:02d}.png'

                dsFieldLow  = xr.open_dataset(outfileLow).isel(Time=0)
                dsFieldHigh = xr.open_dataset(outfileHigh).isel(Time=0)
                fldLow  = dsFieldLow[varmpasname].values
                fldHigh = dsFieldHigh[varmpasname].values

                plt.figure(figsize=figsize, dpi=figdpi)
                ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
                add_land_lakes_coastline(ax)

                data_crs = ccrs.PlateCarree()
                ax.set_extent([-50, 50, 60, 80], crs=data_crs)
                gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6)
                # This will work with cartopy 0.18:
                #gl.xlocator = mticker.FixedLocator(np.arange(-180., 181., 40.))
                #gl.ylocator = mticker.FixedLocator(np.arange(-80., 81., 20.))

                sc = ax.scatter(lonCell, latCell, s=1.2, c=fldLow, cmap=colormap, norm=cnorm,
                                marker='o', transform=data_crs)
                if varname != 'iceArea':
                    cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5, extend='both')
                else:
                    cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5)
                cbar.ax.tick_params(labelsize=16, labelcolor='black')
                cbar.set_label(var['units'], fontsize=14)

                ax.set_title(figtitleLow, y=1.04, fontsize=16)
                plt.savefig(figfileLow, bbox_inches='tight')
                plt.close()

                plt.figure(figsize=figsize, dpi=figdpi)
                ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
                add_land_lakes_coastline(ax)

                data_crs = ccrs.PlateCarree()
                ax.set_extent([-50, 50, 60, 80], crs=data_crs)
                gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6)

                sc = ax.scatter(lonCell, latCell, s=1.2, c=fldHigh, cmap=colormap, norm=cnorm,
                                marker='o', transform=data_crs)
                cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.05, shrink=.5, extend='both')
                cbar.ax.tick_params(labelsize=16, labelcolor='black')
                cbar.set_label(var['units'], fontsize=14)

                ax.set_title(figtitleHigh, y=1.04, fontsize=16)
                plt.savefig(figfileHigh, bbox_inches='tight')
                plt.close()
