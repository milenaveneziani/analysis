from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import subprocess
from distutils.spawn import find_executable
import xarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
from matplotlib.pyplot import cm
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import BoundaryNorm
import cmocean

from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf
from mpas_analysis.shared.io.utility import get_files_year_month, decode_strings

from geometric_features import FeatureCollection, read_feature_collection

from common_functions import hovmoeller_plot, add_inset, compute_regional_maskfile

# Settings for blues
meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/EC30to60E2r2/ocean.EC30to60E2r2.210210.nc'
regionMaskFile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/EC30to60E2r2_arcticRegions20211105.nc'
featureFile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/arcticRegions.geojson'

#runName = '20210413_JRA_tidalMixingBnLwithKPP_EC30to60E2r2'
#runNameShort = 'JRA_tidalMixingBnLwithKPP'
#modeldir = '/lcrc/group/e3sm/ac.milena/scratch/anvil/20210413_JRA_tidalMixingBnLwithKPP_EC30to60E2r2/run'
#
#runName = '20210414_JRA_tidalMixingBnLhighKsWithKPP_EC30to60E2r2'
#runNameShort = 'JRA_tidalMixingBnLhighKsWithKPP'
#modeldir = '/lcrc/group/e3sm/ac.milena/scratch/anvil/20210414_JRA_tidalMixingBnLhighKsWithKPP_EC30to60E2r2/run'
#
#runName = '20210401_JRA_constantMix_EC30to60E2r2'
#runNameShort = 'JRA_constantMix'
#modeldir = '/lcrc/group/e3sm/ac.vanroekel/scratch/anvil/20210401_JRA_constantMix_EC30to60E2r2/run'
#
#runName = 'v2Visbeck_RediequalGM_lowMaxKappa.LR.picontrol'
#runNameShort = 'v2Visbeck_RediequalGM_lowMaxKappa.LR.picontrol'
#modeldir = '/lcrc/group/e3sm/ac.milena/E3SMv2/v2Visbeck_RediequalGM_lowMaxKappa.LR.picontrol/run'
#
#runName = 'v2Visbeck_RediequalGM.LR.picontrol'
#runNameShort = 'v2Visbeck_RediequalGM.LR.picontrol'
#modeldir = '/lcrc/group/e3sm/ac.milena/E3SMv2/v2Visbeck_RediequalGM.LR.picontrol/run'
#
runName = 'v2plusKPP_GM_Redi_mods.LR.piControl'
runNameShort = 'v2plusKPP_GM_Redi_mods.LR.piControl'
modeldir = '/lcrc/group/e3sm/ac.vanroekel/E3SMv2/v2plusKPP_GM_Redi_mods.LR.piControl/run'

# Settings for compy
#meshfile = '/compyfs/inputdata/ocn/mpas-o/EC30to60E2r2/ocean.EC30to60E2r2.200908.nc'
#regionMaskFile = '/compyfs/vene705/mpas-region_masks/EC30to60E2r2_oceanOHCRegions20201120.nc'
#featureFile = '/compyfs/vene705/mpas-region_masks/oceanOHCRegions.geojson'
#
#modeldir = '/compyfs/zhen797/E3SM_simulations/20201108.alpha5_55_fallback.piControl.ne30pg2_r05_EC30to60E2r2-1900_ICG.compy/archive/ocn/hist'
#runName = '20201108.alpha5_55_fallback.piControl.ne30pg2_r05_EC30to60E2r2-1900_ICG.compy'
#runNameShort = 'alpha5_55_fallback'
#modeldir = '/compyfs/zhen797/E3SM_simulations/20201124.alpha5_59_fallback.piControl.ne30pg2_r05_EC30to60E2r2-1900_ICG.compy/archive/ocn/hist/'
#runName = '20201124.alpha5_59_fallback.piControl.ne30pg2_r05_EC30to60E2r2-1900_ICG.compy'
#runNameShort = 'alpha5_59_fallback'

outdir = './timeseries_data/{}'.format(runNameShort)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
figdir = './timeseries/{}'.format(runNameShort)
if not os.path.isdir(figdir):
    os.makedirs(figdir)

if os.path.exists(meshfile):
    dsMesh = xarray.open_dataset(meshfile)
    dsMesh = dsMesh.isel(Time=0)
else:
    raise IOError('No MPAS mesh file found')
areaCell = dsMesh.areaCell
if 'landIceMask' in dsMesh:
    # only the region outside of ice-shelf cavities
    openOceanMask = dsMesh.landIceMask == 0
else:
    openOceanMask = None
refBottomDepth = dsMesh.refBottomDepth
maxLevelCell = dsMesh.maxLevelCell
nVertLevels = dsMesh.sizes['nVertLevels']
vertIndex = xarray.DataArray.from_dict(
    {'dims': ('nVertLevels',), 'data': np.arange(nVertLevels)})
depthMask = (vertIndex < maxLevelCell).transpose('nCells', 'nVertLevels')

if not os.path.exists(regionMaskFile):
    compute_regional_maskfile(meshfile, featureFile, regionMaskFile)
dsRegionMask = xarray.open_dataset(regionMaskFile)
regionNames = decode_strings(dsRegionMask.regionNames)
regionNames.append('Global')
nRegions = np.size(regionNames)

startYear = 1
#endYear = 100
endYear = 69
calendar = 'gregorian'

variables = [{'name': 'kvertical',
              'title': 'Vertical diffusivity',
              'units': 'x1e-4 m$^2$/s',
              'mpas': 'timeMonthly_avg_vertDiffTopOfCell',
              'colormap': plt.get_cmap('viridis'),
              #'colormap': cmocean.cm.thermal,
              #'clevels': np.log10([0.2e-4, 0.5e-4, 1.0e-4, 2.0e-4, 4.0e-4, 6.0e-4, 8.0e-4, 10.0e-4, 15.0e-4]),
              'clevels': [5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0],
              'colorIndices': [0, 28, 57, 85, 113, 142, 170, 198, 227, 255],
              'fac': 1e4}]

startDate = '{:04d}-01-01_00:00:00'.format(startYear)
endDate = '{:04d}-12-31_23:59:59'.format(endYear)
years = range(startYear, endYear + 1)

variableList = [var['mpas'] for var in variables] + \
    ['timeMonthly_avg_layerThickness'] + ['timeMonthly_avg_dThreshMLD'] + \
    ['timeMonthly_avg_boundaryLayerDepth']

timeSeriesFile0 = '{}/Kv_trends_vsdepth'.format(outdir)

# Compute regional averages one year at a time
for year in years:

    timeSeriesFile = '{}_year{:04d}.nc'.format(timeSeriesFile0, year)

    if not os.path.exists(timeSeriesFile):
        print('\nComputing regional time series for year={}'.format(year))

        datasets = []
        for month in range(1, 13):
            inputFile = '{}/{}.mpaso.hist.am.timeSeriesStatsMonthly.{:04d}-{:02d}-01.nc'.format(
                modeldir, runName, year, month)
            #inputFile = '{}/mpaso.hist.am.timeSeriesStatsMonthly.{:04d}-{:02d}-01.nc'.format(
            #    modeldir, year, month)
            if not os.path.exists(inputFile):
                raise IOError('Input file: {} not found'.format(inputFile))

            dsTimeSlice = open_mpas_dataset(fileName=inputFile,
                                            calendar=calendar,
                                            variableList=variableList,
                                            startDate=startDate,
                                            endDate=endDate)
            datasets.append(dsTimeSlice)
        # combine data sets into a single data set
        dsIn = xarray.concat(datasets, 'Time')

        # Global depth-masked layer thickness and layer volume
        layerThickness = dsIn.timeMonthly_avg_layerThickness
        layerThickness = layerThickness.where(depthMask, drop=False)
        layerVol = areaCell*layerThickness

        datasets = []
        regionIndices = []
        for regionName in regionNames:
            print('    region: {}'.format(regionName))

            # Compute region total area and, for regionName
            # other than Global, compute regional mask and
            # regionally masked layer volume
            if regionName=='Global':
                regionIndices.append(nRegions-1)

                totalArea = areaCell.sum()
                if year==years[0]:
                    print('      totalArea: {} mil. km^2'.format(1e-12*totalArea.values))

                regionMaxLevelCell = nVertLevels

                mld = (dsIn['timeMonthly_avg_dThreshMLD']*areaCell).sum(dim='nCells') / totalArea
                bld = (dsIn['timeMonthly_avg_boundaryLayerDepth']*areaCell).sum(dim='nCells') / totalArea
            else:
                regionIndex = regionNames.index(regionName)
                regionIndices.append(regionIndex)

                dsMask = dsRegionMask.isel(nRegions=regionIndex)
                cellMask = dsMask.regionCellMasks == 1
                if openOceanMask is not None:
                    cellMask = np.logical_and(cellMask, openOceanMask)

                localArea = areaCell.where(cellMask, drop=True)
                totalArea = localArea.sum()
                if year==years[0]:
                    print('      totalArea: {} mil. km^2'.format(1e-12*totalArea.values))
                localLayerVol = layerVol.where(cellMask, drop=True)

                regionMaxLevelCell = np.max(maxLevelCell.where(cellMask, drop=True))

                mld = dsIn['timeMonthly_avg_dThreshMLD'].where(cellMask, drop=True)
                mld = (mld*localArea).sum(dim='nCells') / totalArea
                bld = dsIn['timeMonthly_avg_boundaryLayerDepth'].where(cellMask, drop=True)
                bld = (bld*localArea).sum(dim='nCells') / totalArea

            # Temporary dsOut (xarray dataset) containing results for
            # all variables for one single region
            dsOut = xarray.Dataset()
            # Compute layer-volume weighted averages (or sums for OHC)
            for var in variables:
                outName = var['name']
                mpasVarName = var['mpas']
                units = var['units']
                description = var['title']

                timeSeries = dsIn[mpasVarName]
                timeSeries = timeSeries.rolling(nVertLevelsP1=2, center=True).mean().dropna('nVertLevelsP1')
                timeSeries = timeSeries.rename({'nVertLevelsP1': 'nVertLevels'})
                timeSeries = timeSeries.where(depthMask, drop=False)
                if regionName=='Global':
                    timeSeries = (layerVol*timeSeries).sum(dim='nCells') / layerVol.sum(dim='nCells')
                else:
                    timeSeries = timeSeries.where(cellMask, drop=True)
                    timeSeries = (localLayerVol*timeSeries).sum(dim='nCells') / localLayerVol.sum(dim='nCells')

                dsOut[outName] = timeSeries
                dsOut[outName].attrs['units'] = units
                dsOut[outName].attrs['description'] = description

            dsOut['mld'] = mld
            dsOut.mld.attrs['units'] = 'm'
            dsOut['bld'] = bld
            dsOut.bld.attrs['units'] = 'm'
            dsOut['totalArea'] = totalArea
            dsOut.totalArea.attrs['units'] = 'm^2'
            dsOut['regionMaxLevelCell'] = regionMaxLevelCell
            dsOut.regionMaxLevelCell.attrs['description'] = 'Maximum of maxLevelCell for each region'

            datasets.append(dsOut)

        # Combine data sets into a single data set for all regions
        dsOut = xarray.concat(datasets, 'nRegions')
        dsOut['refBottomDepth'] = refBottomDepth

        write_netcdf(dsOut, timeSeriesFile)
    else:
        print('Time series file already exists for year {}. Skipping it...'.format(year))

# Make plot
timeSeriesFiles = []
for year in years:
    timeSeriesFile = '{}_year{:04d}.nc'.format(timeSeriesFile0, year)
    timeSeriesFiles.append(timeSeriesFile)

if os.path.exists(featureFile):
    fcAll = read_feature_collection(featureFile)
else:
    raise IOError('No feature file found')

for regionIndex, regionName in enumerate(regionNames):
    print('    region: {}'.format(regionName))
    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == regionName:
            fc.add_feature(feature)
            break

    dsIn = xarray.open_mfdataset(timeSeriesFiles, combine='nested',
                                 concat_dim='Time', decode_times=False).isel(nRegions=regionIndex)

    #movingAverageMonths = 12
    movingAverageMonths = 1

    depths = dsIn.refBottomDepth.values[0]
    z = np.zeros(depths.shape)
    z[0] = -0.5 * depths[0]
    z[1:] = -0.5 * (depths[0:-1] + depths[1:])

    Time = dsIn.Time.values

    regionMaxLevelCell = dsIn.regionMaxLevelCell.isel(Time=0).values
    regionMaxLevelCell = np.floor(regionMaxLevelCell).astype(int)

    mld = dsIn.mld
    bld = dsIn.bld

    for var in variables:
        varName = var['name']
        factor = var['fac']

        clevels = var['clevels']
        colormap0 = var['colormap']
        colorIndices0 = var['colorIndices']
        underColor = colormap0(colorIndices0[0])
        overColor = colormap0(colorIndices0[-1])
        if len(clevels)+1 == len(colorIndices0):
            # we have 2 extra values for the under/over so make the colormap
            # without these values
            colorIndices = colorIndices0[1:-1]
        else:
            colorIndices = colorIndices0
        colormap = cols.ListedColormap(colormap0(colorIndices))
        colormap.set_under(underColor)
        colormap.set_over(overColor)
        cnorm = mpl.colors.BoundaryNorm(clevels, colormap.N)

        field = factor*dsIn[varName]

        # Compute first-year average (note that this assumes monthly fields)
        fieldMean = field.isel(Time=range(12)).mean(dim='Time')

        # Compute moving average of the anomaly with respect to first-year average
        N = movingAverageMonths
        if movingAverageMonths != 1:
            movingAverageDepthSlices = []
            for nVertLevel in range(nVertLevels):
                depthSlice = field.isel(nVertLevels=nVertLevel) - fieldMean.isel(nVertLevels=nVertLevel)
                mean = pd.Series.rolling(depthSlice.to_series(), N,
                                         center=True).mean()
                mean = xarray.DataArray.from_series(mean)
                mean = mean[int(N / 2.0):-int(round(N / 2.0) - 1)]
                movingAverageDepthSlices.append(mean)
            field = xarray.DataArray(movingAverageDepthSlices)
        else:
            field = field.transpose()

        xLabel = 'Time (yr)'
        yLabel = 'Depth (m)'
        #title = '{} Anomaly, {}'.format(var['title'], regionName)
        title = '{}, {}\n{}'.format(var['title'], regionName, runNameShort)
        figFileName = '{}/{}vsTimeDepth_{}.png'.format(figdir, varName,
                regionName[0].lower()+''.join(e for e in regionName[1:] if e.isalnum()))
                #regionName[0].lower()+regionName[1:].replace(' ', '').replace(')', '').replace('(', '').replace('\\', ''))

        #fig = hovmoeller_plot(Time[N-1:], z, np.log10(field.values), colormap, cnorm, clevels,
        fig = hovmoeller_plot(Time[N-1:], z, field.values, colormap, cnorm, clevels,
                              title, xLabel, yLabel, calendar, kmax=regionMaxLevelCell,
                              mld=mld, bld=bld, colorbarLabel=var['units'], titleFontSize=None,
                              figsize=(15, 6), dpi=None)

        # do this before the inset because otherwise it moves the inset
        # and cartopy doesn't play too well with tight_layout anyway
        plt.tight_layout()

        if regionName!='Global':
            add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1)

        plt.savefig(figFileName, dpi='figure', bbox_inches='tight',
                    pad_inches=0.1)
        plt.close()
