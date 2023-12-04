from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import xarray
import numpy as np
import matplotlib.pyplot as plt

from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf
from mpas_analysis.shared.io.utility import get_files_year_month, decode_strings
from mpas_analysis.ocean.utility import compute_zmid

from geometric_features import FeatureCollection, read_feature_collection

from common_functions import timeseries_analysis_plot, add_inset

startYear = 1
endYear = 55
#startYear = 65
#endYear = 325
calendar = 'gregorian'

# Settings for nersc
#regionMaskDir = '/global/cfs/projectdirs/m1199/milena/mpas-region_masks'
#meshName = 'ARRM60to10'
#meshFile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/oARRM60to10/ocean.ARRM60to10.180715.nc'
#featureFile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/oceanOHCRegions.geojson'
#runName =
#runNameShort =
#rundir =
#isShortTermArchive = False # if True '{modelComp}/hist' will be affixed to rundir later on
 
# Settings for lcrc
#regionMaskDir = '/lcrc/group/e3sm/ac.milena/mpas-region_masks'
#meshName = 'EC30to60E2r2'
#meshFile = f'/lcrc/group/acme/public_html/inputdata/ocn/mpas-o/{meshName}/ocean.EC30to60E2r2.200908.nc'
#runName = '20210127_JRA_POPvertMix_EC30to60E2r2'
#runNameShort = 'JRA_POPvertMix_noSSSrestoring'
#rundir = '/lcrc/group/acme/ac.vanroekel/scratch/anvil/20210127_JRA_POPvertMix_EC30to60E2r2/run'
#isShortTermArchive = False # if True '{modelComp}/hist' will be affixed to rundir later on
 
# Settings for onyx
regionMaskDir = '/p/home/milena/mpas-region_masks'
meshName = 'ARRM10to60E2r1'
meshFile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runName = 'E3SMv2.1B60to10rA02'
runNameShort = 'E3SMv2.1B60to10rA02'
rundir = f'/p/work/osinski/archive/{runName}'
isShortTermArchive = True # if True '{modelComp}/hist' will be affixed to rundir later on

outdir = f'./timeseries_data/{runNameShort}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
figdir = f'./timeseries/{runNameShort}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

if os.path.exists(meshFile):
    dsMesh = xarray.open_dataset(meshFile)
    dsMesh = dsMesh.isel(Time=0)
else:
    raise IOError('No MPAS restart/mesh file found')
if 'landIceMask' in dsMesh:
    # only the region outside of ice-shelf cavities
    openOceanMask = dsMesh.landIceMask == 0
else:
    openOceanMask = None
areaCell = dsMesh.areaCell
globalArea = areaCell.sum()
refBottomDepth = dsMesh.refBottomDepth
maxLevelCell = dsMesh.maxLevelCell

regionGroups = ['Arctic Regions']
#regionGroups = ['OceanOHC Regions']
#regionGroups = ['Antarctic Regions']

# Choose either variables in timeSeriesStatsMonthly (either 3d or 2d) 
# or variables in timeSeriesStatsMonthlyMax (2d only)
#
#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'title': 'Maximum MLD',
#              'units': 'm',
#              'mpas': 'timeMonthlyMax_max_dThreshMLD'}
#            ]
#mpasComp = 'mpaso'
#modelComp = 'ocn'
#
mpasFile = 'timeSeriesStatsMonthly'
variables = [
             {'name': 'sensibleHeatFlux',
              'title': 'Sensible Heat Flux',
              'units': 'W/m$^2$',
              'mpas': 'timeMonthly_avg_sensibleHeatFlux'}
              #'mpas': 'timeMonthly_avg_penetrativeTemperatureFlux'}
              #'mpas': 'timeMonthly_avg_latentHeatFlux'}
              #'mpas': 'timeMonthly_avg_longWaveHeatFluxUp'}
              #'mpas': 'timeMonthly_avg_longWaveHeatFluxDown'}
              #'mpas': 'timeMonthly_avg_shortWaveHeatFlux'}
              #'mpas': 'timeMonthly_avg_seaIceHeatFlux'}
#             {'name': 'temperature',
#              'title': 'Temperature',
#              'units': '$^\circ$C',
#              'mpas': 'timeMonthly_avg_activeTracers_temperature'},
#             {'name': 'salinity',
#              'title': 'Salinity',
#              'units': 'PSU',
#              'mpas': 'timeMonthly_avg_activeTracers_salinity'},
#             {'name': 'potentialDensity',
#              'title': 'Potential Density',
#              'units': 'kg m$^{-3}$',
#              'mpas': 'timeMonthly_avg_potentialDensity'}
            ]
mpasComp = 'mpaso'
modelComp = 'ocn'
#variables = [
#             {'name': '',
#              'title': '',
#              'units': '',
#              'mpas': ''}
#            ]
#mpasComp = 'mpassi'
#modelComp = 'ice'

if isShortTermArchive:
    rundir = f'{rundir}/{modelComp}/hist'

startDate = f'{startYear:04d}-01-01_00:00:00'
endDate = f'{endYear:04d}-12-31_23:59:59'
years = range(startYear, endYear + 1)

# Only needed for 3d monthly averaged variables
zmins = [-100., -6000.]
zmaxs = [0., 0.]

if mpasFile=='timeSeriesStatsMonthly':
    variableList = [var['mpas'] for var in variables] + \
        ['timeMonthly_avg_layerThickness']
    timeVariableNames = ['xtime_startMonthly', 'xtime_endMonthly']
else:
    variableList = [var['mpas'] for var in variables]
    timeVariableNames = ['xtime_startMonthlyMax', 'xtime_endMonthlyMax']

for regionGroup in regionGroups:

    groupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')

    regionMaskFile = f'{regionMaskDir}/{meshName}_{groupName}.nc'
    if os.path.exists(regionMaskFile):
        dsRegionMask = xarray.open_dataset(regionMaskFile)
        regionNames = decode_strings(dsRegionMask.regionNames)
        if regionGroup==regionGroups[0]:
            regionNames.append('Global')
        nRegions = np.size(regionNames)
    else:
        raise IOError('No regional mask file found')

    featureFile = f'{regionMaskDir}/{groupName}.geojson'
    if os.path.exists(featureFile):
        fcAll = read_feature_collection(featureFile)
    else:
        raise IOError('No feature file found for this region group')

    # Compute regional averages one year at a time
    for year in years:

        # Load in entire data set for all chosen variables (all x's, y's, z's)
        # for the indicated time period
        datasets = []
        for month in range(1, 13):
            inputFile = f'{rundir}/{runName}.{mpasComp}.hist.am.{mpasFile}.{year:04d}-{month:02d}-01.nc'
            if not os.path.exists(inputFile):
                raise IOError(f'Input file: {inputFile} not found')

            dsTimeSlice = open_mpas_dataset(fileName=inputFile,
                                            calendar=calendar,
                                            timeVariableNames=timeVariableNames,
                                            variableList=variableList,
                                            startDate=startDate,
                                            endDate=endDate)
            datasets.append(dsTimeSlice)
        # combine data sets into a single data set
        dsIn = xarray.concat(datasets, 'Time')

        if mpasFile=='timeSeriesStatsMonthly': # monthly averages 
            layerThickness = dsIn.timeMonthly_avg_layerThickness
            zMid = compute_zmid(refBottomDepth, maxLevelCell, layerThickness)

            # Compute regional averages one depth range at a time
            for k in range(len(zmins)):

                zmin = zmins[k]
                zmax = zmaxs[k]
                # Global depth-masked layer volume
                depthMask = np.logical_and(zMid >= zmin, zMid <= zmax)
                layerVol = areaCell*layerThickness.where(depthMask, drop=False)
                globalLayerVol = layerVol.sum(dim='nVertLevels').sum(dim='nCells')

                timeSeriesFile = f'{outdir}/{groupName}_z{np.abs(np.int(zmax)):04d}-{np.abs(np.int(zmin)):04d}_year{year:04d}.nc'

                if not os.path.exists(timeSeriesFile):
                    print(f'Computing regional time series for year={year}, depth range= {zmax}, {zmin}')

                    # Compute regional quantities for each depth range
                    datasets = []
                    regionIndices = []
                    for regionName in regionNames:
                        print(f'    region: {regionName}')

                        if regionName=='Global':
                            regionIndices.append(nRegions-1)
                        else:
                            regionIndex = regionNames.index(regionName)
                            regionIndices.append(regionIndex)

                            dsMask = dsRegionMask.isel(nRegions=regionIndex)
                            cellMask = dsMask.regionCellMasks == 1
                            if openOceanMask is not None:
                                cellMask = np.logical_and(cellMask, openOceanMask)

                            localArea = areaCell.where(cellMask, drop=True)
                            regionalArea = localArea.sum()
                            localLayerVol = layerVol.where(cellMask, drop=True)
                            regionalLayerVol = localLayerVol.sum(dim='nVertLevels').sum(dim='nCells')

                        dsOut = xarray.Dataset()
                        for var in variables:
                            outName = var['name']
                            mpasVarName = var['mpas']
                            units = var['units']
                            description = var['title']

                            timeSeries = dsIn[mpasVarName]
                            if 'nVertLevels' in timeSeries.dims:
                                timeSeries = timeSeries.where(depthMask, drop=False)
                                if regionName=='Global':
                                    timeSeries = \
                                        (layerVol*timeSeries).sum(dim='nVertLevels').sum(dim='nCells') / globalLayerVol
                                else:
                                    timeSeries = \
                                        (localLayerVol*timeSeries.where(cellMask, drop=True)).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                            else:
                                if regionName=='Global':
                                    timeSeries = \
                                        (areaCell*timeSeries).sum(dim='nCells') / globalArea
                                else:
                                    timeSeries = \
                                        (localArea*timeSeries.where(cellMask, drop=True)).sum(dim='nCells') / regionalArea

                            dsOut[outName] = timeSeries
                            dsOut[outName].attrs['units'] = units
                            dsOut[outName].attrs['description'] = description

                        if regionName=='Global':
                            dsOut['totalVol'] = globalLayerVol
                            dsOut['totalArea'] = globalArea
                        else:
                            dsOut['totalVol'] = regionalLayerVol
                            dsOut['totalArea'] = regionalArea
                        dsOut.totalVol.attrs['units'] = 'm^3'
                        dsOut.totalArea.attrs['units'] = 'm^2'
                        dsOut['zbounds'] = ('nbounds', [zmin, zmax])
                        dsOut.zbounds.attrs['units'] = 'm'

                        datasets.append(dsOut)

                    # combine data sets into a single data set
                    dsOut = xarray.concat(datasets, 'nRegions')

                    # a few variables have become time or region dependent and shouldn't be
                    dsOut['totalVol'] = dsOut['totalVol'].isel(Time=0, drop=True)
                    dsOut['zbounds'] = dsOut['zbounds'].isel(nRegions=0, drop=True)

                    write_netcdf(dsOut, timeSeriesFile)
                else:
                    print(f'Time series file already exists for year {year} and depth range {zmax}, {zmin}. Skipping it...')
        else: # monthly maxima
            timeSeriesFile = f'{outdir}/{groupName}_max_year{year:04d}.nc'

            if not os.path.exists(timeSeriesFile):
                print(f'Computing regional time series for year={year}')

                # Compute regional quantities for each depth range
                datasets = []
                regionIndices = []
                for regionName in regionNames:
                    print(f'    region: {regionName}')

                    if regionName=='Global':
                        regionIndices.append(nRegions-1)
                    else:
                        regionIndex = regionNames.index(regionName)
                        regionIndices.append(regionIndex)

                        dsMask = dsRegionMask.isel(nRegions=regionIndex)
                        cellMask = dsMask.regionCellMasks == 1
                        if openOceanMask is not None:
                            cellMask = np.logical_and(cellMask, openOceanMask)

                        localArea = areaCell.where(cellMask, drop=True)
                        regionalArea = localArea.sum()

                    dsOut = xarray.Dataset()
                    for var in variables:
                        outName = var['name']
                        mpasVarName = var['mpas']
                        units = var['units']
                        description = var['title']

                        timeSeries = dsIn[mpasVarName]
                        if regionName=='Global':
                            timeSeries = \
                                (areaCell*timeSeries).sum(dim='nCells') / globalArea
                        else:
                            timeSeries = \
                                (localArea*timeSeries.where(cellMask, drop=True)).sum(dim='nCells') / regionalArea

                        dsOut[outName] = timeSeries
                        dsOut[outName].attrs['units'] = units
                        dsOut[outName].attrs['description'] = description

                    if regionName=='Global':
                        dsOut['totalArea'] = globalArea
                    else:
                        dsOut['totalArea'] = regionalArea
                    dsOut.totalArea.attrs['units'] = 'm^2'

                    datasets.append(dsOut)

                # combine data sets into a single data set
                dsOut = xarray.concat(datasets, 'nRegions')

                write_netcdf(dsOut, timeSeriesFile)
            else:
                print(f'Time series file already exists for year {year}. Skipping it...')

    # Time series calculated, now make plots for each variable, region, and depth range (if appropriate)
    if mpasFile=='timeSeriesStatsMonthly': # monthly averages 
        for k in range(len(zmins)):
            zmin = zmins[k]
            zmax = zmaxs[k]
            timeSeriesFiles = []
            for year in years:
                timeSeriesFile = '{outdir}/{groupName}_z{np.abs(np.int(zmax)):04d}-{np.abs(np.int(zmin)):04d}_year{year:04d}.nc'
                timeSeriesFiles.append(timeSeriesFile)
            print('********************', timeSeriesFile)

            for regionIndex, regionName in enumerate(regionNames):
                regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
                fc = FeatureCollection()
                for feature in fcAll.features:
                    if feature['properties']['name'] == regionName:
                        fc.add_feature(feature)
                        break

                dsIn = xarray.open_mfdataset(timeSeriesFiles, combine='nested',
                                             concat_dim='Time', decode_times=False).isel(nRegions=regionIndex)

                zbounds = dsIn.zbounds.values[0]

                #movingAverageMonths = 1
                movingAverageMonths = 12

                for var in variables:
                    # Add something about checking whether var is available in dsIn
                    varName = var['name']
                    title = var['title']
                    units = var['units']
                    field = [dsIn[varName]]
                    xLabel = 'Time (yr)'
                    yLabel = f'{title} ({units})'
                    lineColors = ['k']
                    lineWidths = [2.5]
                    legendText = [runNameShort]
                    title = f'Volume-Mean {title} in {regionName} ({zbounds[0]} < z < {zbounds[1]} m; {np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {units})'
                    figFileName = f'{figdir}/{regionNameShort}_z{np.abs(np.int(zmax)):04d}-{np.abs(np.int(zmin)):04d}_{varName}_years{years[0]}-{years[-1]}.png'

                    fig = timeseries_analysis_plot(field, movingAverageMonths,
                                                   title, xLabel, yLabel,
                                                   calendar=calendar,
                                                   lineColors=lineColors,
                                                   lineWidths=lineWidths,
                                                   legendText=legendText)

                    # do this before the inset because otherwise it moves the inset
                    # and cartopy doesn't play too well with tight_layout anyway
                    plt.tight_layout()

                    if regionName!='Global':
                        add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)

                    plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
                    plt.close()
    
    else: # monthly maxima
    
        timeSeriesFiles = []
        for year in years:
            timeSeriesFile = f'{outdir}/{groupName}_max_year{year:04d}.nc'
            timeSeriesFiles.append(timeSeriesFile)

        for regionIndex, regionName in enumerate(regionNames):
            regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
            fc = FeatureCollection()
            for feature in fcAll.features:
                if feature['properties']['name'] == regionName:
                    fc.add_feature(feature)
                    break

            dsIn = xarray.open_mfdataset(timeSeriesFiles, combine='nested',
                                         concat_dim='Time', decode_times=False).isel(nRegions=regionIndex)

            #movingAverageMonths = 1
            movingAverageMonths = 12

            for var in variables:
                # Add something about checking whether var is available in dsIn
                varName = var['name']
                title = var['title']
                units = var['units']
                field = [dsIn[varName]]
                xLabel = 'Time (yr)'
                yLabel = f'{title} ({units})'
                lineColors = ['k']
                lineWidths = [2.5]
                legendText = [runNameShort]
                title = f'Mean {title} in {regionName} ({np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {units})'
                figFileName = f'{figdir}/{regionNameShort}_{varName}_years{years[0]}-{years[-1]}.png'

                fig = timeseries_analysis_plot(field, movingAverageMonths,
                                               title, xLabel, yLabel,
                                               calendar=calendar,
                                               lineColors=lineColors,
                                               lineWidths=lineWidths,
                                               legendText=legendText)

                # do this before the inset because otherwise it moves the inset
                # and cartopy doesn't play too well with tight_layout anyway
                plt.tight_layout()

                if regionName!='Global':
                    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)

                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
                plt.close()
