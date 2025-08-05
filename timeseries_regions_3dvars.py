from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf_with_fill
#from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf
from mpas_analysis.shared.io.utility import get_files_year_month, decode_strings
from mpas_analysis.ocean.utility import compute_zmid

from geometric_features import FeatureCollection, read_feature_collection

from common_functions import timeseries_analysis_plot, add_inset

startYear = 1
endYear = 59
#endYear = 246 # rA07
#endYear = 386 # rA02
calendar = 'gregorian'

# Settings for nersc
#regionMaskDir = '/global/cfs/cdirs/m1199/milena/mpas-region_masks'
#meshName = 'ARRM10to60E2r1'
#meshFile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runName = 'E3SM-Arcticv2.1_historical0301'
#runNameShort = 'E3SMv2.1-Arctic-historical0301'
#rundir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runName}'
#isShortTermArchive = True # if True 'archive/ocn/hist' will be affixed to rundir later on
 
# Settings for lcrc
#regionMaskDir = '/lcrc/group/e3sm/ac.milena/mpas-region_masks'
#meshName = 'EC30to60E2r2'
#meshFile = f'/lcrc/group/acme/public_html/inputdata/ocn/mpas-o/{meshName}/ocean.EC30to60E2r2.200908.nc'
#runName = '20210127_JRA_POPvertMix_EC30to60E2r2'
#runNameShort = 'JRA_POPvertMix_noSSSrestoring'
#rundir = '/lcrc/group/acme/ac.vanroekel/scratch/anvil/20210127_JRA_POPvertMix_EC30to60E2r2/run'
#isShortTermArchive = False
 
# Settings for erdc.hpc.mil
regionMaskDir = '/p/home/milena/mpas-region_masks'
meshName = 'ARRM10to60E2r1'
meshFile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runName = 'E3SMv2.1G60to10_01'
runNameShort = 'E3SMv2.1G60to10_01'
#runName = 'E3SMv2.1B60to10rA02'
#runNameShort = 'E3SMv2.1B60to10rA02'
rundir = f'/p/cwfs/milena/{runName}'
#runName = 'E3SMv2.1B60to10rA07'
#runNameShort = 'E3SMv2.1B60to10rA07'
#rundir = f'/p/cwfs/apcraig/archive/{runName}'
isShortTermArchive = True # if True 'archive/ocn/hist' will be affixed to rundir later on

# Settings for chicoma
#regionMaskDir = '/users/milena/mpas-region_masks'
#meshName = 'RRSwISC6to18E3r5'
#meshFile = f'/usr/projects/e3sm/inputdata/ocn/mpas-o/{meshName}/mpaso.RRSwISC6to18E3r5.20240327.nc'
#runName = '20240726.icFromLRGcase.GMPAS-JRA1p5.TL319_RRSwISC6to18E3r5.chicoma'
#runNameShort = 'GMPAS-JRA1p5.TL319_RRSwISC6to18E3r5.icFromLRGcase'
#rundir = f'/lustre/scratch4/turquoise/milena/E3SMv3/{runName}/{runName}/run'
#isShortTermArchive = False

computeDepthAvg = True
# Relevant only for computeDepthAvg = True
zmins = [-300.]
zmaxs = [0.]
# Relevant only for computeDepthAvg = False
dlevels = [0.]

#regionGroups = ['Arctic Regions']
#regionGroups = ['arctic_atlantic_budget_regions_new20240408']
#regionGroups = ['OceanOHC Regions']
#regionGroups = ['Antarctic Regions']
regionGroups = ['southAtlantic_eastWest_regions']

# Choose ocean 3d variables to process
#
mpasFile = 'timeSeriesStatsMonthly'
variables = [
             {'name': 'temperature',
              'title': 'Temperature',
              'units': '$^\circ$C',
              'factor': 1,
              'mpas': 'timeMonthly_avg_activeTracers_temperature'},
             {'name': 'salinity',
              'title': 'Salinity',
              'units': 'psu',
              'factor': 1,
              'mpas': 'timeMonthly_avg_activeTracers_salinity'}
            ]

if isShortTermArchive:
    if runName=='E3SMv2.1B60to10rA07':
        rundir = f'{rundir}/ocn/hist'
    else:
        rundir = f'{rundir}/archive/ocn/hist'

outdir = f'./timeseries_data/{runName}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
figdir = f'./timeseries/{runName}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

if os.path.exists(meshFile):
    dsMesh = xr.open_dataset(meshFile)
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
depth = dsMesh.bottomDepth
maxLevelCell = dsMesh.maxLevelCell

# Find model levels for each depth level (relevant if computeDepthAvg = False)
z = dsMesh.refBottomDepth
zlevels = np.zeros(np.shape(dlevels), dtype=np.int64)
for k in range(len(dlevels)):
    dz = np.abs(z.values-dlevels[k])
    zlevels[k] = np.argmin(dz)

startDate = f'{startYear:04d}-01-01_00:00:00'
endDate = f'{endYear:04d}-12-31_23:59:59'
years = range(startYear, endYear + 1)

variableList = [var['mpas'] for var in variables] + ['timeMonthly_avg_layerThickness']
timeVariableNames = ['xtime_startMonthly', 'xtime_endMonthly']

for regionGroup in regionGroups:

    groupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')

    regionMaskFile = f'{regionMaskDir}/{meshName}_{groupName}.nc'
    if os.path.exists(regionMaskFile):
        dsRegionMask = xr.open_dataset(regionMaskFile)
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

        # Load in entire data set for all chosen variables (all x's, y's, z's) for each year
        datasets = []
        for month in range(1, 13):
            inputFile = f'{rundir}/{runName}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
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
        dsIn = xr.concat(datasets, 'Time')

        if computeDepthAvg is True: # computeDepthAvg=True case
            layerThickness = dsIn.timeMonthly_avg_layerThickness
            zMid = compute_zmid(depth, maxLevelCell, layerThickness)

            # Compute regional averages one depth range at a time
            for k in range(len(zmins)):
                zmin = zmins[k]
                zmax = zmaxs[k]

                timeSeriesFile = f'{outdir}/{groupName}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_year{year:04d}.nc'

                if not os.path.exists(timeSeriesFile):
                    print(f'Computing regional time series for year={year}, depth range= {zmax}, {zmin}')

                    # Global depth-masked layer volume
                    depthMask = np.logical_and(zMid >= zmin, zMid <= zmax)
                    layerVol = areaCell * (layerThickness.where(depthMask, drop=False))
                    globalLayerVol = layerVol.sum(dim='nVertLevels').sum(dim='nCells')

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

                        dsOut = xr.Dataset()
                        for var in variables:
                            outName = var['name']
                            mpasVarName = var['mpas']
                            units = var['units']
                            description = var['title']

                            timeSeries = dsIn[mpasVarName]
                            timeSeries = timeSeries.where(depthMask, drop=False)
                            if regionName=='Global':
                                timeSeries = \
                                    (layerVol*timeSeries).sum(dim='nVertLevels').sum(dim='nCells') / globalLayerVol
                            else:
                                timeSeries = \
                                    (localLayerVol*timeSeries.where(cellMask, drop=True)).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol

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

                        dsOut['regionNames'] = regionName

                        datasets.append(dsOut)

                    # combine data sets into a single data set
                    dsOut = xr.concat(datasets, 'nRegions')

                    # zbounds has become region dependent and shouldn't be
                    dsOut['zbounds'] = dsOut['zbounds'].isel(nRegions=0, drop=True)

                    #write_netcdf(dsOut, timeSeriesFile)
                    write_netcdf_with_fill(dsOut, timeSeriesFile)
                else:
                    print(f'Time series file already exists for year {year} and depth range {zmax}, {zmin}. Skipping it...')

        else:  # computeDepthAvg=False case

            # Compute regional averages one depth level at a time
            for k in range(len(dlevels)):

                timeSeriesFile = f'{outdir}/{groupName}_depth{int(dlevels[k]):04d}_year{year:04d}.nc'

                if not os.path.exists(timeSeriesFile):
                    print(f'Computing regional time series for year={year}, depth level={int(dlevels[k])}')


                    # Compute regional quantities for each depth level
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

                        dsOut = xr.Dataset()
                        for var in variables:
                            outName = var['name']
                            mpasVarName = var['mpas']
                            units = var['units']
                            description = var['title']

                            timeSeries = dsIn[mpasVarName].isel(nVertLevels=zlevels[k])
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

                        dsOut['regionNames'] = regionName

                        datasets.append(dsOut)

                    # combine data sets into a single data set
                    dsOut = xr.concat(datasets, 'nRegions')

                    #write_netcdf(dsOut, timeSeriesFile)
                    write_netcdf_with_fill(dsOut, timeSeriesFile)
                else:
                    print(f'Time series file already exists for year {year}. Skipping it...')

    # Time series calculated, now make plots for each variable, region, and depth range (if appropriate)
    if computeDepthAvg is True:
        for k in range(len(zmins)):
            zmin = zmins[k]
            zmax = zmaxs[k]
            timeSeriesFiles = []
            for year in years:
                timeSeriesFile = f'{outdir}/{groupName}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_year{year:04d}.nc'
                timeSeriesFiles.append(timeSeriesFile)

            for regionIndex, regionName in enumerate(regionNames):
                regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
                fc = FeatureCollection()
                for feature in fcAll.features:
                    if feature['properties']['name'] == regionName:
                        fc.add_feature(feature)
                        break

                dsIn = xr.open_mfdataset(timeSeriesFiles, combine='nested',
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
                    figFileName = f'{figdir}/{regionNameShort}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{varName}_years{years[0]}-{years[-1]}.png'

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
    
    else:
      
        for k in range(len(dlevels)):
            timeSeriesFiles = []
            for year in years:
                timeSeriesFile = f'{outdir}/{groupName}_depth{int(dlevels[k]):04d}_year{year:04d}.nc'
                timeSeriesFiles.append(timeSeriesFile)

            for regionIndex, regionName in enumerate(regionNames):
                regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
                fc = FeatureCollection()
                for feature in fcAll.features:
                    if feature['properties']['name'] == regionName:
                        fc.add_feature(feature)
                        break

                dsIn = xr.open_mfdataset(timeSeriesFiles, combine='nested',
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
                    title = f'Mean {title} in {regionName} (z={z[zlevels[k]]:5.1f} m; {np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {units})'
                    figFileName = f'{figdir}/{regionNameShort}_depth{int(dlevels[k]):04d}_{varName}_years{years[0]}-{years[-1]}.png'

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
