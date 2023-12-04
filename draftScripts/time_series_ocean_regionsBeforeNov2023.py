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
endYear = 54
calendar = 'gregorian'

# Settings for nersc
#regionMaskDir = '/global/cfs/projectdirs/m1199/milena/mpas-region_masks'
#meshName = 'ARRM60to10'
#restartFile = '/global/cfs/projectdirs/e3sm/inputdata/ocn/mpas-o/oARRM60to10/ocean.ARRM60to10.180715.nc'
#regionMaskFile = '/global/cfs/projectdirs/m1199/milena/mpas-region_masks/{}_oceanOHCRegions.nc'.format(meshName)
#featureFile = '/global/cfs/projectdirs/m1199/milena/mpas-region_masks/oceanOHCRegions.geojson'
#runName = 'ARRM60to10_JRA_GM_ramp'
#runNameShort = 'E3SM-Arctic-OSI'
#rundir = '/global/cscratch1/sd/milena/E3SM_simulations/ARRM60to10_JRA_GM_ramp/run'
#runName = '20210204.A_WCYCL1850S_CMIP6.ne30pg2_oARRM60to10_ICG.beta1.cori-knl'
#runNameShort = 'E3SM-Arctic-coupled-beta1'
#rundir = '/global/cscratch1/sd/dcomeau/e3sm_scratch/cori-knl/20210204.A_WCYCL1850S_CMIP6.ne30pg2_oARRM60to10_ICG.beta1.cori-knl/run'
 
# Settings for lcrc
#regionMaskDir = '/lcrc/group/e3sm/ac.milena/mpas-region_masks'
#meshName = 'EC30to60E2r2'
#restartFile = '/lcrc/group/acme/public_html/inputdata/ocn/mpas-o/{}/ocean.EC30to60E2r2.200908.nc'.format(meshName)
#runName = '20210127_JRA_POPvertMix_EC30to60E2r2'
#runNameShort = 'JRA_POPvertMix_noSSSrestoring'
#rundir = '/lcrc/group/acme/ac.vanroekel/scratch/anvil/20210127_JRA_POPvertMix_EC30to60E2r2/run'
 
# Settings for onyx
regionMaskDir = '/p/home/milena/mpas-region_masks'
meshName = 'ARRM10to60E2r1'
restartFile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runName = 'E3SMv2.1B60to10rA02'
runNameShort = 'E3SMv2.1B60to10rA02'
rundir = f'/p/work/osinski/archive/{runName}/ocn/hist'

outdir = './timeseries_data/{}'.format(runNameShort)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
figdir = './timeseries/{}'.format(runNameShort)
if not os.path.isdir(figdir):
    os.makedirs(figdir)

if os.path.exists(restartFile):
    dsRestart = xarray.open_dataset(restartFile)
    dsRestart = dsRestart.isel(Time=0)
else:
    raise IOError('No MPAS restart/mesh file found')
if 'landIceMask' in dsRestart:
    # only the region outside of ice-shelf cavities
    openOceanMask = dsRestart.landIceMask == 0
else:
    openOceanMask = None
areaCell = dsRestart.areaCell
globalArea = areaCell.sum()
refBottomDepth = dsRestart.refBottomDepth
maxLevelCell = dsRestart.maxLevelCell

regionGroups = ['Arctic Regions']
#regionGroups = ['OceanOHC Regions']
#regionGroups = ['Antarctic Regions']
# Choose 3d variables for z range greater than upper ocean
#variables = [{'name': 'salinity',
#              'title': 'Salinity',
#              'units': 'PSU',
#              'mpas': 'timeMonthly_avg_activeTracers_salinity'}]
variables = [{'name': 'temperature',
              'title': 'Temperature',
              'units': '$^\circ$C',
              'mpas': 'timeMonthly_avg_activeTracers_temperature'},
             {'name': 'salinity',
              'title': 'Salinity',
              'units': 'PSU',
              'mpas': 'timeMonthly_avg_activeTracers_salinity'},
             {'name': 'potentialDensity',
              'title': 'Potential Density',
              'units': 'kg m$^{-3}$',
              'mpas': 'timeMonthly_avg_potentialDensity'}]

startDate = '{:04d}-01-01_00:00:00'.format(startYear)
endDate = '{:04d}-12-31_23:59:59'.format(endYear)
years = range(startYear, endYear + 1)

zmins = [-100., -6000.]
zmaxs = [0., 0.]

variableList = [var['mpas'] for var in variables] + \
    ['timeMonthly_avg_layerThickness']

for regionGroup in regionGroups:

    groupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')

    regionMaskFile = '{}/{}_{}.nc'.format(regionMaskDir, meshName, groupName)
    if os.path.exists(regionMaskFile):
        dsRegionMask = xarray.open_dataset(regionMaskFile)
        regionNames = decode_strings(dsRegionMask.regionNames)
        if regionGroup==regionGroups[0]:
            regionNames.append('Global')
        nRegions = np.size(regionNames)
    else:
        raise IOError('No regional mask file found')

    featureFile = '{}/{}.geojson'.format(regionMaskDir, groupName)
    if os.path.exists(featureFile):
        fcAll = read_feature_collection(featureFile)
    else:
        raise IOError('No feature file found for this region group')

    # Compute regional averages one year at a time
    for year in years:

        # Load in entire data set (all variables, all x's, y's, z's,
        # for the indicated time period)
        datasets = []
        for month in range(1, 13):
            inputFile = '{}/{}.mpaso.hist.am.timeSeriesStatsMonthly.{:04d}-{:02d}-01.nc'.format(
                rundir, runName, year, month)
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

            timeSeriesFile = '{}/{}_z{:04d}-{:04d}_year{:04d}.nc'.format(outdir,
                              groupName, np.abs(np.int(zmax)), np.abs(np.int(zmin)), year)

            if not os.path.exists(timeSeriesFile):
                print('Computing regional time series for year={}, depth range= {}, {}'.format(
                    year, zmax, zmin))

                # Compute regional quantities for each depth range
                datasets = []
                regionIndices = []
                for regionName in regionNames:
                    print('    region: {}'.format(regionName))

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
                print('Time series file already exists for year {} and depth range {}, {}. Skipping it...'.format(
                    year, zmax, zmin))

    # Make plots for each depth range (and each region and each variable)
    for k in range(len(zmins)):
        zmin = zmins[k]
        zmax = zmaxs[k]
        timeSeriesFiles = []
        for year in years:
            timeSeriesFile = '{}/{}_z{:04d}-{:04d}_year{:04d}.nc'.format(outdir,
                              groupName, np.abs(np.int(zmax)), np.abs(np.int(zmin)), year)
            timeSeriesFiles.append(timeSeriesFile)

        for regionIndex, regionName in enumerate(regionNames):
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
                varName = var['name']
                field = [dsIn[varName]]
                xLabel = 'Time (yr)'
                yLabel = '{} ({})'.format(var['title'], var['units'])
                lineColors = ['k']
                lineWidths = [2.5]
                legendText = [runNameShort]
                title = 'Volume-Mean {} in {} ({} < z < {} m; {:5.2f} $\pm$ {:5.2f} {})'.format(
                        var['title'], regionName, zbounds[0], zbounds[1], np.nanmean(field), 
                        np.nanstd(field), var['units'])
                figFileName = '{}/{}_z{:04d}-{:04d}_{}.png'.format(
                    figdir,
                    regionName[0].lower() + \
                    regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_'),
                    np.abs(np.int(zmax)), np.abs(np.int(zmin)),
                    varName)

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

                plt.savefig(figFileName, dpi='figure', bbox_inches='tight',
                            pad_inches=0.1)
                plt.close()
