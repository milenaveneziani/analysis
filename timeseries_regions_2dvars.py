from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import netCDF4

from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf_with_fill
#from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf
from mpas_analysis.shared.io.utility import get_files_year_month, decode_strings

from geometric_features import FeatureCollection, read_feature_collection

from common_functions import timeseries_analysis_plot, add_inset, days_to_datetime

#startYear = 1950
#endYear = 2014
startYear = 1
endYear = 31
#startYear = 65
#endYear = 325
calendar = 'gregorian'

# Settings for nersc
#regionMaskDir = '/global/cfs/cdirs/m1199/milena/mpas-region_masks'
#meshName = 'ARRM10to60E2r1'
#meshFile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runName = 'E3SM-Arcticv2.1_historical0101'
#runNameShort = 'E3SMv2.1-Arctic-historical0101'
#rundir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runName}'
#isShortTermArchive = True # if True '{modelComp}/hist' will be affixed to rundir later on
 
# Settings for lcrc
#regionMaskDir = '/lcrc/group/e3sm/ac.milena/mpas-region_masks'
##meshName = 'EC30to60E2r2'
##meshFile = f'/lcrc/group/acme/public_html/inputdata/ocn/mpas-o/{meshName}/ocean.EC30to60E2r2.200908.nc'
##runName = '20210127_JRA_POPvertMix_EC30to60E2r2'
##runNameShort = 'JRA_POPvertMix_noSSSrestoring'
##rundir = '/lcrc/group/acme/ac.vanroekel/scratch/anvil/20210127_JRA_POPvertMix_EC30to60E2r2/run'
##isShortTermArchive = False # if True '{modelComp}/hist' will be affixed to rundir later on
#meshName = 'RRSwISC6to18E3r5'
#meshFile = f'/lcrc/group/acme/public_html/inputdata/ocn/mpas-o/{meshName}/mpaso.RRSwISC6to18E3r5.20240327.nc'
#runName = '20240729.HRr5-test12.chrysalis'
#runNameShort = '20240729.HRr5-test12'
#rundir = '/lcrc/group/e3sm2/ac.jwolfe/E3SMv3_dev/20240729.HRr5-test12.chrysalis/archive'
#isShortTermArchive = True
 
# Settings for erdc.hpc.mil
#regionMaskDir = '/p/home/milena/mpas-region_masks'
#meshName = 'ARRM10to60E2r1'
#meshFile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runName = 'E3SMv2.1B60to10rA02'
#runNameShort = 'E3SMv2.1B60to10rA02'
#rundir = f'/p/work/milena/{runName}'
#isShortTermArchive = True # if True 'archive/{modelComp}/hist' will be affixed to rundir later on

# Settings for chicoma
regionMaskDir = '/users/milena/mpas-region_masks'
meshName = 'RRSwISC6to18E3r5'
meshFile = f'/usr/projects/e3sm/inputdata/ocn/mpas-o/{meshName}/mpaso.RRSwISC6to18E3r5.20240327.nc'
runName = '20240726.icFromLRGcase.GMPAS-JRA1p5.TL319_RRSwISC6to18E3r5.chicoma'
runNameShort = 'GMPAS-JRA1p5.TL319_RRSwISC6to18E3r5.icFromLRGcase'
rundir = f'/lustre/scratch4/turquoise/milena/E3SMv3/{runName}/{runName}/run'
isShortTermArchive = False

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

sref = 34.8 # needed for Arctic fwc calculation

# Quantities needed for plotting only
#movingAverageMonths = 12
#monthsToPlot = range(1, 13)
#titleMonthsToPlot = None
movingAverageMonths = 1
monthsToPlot = [1, 2, 3, 4] # JFMA only (movingAverageMonths is changed to 1 later on)
titleMonthsToPlot = 'JFMA'

# region mask file will be $meshname_$regionGroups.nc
regionGroups = ['arctic_atlantic_budget_regions_new20240408']
#regionGroups = ['Arctic Regions']
#regionGroups = ['OceanOHC Regions']
#regionGroups = ['Antarctic Regions']

# Choose either 2d variables in timeSeriesStatsMonthly
# or variables in timeSeriesStatsMonthlyMax (2d only) or
# ice variables (2d only)
#
#   Ocean variables
#mpasComp = 'mpaso'
#modelComp = 'ocn'
#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'title': 'Maximum MLD',
#              'units': 'm',
#              'factor': 1,
#              'mpas': 'timeMonthlyMax_max_dThreshMLD'}
#            ]
#
#mpasFile = 'timeSeriesStatsMonthly'
#variables = [
#             {'name': 'sensibleHeatFlux',
#              'title': 'Sensible Heat Flux',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'mpas': 'timeMonthly_avg_sensibleHeatFlux'},
#             {'name': 'fwc',
#              'title': 'Freshwater content',
#              'units': '10$^3$ km$^3$',
#              'factor': 1e-12,
#              'mpas': 'timeMonthly_avg_activeTracers_salinity'}
#            ]
             #{'name': 'latentHeatFlux',
             # 'title': 'Latent Heat Flux',
             # 'units': 'W/m$^2$',
             # 'factor': 1,
             # 'mpas': 'timeMonthly_avg_latentHeatFlux'}
             #{'name': 'longWaveHeatFluxUp',
             # 'title': 'Longwave Up Heat Flux',
             # 'units': 'W/m$^2$',
             # 'factor': 1,
             # 'mpas': 'timeMonthly_avg_longWaveHeatFluxUp'}
             #{'name': 'longWaveHeatFluxDown',
             # 'title': 'Longwave Down Heat Flux',
             # 'units': 'W/m$^2$',
             # 'factor': 1,
             # 'mpas': 'timeMonthly_avg_longWaveHeatFluxDown'}
             #{'name': 'shortWaveHeatFlux',
             # 'title': 'Shortwave Heat Flux',
             # 'units': 'W/m$^2$',
             # 'factor': 1,
             # 'mpas': 'timeMonthly_avg_shortWaveHeatFlux'}
             # 'mpas': 'timeMonthly_avg_seaIceHeatFlux'}
             # 'mpas': 'timeMonthly_avg_penetrativeTemperatureFlux'}
             #{'name': 'surfaceBuoyancyForcing',
             # 'title': 'Surface buoyancy flux',
             # 'units': 'm$^2$ s$^{-3}$',
             # 'factor': 1,
             # 'mpas': 'timeMonthly_avg_surfaceBuoyancyForcing'}
#   Sea ice variables
mpasComp = 'mpassi'
modelComp = 'ice'
mpasFile = 'timeSeriesStatsMonthly'
variables = [
             {'name': 'iceArea',
              'title': 'Integrated Ice Area',
              'units': 'km$^2$',
              'factor': 1e-6,
              'mpas': 'timeMonthly_avg_iceAreaCell'},
             {'name': 'iceVolume',
              'title': 'Integrated Ice Volume',
              'units': 'km$^3$',
              'factor': 1e-9,
              'mpas': 'timeMonthly_avg_iceVolumeCell'}
            ]

if isShortTermArchive:
    rundir = f'{rundir}/archive/{modelComp}/hist'

startDate = f'{startYear:04d}-01-01_00:00:00'
endDate = f'{endYear:04d}-12-31_23:59:59'
years = range(startYear, endYear + 1)

if mpasFile=='timeSeriesStatsMonthly':
    timeVariableNames = ['xtime_startMonthly', 'xtime_endMonthly']
else:
    timeVariableNames = ['xtime_startMonthlyMax', 'xtime_endMonthlyMax']

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

    if mpasFile=='timeSeriesStatsMonthly': # monthly averages 
        outfile = f'{groupName}_'
    else: # monthly maxima
        outfile = f'{groupName}_max_'

    for var in variables:
        varname = var['name']
        varmpasname = var['mpas']
        varfactor = var['factor']
        varunits = var['units']
        vartitle = var['title']
        variableList = [varmpasname]
        if varname=='fwc':
            variableList = variableList + ['timeMonthly_avg_layerThickness']

        outdirvar = f'{outdir}/{varname}'
        if not os.path.isdir(outdirvar):
            os.makedirs(outdirvar) 

        print('')
        for year in years:

            timeSeriesFile = f'{outdirvar}/{outfile}year{year:04d}.nc'

            if not os.path.exists(timeSeriesFile):
                print(f'Processing variable = {vartitle},  year={year}')
                # Load in yearly data set for chosen variable
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
                dsIn = xr.concat(datasets, 'Time')

                datasets = []
                regionIndices = []
                for regionName in regionNames:
                    print(f'    region: {regionName}')
                    regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')

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

                    if varname=='fwc':
                        nTimes = dsIn.dims['Time']
                        nCells = dsIn.dims['nCells']
                        if regionNameShort=='arcticOcean_noBarents_KaraSeas' or \
                           regionNameShort=='beaufortGyre' or regionNameShort=='canadaBasin':
                            salinity = dsIn[varmpasname].values
                            layerThickness = dsIn['timeMonthly_avg_layerThickness'].values
                            fwc = np.zeros([nTimes, nCells])
                            for itime in range(nTimes):
                                for icell in range(nCells):
                                    zindex = np.where(salinity[itime, icell, :]<=sref)[0]
                                    if zindex.any():
                                        fwc_tmp = (sref - salinity[itime, icell, zindex])/sref * \
                                                  layerThickness[itime, icell, zindex]
                                        fwc[itime, icell] = fwc_tmp.sum()
                            dsIn[varname] = (['Time', 'nCells'], fwc)
                            dsIn = dsIn[varname]
                            dsOut = (localArea*dsIn.where(cellMask, drop=True)).sum(dim='nCells')
                            dsOut = dsOut.rename(varname)
                        else:
                            print('    Warning: Freshwater content is not computed for this region')
                            continue
                    elif varname=='iceArea' or varname=='iceVolume':
                        if regionName=='Global':
                            dsOut = (areaCell*dsIn).sum(dim='nCells')
                        else:
                            dsOut = (localArea*dsIn.where(cellMask, drop=True)).sum(dim='nCells')
                        dsOut = dsOut.rename({varmpasname: varname})
                    else:
                        if regionName=='Global':
                            dsOut = (areaCell*dsIn).sum(dim='nCells') / globalArea
                        else:
                            dsOut = (localArea*dsIn.where(cellMask, drop=True)).sum(dim='nCells') / regionalArea
                        dsOut = dsOut.rename({varmpasname: varname})

                    dsOut = varfactor * dsOut
                    #print(dsOut)
                    #print(dsOut[varname])
                    dsOut[varname].attrs['units'] = varunits
                    dsOut[varname].attrs['description'] = vartitle

                    if regionName=='Global':
                        dsOut['totalArea'] = globalArea
                    else:
                        dsOut['totalArea'] = regionalArea
                    dsOut['totalArea'].attrs['units'] = 'm^2'

                    dsOut['regionNames'] = regionName

                    datasets.append(dsOut)

                # combine data sets into a single data set
                dsOut = xr.concat(datasets, 'nRegions')

                write_netcdf_with_fill(dsOut, timeSeriesFile)
                #write_netcdf(dsOut, timeSeriesFile)
            else:
                print(f'Time series file already exists for {varname} and year {year}. Skipping it...')

        # Time series calculated ==> make plots
        print(f'\n  now plot {varname} for each region\n')
        timeSeriesFiles = []
        for year in years:
            timeSeriesFile = f'{outdirvar}/{outfile}year{year:04d}.nc'
            timeSeriesFiles.append(timeSeriesFile)

        for regionIndex, regionName in enumerate(regionNames):
            print(f'    region: {regionName}')
            regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
            fc = FeatureCollection()
            for feature in fcAll.features:
                if feature['properties']['name'] == regionName:
                    fc.add_feature(feature)
                    break

            dsIn = xr.open_mfdataset(timeSeriesFiles, combine='nested',
                                     concat_dim='Time', decode_times=False).isel(nRegions=regionIndex)

            if len(monthsToPlot)!=12:
                # Subset time series (making sure that movingAverageMonths is set to 1
                # (no running average))
                movingAverageMonths = 1
                referenceDate = '0001-01-01'
                datetimes = netCDF4.num2date(dsIn.Time, f'days since {referenceDate}', calendar=calendar)
                timemonths = []
                for date in datetimes.flat:
                    timemonths.append(date.month)
                mask = np.logical_and(timemonths>=np.min(monthsToPlot), timemonths<=np.max(monthsToPlot))
                #mask = xr.Dataset(data_vars=dict(mask=(['Time'], mask)))
                dsIn['timeMonthlyMask'] = ('Time', mask)
                dsIn = dsIn.where(dsIn.timeMonthlyMask, drop=True)

            field = [dsIn[varname]]
            xLabel = 'Time (yr)'
            yLabel = f'{vartitle} ({varunits})'
            lineColors = ['k']
            lineWidths = [2.5]
            legendText = [runNameShort]
            if titleMonthsToPlot is None:
                title = f'1-year running mean {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'
            else:
                title = f'{titleMonthsToPlot} {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'
            figFileName = f'{figdir}/{regionNameShort}_{varname}_years{years[0]}-{years[-1]}.png'

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
