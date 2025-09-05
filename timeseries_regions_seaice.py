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

from common_functions import timeseries_analysis_plot, add_inset

#startYear = 2000
startYear = 2010
endYear = 2014
#startYear = 1
#endYear = 1
#endYear = 50
#endYear = 246 # rA07
#endYear = 386 # rA02
calendar = 'gregorian'

# Settings for nersc
regionMaskDir = '/global/cfs/cdirs/m1199/milena/mpas-region_masks'
meshName = 'ARRM10to60E2r1'
meshFile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runName = 'E3SM-Arcticv2.1_historical0101'
runNameShort = 'E3SMv2.1-Arctic-historical0101'
rundir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runName}'
isShortTermArchive = True # if True '{modelComp}/hist' will be affixed to rundir later on
 
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
#runName = 'E3SMv2.1G60to10_01'
#runNameShort = 'E3SMv2.1G60to10_01'
#runName = 'E3SMv2.1B60to10rA02'
#runNameShort = 'E3SMv2.1B60to10rA02'
#rundir = f'/p/cwfs/milena/{runName}'
#runName = 'E3SMv2.1B60to10rA07'
#runNameShort = 'E3SMv2.1B60to10rA07'
#rundir = f'/p/cwfs/apcraig/archive/{runName}'
#isShortTermArchive = True # if True 'archive/{modelComp}/hist' will be affixed to rundir later on

# Settings for chicoma
#regionMaskDir = '/users/milena/mpas-region_masks'
#meshName = 'RRSwISC6to18E3r5'
#meshFile = f'/usr/projects/e3sm/inputdata/ocn/mpas-o/{meshName}/mpaso.RRSwISC6to18E3r5.20240327.nc'
#runName = '20240726.icFromLRGcase.GMPAS-JRA1p5.TL319_RRSwISC6to18E3r5.chicoma'
#runNameShort = 'GMPAS-JRA1p5.TL319_RRSwISC6to18E3r5.icFromLRGcase'
#rundir = f'/lustre/scratch4/turquoise/milena/E3SMv3/{runName}/{runName}/run'
#isShortTermArchive = False

sref = 34.8 # needed for Arctic fwc calculation

# Quantities needed for plotting only
#movingAverageMonths = 12 # use this for monthly fields only
movingAverageMonths = 1
monthsToPlot = range(1, 13)
#monthsToPlot = [1, 2, 3, 4] # JFMA only (movingAverageMonths is changed to 1 later on)
#titleMonthsToPlot = 'JFMA'
titleMonthsToPlot = None

# region mask file will be $meshname_$regionGroups.nc
#regionGroups = ['oceanSubBasins20210315']
#regionGroups = ['arctic_atlantic_budget_regions_new20240408']
#regionGroups = ['Arctic Regions']
##regionGroups = ['OceanOHC Regions']
##regionGroups = ['Antarctic Regions']
regionGroups = ['Beaufort Sea Siobhan']

# Choose either 2d variables in timeSeriesStatsMonthly
# or variables in timeSeriesStatsMonthlyMax (2d only) or
# ice variables (2d only)
#
#   Ocean variables
#mpasComp = 'mpaso'
#modelComp = 'ocn'
#mpasFile = 'timeSeriesStatsMonthlyMax'
#mpasvar = 'timeMonthlyMax_max'
#variables = [
#             {'name': 'maxMLD',
#              'title': 'Maximum MLD',
#              'units': 'm',
#              'factor': 1,
#              'mpas': f'{mpasvar}_dThreshMLD'}
#            ]

#mpasFile = 'timeSeriesStatsMonthly'
#mpasvar = 'timeMonthly_avg'
#variables = [
#             {'name': 'dThreshMLD',
#              'title': 'Mean MLD',
#              'units': 'm',
#              'factor': 1,
#              'mpas': f'{mpasvar}_dThreshMLD'},
#             {'name': 'sensibleHeatFlux',
#              'title': 'Sensible Heat Flux',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_sensibleHeatFlux'},
#             {'name': 'latentHeatFlux',
#              'title': 'Latent Heat Flux',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_latentHeatFlux'},
#             {'name': 'longWaveHeatFluxUp',
#              'title': 'Longwave Up Heat Flux',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_longWaveHeatFluxUp'},
#             {'name': 'longWaveHeatFluxDown',
#              'title': 'Longwave Down Heat Flux',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_longWaveHeatFluxDown'},
#             {'name': 'shortWaveHeatFlux',
#              'title': 'Shortwave Heat Flux',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_shortWaveHeatFlux'},
#             {'name': 'evaporationFlux',
#              'title': 'Evaporation Flux',
#              'units': 'kg m^$-2$ s^$-1$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_evaporationFlux'},
#             {'name': 'rainFlux',
#              'title': 'Rain Flux',
#              'units': 'kg m^$-2$ s^$-1$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_rainFlux'},
#             {'name': 'snowFlux',
#              'title': 'Snow Flux',
#              'units': 'kg m^$-2$ s^$-1$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_snowFlux'},
#             {'name': 'riverRunoffFlux',
#              'title': 'River Runoff Flux',
#              'units': 'kg m^$-2$ s^$-1$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_riverRunoffFlux'},
#             {'name': 'iceRunoffFlux',
#              'title': 'Ice Runoff Flux',
#              'units': 'kg m^$-2$ s^$-1$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_iceRunoffFlux'},
#             {'name': 'seaIceFreshWaterFlux',
#              'title': 'Sea Ice Freshwater Flux',
#              'units': 'kg m^$-2$ s^$-1$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_seaIceFreshWaterFlux'},
#             {'name': 'surfaceBuoyancyForcing',
#              'title': 'Surface buoyancy flux',
#              'units': 'm$^2$ s$^{-3}$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_surfaceBuoyancyForcing'},
#             {'name': 'totalHeatFlux',
#              'title': 'Total Heat Flux (Sen+Lat+SWnet+LWnet)',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'mpas': None},
#             {'name': 'totalFWFlux',
#              'title': 'Total FW Flux (E-P+Runoff+SeaIce)',
#              'units': 'kg m^$-2$ s^$-1$',
#              'factor': 1,
#              'mpas': None},
#             {'name': 'fwc',
#              'title': 'Freshwater content',
#              'units': '10$^3$ km$^3$',
#              'factor': 1e-12,
#              'mpas': None}
#            ]

#   Sea ice variables
mpasComp = 'mpassi'
modelComp = 'ice'
#mpasFile = 'timeSeriesStatsMonthly'
#mpasvar = 'timeMonthly_avg'
mpasFile = 'timeSeriesStatsDaily'
mpasvar = 'timeDaily_avg'
variables = [
             {'name': 'icePressure',
              'title': 'sea ice pressure',
              'units': 'N m$^{-1}$',
              'factor': 1,
              'mpas': f'{mpasvar}_icePressure'},
             {'name': 'iceConcentration',
              'title': 'sea ice concentration',
              'units': 'fraction',
              'factor': 1,
              'mpas': f'{mpasvar}_iceAreaCell'},
             {'name': 'iceArea',
              'title': 'integrated sea ice area',
              'units': 'km$^2$',
              'factor': 1e-6,
              'mpas': f'{mpasvar}_iceAreaCell'},
             {'name': 'iceVolume',
              'title': 'integrated sea ice volume',
              'units': 'km$^3$',
              'factor': 1e-9,
              'mpas': f'{mpasvar}_iceVolumeCell'},
             {'name': 'uVelocityGeo',
              'title': 'sea ice velocity zonal',
              'units': 'm/s',
              'factor': 1,
              'mpas': f'{mpasvar}_uVelocityGeo'},
             {'name': 'vVelocityGeo',
              'title': 'sea ice velocity meridional',
              'units': 'm/s',
              'factor': 1,
              'mpas': f'{mpasvar}_vVelocityGeo'},
# the following are only available as monthly (for E3SMv2.1 runs):
#             {'name': 'iceDivergence',
#              'title': 'sea ice divergence',
#              'units': '%/day',
#              'factor': 1,
#              'mpas': f'{mpasvar}_divergence'},
#             {'name': 'firstYearIceConcentration',
#              'title': 'first-year sea ice concentration',
#              'units': 'fraction',
#              'factor': 1,
#              'mpas': f'{mpasvar}_firstYearIceAreaCell'},
#             {'name': 'firstYearIceArea',
#              'title': 'first-year integrated sea ice area',
#              'units': 'km$^2$',
#              'factor': 1e-6,
#              'mpas': f'{mpasvar}_firstYearIceAreaCell'},
#             {'name': 'levelIceConcentration',
#              'title': 'level-ice concentration',
#              'units': 'fraction',
#              'factor': 1,
#              'mpas': f'{mpasvar}_levelIceAreaCell'},
#             {'name': 'levelIceArea',
#              'title': 'integrated level-ice area',
#              'units': 'km$^2$',
#              'factor': 1e-6,
#              'mpas': f'{mpasvar}_levelIceAreaCell'},
#             {'name': 'ridgedIceConcentration',
#              'title': 'ridged-ice concentration',
#              'units': 'fraction',
#              'factor': 1,
#              'mpas': f'{mpasvar}_ridgedIceAreaAverage'},
#             {'name': 'ridgedIceArea',
#              'title': 'integrated ridged-ice area',
#              'units': 'km$^2$',
#              'factor': 1e-6,
#              'mpas': f'{mpasvar}_ridgedIceAreaAverage'},
#             {'name': 'levelIceVolume',
#              'title': 'integrated level-ice volume',
#              'units': 'km$^3$',
#              'factor': 1e-9,
#              'mpas': f'{mpasvar}_levelIceVolumeCell'},
#             {'name': 'ridgedIceVolume',
#              'title': 'integrated ridged-ice volume',
#              'units': 'km$^3$',
#              'factor': 1e-9,
#              'mpas': f'{mpasvar}_ridgedIceVolumeAverage'},
#             {'name': 'iceAgeCell',
#              'title': 'sea ice age',
#              'units': 'years',
#              'factor': 1/(86400*365.35),
#              'mpas': f'{mpasvar}_iceAgeCell'},
#             {'name': 'uAirVelocity',
#              'title': 'air velocity zonal',
#              'units': 'm/s',
#              'factor': 1,
#              'mpas': f'{mpasvar}_uAirVelocity'},
#             {'name': 'vAirVelocity',
#              'title': 'air velocity meridional',
#              'units': 'm/s',
#              'factor': 1,
#              'mpas': f'{mpasvar}_vAirVelocity'},
#             {'name': 'airStressVertexUGeo',
#              'title': 'ice-air stress zonal',
#              'units': 'N/m$^2$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_airStressVertexUGeo'},
#             {'name': 'airStressVertexVGeo',
#              'title': 'ice-air stress meridional',
#              'units': 'N/m$^2$',
#              'factor': 1,
#              'mpas': f'{mpasvar}_airStressVertexVGeo'}
            ]

if isShortTermArchive:
    if runName=='E3SMv2.1B60to10rA07':
        rundir = f'{rundir}/{modelComp}/hist'
    else:
        rundir = f'{rundir}/archive/{modelComp}/hist'

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
areaTriangle = dsMesh.areaTriangle
verticesOnCell = dsMesh.verticesOnCell
globalAreaCell = areaCell.sum()
globalAreaTriangle = areaTriangle.sum()

startDate = f'{startYear:04d}-01-01_00:00:00'
endDate = f'{endYear:04d}-12-31_23:59:59'
years = range(startYear, endYear + 1)

if mpasFile=='timeSeriesStatsMonthly':
    timeVariableNames = ['xtime_startMonthly', 'xtime_endMonthly']
if mpasFile=='timeSeriesStatsMonthlyMax':
    timeVariableNames = ['xtime_startMonthlyMax', 'xtime_endMonthlyMax']
if mpasFile=='timeSeriesStatsDaily':
    timeVariableNames = ['xtime_startDaily', 'xtime_endDaily']

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
    if mpasFile=='timeSeriesStatsMonthlyMax': # monthly maxima
        outfile = f'{groupName}_max_'
    if mpasFile=='timeSeriesStatsDaily': # daily averages 
        outfile = f'{groupName}_daily_'

    for var in variables:
        varname = var['name']
        varmpasname = var['mpas']
        varfactor = var['factor']
        varunits = var['units']
        vartitle = var['title']

        if varname=='fwc':
            variableList = [f'{mpasvar}_activeTracers_salinity', f'{mpasvar}_layerThickness']
        elif varname=='totalHeatFlux':
            variableList = [f'{mpasvar}_sensibleHeatFlux', f'{mpasvar}_latentHeatFlux',
                            f'{mpasvar}_shortWaveHeatFlux', f'{mpasvar}_longWaveHeatFluxDown',
                            f'{mpasvar}_longWaveHeatFluxUp']
        elif varname=='totalFWFlux':
            variableList = [f'{mpasvar}_evaporationFlux', f'{mpasvar}_rainFlux',
                            f'{mpasvar}_snowFlux', f'{mpasvar}_riverRunoffFlux',
                            f'{mpasvar}_iceRunoffFlux', f'{mpasvar}_seaIceFreshWaterFlux']
        else:
            variableList = [varmpasname]

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
                        # regionVertexMasks does not seem to work (all values are 0), so had to find
                        # an alternative method:
                        #triangleMask = dsMask.regionVertexMasks == 1
                        vertices_inregion = verticesOnCell.where(cellMask, drop=True)
                        vertices_inregion = vertices_inregion.stack(nVertices_inregion=('nCells', 'maxEdges'))
                        vertices_inregion = vertices_inregion.where(vertices_inregion!=0, drop=True).astype('int64')
                        vertices_inregion = np.unique(vertices_inregion)

                        localAreaCell = areaCell.where(cellMask, drop=True)
                        regionalAreaCell = localAreaCell.sum()
                        localAreaTriangle = areaTriangle.isel(nVertices=vertices_inregion-1, drop=True)
                        regionalAreaTriangle = localAreaTriangle.sum()

                    if varname=='fwc':
                        nTimes = dsIn.dims['Time']
                        nCells = dsIn.dims['nCells']
                        if regionNameShort=='arcticOcean_noBarents_KaraSeas' or \
                           regionNameShort=='beaufortGyre' or regionNameShort=='canadaBasin':
                            salinity = dsIn['timeMonthly_avg_activeTracers_salinity'].values
                            layerThickness = dsIn[f'{mpasvar}_layerThickness'].values
                            fwc = np.zeros([nTimes, nCells])
                            for itime in range(nTimes):
                                for icell in range(nCells):
                                    zindex = np.where(salinity[itime, icell, :]<=sref)[0]
                                    if zindex.any():
                                        fwc_tmp = (sref - salinity[itime, icell, zindex])/sref * \
                                                  layerThickness[itime, icell, zindex]
                                        fwc[itime, icell] = fwc_tmp.sum()
                            fwc = xr.DataArray(data=fwc, dims=('Time', 'nCells'))
                            fwc = (localAreaCell*fwc.where(cellMask, drop=True)).sum(dim='nCells')
                            fwc = varfactor * fwc
                            dsOut = xr.Dataset(data_vars={varname: fwc},
                                               coords={'Time': dsIn.Time},
                                               attrs={'units': varunits, 'description': vartitle})
                        else:
                            print('    Warning: Freshwater content is not computed for this region')
                            continue
                    elif varname=='totalHeatFlux':
                        totalHeatFlux = dsIn[f'{mpasvar}_sensibleHeatFlux']     + \
                                        dsIn[f'{mpasvar}_latentHeatFlux']       + \
                                        dsIn[f'{mpasvar}_shortWaveHeatFlux']    + \
                                        dsIn[f'{mpasvar}_longWaveHeatFluxDown'] + \
                                        dsIn[f'{mpasvar}_longWaveHeatFluxUp']
                        if regionName=='Global':
                            totalHeatFlux = (areaCell*totalHeatFlux).sum(dim='nCells') / globalAreaCell
                        else:
                            totalHeatFlux = (localAreaCell*totalHeatFlux.where(cellMask, drop=True)).sum(dim='nCells') / regionalAreaCell
                        totalHeatFlux = varfactor * totalHeatFlux
                        dsOut = xr.Dataset(data_vars={varname: totalHeatFlux},
                                           coords={'Time': dsIn.Time},
                                           attrs={'units': varunits, 'description': vartitle})
                    elif varname=='totalFWFlux':
                        totalFWFlux = dsIn[f'{mpasvar}_evaporationFlux'] + \
                                      dsIn[f'{mpasvar}_rainFlux']        + \
                                      dsIn[f'{mpasvar}_snowFlux']        + \
                                      dsIn[f'{mpasvar}_riverRunoffFlux'] + \
                                      dsIn[f'{mpasvar}_iceRunoffFlux']   + \
                                      dsIn[f'{mpasvar}_seaIceFreshWaterFlux']
                        if regionName=='Global':
                            totalFWFlux = (areaCell*totalFWFlux).sum(dim='nCells') / globalAreaCell
                        else:
                            totalFWFlux = (localAreaCell*totalFWFlux.where(cellMask, drop=True)).sum(dim='nCells') / regionalAreaCell
                        totalFWFlux = varfactor * totalFWFlux
                        dsOut = xr.Dataset(data_vars={varname: totalFWFlux},
                                           coords={'Time': dsIn.Time},
                                           attrs={'units': varunits, 'description': vartitle})
                    elif varname=='iceArea' or varname=='iceVolume' or varname=='firstYearIceArea' or \
                         varname=='levelIceArea' or varname=='ridgedIceArea' or varname=='levelIceVolume' or \
                         varname=='ridgedIceVolume':
                        if regionName=='Global':
                            dsOut = (areaCell*dsIn).sum(dim='nCells')
                        else:
                            dsOut = (localAreaCell*dsIn.where(cellMask, drop=True)).sum(dim='nCells')
                        dsOut = varfactor * dsOut
                        dsOut = dsOut.rename({varmpasname: varname})
                        dsOut[varname].attrs['units'] = varunits
                        dsOut[varname].attrs['description'] = vartitle
                    elif varname=='uVelocityGeo' or varname=='vVelocityGeo' or \
                         varname=='airStressVertexUGeo' or varname=='airStressVertexVGeo':
                        if regionName=='Global':
                            dsOut = (areaTriangle*dsIn).sum(dim='nVertices') / globalAreaTriangle
                        else:
                            dsOut = (localAreaTriangle*dsIn.isel(nVertices=vertices_inregion-1, drop=True)).sum(dim='nVertices') / regionalAreaTriangle
                        dsOut = varfactor * dsOut
                        dsOut = dsOut.rename({varmpasname: varname})
                        dsOut[varname].attrs['units'] = varunits
                        dsOut[varname].attrs['description'] = vartitle
                    else:
                        if regionName=='Global':
                            dsOut = (areaCell*dsIn).sum(dim='nCells') / globalAreaCell
                        else:
                            dsOut = (localAreaCell*dsIn.where(cellMask, drop=True)).sum(dim='nCells') / regionalAreaCell
                        dsOut = varfactor * dsOut
                        dsOut = dsOut.rename({varmpasname: varname})
                        dsOut[varname].attrs['units'] = varunits
                        dsOut[varname].attrs['description'] = vartitle

                    if regionName=='Global':
                        dsOut['totalAreaCell'] = globalAreaCell
                        dsOut['totalAreaTriangle'] = globalAreaTriangle
                    else:
                        dsOut['totalAreaCell'] = regionalAreaCell
                        dsOut['totalAreaTriangle'] = regionalAreaTriangle
                    dsOut['totalAreaCell'].attrs['units'] = 'm^2'
                    dsOut['totalAreaTriangle'].attrs['units'] = 'm^2'

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
                dsIn['timeMask'] = ('Time', mask)
                dsIn = dsIn.where(dsIn.timeMask, drop=True)

            field = [dsIn[varname]]
            xLabel = 'Time (yr)'
            yLabel = f'{vartitle} ({varunits})'
            lineColors = ['k']
            lineWidths = [2.5]
            legendText = [runNameShort]
            if titleMonthsToPlot is None:
                if movingAverageMonths==1:
                    if mpasFile=='timeSeriesStatsMonthly' or mpasFile=='timeSeriesStatsMonthlyMax':
                        title = f'Monthly {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'
                    if mpasFile=='timeSeriesStatsDaily':
                        title = f'Daily {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'
                else:
                    movingAverageYears = movingAverageMonths/12
                    title = f'{movingAverageYears}-year running mean {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'
            else:
                title = f'{titleMonthsToPlot} {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'

            if mpasFile=='timeSeriesStatsDaily':
                figFileName = f'{figdir}/{regionNameShort}_{varname}_years{years[0]}-{years[-1]}_daily.png'
            else:
                figFileName = f'{figdir}/{regionNameShort}_{varname}_years{years[0]}-{years[-1]}.png'

            fig = timeseries_analysis_plot(field, movingAverageMonths,
                                           title, xLabel, yLabel,
                                           calendar=calendar,
                                           timevarname = 'Time',
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
