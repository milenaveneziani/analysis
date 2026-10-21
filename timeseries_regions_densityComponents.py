from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import xarray as xr
import numpy as np
import gsw
import matplotlib.pyplot as plt
import netCDF4
from sympy import Symbol, latex
import time

from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf_with_fill
from mpas_analysis.shared.io.utility import decode_strings
from mpas_analysis.ocean.utility import compute_zmid

from geometric_features import FeatureCollection, read_feature_collection

from common_functions import timeseries_analysis_plot, add_inset
from make_plots import make_scatter_plot


startYear = 1
endYear = 50
#endYear = 2
#endYear = 246 # rA07
#endYear = 386 # rA02
startYear = 1950
endYear = 1970
year0 = startYear # Plotting of anomalies wrt month 1 of year0
referenceDate = '0001-01-01 00:00:00'
calendar = 'noleap'
daysInYear = 365.0

# Settings for nersc
regionMaskDir = '/global/cfs/cdirs/m1199/milena/mpas-region_masks'
meshName = 'ARRM10to60E2r1'
meshFile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runName = 'E3SM-Arcticv2.1_historical0101'
runNameShort = 'E3SMv2.1-Arctic-historical0101'
rundir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runName}'
isShortTermArchive = True # if True 'archive/ocn/hist' will be affixed to rundir later on
 
# Settings for lcrc
#regionMaskDir = '/lcrc/group/e3sm/ac.milena/mpas-region_masks'
#meshName = 'EC30to60E2r2'
#meshFile = f'/lcrc/group/acme/public_html/inputdata/ocn/mpas-o/{meshName}/ocean.EC30to60E2r2.200908.nc'
#runName = '20210127_JRA_POPvertMix_EC30to60E2r2'
#runNameShort = 'JRA_POPvertMix_noSSSrestoring'
#rundir = '/lcrc/group/acme/ac.vanroekel/scratch/anvil/20210127_JRA_POPvertMix_EC30to60E2r2/run'
#isShortTermArchive = False
 
# Settings for erdc.hpc.mil
#regionMaskDir = '/p/home/milena/mpas-region_masks'
#meshName = 'ARRM10to60E2r1'
#meshFile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runName = 'E3SMv2.1G60to10_01'
#runNameShort = 'E3SMv2.1G60to10_01'
#runName = 'E3SMv2.1B60to10rA02'
#runNameShort = 'E3SMv2.1B60to10rA02'
#rundir = f'/p/global/milena/{runName}'
#runName = 'E3SMv2.1B60to10rA07'
#runNameShort = 'E3SMv2.1B60to10rA07'
#rundir = f'/p/global/apcraig/archive/{runName}'
#isShortTermArchive = True # if True 'archive/ocn/hist' will be affixed to rundir later on

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
zmins = [-800.]
zmaxs = [10.]
# Relevant only for computeDepthAvg = False
dlevels = [0.]

plotAnnualAvg = True
# Relevant only for plotAnnualAvg = False
movingAverageMonths = 1
#movingAverageMonths = 12

#regionGroups = ['Arctic Regions']
#regionGroups = ['arctic_atlantic_budget_regions_new20240408']
regionGroups = ['arctic_atlantic_budget_regions_20260827']
#regionGroups = ['OceanOHC Regions']
#regionGroups = ['Antarctic Regions']
#regionGroups = ['southAtlantic_eastWest_regions']

# MPAS variables needed
#
mpasFile = 'timeSeriesStatsMonthly'

# For debugging only and assuming the timeseries has already been calculated
#variableList = ['timeMonthly_avg_layerThickness']
variableList = ['timeMonthly_avg_activeTracers_temperature',
                'timeMonthly_avg_activeTracers_salinity',
                'timeMonthly_avg_potentialDensity',
                'timeMonthly_avg_layerThickness']
timeVariableNames = ['xtime_startMonthly', 'xtime_endMonthly']

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
maxLevelCell = dsMesh.maxLevelCell - 1 # now compute_zmid uses 0-based indexing
latCell = 180.0/np.pi * dsMesh.latCell
lonCell = 180.0/np.pi * dsMesh.lonCell

# Find model levels for each depth level (relevant if computeDepthAvg = False)
z = dsMesh.refBottomDepth
zlevels = np.zeros(np.shape(dlevels), dtype=np.int64)
for k in range(len(dlevels)):
    dz = np.abs(z.values-dlevels[k])
    zlevels[k] = np.argmin(dz)

startDate = f'{startYear:04d}-01-01_00:00:00'
endDate = f'{endYear:04d}-12-31_23:59:59'
years = range(startYear, endYear + 1)

# Read in month1 of year0 (wrt anomalies will be calculated)
inputFile = f'{rundir}/{runName}.mpaso.hist.am.timeSeriesStatsMonthly.{year0:04d}-01-01.nc'
if not os.path.exists(inputFile):
    raise IOError(f'Input file: {inputFile} not found')
dsIn0 = open_mpas_dataset(fileName=inputFile,
                          calendar=calendar,
                          timeVariableNames=timeVariableNames,
                          variableList=variableList,
                          startDate=startDate,
                          endDate=endDate)
dsIn0 = dsIn0.isel(Time=0)

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
    regionNames.remove('Global')

    featureFile = f'{regionMaskDir}/{groupName}.geojson'
    if os.path.exists(featureFile):
        fcAll = read_feature_collection(featureFile)
    else:
        raise IOError('No feature file found for this region group')

    # Compute regional averages one year at a time
    for year in years:

        print('start loading yearly data')
        t0 = time.time()
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
        print(f'end loading yearly data. Time taken={time.time()-t0} seconds')

        if computeDepthAvg is True: # computeDepthAvg=True case
            layerThickness = dsIn.timeMonthly_avg_layerThickness
            zMid = compute_zmid(depth, maxLevelCell, layerThickness)

            # Compute regional averages one depth range at a time
            for k in range(len(zmins)):
                zmin = zmins[k]
                zmax = zmaxs[k]

                if zmax>0:
                    timeSeriesFile = f'{outdir}/{groupName}_z0000-{np.abs(np.int32(zmin)):04d}_anomalieswrtYear{year0}_year{year:04d}.nc'
                else:
                    timeSeriesFile = f'{outdir}/{groupName}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_anomalieswrtYear{year0}_year{year:04d}.nc'

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

                        # Focus on T, S, potential density (wrt surface), GSW potential density, alpha, 
                        # and beta fields on each cell center within the region and on each depth level
                        # within the indicated depth range
                        latRegion = latCell.where(cellMask, drop=True)
                        lonRegion = lonCell.where(cellMask, drop=True)
                        layerDepth = zMid.where(depthMask, drop=False).where(cellMask, drop=True)
                        temp = dsIn['timeMonthly_avg_activeTracers_temperature'].where(depthMask, drop=False).where(cellMask, drop=True)
                        salt = dsIn['timeMonthly_avg_activeTracers_salinity'].where(depthMask, drop=False).where(cellMask, drop=True)
                        rho = dsIn['timeMonthly_avg_potentialDensity'].where(depthMask, drop=False).where(cellMask, drop=True)
                        SA = gsw.SA_from_SP(salt, -layerDepth, lonRegion, latRegion)
                        CT = gsw.CT_from_pt(SA, temp)
                        [gsw_rho, alpha, beta] = gsw.density.rho_alpha_beta(SA, CT, 0.0)
                        # Calculate regional and depth-range quantities at time=0
                        temp0 = dsIn0['timeMonthly_avg_activeTracers_temperature'].where(depthMask, drop=False).where(cellMask, drop=True)
                        salt0 = dsIn0['timeMonthly_avg_activeTracers_salinity'].where(depthMask, drop=False).where(cellMask, drop=True)
                        rho0 = dsIn0['timeMonthly_avg_potentialDensity'].where(depthMask, drop=False).where(cellMask, drop=True)
                        SA0 = gsw.SA_from_SP(salt0, -layerDepth, lonRegion, latRegion)
                        CT0 = gsw.CT_from_pt(SA0, temp0)
                        gsw_rho0 = gsw.density.rho(SA0, CT0, 0.0)
                        # Calculate contributions to gsw_rho from alpha and beta oceans
                        dCT = CT - CT0
                        dSA = SA - SA0
                        dgswrho = gsw_rho - gsw_rho0
                        dgswrho_alpha = - gsw_rho * alpha * dCT
                        dgswrho_beta  =   gsw_rho * beta  * dSA
                        alphaBetaRatio = dgswrho_alpha/dgswrho_beta.clip(min=1e-15)

                        # Debugging
                        #dCT = CT.isel(Time=11) - CT.isel(Time=10)
                        #dSA = SA.isel(Time=11) - SA.isel(Time=10)
                        #drho = gsw_rho.isel(Time=11) - gsw_rho.isel(Time=10)
                        #drho_alpha = - gsw_rho.isel(Time=10) * alpha.isel(Time=10) * dCT
                        #drho_beta  =   gsw_rho.isel(Time=10) * beta.isel(Time=10)  * dSA
                        #delta = drho-(drho_alpha+drho_beta)
                        #print(np.nanmin(delta))
                        #print(np.nanmax(delta))
                        #print(np.nanmin(drho), np.nanmin(drho_alpha+drho_beta))
                        #print(np.nanmax(drho), np.nanmax(drho_alpha+drho_beta))
                        #colorindices = [0, 14, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]
                        #clevels = np.linspace(np.nanmin(delta.isel(nVertLevels=0)), np.nanmax(delta.isel(nVertLevels=0)), num=14)
                        #make_scatter_plot(lonRegion, latRegion, 0.5, '', './tmp1.png', projectionName='NorthPolarStereo',
                        #                  lon0=-100, lon1=20, dlon=20.0, lat0=44, lat1=65, dlat=5.0,
                        #                  fld=delta.isel(nVertLevels=0), cmap=plt.get_cmap('RdBu_r'), clevels=clevels,
                        #                  cindices=colorindices)
                        #clevels = np.linspace(np.nanmin(drho.isel(nVertLevels=0)), np.nanmax(drho.isel(nVertLevels=0)), num=14)
                        #make_scatter_plot(lonRegion, latRegion, 0.5, '', './tmp2.png', projectionName='NorthPolarStereo',
                        #                  lon0=-100, lon1=20, dlon=20.0, lat0=44, lat1=65, dlat=5.0,
                        #                  fld=drho.isel(nVertLevels=0), cmap=plt.get_cmap('RdBu_r'), clevels=clevels,
                        #                  cindices=colorindices)
                        #clevels = np.linspace(np.nanmin((drho_alpha+drho_beta).isel(nVertLevels=0)), np.nanmax((drho_alpha+drho_beta).isel(nVertLevels=0)), num=14)
                        #make_scatter_plot(lonRegion, latRegion, 0.5, '', './tmp3.png', projectionName='NorthPolarStereo',
                        #                  lon0=-100, lon1=20, dlon=20.0, lat0=44, lat1=65, dlat=5.0,
                        #                  fld=(drho_alpha+drho_beta).isel(nVertLevels=0), cmap=plt.get_cmap('RdBu_r'), clevels=clevels,
                        #                  cindices=colorindices)

                        # Now weight-average horizontally and vertically
                        dtemp = (localLayerVol * (temp-temp0)).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        dsalt = (localLayerVol * (salt-salt0)).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        drho = (localLayerVol * (rho-rho0)).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        dgswrho = (localLayerVol * dgswrho).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        dgswrho_alpha = (localLayerVol * dgswrho_alpha).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        dgswrho_beta = (localLayerVol * dgswrho_beta).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        alphaBetaRatio = (localLayerVol * alphaBetaRatio).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol

                        # Save to dataset
                        dsOut = xr.Dataset()
                        dsOut['temperature'] = dtemp
                        dsOut['temperature'].attrs['units'] = r'$^\circ$C'
                        dsOut['temperature'].attrs['description'] = f'Potential temperature (MPAS-computed) anomaly wrt Jan of year {year0}'
                        dsOut['salinity'] = dsalt
                        dsOut['salinity'].attrs['units'] = 'psu'
                        dsOut['salinity'].attrs['description'] = f'Salinity (MPAS-computed) anomaly wrt Jan of year {year0}'
                        dsOut['potentialDensity'] = drho
                        dsOut['potentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['potentialDensity'].attrs['description'] = f'Potential density (MPAS-computed) anomaly wrt Jan of year {year0}'
                        dsOut['gswPotentialDensity'] = dgswrho
                        dsOut['gswPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['gswPotentialDensity'].attrs['description'] = f'Potential density (GSW-computed; pref=0) anomaly wrt Jan of year {year0}'
                        dsOut['alphaPotentialDensity'] = dgswrho_alpha
                        dsOut['alphaPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['alphaPotentialDensity'].attrs['description'] = f'alpha component of potential density (-gswrho*alpha*dCT; GSW-computed; pref=0) anomaly wrt Jan of year {year0}'
                        dsOut['betaPotentialDensity'] = dgswrho_beta
                        dsOut['betaPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['betaPotentialDensity'].attrs['description'] = f'beta component of potential density (gswrho*beta*dSA; GSW-computed; pref=0) anomaly wrt Jan of year {year0}'
                        dsOut['alphaBetaRatio'] = alphaBetaRatio
                        dsOut['alphaBetaRatio'].attrs['units'] = 'unitless'
                        dsOut['alphaBetaRatio'].attrs['description'] = f'ratio of alpha/beta components of potential density (-alpha*dCT/(beta*dSA); GSW-computed; pref=0) anomaly wrt Jan of year {year0}'

                        dsOut['totalVol'] = regionalLayerVol
                        dsOut.totalVol.attrs['units'] = 'm^3'
                        dsOut['totalArea'] = regionalArea
                        dsOut.totalArea.attrs['units'] = 'm^2'
                        dsOut['zbounds'] = ('nbounds', [zmin, zmax])
                        dsOut.zbounds.attrs['units'] = 'm'

                        dsOut['regionNames'] = regionName

                        datasets.append(dsOut)

                    # combine data sets into a single data set
                    dsOut = xr.concat(datasets, 'nRegions')

                    # This is necessary because the time variable was changed
                    # in ways that I do not quite understand..
                    dsOut['Time'] = dsIn['Time']

                    # zbounds has become region dependent and shouldn't be
                    dsOut['zbounds'] = dsOut['zbounds'].isel(nRegions=0, drop=True)

                    #write_netcdf(dsOut, timeSeriesFile)
                    write_netcdf_with_fill(dsOut, timeSeriesFile)
                else:
                    print(f'Time series file already exists for year {year} and depth range {zmax}, {zmin}. Skipping it...')

        else:  # computeDepthAvg=False case

            # Compute regional averages one depth level at a time
            for k in range(len(dlevels)):

                timeSeriesFile = f'{outdir}/{groupName}_depth{int(dlevels[k]):04d}_anomalieswrtYear{year0}_year{year:04d}.nc'

                if not os.path.exists(timeSeriesFile):
                    print(f'Computing regional time series for year={year}, depth level={int(dlevels[k])}')


                    # Compute regional quantities for each depth level
                    datasets = []
                    regionIndices = []
                    for regionName in regionNames:
                        print(f'    region: {regionName}')

                        regionIndex = regionNames.index(regionName)
                        regionIndices.append(regionIndex)

                        dsMask = dsRegionMask.isel(nRegions=regionIndex)
                        cellMask = dsMask.regionCellMasks == 1
                        if openOceanMask is not None:
                            cellMask = np.logical_and(cellMask, openOceanMask)

                        localArea = areaCell.where(cellMask, drop=True)
                        regionalArea = localArea.sum()

                        # Focus on T, S, potential density (wrt surface), GSW potential density, alpha, 
                        # and beta fields on each cell center within the region
                        latRegion = latCell.where(cellMask, drop=True)
                        lonRegion = lonCell.where(cellMask, drop=True)
                        temp = dsIn['timeMonthly_avg_activeTracers_temperature'].isel(nVertLevels=zlevels[k]).where(cellMask, drop=True)
                        salt = dsIn['timeMonthly_avg_activeTracers_salinity'].isel(nVertLevels=zlevels[k]).where(cellMask, drop=True)
                        rho = dsIn['timeMonthly_avg_potentialDensity'].isel(nVertLevels=zlevels[k]).where(cellMask, drop=True)
                        SA = gsw.SA_from_SP(salt, dlevels[k], lonRegion, latRegion)
                        CT = gsw.CT_from_pt(SA, temp)
                        [gsw_rho, alpha, beta] = gsw.density.rho_alpha_beta(SA, CT, 0.0)
                        # Calculate regional and depth-range quantities at time=0
                        temp0 = dsIn0['timeMonthly_avg_activeTracers_temperature'].where(depthMask, drop=False).where(cellMask, drop=True)
                        salt0 = dsIn0['timeMonthly_avg_activeTracers_salinity'].where(depthMask, drop=False).where(cellMask, drop=True)
                        rho0 = dsIn0['timeMonthly_avg_potentialDensity'].where(depthMask, drop=False).where(cellMask, drop=True)
                        SA0 = gsw.SA_from_SP(salt0, dlevels[k], lonRegion, latRegion)
                        CT0 = gsw.CT_from_pt(SA0, temp0)
                        gsw_rho0 = gsw.density.rho(SA0, CT0, 0.0)
                        # Calculate contributions to gsw_rho from alpha and beta oceans
                        dCT = CT - CT0
                        dSA = SA - SA0
                        dgswrho = gsw_rho - gsw_rho0
                        dgswrho_alpha = - gsw_rho * alpha * dCT
                        dgswrho_beta  =   gsw_rho * beta  * dSA
                        alphaBetaRatio = dgswrho_alpha/dgswrho_beta.clip(min=1e-15)

                        # Now weight-average horizontally and vertically
                        dtemp = (localArea * (temp-temp0)).sum(dim='nCells') / regionalArea
                        dsalt = (localArea * (salt-salt0)).sum(dim='nCells') / regionalArea
                        drho = (localArea * (rho-rho0)).sum(dim='nCells') / regionalArea
                        dgswrho = (localArea * dgswrho).sum(dim='nCells') / regionalArea
                        dgswrho_alpha = (localArea * dgswrho_alpha).sum(dim='nCells') / regionalArea
                        dgswrho_beta = (localArea * dgswrho_beta).sum(dim='nCells') / regionalArea
                        alphaBetaRatio = (localLayerVol * alphaBetaRatio).sum(dim='nCells') / regionalArea

                        # Save to dataset
                        dsOut = xr.Dataset()
                        dsOut['temperature'] = dtemp
                        dsOut['temperature'].attrs['units'] = r'$^\circ$C'
                        dsOut['temperature'].attrs['description'] = f'Potential temperature (MPAS-computed) anomaly wrt Jan of year {year0}'
                        dsOut['salinity'] = dsalt
                        dsOut['salinity'].attrs['units'] = 'psu'
                        dsOut['salinity'].attrs['description'] = f'Salinity (MPAS-computed) anomaly wrt Jan of year {year0}'
                        dsOut['potentialDensity'] = drho
                        dsOut['potentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['potentialDensity'].attrs['description'] = f'Potential density (MPAS-computed) anomaly wrt Jan of year {year0}'
                        dsOut['gswPotentialDensity'] = dgswrho
                        dsOut['gswPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['gswPotentialDensity'].attrs['description'] = f'Potential density (GSW-computed; pref=0) anomaly wrt Jan of year {year0}'
                        dsOut['alphaPotentialDensity'] = dgswrho_alpha
                        dsOut['alphaPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['alphaPotentialDensity'].attrs['description'] = f'alpha component of potential density (-gswrho*alpha*CT; GSW-computed; pref=0) anomaly wrt Jan of year {year0}'
                        dsOut['betaPotentialDensity'] = dgswrho_beta
                        dsOut['betaPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['betaPotentialDensity'].attrs['description'] = f'beta component of potential density (gswrho*beta*SA; GSW-computed; pref=0) anomaly wrt Jan of year {year0}'
                        dsOut['alphaBetaRatio'] = alphaBetaRatio
                        dsOut['alphaBetaRatio'].attrs['units'] = 'unitless'
                        dsOut['alphaBetaRatio'].attrs['description'] = f'ratio of alpha/beta components of potential density (-alpha*dCT/(beta*dSA); GSW-computed; pref=0) anomaly wrt Jan of year {year0}'

                        dsOut['totalArea'] = regionalArea
                        dsOut.totalArea.attrs['units'] = 'm^2'

                        dsOut['regionNames'] = regionName

                        datasets.append(dsOut)

                    # combine data sets into a single data set
                    dsOut = xr.concat(datasets, 'nRegions')

                    # This is necessary because the time variable was changed
                    # in ways that I do not quite understand..
                    dsOut['Time'] = dsIn['Time']

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
                if zmax>0:
                    timeSeriesFile = f'{outdir}/{groupName}_z0000-{np.abs(np.int32(zmin)):04d}_anomalieswrtYear{year0}_year{year:04d}.nc'
                else:
                    timeSeriesFile = f'{outdir}/{groupName}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_anomalieswrtYear{year0}_year{year:04d}.nc'
                timeSeriesFiles.append(timeSeriesFile)

            for regionIndex, regionName in enumerate(regionNames):
                regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
                fc = FeatureCollection()
                for feature in fcAll.features:
                    if feature['properties']['name'] == regionName:
                        fc.add_feature(feature)
                        break

                dsIn  = xr.open_mfdataset(timeSeriesFiles, combine='nested', data_vars='all', 
                                          concat_dim='Time', decode_times=False).isel(nRegions=regionIndex)

                zbounds = dsIn.zbounds.values[0]
                xLabel = 'Time (yr)'
                legendText = ['']
                pmTex = latex(Symbol('pm'))

                if plotAnnualAvg is True:
                    dsToPlot = []
                    t = dsIn.Time
                    datetimes = netCDF4.num2date(t, f'days since {referenceDate}', calendar=calendar)
                    #print(datetimes)
                    t_years = np.empty(np.shape(t), dtype=np.int64)
                    for i, date in enumerate(datetimes.flat):
                        t_years[i] = date.year
                    #print(t_years)
                    weights = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])/daysInYear
                    #weights = xr.Dataset(data_vars={'weights': (('Time', ), weights)},
                    #                     coords={'Time': (('Time', ), t.values)})
                    weights = xr.DataArray(weights, dims=('Time', ), coords={'Time': t.values})
                    print(weights.values)
                    dsAnnual = xr.Dataset()
                    for year in years:
                        indexYears = np.where(t_years==year)[0]
                        #print(indexYears)
                        for var in ['temperature', 'salinity', 'potentialDensity',
                                    'gswPotentialDensity', 'alphaPotentialDensity',
                                    'betaPotentialDensity', 'alphaBetaRatio']:
                            annualData = dsIn[var].isel(Time=indexYears)
                            annualData = (weights * annualData).sum(dim='Time', keepdims=True)
                            annualData.name = var
                            dsAnnual[var] = xr.DataArray(
                                                         data=annualData,
                                                         dims=('Time', ),
                                                         attrs=dsIn[var].attrs
                                                        )
                        dsToPlot.append(dsAnnual)
                    dsToPlot = xr.concat(dsToPlot, dim='Time')
                    print(dsToPlot)
                else:
                    dsToPlot = dsIn
                movingAverageMonths = 1

                # Plot temperature anomaly first
                field = [dsToPlot['temperature']]
                units = dsToPlot['temperature'].attrs['units']
                yLabel = f'Temperature ({units})'
                title = f'Volume-Mean temperature anomaly wrt year {year0} in {regionName} ({zbounds[0]} < z < {zbounds[1]} m; {np.nanmean(field):5.2f} {pmTex} {np.nanstd(field):5.2f})'
                lineColors = ['k']
                lineWidths = [2.5]
                if zmax>0:
                    if plotAnnualAvg is True:
                        figFileName = f'{figdir}/{regionNameShort}_z0000-{np.abs(np.int32(zmin)):04d}_tempAnomalywrtYear{year0}Annual_years{years[0]}-{years[-1]}.png'
                    else:
                        figFileName = f'{figdir}/{regionNameShort}_z0000-{np.abs(np.int32(zmin)):04d}_tempAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                else:
                    if plotAnnualAvg is True:
                        figFileName = f'{figdir}/{regionNameShort}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_tempAnomalywrtYear{year0}Annual_years{years[0]}-{years[-1]}.png'
                    else:
                        figFileName = f'{figdir}/{regionNameShort}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_tempAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'

                [fig, ax] = timeseries_analysis_plot(field, movingAverageMonths, title, xLabel, yLabel,
                                                     calendar=calendar, timevarname = 'Time', lineColors=lineColors,
                                                     lineWidths=lineWidths, legendText=legendText)

                # do this before the inset because otherwise it moves the inset
                # and cartopy doesn't play too well with tight_layout anyway
                plt.tight_layout()

                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)

                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
                plt.close()

                # Then plot salinity anomaly
                field = [dsToPlot['salinity']]
                units = dsToPlot['salinity'].attrs['units']
                yLabel = f'Salinity ({units})'
                title = f'Volume-Mean salinity anomaly wrt year {year0} in {regionName} ({zbounds[0]} < z < {zbounds[1]} m; {np.nanmean(field):5.2f} {pmTex} {np.nanstd(field):5.2f})'
                lineColors = ['k']
                lineWidths = [2.5]
                if zmax>0:
                    if plotAnnualAvg is True:
                        figFileName = f'{figdir}/{regionNameShort}_z0000-{np.abs(np.int32(zmin)):04d}_saltAnomalywrtYear{year0}Annual_years{years[0]}-{years[-1]}.png'
                    else:
                        figFileName = f'{figdir}/{regionNameShort}_z0000-{np.abs(np.int32(zmin)):04d}_saltAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                else:
                    if plotAnnualAvg is True:
                        figFileName = f'{figdir}/{regionNameShort}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_saltAnomalywrtYear{year0}Annual_years{years[0]}-{years[-1]}.png'
                    else:
                        figFileName = f'{figdir}/{regionNameShort}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_saltAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                [fig, ax] = timeseries_analysis_plot(field, movingAverageMonths, title, xLabel, yLabel,
                                                     calendar=calendar, timevarname = 'Time', lineColors=lineColors,
                                                     lineWidths=lineWidths, legendText=legendText)
                plt.tight_layout()
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)
                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
                plt.close()

                # Finally plot density anomaly
                units = dsToPlot['potentialDensity'].attrs['units']
                yLabel = f'Potential density ({units})'
                title = f'Volume-Mean potential density anomaly contributions wrt year {year0} in {regionName} ({zbounds[0]} < z < {zbounds[1]} m)'
                if zmax>0:
                    if plotAnnualAvg is True:
                        figFileName = f'{figdir}/{regionNameShort}_z0000-{np.abs(np.int32(zmin)):04d}_rhoAnomalywrtYear{year0}Annual_years{years[0]}-{years[-1]}.png'
                    else:
                        figFileName = f'{figdir}/{regionNameShort}_z0000-{np.abs(np.int32(zmin)):04d}_rhoAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                else:
                    if plotAnnualAvg is True:
                        figFileName = f'{figdir}/{regionNameShort}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_rhoAnomalywrtYear{year0}Annual_years{years[0]}-{years[-1]}.png'
                    else:
                        figFileName = f'{figdir}/{regionNameShort}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_rhoAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                lineColors = ['gray', 'k', 'red', 'blue']
                lineWidths = [2.5, 2.5, 2.5, 2.5]

                drho = dsToPlot['potentialDensity']
                dgswrho  = dsToPlot['gswPotentialDensity']
                dgswrho_alpha = dsToPlot['alphaPotentialDensity']
                dgswrho_beta = dsToPlot['betaPotentialDensity']
                alphaBetaRatio = dsToPlot['alphaBetaRatio']
                legendText = [f'rho ({np.nanmean(drho):5.2f} {pmTex} {np.nanstd(drho):5.2f})']
                legendText.append(f'gsw_rho ({np.nanmean(dgswrho):5.2f} {pmTex} {np.nanstd(dgswrho):5.2f})')
                legendText.append(f'gsw_rhoAlpha ({np.nanmean(dgswrho_alpha):5.2f} {pmTex} {np.nanstd(dgswrho_alpha):5.2f})')
                legendText.append(f'gsw_rhoBeta ({np.nanmean(dgswrho_beta):5.2f} {pmTex} {np.nanstd(dgswrho_beta):5.2f})')
                field = [drho, dgswrho, dgswrho_alpha, dgswrho_beta]
                [fig, ax] = timeseries_analysis_plot(field, movingAverageMonths, title, xLabel, yLabel,
                                                     calendar=calendar, timevarname = 'Time', lineColors=lineColors,
                                                     lineWidths=lineWidths, legendText=legendText)
                #ax2 = ax.twinx()
                #ax2color = 'green'
                #ax2.plot(dsToPlot.Time, alphaBetaRatio, '-', color=ax2color, linewidth=2.5)
                #ax2.tick_params(axis='y', labelcolor=ax2color)
                #ax2.spines['right'].set_color(ax2color)
                #ax2.set_ylabel('alpha/beta ratio', color=ax2color)

                plt.tight_layout()
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)
                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
                plt.close()
    
    else:
      
        for k in range(len(dlevels)):
            timeSeriesFiles = []
            for year in years:
                timeSeriesFile = f'{outdir}/{groupName}_depth{int(dlevels[k]):04d}_anomalieswrtYear{year0}_year{year:04d}.nc'
                timeSeriesFiles.append(timeSeriesFile)

            for regionIndex, regionName in enumerate(regionNames):
                regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
                fc = FeatureCollection()
                for feature in fcAll.features:
                    if feature['properties']['name'] == regionName:
                        fc.add_feature(feature)
                        break

                dsIn = xr.open_mfdataset(timeSeriesFiles, combine='nested', data_vars='all',
                                         concat_dim='Time', decode_times=False).isel(nRegions=regionIndex)

                movingAverageMonths = 1
                #movingAverageMonths = 12

                xLabel = 'Time (yr)'
                legendText = ['']

                # Plot temperature anomaly first
                field = [dsIn['temperature']]
                units = dsIn['temperature'].attrs['units']
                yLabel = f'Temperature ({units})'
                title = f'Mean temperature anomaly wrt year {year0} in {regionName} (z={z[zlevels[k]]:5.1f} m; {np.nanmean(field):5.2f} {pmTex} {np.nanstd(field):5.2f})'
                lineColors = ['k']
                lineWidths = [2.5]
                figFileName = f'{figdir}/{regionNameShort}_depth{int(dlevels[k]):04d}_tempAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                [fig, ax] = timeseries_analysis_plot(field, movingAverageMonths, title, xLabel, yLabel,
                                                     calendar=calendar, timevarname = 'Time', lineColors=lineColors,
                                                     lineWidths=lineWidths, legendText=legendText)
                plt.tight_layout()
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)
                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)

                # Then plot salinity anomaly
                field = [dsIn['salinity']]
                units = dsIn['salinity'].attrs['units']
                yLabel = f'Salinity ({units})'
                title = f'Mean salinity anomaly wrt year {year0} in {regionName} (z={z[zlevels[k]]:5.1f} m; {np.nanmean(field):5.2f} {pmTex} {np.nanstd(field):5.2f})'
                lineColors = ['k']
                lineWidths = [2.5]
                figFileName = f'{figdir}/{regionNameShort}_depth{int(dlevels[k]):04d}_saltAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                [fig, ax] = timeseries_analysis_plot(field, movingAverageMonths, title, xLabel, yLabel,
                                                     calendar=calendar, timevarname = 'Time', lineColors=lineColors,
                                                     lineWidths=lineWidths, legendText=legendText)
                plt.tight_layout()
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)
                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)

                # Finally plot density anomaly
                units = dsIn['potentialDensity'].attrs['units']
                yLabel = f'Potential density ({units})'
                title = f'Mean potential density anomaly contributions wrt year {year0} in {regionName} (z={z[zlevels[k]]:5.1f} m)'
                figFileName = f'{figdir}/{regionNameShort}_depth{int(dlevels[k]):04d}_rhoAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                lineColors = ['gray', 'k', 'red', 'blue']
                lineWidths = [2.5, 2.5, 2.5, 2.5]

                drho = dsIn['potentialDensity']
                dgswrho  = dsIn['gswPotentialDensity']
                dgswrho_alpha = dsIn['alphaPotentialDensity']
                dgswrho_beta = dsIn['betaPotentialDensity']
                legendText = [f'rho ({np.nanmean(drho):5.2f} {pmTex} {np.nanstd(drho):5.2f})']
                legendText.append(f'gsw_rho ({np.nanmean(dgswrho):5.2f} {pmTex} {np.nanstd(dgswrho):5.2f})')
                legendText.append(f'gsw_rhoAlpha ({np.nanmean(dgswrho_alpha):5.2f} {pmTex} {np.nanstd(dgswrho_alpha):5.2f})')
                legendText.append(f'gsw_rhoBeta ({np.nanmean(dgswrho_beta):5.2f} {pmTex} {np.nanstd(dgswrho_beta):5.2f})')
                field = [drho, dgswrho, dgswrho_alpha, dgswrho_beta]
                [fig, ax] = timeseries_analysis_plot(field, movingAverageMonths, title, xLabel, yLabel,
                                                     calendar=calendar, timevarname = 'Time', lineColors=lineColors,
                                                     lineWidths=lineWidths, legendText=legendText)
                plt.tight_layout()
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)
                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
                plt.close()
