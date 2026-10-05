from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import xarray as xr
import numpy as np
import gsw
import matplotlib.pyplot as plt

from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf_with_fill
#from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf
from mpas_analysis.shared.io.utility import get_files_year_month, decode_strings
from mpas_analysis.ocean.utility import compute_zmid

from geometric_features import FeatureCollection, read_feature_collection

from common_functions import timeseries_analysis_plot, add_inset

#startYear = 1
#endYear = 50
#endYear = 246 # rA07
#endYear = 386 # rA02
startYear = 1950
endYear = 1950
year0 = startYear # Plotting of anomalies wrt month 1 of year0
calendar = 'gregorian'

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

#regionGroups = ['Arctic Regions']
#regionGroups = ['arctic_atlantic_budget_regions_new20240408']
regionGroups = ['arctic_atlantic_budget_regions_20260827']
#regionGroups = ['OceanOHC Regions']
#regionGroups = ['Antarctic Regions']
#regionGroups = ['southAtlantic_eastWest_regions']

# MPAS variables needed
#
mpasFile = 'timeSeriesStatsMonthly'
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

                if zmax>0:
                    timeSeriesFile = f'{outdir}/{groupName}_z0000-{np.abs(np.int32(zmin)):04d}_year{year:04d}.nc'
                else:
                    timeSeriesFile = f'{outdir}/{groupName}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_year{year:04d}.nc'

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
                        # Contributions to gsw_rho from alpha and beta oceans
                        gsw_rhoAlpha = - gsw_rho * alpha * CT
                        gsw_rhoBeta = gsw_rho * beta * SA

                        # Now weight-average horizontally and vertically
                        temp = (localLayerVol*temp).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        salt = (localLayerVol*salt).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        rho = (localLayerVol*rho).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        gsw_rho = (localLayerVol*gsw_rho).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        gsw_rhoAlpha = (localLayerVol*gsw_rhoAlpha).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol
                        gsw_rhoBeta = (localLayerVol*gsw_rhoBeta).sum(dim='nVertLevels').sum(dim='nCells') / regionalLayerVol

                        # Save to dataset
                        dsOut = xr.Dataset()
                        dsOut['temperature'] = temp
                        dsOut['temperature'].attrs['units'] = r'$^\circ$C'
                        dsOut['temperature'].attrs['description'] = 'Potential temperature (MPAS-computed)'
                        dsOut['salinity'] = salt
                        dsOut['salinity'].attrs['units'] = 'psu'
                        dsOut['salinity'].attrs['description'] = 'Salinity (MPAS-computed)'
                        dsOut['potentialDensity'] = rho
                        dsOut['potentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['potentialDensity'].attrs['description'] = 'Potential density (MPAS-computed)'
                        dsOut['gswPotentialDensity'] = gsw_rho
                        dsOut['gswPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['gswPotentialDensity'].attrs['description'] = 'Potential density (GSW-computed; pref=0)'
                        dsOut['alphaPotentialDensity'] = gsw_rhoAlpha
                        dsOut['alphaPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['alphaPotentialDensity'].attrs['description'] = 'alpha component of potential density (-gsw_rho*alpha*CT; GSW-computed; pref=0)'
                        dsOut['betaPotentialDensity'] = gsw_rhoBeta
                        dsOut['betaPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['betaPotentialDensity'].attrs['description'] = 'beta component of potential density (gsw_rho*beta*SA; GSW-computed; pref=0)'

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
                        # Contributions to gsw_rho from alpha and beta oceans
                        gsw_rhoAlpha = - gsw_rho * alpha * CT
                        gsw_rhoBeta = gsw_rho * beta * SA

                        # Now weight-average horizontally and vertically
                        temp = (localArea*temp).sum(dim='nCells') / regionalArea
                        salt = (localArea*salt).sum(dim='nCells') / regionalArea
                        rho = (localArea*rho).sum(dim='nCells') / regionalArea
                        gsw_rho = (localArea*gsw_rho).sum(dim='nCells') / regionalArea
                        gsw_rhoAlpha = (localArea*gsw_rhoAlpha).sum(dim='nCells') / regionalArea
                        gsw_rhoBeta = (localArea*gsw_rhoBeta).sum(dim='nCells') / regionalArea

                        # Save to dataset
                        dsOut = xr.Dataset()
                        dsOut['temperature'] = temp
                        dsOut['temperature'].attrs['units'] = r'$^\circ$C'
                        dsOut['temperature'].attrs['description'] = 'Potential temperature (MPAS-computed)'
                        dsOut['salinity'] = salt
                        dsOut['salinity'].attrs['units'] = 'psu'
                        dsOut['salinity'].attrs['description'] = 'Salinity (MPAS-computed)'
                        dsOut['potentialDensity'] = rho
                        dsOut['potentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['potentialDensity'].attrs['description'] = 'Potential density (MPAS-computed)'
                        dsOut['gswPotentialDensity'] = gsw_rho
                        dsOut['gswPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['gswPotentialDensity'].attrs['description'] = 'Potential density (GSW-computed; pref=0)'
                        dsOut['alphaPotentialDensity'] = gsw_rhoAlpha
                        dsOut['alphaPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['alphaPotentialDensity'].attrs['description'] = 'alpha component of potential density (-gsw_rho*alpha*CT; GSW-computed; pref=0)'
                        dsOut['betaPotentialDensity'] = gsw_rhoBeta
                        dsOut['betaPotentialDensity'].attrs['units'] = r'Kg/m$^3$'
                        dsOut['betaPotentialDensity'].attrs['description'] = 'beta component of potential density (gsw_rho*beta*SA; GSW-computed; pref=0)'


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
            if zmax>0:
                timeSeriesFile0 = f'{outdir}/{groupName}_z0000-{np.abs(np.int32(zmin)):04d}_year{year0:04d}.nc'
            else:
                timeSeriesFile0 = f'{outdir}/{groupName}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_year{year0:04d}.nc'
            timeSeriesFiles = []
            for year in years:
                if zmax>0:
                    timeSeriesFile = f'{outdir}/{groupName}_z0000-{np.abs(np.int32(zmin)):04d}_year{year:04d}.nc'
                else:
                    timeSeriesFile = f'{outdir}/{groupName}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_year{year:04d}.nc'
                timeSeriesFiles.append(timeSeriesFile)

            for regionIndex, regionName in enumerate(regionNames):
                regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
                fc = FeatureCollection()
                for feature in fcAll.features:
                    if feature['properties']['name'] == regionName:
                        fc.add_feature(feature)
                        break

                dsIn0 = xr.open_dataset(timeSeriesFile0, decode_times=False).isel(nRegions=regionIndex)
                dsIn  = xr.open_mfdataset(timeSeriesFiles, combine='nested',
                                          concat_dim='Time', decode_times=False).isel(nRegions=regionIndex)

                zbounds = dsIn.zbounds.values[0]

                movingAverageMonths = 1
                #movingAverageMonths = 12

                xLabel = 'Time (yr)'
                legendText = ['']

                # Plot temperature anomaly first
                field = [dsIn['temperature'] - dsIn0['temperature'].isel(Time=0)]
                units = dsIn['temperature'].attrs['units']
                yLabel = f'Temperature ({units})'
                title = f'Volume-Mean temperature anomaly wrt year {year0} in {regionName} ({zbounds[0]} < z < {zbounds[1]} m; {np.nanmean(field):5.2f} {r'$\pm$'} {np.nanstd(field):5.2f})'
                lineColors = ['k']
                lineWidths = [2.5]
                if zmax>0:
                    figFileName = f'{figdir}/{regionNameShort}_z0000-{np.abs(np.int32(zmin)):04d}_tempAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                else:
                    figFileName = f'{figdir}/{regionNameShort}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_tempAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'

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

                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)

                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
                plt.close()

                # Then plot salinity anomaly
                field = [dsIn['salinity'] - dsIn0['salinity'].isel(Time=0)]
                units = dsIn['salinity'].attrs['units']
                yLabel = f'Salinity ({units})'
                title = f'Volume-Mean salinity anomaly wrt year {year0} in {regionName} ({zbounds[0]} < z < {zbounds[1]} m; {np.nanmean(field):5.2f} {r'$\pm$'} {np.nanstd(field):5.2f})'
                lineColors = ['k']
                lineWidths = [2.5]
                if zmax>0:
                    figFileName = f'{figdir}/{regionNameShort}_z0000-{np.abs(np.int32(zmin)):04d}_saltAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                else:
                    figFileName = f'{figdir}/{regionNameShort}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_saltAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                fig = timeseries_analysis_plot(field, movingAverageMonths,
                                               title, xLabel, yLabel,
                                               calendar=calendar,
                                               timevarname = 'Time',
                                               lineColors=lineColors,
                                               lineWidths=lineWidths,
                                               legendText=legendText)
                plt.tight_layout()
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)
                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
                plt.close()

                # Finally plot density anomaly
                units = dsIn['potentialDensity'].attrs['units']
                yLabel = f'Potential density ({units})'
                title = f'Volume-Mean potential density anomaly contributions wrt year {year0} in {regionName} ({zbounds[0]} < z < {zbounds[1]} m)'
                if zmax>0:
                    figFileName = f'{figdir}/{regionNameShort}_z0000-{np.abs(np.int32(zmin)):04d}_rhoAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                else:
                    figFileName = f'{figdir}/{regionNameShort}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_rhoAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                lineColors = ['k', 'gray', 'red', 'blue']
                lineWidths = [2.5, 2.5, 2.5, 2.5]

                drho = dsIn['potentialDensity'] - dsIn0['potentialDensity'].isel(Time=0)
                dgsw_rho = dsIn['gswPotentialDensity'] - dsIn0['gswPotentialDensity'].isel(Time=0)
                dgsw_rhoAlpha = dsIn['alphaPotentialDensity'] - dsIn0['alphaPotentialDensity'].isel(Time=0)
                dgsw_rhoBeta = dsIn['betaPotentialDensity'] - dsIn0['betaPotentialDensity'].isel(Time=0)
                legendText = [f'rho ({np.nanmean(drho):5.2f} {r'$\pm$'} {np.nanstd(drho):5.2f})']
                legendText.append(f'gsw_rho ({np.nanmean(dgsw_rho):5.2f} {r'$\pm$'} {np.nanstd(dgsw_rho):5.2f})')
                legendText.append(f'gsw_rhoAlpha ({np.nanmean(dgsw_rhoAlpha):5.2f} {r'$\pm$'} {np.nanstd(dgsw_rhoAlpha):5.2f})')
                legendText.append(f'gsw_rhoBeta ({np.nanmean(dgsw_rhoBeta):5.2f} {r'$\pm$'} {np.nanstd(dgsw_rhoBeta):5.2f})')
                field = [drho, dgsw_rho, dgsw_rhoAlpha, dgsw_rhoBeta]
                fig = timeseries_analysis_plot(field, movingAverageMonths,
                                               title, xLabel, yLabel,
                                               calendar=calendar,
                                               timevarname = 'Time',
                                               lineColors=lineColors,
                                               lineWidths=lineWidths,
                                               legendText=legendText)
                plt.tight_layout()
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)
                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
                plt.close()
    
    else:
      
        for k in range(len(dlevels)):
            timeSeriesFile0 = f'{outdir}/{groupName}_depth{int(dlevels[k]):04d}_year{year0:04d}.nc'
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

                dsIn0 = xr.open_dataset(timeSeriesFile0, decode_times=False).isel(nRegions=regionIndex)
                dsIn = xr.open_mfdataset(timeSeriesFiles, combine='nested',
                                         concat_dim='Time', decode_times=False).isel(nRegions=regionIndex)

                movingAverageMonths = 1
                #movingAverageMonths = 12

                xLabel = 'Time (yr)'
                legendText = ['']

                # Plot temperature anomaly first
                field = [dsIn['temperature'] - dsIn0['temperature'].isel(Time=0)]
                units = dsIn['temperature'].attrs['units']
                yLabel = f'Temperature ({units})'
                title = f'Mean temperature anomaly wrt year {year0} in {regionName} (z={z[zlevels[k]]:5.1f} m; {np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f})'
                lineColors = ['k']
                lineWidths = [2.5]
                figFileName = f'{figdir}/{regionNameShort}_depth{int(dlevels[k]):04d}_tempAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                fig = timeseries_analysis_plot(field, movingAverageMonths,
                                               title, xLabel, yLabel,
                                               calendar=calendar,
                                               timevarname = 'Time',
                                               lineColors=lineColors,
                                               lineWidths=lineWidths,
                                               legendText=legendText)
                plt.tight_layout()
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)
                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)

                # Then plot salinity anomaly
                field = [dsIn['salinity'] - dsIn0['salinity'].isel(Time=0)]
                units = dsIn['salinity'].attrs['units']
                yLabel = f'Salinity ({units})'
                title = f'Mean salinity anomaly wrt year {year0} in {regionName} (z={z[zlevels[k]]:5.1f} m; {np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f})'
                lineColors = ['k']
                lineWidths = [2.5]
                figFileName = f'{figdir}/{regionNameShort}_depth{int(dlevels[k]):04d}_saltAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                fig = timeseries_analysis_plot(field, movingAverageMonths,
                                               title, xLabel, yLabel,
                                               calendar=calendar,
                                               timevarname = 'Time',
                                               lineColors=lineColors,
                                               lineWidths=lineWidths,
                                               legendText=legendText)
                plt.tight_layout()
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)
                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)

                # Finally plot density anomaly
                units = dsIn['potentialDensity'].attrs['units']
                yLabel = f'Potential density ({units})'
                title = f'Volume-Mean potential density anomaly contributions wrt year {year0} in {regionName} ({zbounds[0]} < z < {zbounds[1]} m)'
                title = f'Mean potential density anomaly contributions wrt year {year0} in {regionName} (z={z[zlevels[k]]:5.1f} m)'
                figFileName = f'{figdir}/{regionNameShort}_depth{int(dlevels[k]):04d}_rhoAnomalywrtYear{year0}_years{years[0]}-{years[-1]}.png'
                lineColors = ['k', 'gray', 'red', 'blue']
                lineWidths = [2.5, 2.5, 2.5, 2.5]

                drho = dsIn['potentialDensity'] - dsIn0['potentialDensity'].isel(Time=0)
                dgsw_rho = dsIn['gswPotentialDensity'] - dsIn0['gswPotentialDensity'].isel(Time=0)
                dgsw_rhoAlpha = dsIn['alphaPotentialDensity'] - dsIn0['alphaPotentialDensity'].isel(Time=0)
                dgsw_rhoBeta = dsIn['betaPotentialDensity'] - dsIn0['betaPotentialDensity'].isel(Time=0)
                legendText = [f'rho ({np.nanmean(drho):5.2f} {r'$\pm$'} {np.nanstd(drho):5.2f})']
                legendText.append(f'gsw_rho ({np.nanmean(dgsw_rho):5.2f} {r'$\pm$'} {np.nanstd(dgsw_rho):5.2f})')
                legendText.append(f'gsw_rhoAlpha ({np.nanmean(dgsw_rhoAlpha):5.2f} {r'$\pm$'} {np.nanstd(dgsw_rhoAlpha):5.2f})')
                legendText.append(f'gsw_rhoBeta ({np.nanmean(dgsw_rhoBeta):5.2f} {r'$\pm$'} {np.nanstd(dgsw_rhoBeta):5.2f})')
                field = [drho, dgsw_rho, dgsw_rhoAlpha, dgsw_rhoBeta]
                fig = timeseries_analysis_plot(field, movingAverageMonths,
                                               title, xLabel, yLabel,
                                               calendar=calendar,
                                               timevarname = 'Time',
                                               lineColors=lineColors,
                                               lineWidths=lineWidths,
                                               legendText=legendText)
                plt.tight_layout()
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)
                plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
                plt.close()
