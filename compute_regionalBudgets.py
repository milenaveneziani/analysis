#!/usr/bin/env python
"""
Name: compute_regionalBudgets.py
Author: Milena Veneziani


"""

# ensure plots are rendered on ICC
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
import time

from mpas_analysis.shared.io.utility import decode_strings

from common_functions import extract_openBoundaries, add_inset, plot_xtick_format
from geometric_features import FeatureCollection, read_feature_collection


# Settings for lcrc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#featurefile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/arctic_atlantic_budget_regions.geojson'
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#regionmaskfile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/EC30to60E2r2_arctic_atlantic_budget_regions20230313.nc'
#casenameFull = 'v2_1.LR.historical_0101'
#casename = 'v2_1.LR.historical_0101'
#modeldir = f'/lcrc/group/e3sm/ac.golaz/E3SMv2_1/{casenameFull}/archive/ocn/hist'

# Settings for nersc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
featurefile = '/global/cfs/projectdirs/e3sm/milena/mpas-region_masks/arctic_atlantic_budget_regions.geojson'
meshfile = '/global/cfs/projectdirs/e3sm/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
regionmaskfile = '/global/cfs/projectdirs/e3sm/milena/mpas-region_masks/EC30to60E2r2_arctic_atlantic_budget_regions20230313.nc'
casename = 'GM600_Redi600'
casenameFull = 'GMPAS-JRA1p4_EC30to60E2r2_GM600_Redi600_perlmutter'
modeldir = f'/global/cfs/projectdirs/e3sm/maltrud/archive/onHPSS/{casenameFull}/ocn/hist'
#meshfile = '/global/cfs/projectdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
#regionmaskfile = '/global/cfs/projectdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regions20230313.nc'
#casenameFull = '20221201.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.lat-dep-bd-submeso.cori-knl'
#casename = 'fullyRRM_lat-dep-bd-submeso'
#modeldir = f'/global/cscratch1/sd/milena/e3sm_scratch/cori-knl/{casenameFull}/run'

#regionNames = ['all']
#regionNames = ['Greater Arctic']
regionNames = ['North Atlantic subpolar gyre']

# Choose years
year1 = 1
year2 = 65
#year2 = 500
years = range(year1, year2+1)

referenceDate = '0001-01-01'

#movingAverageMonths = 1
movingAverageMonths = 12

m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
rho0 = 1027.0 # kg/m^3

figdir = f'./volBudget/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)
outdir = f'./volBudget_data/{casename}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found for this region group')

nTime = 12*len(years)

###
### PART 1 -- Read/compute mesh and regional mask quantities
###

# Read in regions information
dsRegionMask = xr.open_dataset(regionmaskfile)
regions = decode_strings(dsRegionMask.regionNames)
if regionNames[0]=='all':
    regionNames = regions
nRegions = np.size(regionNames)

# Read in relevant global mesh information
dsMesh = xr.open_dataset(meshfile)
areaCell = dsMesh.areaCell

# The following was originally created to properly mask points on land and 
# topography when computing the lateral fluxes. But I believe this is no
# longer necessary because 1) we only consider the open boundary edges
# (which are open ocean edges by definition) and 2) the layerThickness is
# already masked (nan) below maxLevelCell.
#
# Create land/bathymetry mask separately for the two neighboring cells of each edge
#  Landmask
#cellsOnEdge = dsMesh.cellsOnEdge # cellID of the 2 cells straddling each edge. If 0, cell is on land.
#coe0 = cellsOnEdge.isel(TWO=0).values - 1
#coe1 = cellsOnEdge.isel(TWO=1).values - 1
#landmask0 = coe0==0
#landmask1 = coe1==0
#  Topomask
#nLevels = dsMesh.dims['nVertLevels']
#maxLevelCell = dsMesh.maxLevelCell.values
#kmaxOnCells0 = maxLevelCell[coe0]
#kmaxOnCells1 = maxLevelCell[coe1]
#karray = np.array(range(nLevels))
#topomask0 = np.ones((len(landmask0), nLevels)) * karray[np.newaxis, :]
#topomask1 = np.ones((len(landmask1), nLevels)) * karray[np.newaxis, :]
#for k in range(len(kmaxOnCells0)):
#    topomask0[k, kmaxOnCells0[k]:] = 0
#for k in range(len(kmaxOnCells1)):
#    topomask1[k, kmaxOnCells1[k]:] = 0
# Save to mesh dataset
#dsMesh['cellsOnEdge0'] = (('nEdges'), coe0)
#dsMesh['cellsOnEdge1'] = (('nEdges'), coe1)
#dsMesh['landmask0'] = (('nEdges'), landmask0)
#dsMesh['landmask1'] = (('nEdges'), landmask1)
#dsMesh['topomask0'] = (('nEdges', 'nVertLevels'), topomask0)
#dsMesh['topomask1'] = (('nEdges', 'nVertLevels'), topomask1)

for n in range(nRegions):
    regionName = regionNames[n]
    rname = regionName.replace(' ', '')
    regionIndex = regions.index(regionName)
    print(f'\nRegion: {regionName}')

    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == regionName:
            fc.add_feature(feature)
            break

    ###
    ### PART 2 -- Compute budget terms if outfile does not exist
    ###           (note: could save yearly files instead)
    ###
    outfile = f'{outdir}/volBudget_{rname}_years{year1:04d}-{year2:04d}.nc'
    if not os.path.exists(outfile):
        # Initialize volume budget terms
        t = np.zeros(nTime)
        volNetLateralFlux = np.zeros(nTime)
        evapFlux = np.zeros(nTime)
        rainFlux = np.zeros(nTime)
        snowFlux = np.zeros(nTime)
        riverRunoffFlux = np.zeros(nTime)
        iceRunoffFlux = np.zeros(nTime)
        seaIceFreshWaterFlux = np.zeros(nTime)
        layerThick = np.zeros(nTime)
        #frazilThick = np.zeros(nTime)

        # Get regional mask quantities
        dsMask = dsRegionMask.isel(nRegions=n)
        [openBryEdges, openBrySigns] = extract_openBoundaries(dsMask.regionCellMasks.values, dsMesh)
        cellMask = dsMask.regionCellMasks == 1
        regionArea = areaCell.where(cellMask, drop=True)
        dvEdge = dsMesh.dvEdge[openBryEdges]
        coe0 = dsMesh.cellsOnEdge.isel(TWO=0, nEdges=openBryEdges) - 1
        coe1 = dsMesh.cellsOnEdge.isel(TWO=1, nEdges=openBryEdges) - 1

        # Compute budget terms
        ktime = 0
        for year in years:
            print(f'Year = {year:04d} out of {len(years)} years total')
            for month in range(1, 13):
                print(f'  Month= {month:02d}')
                modelfile = f'{modeldir}/{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

                ds = xr.open_dataset(modelfile, decode_times=False)

                t[ktime] = ds.Time.isel(Time=0).values

                # Compute net lateral fluxes
                t0 = time.time()
                if 'timeMonthly_avg_normalTransportVelocity' in ds.keys():
                    vel = ds.timeMonthly_avg_normalTransportVelocity.isel(Time=0, nEdges=openBryEdges)
                elif 'timeMonthly_avg_normalVelocity' in ds.keys():
                    vel = ds.timeMonthly_avg_normalVelocity.isel(Time=0, nEdges=openBryEdges)
                    if 'timeMonthly_avg_normalGMBolusVelocity' in ds.keys():
                        vel = vel + ds.timeMonthly_avg_normalGMBolusVelocity.isel(Time=0, nEdges=openBryEdges)
                    if 'timeMonthly_avg_normalMLEvelocity' in ds.keys():
                        vel = vel + ds.timeMonthly_avg_normalMLEvelocity.isel(Time=0, nEdges=openBryEdges)
                else:
                    raise KeyError('no appropriate normalVelocity variable found')
                #  First, get dz's from two neighboring cells for each edge
                dzOnCells0 = ds.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe0)
                dzOnCells1 = ds.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe1)
                #  Then, interpolate dz's onto edges, also considering the topomask
                dzOnEdges = 0.5 * (dzOnCells0 + dzOnCells1)
                dArea = dvEdge * dzOnEdges
                normalVel = vel * xr.DataArray(openBrySigns, dims='nEdges')
                lateralFlux = (normalVel * dArea).sum(dim='nVertLevels', skipna=True).sum(dim='nEdges')
                volNetLateralFlux[ktime] = lateralFlux.values
                t1 = time.time()
                print('   Lateral flux calculation, #seconds = ', t1-t0)

                # Compute net surface fluxes
                if 'timeMonthly_avg_evaporationFlux' in ds.keys():
                   flux = ds.timeMonthly_avg_evaporationFlux.isel(Time=0).where(cellMask, drop=True)
                   evapFlux[ktime] = (flux * regionArea).sum(dim='nCells').values
                else:
                   raise KeyError('no evaporation flux variable found')
                if 'timeMonthly_avg_rainFlux' in ds.keys():
                   flux = ds.timeMonthly_avg_rainFlux.isel(Time=0).where(cellMask, drop=True)
                   rainFlux[ktime] = (flux * regionArea).sum(dim='nCells').values
                else:
                   raise KeyError('no rain flux variable found')
                if 'timeMonthly_avg_snowFlux' in ds.keys():
                   flux = ds.timeMonthly_avg_snowFlux.isel(Time=0).where(cellMask, drop=True)
                   snowFlux[ktime] = (flux * regionArea).sum(dim='nCells').values
                else:
                   raise KeyError('no snow flux variable found')
                if 'timeMonthly_avg_riverRunoffFlux' in ds.keys():
                   flux = ds.timeMonthly_avg_riverRunoffFlux.isel(Time=0).where(cellMask, drop=True)
                   riverRunoffFlux[ktime] = (flux * regionArea).sum(dim='nCells').values
                else:
                   raise KeyError('no river runoff flux variable found')
                if 'timeMonthly_avg_iceRunoffFlux' in ds.keys():
                   flux = ds.timeMonthly_avg_iceRunoffFlux.isel(Time=0).where(cellMask, drop=True)
                   iceRunoffFlux[ktime] = (flux * regionArea).sum(dim='nCells').values
                else:
                   raise KeyError('no ice runoff flux variable found')
                if 'timeMonthly_avg_seaIceFreshWaterFlux' in ds.keys():
                   flux = ds.timeMonthly_avg_seaIceFreshWaterFlux.isel(Time=0).where(cellMask, drop=True)
                   seaIceFreshWaterFlux[ktime] = (flux * regionArea).sum(dim='nCells').values
                else:
                    raise KeyError('no sea ice freshwater flux variable found')
                if 'timeMonthly_avg_icebergFlux' in ds.keys():
                   flux = ds.timeMonthly_avg_icebergFlux.isel(Time=0).where(cellMask, drop=True)
                   icebergFlux[ktime] = (flux * regionArea).sum(dim='nCells').values
                if 'timeMonthly_avg_landIceFlux' in ds.keys():
                   flux = ds.timeMonthly_avg_landIceFlux.isel(Time=0).where(cellMask, drop=True)
                   landIceFlux[ktime] = (flux * regionArea).sum(dim='nCells').values
                t2 = time.time()
                print('   Surface fluxes calculation, #seconds = ', t2-t1)

                # Compute layer thickness tendencies
                if 'timeMonthly_avg_tendLayerThickness' in ds.keys():
                   layerThickTend = ds.timeMonthly_avg_tendLayerThickness.isel(Time=0).where(cellMask, drop=True)
                   layerThick[ktime] = (layerThickTend.sum(dim='nVertLevels', skipna=True) * regionArea).sum(dim='nCells').values
                else:
                   raise KeyError('no layer thickness tendency variable found')
                #if 'timeMonthly_avg_frazilLayerThicknessTendency' in ds.keys():
                #   frazilThickTend = ds.timeMonthly_avg_frazilLayerThicknessTendency.isel(Time=0).where(cellMask, drop=True)
                #   frazilThick[ktime] = (frazilThickTend.sum(dim='nVertLevels', skipna=True) * regionArea).sum(dim='nCells').values
                #else:
                #   raise KeyError('no frazil layer thickness tendency variable found')
                t3 = time.time()
                print('   Layer thickness tendencies calculation, #seconds = ', t3-t2)

                ktime = ktime + 1

        volNetLateralFlux = m3ps_to_Sv * volNetLateralFlux
        evapFlux = 1/rho0 * m3ps_to_Sv * evapFlux
        rainFlux = 1/rho0 * m3ps_to_Sv * rainFlux
        snowFlux = 1/rho0 * m3ps_to_Sv * snowFlux
        riverRunoffFlux = 1/rho0 * m3ps_to_Sv * riverRunoffFlux
        iceRunoffFlux = 1/rho0 * m3ps_to_Sv * iceRunoffFlux
        seaIceFreshWaterFlux = 1/rho0 * m3ps_to_Sv * seaIceFreshWaterFlux
        thickTend = m3ps_to_Sv * layerThick
        #frazilTend = m3ps_to_Sv * frazilThick

        # Save to file
        ncid = Dataset(outfile, mode='w', clobber=True, format='NETCDF3_CLASSIC')
        ncid.createDimension('Time', None)

        times = ncid.createVariable('Time', 'f8', 'Time')
        volNetLateralFluxVar = ncid.createVariable('volNetLateralFlux', 'f8', ('Time'))
        evapVar = ncid.createVariable('evapFlux', 'f8', ('Time'))
        rainVar = ncid.createVariable('rainFlux', 'f8', ('Time'))
        snowVar = ncid.createVariable('snowFlux', 'f8', ('Time'))
        lrunoffVar = ncid.createVariable('riverRunoffFlux', 'f8', ('Time'))
        srunoffVar = ncid.createVariable('iceRunoffFlux', 'f8', ('Time'))
        icefreshVar = ncid.createVariable('seaIceFreshWaterFlux', 'f8', ('Time'))
        thickTendVar = ncid.createVariable('thicknessTendency', 'f8', ('Time'))
        #frazilTendVar = ncid.createVariable('frazilTendency', 'f8', ('Time'))

        volNetLateralFluxVar.units = 'Sv'
        evapVar.units = 'Sv'
        rainVar.units = 'Sv'
        snowVar.units = 'Sv'
        lrunoffVar.units = 'Sv'
        srunoffVar.units = 'Sv'
        icefreshVar.units = 'Sv'
        thickTendVar.units = 'Sv'
        #frazilTendVar.units = 'Sv'

        volNetLateralFluxVar.description = 'Net lateral volume transport across all open boundaries'
        evapVar.description = 'Volume change due to region integrated evaporation'
        rainVar.description = 'Volume change due to region integrated rain precipitation'
        snowVar.description = 'Volume change due to region integrated snow precipitation'
        lrunoffVar.description = 'Volume change due to region integrated liquid runoff'
        srunoffVar.description = 'Volume change due to region integrated solid runoff'
        icefreshVar.description = 'Volume change due to region integrated sea-ice freshwater flux'
        thickTendVar.description = 'Volume change due to total water column tendency (SSH changes)'
        #frazilTendVar.description = 'Volume change due to frazil ice formation'

        times[:] = t
        volNetLateralFluxVar[:] = volNetLateralFlux
        evapVar[:] = evapFlux
        rainVar[:] = rainFlux
        snowVar[:] = snowFlux
        lrunoffVar[:] = riverRunoffFlux
        srunoffVar[:] = iceRunoffFlux
        icefreshVar[:] = seaIceFreshWaterFlux
        thickTendVar[:] = thickTend
        #frazilTendVar[:] = frazilTend
        ncid.close()
    else:
        print(f'\nFile {outfile} already exists. Plotting only...\n')

    ###
    ### PART 3 -- Plotting
    ###
    # Read in previously computed volume budget quantities
    ncid = Dataset(outfile, mode='r')
    t = ncid.variables['Time'][:]
    volNetLateralFlux = ncid.variables['volNetLateralFlux'][:]
    evapFlux = ncid.variables['evapFlux'][:]
    rainFlux = ncid.variables['rainFlux'][:]
    snowFlux = ncid.variables['snowFlux'][:]
    riverRunoffFlux = ncid.variables['riverRunoffFlux'][:]
    iceRunoffFlux = ncid.variables['iceRunoffFlux'][:]
    seaIceFreshWaterFlux = ncid.variables['seaIceFreshWaterFlux'][:]
    thickTend = ncid.variables['thicknessTendency'][:]
    #frazilTend = ncid.variables['frazilTendency'][:]
    ncid.close()
    res = thickTend - (volNetLateralFlux + evapFlux + rainFlux + snowFlux + riverRunoffFlux + iceRunoffFlux + seaIceFreshWaterFlux)
    #res = thickTend + frazilTend - (volNetLateralFlux + evapFlux + rainFlux + snowFlux + riverRunoffFlux + iceRunoffFlux + seaIceFreshWaterFlux)

    # Compute running averages
    volNetLateralFlux_runavg = pd.Series.rolling(pd.DataFrame(volNetLateralFlux), movingAverageMonths, center=True).mean()
    evapFlux_runavg = pd.Series.rolling(pd.DataFrame(evapFlux), movingAverageMonths, center=True).mean()
    rainFlux_runavg = pd.Series.rolling(pd.DataFrame(rainFlux), movingAverageMonths, center=True).mean()
    snowFlux_runavg = pd.Series.rolling(pd.DataFrame(snowFlux), movingAverageMonths, center=True).mean()
    riverRunoffFlux_runavg = pd.Series.rolling(pd.DataFrame(riverRunoffFlux), movingAverageMonths, center=True).mean()
    iceRunoffFlux_runavg = pd.Series.rolling(pd.DataFrame(iceRunoffFlux), movingAverageMonths, center=True).mean()
    seaIceFreshWaterFlux_runavg = pd.Series.rolling(pd.DataFrame(seaIceFreshWaterFlux), movingAverageMonths, center=True).mean()
    thickTend_runavg = pd.Series.rolling(pd.DataFrame(thickTend), movingAverageMonths, center=True).mean()
    #frazilTend_runavg = pd.Series.rolling(pd.DataFrame(frazilTend), movingAverageMonths, center=True).mean()
    res_runavg = pd.Series.rolling(pd.DataFrame(res), movingAverageMonths, center=True).mean()
    volNetLateralFluxMean = np.mean(volNetLateralFlux_runavg)

    evapFluxMean = np.mean(evapFlux_runavg)
    rainFluxMean = np.mean(rainFlux_runavg)
    snowFluxMean = np.mean(snowFlux_runavg)
    riverRunoffFluxMean = np.mean(riverRunoffFlux_runavg)
    iceRunoffFluxMean = np.mean(iceRunoffFlux_runavg)
    seaIceFreshWaterFluxMean = np.mean(seaIceFreshWaterFlux_runavg)
    #frazilTendMean = np.mean(frazilTend_runavg)
    thickTendMean = np.mean(thickTend_runavg)
    resMean = np.mean(res_runavg)

    figdpi = 300
    figsize = (16, 16)
    figfile = f'{figdir}/volBudget_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'

    fig, ax = plt.subplots(5, 2, figsize=figsize)
    ax[0, 0].plot(t, volNetLateralFlux, 'k', alpha=0.5, linewidth=1.5)
    ax[0, 1].plot(t, evapFlux, 'k', alpha=0.5, linewidth=1.5)
    ax[1, 0].plot(t, rainFlux, 'k', alpha=0.5, linewidth=1.5)
    ax[1, 1].plot(t, snowFlux, 'k', alpha=0.5, linewidth=1.5)
    ax[2, 0].plot(t, riverRunoffFlux, 'k', alpha=0.5, linewidth=1.5)
    ax[2, 1].plot(t, iceRunoffFlux, 'k', alpha=0.5, linewidth=1.5)
    ax[3, 0].plot(t, seaIceFreshWaterFlux, 'k', alpha=0.5, linewidth=1.5)
    #ax[3, 1].plot(t, frazilTend, 'k', alpha=0.5, linewidth=1.5)
    ax[4, 0].plot(t, thickTend, 'k', alpha=0.5, linewidth=1.5)
    ax[4, 1].plot(t, res, 'k', alpha=0.5, linewidth=1.5)
    if movingAverageMonths!=1:
        ax[0, 0].plot(t, volNetLateralFlux_runavg, 'k', linewidth=3)
        ax[0, 1].plot(t, evapFlux_runavg, 'k', linewidth=3)
        ax[1, 0].plot(t, rainFlux_runavg, 'k', linewidth=3)
        ax[1, 1].plot(t, snowFlux_runavg, 'k', linewidth=3)
        ax[2, 0].plot(t, riverRunoffFlux_runavg, 'k', linewidth=3)
        ax[2, 1].plot(t, iceRunoffFlux_runavg, 'k', linewidth=3)
        ax[3, 0].plot(t, seaIceFreshWaterFlux_runavg, 'k', linewidth=3)
        #ax[3, 1].plot(t, frazilTend_runavg, 'k', linewidth=3)
        ax[4, 0].plot(t, thickTend_runavg, 'k', linewidth=3)
        ax[4, 1].plot(t, res_runavg, 'k', linewidth=3)
     
    ax[0, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax[0, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax[1, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax[1, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax[2, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax[2, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax[3, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax[3, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax[4, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax[4, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)

    ax[0, 0].autoscale(enable=True, axis='x', tight=True)
    ax[0, 1].autoscale(enable=True, axis='x', tight=True)
    ax[1, 0].autoscale(enable=True, axis='x', tight=True)
    ax[1, 1].autoscale(enable=True, axis='x', tight=True)
    ax[2, 0].autoscale(enable=True, axis='x', tight=True)
    ax[2, 1].autoscale(enable=True, axis='x', tight=True)
    ax[3, 0].autoscale(enable=True, axis='x', tight=True)
    ax[3, 1].autoscale(enable=True, axis='x', tight=True)
    ax[4, 0].autoscale(enable=True, axis='x', tight=True)
    ax[4, 1].autoscale(enable=True, axis='x', tight=True)

    ax[0, 0].grid(color='k', linestyle=':', linewidth=0.5)
    ax[0, 1].grid(color='k', linestyle=':', linewidth=0.5)
    ax[1, 0].grid(color='k', linestyle=':', linewidth=0.5)
    ax[1, 1].grid(color='k', linestyle=':', linewidth=0.5)
    ax[2, 0].grid(color='k', linestyle=':', linewidth=0.5)
    ax[2, 1].grid(color='k', linestyle=':', linewidth=0.5)
    ax[3, 0].grid(color='k', linestyle=':', linewidth=0.5)
    ax[3, 1].grid(color='k', linestyle=':', linewidth=0.5)
    ax[4, 0].grid(color='k', linestyle=':', linewidth=0.5)
    ax[4, 1].grid(color='k', linestyle=':', linewidth=0.5)

    ax[0, 0].set_title(f'mean={volNetLateralFluxMean:.2e}', fontsize=16, fontweight='bold')
    ax[0, 1].set_title(f'mean={evapFluxMean:.2e}', fontsize=16, fontweight='bold')
    ax[1, 0].set_title(f'mean={rainFluxMean:.2e}', fontsize=16, fontweight='bold')
    ax[1, 1].set_title(f'mean={snowFluxMean:.2e}', fontsize=16, fontweight='bold')
    ax[2, 0].set_title(f'mean={riverRunoffFluxMean:.2e}', fontsize=16, fontweight='bold')
    ax[2, 1].set_title(f'mean={iceRunoffFluxMean:.2e}', fontsize=16, fontweight='bold')
    ax[3, 0].set_title(f'mean={seaIceFreshWaterFluxMean:.2e}', fontsize=16, fontweight='bold')
    #ax[3, 1].set_title(f'mean={frazilTendMean:.2e}', fontsize=16, fontweight='bold')
    ax[4, 0].set_title(f'mean={thickTendMean:.2e}', fontsize=16, fontweight='bold')
    ax[4, 1].set_title(f'mean={resMean:.2e}', fontsize=16, fontweight='bold')

    ax[4, 0].set_xlabel('Time (Days)', fontsize=12, fontweight='bold')
    ax[4, 1].set_xlabel('Time (Days)', fontsize=12, fontweight='bold')

    ax[0, 0].set_ylabel('Lateral flux (Sv)', fontsize=12, fontweight='bold')
    ax[0, 1].set_ylabel('Evap flux (Sv)', fontsize=12, fontweight='bold')
    ax[1, 0].set_ylabel('Rain flux (Sv)', fontsize=12, fontweight='bold')
    ax[1, 1].set_ylabel('Snow flux (Sv)', fontsize=12, fontweight='bold')
    ax[2, 0].set_ylabel('River runoff flux (Sv)', fontsize=12, fontweight='bold')
    ax[2, 1].set_ylabel('Ice runoff flux (Sv)', fontsize=12, fontweight='bold')
    ax[3, 0].set_ylabel('Sea ice FW flux (Sv)', fontsize=12, fontweight='bold')
    ax[3, 1].set_ylabel('Frazil thickness tend (Sv)', fontsize=12, fontweight='bold')
    ax[4, 0].set_ylabel('Layer thickness tend (Sv)', fontsize=12, fontweight='bold')
    ax[4, 1].set_ylabel('Residual (Sv)', fontsize=12, fontweight='bold')

    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Region = {regionName}, runname = {casename}', \
                 fontsize=14, fontweight='bold', y=1.025)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1)
    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

    figsize = (14, 8)
    figfile = f'{figdir}/volBudgetSummary_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    if movingAverageMonths==1:
        ax.plot(t, volNetLateralFlux, 'r', linewidth=2, label='netLateral')
        ax.plot(t, evapFlux+rainFlux+snowFlux, 'c', linewidth=2, label='E-P')
        ax.plot(t, riverRunoffFlux+iceRunoffFlux, 'g', linewidth=2, label='runoff')
        ax.plot(t, seaIceFreshWaterFlux, 'b', linewidth=2, label='seaiceFW')
        ax.plot(t, thickTend, 'm', linewidth=2, label='thickTend')
        #ax.plot(t, thickTend+frazilTend, 'm', linewidth=2, label='thick+frazil')
        ax.plot(t, res, 'k', linewidth=2, label='res')
    else:
        ax.plot(t, volNetLateralFlux_runavg, 'r', linewidth=2, label='netLateral')
        ax.plot(t, evapFlux_runavg+rainFlux_runavg+snowFlux_runavg, 'c', linewidth=2, label='E-P')
        ax.plot(t, riverRunoffFlux_runavg+iceRunoffFlux_runavg, 'g', linewidth=2, label='runoff')
        ax.plot(t, seaIceFreshWaterFlux_runavg, 'b', linewidth=2, label='seaiceFW')
        ax.plot(t, thickTend_runavg, 'm', linewidth=2, label='thickTend')
        #ax.plot(t, thickTend_runavg+frazilTend_runavg, 'm', linewidth=2, label='thick+frazil')
        ax.plot(t, res_runavg, 'k', linewidth=2, label='res')
        ax.set_title(f'{movingAverageMonths}-month running averages', fontsize=16, fontweight='bold')
    ax.plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.grid(color='k', linestyle=':', linewidth=0.5)
    ax.legend(loc='lower left')
    plot_xtick_format('gregorian', np.min(t), np.max(t), maxXTicks=20)
    ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sv', fontsize=12, fontweight='bold')
    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Region = {regionName}, runname = {casename}', \
                 fontsize=14, fontweight='bold', y=1.025)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1)
    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
