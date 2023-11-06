#!/usr/bin/env python
"""
Name: compute_volumeBudget.py
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
import platform
import time

from mpas_analysis.shared.io.utility import decode_strings

from common_functions import extract_openBoundaries, add_inset
from geometric_features import FeatureCollection, read_feature_collection


def get_mask_short_names(mask):
    # This is the right way to handle transects defined in the main transect file mask:
    shortnames = [str(aname.values)[:str(aname.values).find(',')].strip()
                  for aname in mask.transectNames]
    mask['shortNames'] = xr.DataArray(shortnames, dims='nTransects')
    mask = mask.set_index(nTransects=['transectNames', 'shortNames'])
    return mask


# Choose years
year1 = 1
year2 = 1
#year2 = 500
years = range(year1, year2+1)

# Settings for lcrc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#regionmaskfile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/EC30to60E2r2_arctic_atlantic_budget_regions20230313.nc'
#regionfeaturefile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/arctic_atlantic_budget_regions.geojson'
#casenameFull = 'v2_1.LR.historical_0101'
#casename = 'v2_1.LR.historical_0101'
#modeldir = f'/lcrc/group/e3sm/ac.golaz/E3SMv2_1/{casenameFull}/archive/ocn/hist'

# Settings for nersc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
projectdir = '/global/cfs/projectdirs/e3sm'
meshfile = f'{projectdir}/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
regionmaskfile = './test.nc'
regionfeaturefile = './test.geojson'
casename = 'GM600_Redi600'
casenameFull = 'GMPAS-JRA1p4_EC30to60E2r2_GM600_Redi600_perlmutter'
modeldir = f'{projectdir}/maltrud/archive/onHPSS/{casenameFull}/ocn/hist'
#meshfile = f'{projectdir}/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
#regionmaskfile = f'{projectdir}/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regions20230313.nc'
#regionfeaturefile = f'{projectdir}/milena/mpas-region_masks/arctic_atlantic_budget_regions.geojson'
#casenameFull = '20221201.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.lat-dep-bd-submeso.cori-knl'
#casename = 'fullyRRM_lat-dep-bd-submeso'
#modeldir = f'/global/cscratch1/sd/milena/e3sm_scratch/cori-knl/{casenameFull}/run'

m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
rho0 = 1027.0 # kg/m^3
dt = 30.0*86400.0 # 1 month dt in seconds for sshTend calculation

regionVariables = [{'name': 'evap',
                    'title': 'Volume change due to evaporation',
                    'mpasName': 'timeMonthly_avg_evaporationFlux'},
                   {'name': 'rain',
                    'title': 'Volume change due to rain',
                    'mpasName': 'timeMonthly_avg_rainFlux'},
                   {'name': 'snow',
                    'title': 'Volume change due to snow',
                    'mpasName': 'timeMonthly_avg_snowFlux'},
                   {'name': 'riverRunoff',
                    'title': 'Volume change due to liquid runoff',
                    'mpasName': 'timeMonthly_avg_riverRunoffFlux'},
                   {'name': 'iceRunoff',
                    'title': 'Volume change due to solid runoff',
                    'mpasName': 'timeMonthly_avg_iceRunoffFlux'},
                   {'name': 'seaIceFreshwater',
                    'title': 'Volume change due to sea ice melting/forming',
                    'mpasName': 'timeMonthly_avg_seaIceFreshWaterFlux'},
                   {'name': 'layerThicknessTend',
                    'title': 'Volume change due to layer thickness tendency',
                    'mpasName': 'timeMonthly_avg_tendLayerThickness'},
                   {'name': 'frazilTend',
                    'title': 'Volume change due to frazil formation tendency',
                    'mpasName': 'timeMonthly_avg_frazilLayerThicknessTendency'}]
# other fluxes to consider if icerberg and/or freshwater flux
# from ice cavities are included: icebergFlux, landIceFlux
#                   {'name': 'icebergFlux',
#                    'title': 'Volume change due to iceberg flux',
#                    'mpasName': 'timeMonthly_avg_icebergFlux'},
#                   {'name': 'landIceFlux',
#                    'title': 'Volume change due to land ice flux',
#                    'mpasName': 'timeMonthly_avg_landIceFlux'}]
varlist = [var['mpasName'] for var in regionVariables]

outfile = f'volBudget_{casename}_years{year1:04d}-{year2:04d}.nc'

transectName = 'all'
figdir = f'./volBudget/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

nTime = 12*len(years)

t0 = time.time()
# Read in regions information
dsRegionMask = xr.open_dataset(regionmaskfile)
regions = decode_strings(dsRegionMask.regionNames)
nRegions = np.size(regions)

# Read in relevant global mesh information
dsMesh = xr.open_dataset(meshfile)
edgesOnCell = dsMesh.edgesOnCell # edgeID of all edges bordering each cell. If 0, edge is on land.
cellsOnEdge = dsMesh.cellsOnEdge # cellID of the 2 cells straddling each edge. If 0, cell is on land.
areaCell = dsMesh.areaCell
nLevels = dsMesh.dims['nVertLevels']
nCells = dsMesh.dims['nCells']
maxEdges = dsMesh.dims['maxEdges']
maxLevelCell = dsMesh.maxLevelCell.values

# Create land/bathymetry mask separately for the two neighboring cells of each edge
#  Landmask
coe0 = cellsOnEdge.values[:, 0] - 1
coe1 = cellsOnEdge.values[:, 1] - 1
landmask0 = coe0==0
landmask1 = coe1==0
#  Topomask
kmaxOnCells0 = maxLevelCell[coe0]
kmaxOnCells1 = maxLevelCell[coe1]
karray = np.array(range(nLevels))
topomask0 = np.ones((len(landmask0), nLevels)) * karray[np.newaxis, :]
topomask1 = np.ones((len(landmask1), nLevels)) * karray[np.newaxis, :]
for k in range(len(kmaxOnCells0)):
    topomask0[k, kmaxOnCells0[k]:] = 0
for k in range(len(kmaxOnCells1)):
    topomask1[k, kmaxOnCells1[k]:] = 0

# Save to mesh dataset
dsMesh['cellsOnEdge0'] = (('nEdges'), coe0)
dsMesh['cellsOnEdge1'] = (('nEdges'), coe1)
dsMesh['landmask0'] = (('nEdges'), landmask0)
dsMesh['landmask1'] = (('nEdges'), landmask1)
t1 = time.time()
print('Mesh reading/preprocessing, #seconds = ', t1-t0)

# Compute budget if outfile does not exist
if not os.path.exists(outfile):
    # Initialize volume budget terms
    volNetLateralFlux = np.zeros((nTime, nRegions))
    evapFlux = np.zeros((nTime, nRegions))
    rainFlux = np.zeros((nTime, nRegions))
    snowFlux = np.zeros((nTime, nRegions))
    riverRunoffFlux = np.zeros((nTime, nRegions))
    iceRunoffFlux = np.zeros((nTime, nRegions))
    seaiceFlux = np.zeros((nTime, nRegions))
    layerThick = np.zeros((nTime, nRegions))
    frazilThick = np.zeros((nTime, nRegions))
    t = np.zeros(nTime)

    # Loop over regions
    for j in range(nRegions):
        t0 = time.time()
        print('regionName=', regions[j])
        # Get regional mesh quantities
        dsMask = dsRegionMask.isel(nRegions=j)
        #extract_openBoundaries(dsMask, dsMesh)
        [openBryEdges, openBrySigns] = extract_openBoundaries(dsMask.regionCellMasks.values, dsMesh)
        cellMask = dsMask.regionCellMasks == 1
        regionArea = areaCell.where(cellMask, drop=True)
        dvEdge = dsMesh.dvEdge[openBryEdges]
        coe0 = dsMesh.cellsOnEdge.isel(TWO=0)[openBryEdges]-1
        coe1 = dsMesh.cellsOnEdge.isel(TWO=1)[openBryEdges]-1
        t1 = time.time()
        print('Initial mesh-related processing, #seconds = ', t1-t0)

        ktime = 0
        for year in years:
            print(f'Year = {year:04d} out of {len(years)} years total')
            for month in range(1, 13):
                print(f'  Month= {month:02d}')
                modelfile = f'{modeldir}/{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

                ds = xr.open_dataset(modelfile)
                # ** Get time, velocities and layer thickness **
                t[ktime] = ds.timeMonthly_avg_daysSinceStartOfSim.values/365.
                if 'timeMonthly_avg_normalTransportVelocity' in ds.keys():
                    vel = ds.timeMonthly_avg_normalTransportVelocity.isel(Time=0)[openBryEdges, :]
                elif 'timeMonthly_avg_normalVelocity' in ds.keys():
                    vel = ds.timeMonthly_avg_normalVelocity.isel(Time=0)[openBryEdges, :]
                    if 'timeMonthly_avg_normalGMBolusVelocity' in ds.keys():
                        vel = vel + ds.timeMonthly_avg_normalGMBolusVelocity.isel(Time=0)[openBryEdges, :]
                    if 'timeMonthly_avg_normalMLEvelocity' in ds.keys():
                        vel = vel + ds.timeMonthly_avg_normalMLEvelocity.isel(Time=0)[openBryEdges, :]
                else:
                    raise KeyError('no appropriate normalVelocity variable found')
                dzOnCells = ds.timeMonthly_avg_layerThickness.isel(Time=0)
                # Compute dz for each edge in the region
                #  First, get dz's from two neighboring cells for each edge
                dzOnCells0 = dzOnCells.isel(nCells=coe0)
                dzOnCells1 = dzOnCells.isel(nCells=coe1)
                #  Then, interpolate dz's onto edges, also considering the topomask
                dzOnEdges = 0.5 * (dzOnCells0 + dzOnCells1)
                # Compute dArea for each edge in the region
                dArea = dvEdge * dzOnEdges

                # Compute net lateral fluxes:
                normalVel = vel * xr.DataArray(openBrySigns, dims='nEdges')
                lateralFlux = (normalVel * dArea).sum(dim='nEdges').sum(dim='nVertLevels')
                volNetLateralFlux[ktime, j] = lateralFlux.values
                t2 = time.time()
                print('Lateral flux calculation, #seconds = ', t2-t1)

                # Reduce data set and apply regional mask
                ds = ds[varlist].where(cellMask, drop=True)

                # Compute net surface fluxes:
                flux = ds[[var['mpasName'] for var in regionVariables if var['name']=='evap'][0]]
                evapFlux[ktime, j] = (flux * regionArea).sum(dim='nCells').values
                flux = ds[[var['mpasName'] for var in regionVariables if var['name']=='rain'][0]]
                rainFlux[ktime, j] = (flux * regionArea).sum(dim='nCells').values
                flux = ds[[var['mpasName'] for var in regionVariables if var['name']=='snow'][0]]
                snowFlux[ktime, j] = (flux * regionArea).sum(dim='nCells').values
                flux = ds[[var['mpasName'] for var in regionVariables if var['name']=='riverRunoff'][0]]
                riverRunoffFlux[ktime, j] = (flux * regionArea).sum(dim='nCells').values
                flux = ds[[var['mpasName'] for var in regionVariables if var['name']=='iceRunoff'][0]]
                iceRunoffFlux[ktime, j] = (flux * regionArea).sum(dim='nCells').values
                flux = ds[[var['mpasName'] for var in regionVariables if var['name']=='seaIceFreshwater'][0]]
                seaiceFlux[ktime, j] = (flux * regionArea).sum(dim='nCells').values
                t3 = time.time()
                print('Surface fluxes calculation, #seconds = ', t3-t2)

                # Compute layer thickness tendencies:
                layerThickTend = ds[[var['mpasName'] for var in regionVariables if var['name']=='layerThicknessTend'][0]]
                layerThickTend = ((layerThickTend * regionArea).sum(dim='nCells')).sum(dim='nVertLevels')
                layerThick[ktime, j] = layerThickTend.values
                frazilThickTend = ds[[var['mpasName'] for var in regionVariables if var['name']=='frazilTend'][0]]
                frazilThickTend = ((frazilThickTend * regionArea).sum(dim='nCells')).sum(dim='nVertLevels')
                frazilThick[ktime, j] = frazilThickTend.values
                t4 = time.time()
                print('Layer thickness tendencies calculation, #seconds = ', t4-t3)
                print('\nnetlateral, evap, rain, snow, riverrunoff, icerunoff, seaiceflux, layerThickTend, frazilThickTend')
                print(m3ps_to_Sv*volNetLateralFlux[0, 0]+1/rho0*m3ps_to_Sv*evapFlux[0, 0]+1/rho0*m3ps_to_Sv*rainFlux[0, 0]+1/rho0*m3ps_to_Sv*snowFlux[0, 0]+1/rho0*m3ps_to_Sv*riverRunoffFlux[0, 0]+1/rho0*m3ps_to_Sv*iceRunoffFlux[0, 0]+1/rho0*m3ps_to_Sv*seaiceFlux[0, 0], m3ps_to_Sv*layerThick[0, 0]+m3ps_to_Sv*frazilThick[0, 0])
                boh

            ktime = ktime + 1

    volNetLateralFlux = m3ps_to_Sv*volNetLateralFlux
    evapFlux = 1/rho0*m3ps_to_Sv*evapFlux
    rainFlux = 1/rho0*m3ps_to_Sv*rainFlux
    snowFlux = 1/rho0*m3ps_to_Sv*snowFlux
    riverRunoffFlux = 1/rho0*m3ps_to_Sv*riverRunoffFlux
    iceRunoffFlux = 1/rho0*m3ps_to_Sv*iceRunoffFlux
    seaiceFlux = 1/rho0*m3ps_to_Sv*seaiceFlux
    #sshTend = 1/dt*m3ps_to_Sv*np.diff(ssh, n=1, axis=0, prepend=np.nan)

    # Save to file
    ncid = Dataset(outfile, mode='w', clobber=True, format='NETCDF3_CLASSIC')
    ncid.createDimension('Time', None)
    ncid.createDimension('nRegions', nRegions)
    ncid.createDimension('nLevels', nLevels)
    ncid.createDimension('StrLen', 64)

    regionNames = ncid.createVariable('RegionNames', 'c', ('nRegions', 'StrLen'))
    times = ncid.createVariable('Time', 'f8', 'Time')
    volNetLateralFluxVar = ncid.createVariable('volNetLateralFlux', 'f8', ('Time', 'nRegions'))
    evapVar = ncid.createVariable('evapFlux', 'f8', ('Time', 'nRegions'))
    rainVar = ncid.createVariable('rainFlux', 'f8', ('Time', 'nRegions'))
    snowVar = ncid.createVariable('snowFlux', 'f8', ('Time', 'nRegions'))
    lrunoffVar = ncid.createVariable('riverRunoffFlux', 'f8', ('Time', 'nRegions'))
    srunoffVar = ncid.createVariable('iceRunoffFlux', 'f8', ('Time', 'nRegions'))
    icefreshVar = ncid.createVariable('seaiceFlux', 'f8', ('Time', 'nRegions'))
    #sshTendVar = ncid.createVariable('sshTend', 'f8', ('Time', 'nRegions'))

    volNetLateralFluxVar.units = 'Sv'
    evapVar.units = 'Sv'
    rainVar.units = 'Sv'
    snowVar.units = 'Sv'
    lrunoffVar.units = 'Sv'
    srunoffVar.units = 'Sv'
    icefreshVar.units = 'Sv'
    #sshTendVar.units = 'Sv'

    volNetLateralFluxVar.description = 'Net lateral volume transport across all open boundaries'
    evapVar.description = 'Volume change due to region integrated evaporation'
    rainVar.description = 'Volume change due to region integrated rain precipitation'
    snowVar.description = 'Volume change due to region integrated snow precipitation'
    lrunoffVar.description = 'Volume change due to region integrated liquid runoff'
    srunoffVar.description = 'Volume change due to region integrated solid runoff'
    icefreshVar.description = 'Volume change due to region integrated sea-ice freshwater'
    #sshTendVar.description = 'Volume change due to SSH tendency'

    times[:] = t
    volNetLateralFluxVar[:, :] = volNetLateralFlux
    evapVar[:, :] = evapFlux
    rainVar[:, :] = rainFlux
    snowVar[:, :] = snowFlux
    lrunoffVar[:, :] = riverRunoffFlux
    srunoffVar[:, :] = iceRunoffFlux
    icefreshVar[:, :] = seaiceFlux
    #sshTendVar[:, :] = sshTend
    for j in range(nRegions):
        nLetters = len(regions[j])
        regionNames[j, :nLetters] = regions[j]
    ncid.close()
else:
    print(f'File {outfile} already exists. Plotting only...')

# Read in previously computed volume budget quantities
ncid = Dataset(outfile, mode='r')
t = ncid.variables['Time'][:]
print(t)
lateralFlux = ncid.variables['volNetLateralFlux'][:, :]
evap = ncid.variables['evapFlux'][:, :]
rain = ncid.variables['rainFlux'][:, :]
snow = ncid.variables['snowFlux'][:, :]
lrunoff = ncid.variables['riverRunoffFlux'][:, :]
srunoff = ncid.variables['iceRunoffFlux'][:, :]
seaiceFW = ncid.variables['seaiceFlux'][:, :]
#sshTend = ncid.variables['sshTend'][:, :]
ncid.close()
tot = lateralFlux + evap + rain + snow + lrunoff + srunoff + seaiceFW + sshTend

if os.path.exists(regionfeaturefile):
    fcregion = read_feature_collection(regionfeaturefile)
else:
    raise IOError('No region feature file found')

figdpi = 300
figsize = (16, 16)
for j in range(nRegions):
    regionName = regions[j]
    regionName_forfigfile = regionName.replace(" ", "")

    fc = FeatureCollection()
    for feature in fcregion.features:
        if feature['properties']['name'] == regionName:
            fc.add_feature(feature)
            break

    lateralFlux_runavg = pd.Series.rolling(pd.DataFrame(lateralFlux[:, j]), 12, center=True).mean()
    evap_runavg = pd.Series.rolling(pd.DataFrame(evap[:, j]), 12, center=True).mean()
    rain_runavg = pd.Series.rolling(pd.DataFrame(rain[:, j]), 12, center=True).mean()
    snow_runavg = pd.Series.rolling(pd.DataFrame(snow[:, j]), 12, center=True).mean()
    lrunoff_runavg = pd.Series.rolling(pd.DataFrame(lrunoff[:, j]), 12, center=True).mean()
    srunoff_runavg = pd.Series.rolling(pd.DataFrame(srunoff[:, j]), 12, center=True).mean()
    seaiceFW_runavg = pd.Series.rolling(pd.DataFrame(seaiceFW[:, j]), 12, center=True).mean()
    sshTend_runavg = pd.Series.rolling(pd.DataFrame(sshTend[:, j]), 12, center=True).mean()
    tot_runavg = pd.Series.rolling(pd.DataFrame(tot[:, j]), 12, center=True).mean()
    figfile = f'{figdir}/volBudget_{regionName_forfigfile}_{casename}_years{year1:04d}-{year2:04d}.png'
    fig, ax = plt.subplots(5, 2, figsize=figsize)
    ax[0, 0].plot(t, lateralFlux[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[0, 0].plot(t, lateralFlux_runavg, 'k', linewidth=3)
    ax[0, 1].plot(t, evap[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[0, 1].plot(t, evap_runavg, 'k', linewidth=3)
    ax[1, 0].plot(t, rain[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[1, 0].plot(t, rain_runavg, 'k', linewidth=3)
    ax[1, 1].plot(t, snow[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[1, 1].plot(t, snow_runavg, 'k', linewidth=3)
    ax[2, 0].plot(t, lrunoff[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[2, 0].plot(t, lrunoff_runavg, 'k', linewidth=3)
    ax[2, 1].plot(t, srunoff[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[2, 1].plot(t, srunoff_runavg, 'k', linewidth=3)
    ax[3, 0].plot(t, seaiceFW[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[3, 0].plot(t, seaiceFW_runavg, 'k', linewidth=3)
    ax[3, 1].plot(t, sshTend[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[3, 1].plot(t, sshTend_runavg, 'k', linewidth=3)
    ax[4, 0].plot(t, tot[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[4, 0].plot(t, tot_runavg, 'k', linewidth=3)

    ax[0, 0].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[0, 1].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[1, 0].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[1, 1].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[2, 0].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[2, 1].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[3, 0].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[3, 1].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[4, 0].plot(t, np.zeros_like(t), 'k', linewidth=1)

    ax[0, 0].autoscale(enable=True, axis='x', tight=True)
    ax[0, 1].autoscale(enable=True, axis='x', tight=True)
    ax[1, 0].autoscale(enable=True, axis='x', tight=True)
    ax[1, 1].autoscale(enable=True, axis='x', tight=True)
    ax[2, 0].autoscale(enable=True, axis='x', tight=True)
    ax[2, 1].autoscale(enable=True, axis='x', tight=True)
    ax[3, 0].autoscale(enable=True, axis='x', tight=True)
    ax[3, 1].autoscale(enable=True, axis='x', tight=True)
    ax[4, 0].autoscale(enable=True, axis='x', tight=True)

    ax[0, 0].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[0, 1].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[1, 0].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[1, 1].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[2, 0].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[2, 1].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[3, 0].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[3, 1].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[4, 0].grid(color='k', linestyle=':', linewidth = 0.5)

    ax[0, 0].set_title(f'mean={np.nanmean(lateralFlux[:, j]):6.4f} $\pm$ {np.nanstd(lateralFlux[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[0, 1].set_title(f'mean={np.nanmean(evap[:, j]):6.4f} $\pm$ {np.nanstd(evap[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[1, 0].set_title(f'mean={np.nanmean(rain[:, j]):6.4f} $\pm$ {np.nanstd(rain[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[1, 1].set_title(f'mean={np.nanmean(snow[:, j]):6.4f} $\pm$ {np.nanstd(snow[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[2, 0].set_title(f'mean={np.nanmean(lrunoff[:, j]):6.4f} $\pm$ {np.nanstd(lrunoff[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[2, 1].set_title(f'mean={np.nanmean(srunoff[:, j]):6.4f} $\pm$ {np.nanstd(srunoff[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[3, 0].set_title(f'mean={np.nanmean(seaiceFW[:, j]):6.4f} $\pm$ {np.nanstd(seaiceFW[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[3, 1].set_title(f'mean={np.nanmean(sshTend[:, j]):6.4f} $\pm$ {np.nanstd(sshTend[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[4, 0].set_title(f'mean={np.nanmean(tot[:, j]):.2e} $\pm$ {np.nanstd(tot[:, j]):.2e})', \
                       fontsize=16, fontweight='bold')

    ax[3, 1].set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax[4, 0].set_xlabel('Time (Years)', fontsize=12, fontweight='bold')

    ax[0, 0].set_ylabel('Lateral flux (Sv)', fontsize=12, fontweight='bold')
    ax[0, 1].set_ylabel('Evap flux (Sv)', fontsize=12, fontweight='bold')
    ax[1, 0].set_ylabel('Rain flux (Sv)', fontsize=12, fontweight='bold')
    ax[1, 1].set_ylabel('Snow flux (Sv)', fontsize=12, fontweight='bold')
    ax[2, 0].set_ylabel('River runoff flux (Sv)', fontsize=12, fontweight='bold')
    ax[2, 1].set_ylabel('Ice runoff flux (Sv)', fontsize=12, fontweight='bold')
    ax[3, 0].set_ylabel('Sea ice FW flux (Sv)', fontsize=12, fontweight='bold')
    ax[3, 1].set_ylabel('SSH tendency (Sv)', fontsize=12, fontweight='bold')
    ax[4, 0].set_ylabel('Sum of all terms (Sv)', fontsize=12, fontweight='bold')

    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Region = {regionName}, runname = {casename}', \
                 fontsize=14, fontweight='bold', y=1.025)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
