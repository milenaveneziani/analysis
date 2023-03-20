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

from mpas_analysis.shared.io.utility import decode_strings

from common_functions import add_inset
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
years = range(year1, year2+1)

# Settings for anvil/chrysalis:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#regionmaskfile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/EC30to60E2r2_arctic_atlantic_budget_regions20230313.nc'
#transectmaskfile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/EC30to60E2r2_arctic_atlantic_budget_regionsTransects20230313.nc'
#regionfeaturefile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/arctic_atlantic_budget_regions.geojson'
#transectfeaturefile = 'lcrc/group/e3sm/ac.milena/mpas-region_masks/arctic_atlantic_budget_regionsTransects.geojson'
#casenameFull = 'v2_1.LR.piControl'
#casename = 'v2_1.LR.piControl'
#modeldir = f'/lcrc/group/e3sm/ac.golaz/E3SMv2_1/{casenameFull}/archive/ocn/hist'

# Settings for cori:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
projectdir = '/global/project/projectdirs/e3sm'
#projectdir = '/global/cfs/projectdirs/e3sm'
meshfile = f'{projectdir}/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
regionmaskfile = f'{projectdir}/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regions20230313.nc'
transectmaskfile = f'{projectdir}/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regionsTransects20230313.nc'
regionfeaturefile = f'{projectdir}/milena/mpas-region_masks/arctic_atlantic_budget_regions.geojson'
transectfeaturefile = f'{projectdir}/milena/mpas-region_masks/arctic_atlantic_budget_regionsTransects.geojson'
casenameFull = '20221201.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.lat-dep-bd-submeso.cori-knl'
casename = 'fullyRRM_lat-dep-bd-submeso'
modeldir = f'/global/cscratch1/sd/milena/e3sm_scratch/cori-knl/{casenameFull}/run'

m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
rho0 = 1027.0 # kg/m^3
dt = 30.0*86400.0 # 1 month dt in seconds for sshTend calculation

# others: frazilTend, icebergFlux, landIceFlux
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
                   {'name': 'ssh',
                    'title': 'Volume change due to ssh tendency',
                    'mpasName': 'timeMonthly_avg_ssh'}]
varlist = [var['mpasName'] for var in regionVariables]

outfile = f'volBudget_{casename}_years{year1:04d}-{year2:04d}.nc'

transectName = 'all'
figdir = f'./volBudget/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

nTime = 12*len(years)

# Read in regions information
dsRegionMask = xr.open_dataset(regionmaskfile)
regions = decode_strings(dsRegionMask.regionNames)
nRegions = np.size(regions)

# Read in transects information
mask = get_mask_short_names(xr.open_dataset(transectmaskfile))
transectList = mask.shortNames[:].values
nTransects = len(transectList)
maxEdges = mask.dims['maxEdgesInTransect']
print(f'\nComputing/plotting time series for these transects: {transectList}')
print(f' and these regions: {regions}\n')

# Create a list of edges and total edges in each transect
nEdgesInTransect = np.zeros(nTransects)
edgeVals = np.zeros((nTransects, maxEdges))
for j in range(nTransects):
    amask = mask.sel(shortNames=transectList[j]).squeeze()
    transectEdges = amask.transectEdgeGlobalIDs.values
    inds = np.where(transectEdges > 0)[0]
    nEdgesInTransect[j] = len(inds)
    transectEdges = transectEdges[inds]
    edgeVals[j, :len(inds)] = np.asarray(transectEdges-1, dtype='i')
nEdgesInTransect = np.asarray(nEdgesInTransect, dtype='i')
edgesToRead = edgeVals[0, :nEdgesInTransect[0]]
for j in range(1, nTransects):
    edgesToRead = np.hstack([edgesToRead, edgeVals[j, :nEdgesInTransect[j]]])
edgesToRead = np.asarray(edgesToRead, dtype='i')

# Create a list with the start and stop for transect bounds
nTransectStartStop = np.zeros(nTransects+1)
for j in range(1, nTransects+1):
    nTransectStartStop[j] = nTransectStartStop[j-1] + nEdgesInTransect[j-1]

# Read in relevant mesh information
mesh = xr.open_dataset(meshfile)
dvEdge = mesh.dvEdge.sel(nEdges=edgesToRead).values
cellsOnEdge = mesh.cellsOnEdge.sel(nEdges=edgesToRead).values
maxLevelCell = mesh.maxLevelCell.values
kmaxOnCells1 = maxLevelCell[cellsOnEdge[:, 0]-1]
kmaxOnCells2 = maxLevelCell[cellsOnEdge[:, 1]-1]
areaCell = mesh.areaCell
edgeSigns = np.zeros((nTransects, len(edgesToRead)))
for j in range(nTransects):
    edgeSigns[j, :] = mask.sel(nEdges=edgesToRead, shortNames=transectList[j]).squeeze().transectEdgeMaskSigns.values
nLevels = mesh.dims['nVertLevels']

# Compute budget if outfile does not exist
if not os.path.exists(outfile):
    # Volume transport across the open transects bounding the budget regions
    vol = np.zeros((nTime, nTransects))
    # Volume changes integrated over the budget regions due to various processes
    evapFlux = np.zeros((nTime, nRegions))
    rainFlux = np.zeros((nTime, nRegions))
    snowFlux = np.zeros((nTime, nRegions))
    riverRunoffFlux = np.zeros((nTime, nRegions))
    iceRunoffFlux = np.zeros((nTime, nRegions))
    seaiceFlux = np.zeros((nTime, nRegions))
    ssh = np.zeros((nTime, nRegions))
    t = np.zeros(nTime)
    i = 0
    for year in years:
        print(f'Year = {year:04d} out of {len(years)} years total')
        for month in range(1, 13):
            print(f'  Month= {month:02d}')
            modelfile = f'{modeldir}/{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
            # Read in model data
            ncid = Dataset(modelfile, 'r')
            if 'timeMonthly_avg_normalTransportVelocity' in ncid.variables.keys():
                vel = ncid.variables['timeMonthly_avg_normalTransportVelocity'][0, edgesToRead, :]
            elif 'timeMonthly_avg_normalVelocity' in ncid.variables.keys():
                vel = ncid.variables['timeMonthly_avg_normalVelocity'][0, edgesToRead, :]
                if 'timeMonthly_avg_normalGMBolusVelocity' in ncid.variables.keys():
                    vel += ncid.variables['timeMonthly_avg_normalGMBolusVelocity'][0, edgesToRead, :]
            else:
                raise KeyError('no appropriate normalVelocity variable found')
            # Note that the following is incorrect when cellsOnEdge is zero (that cell bordering the
            # transect edge is on land), but that is OK because the value will be masked during
            # land-sea masking below
            dzOnCells1 = ncid.variables['timeMonthly_avg_layerThickness'][0, cellsOnEdge[:, 0]-1, :]
            dzOnCells2 = ncid.variables['timeMonthly_avg_layerThickness'][0, cellsOnEdge[:, 1]-1, :]
            t[i] = ncid.variables['timeMonthly_avg_daysSinceStartOfSim'][:]/365.
            ncid.close()

            # ** Compute net transports across transects **
            # Mask transect values that fall on land
            landmask1 = cellsOnEdge[:, 0]==0
            landmask2 = cellsOnEdge[:, 1]==0
            dzOnCells1[landmask1, :] = np.nan
            dzOnCells2[landmask2, :] = np.nan
            # Mask transect values that fall onto topography
            for k in range(len(kmaxOnCells1)):
                dzOnCells1[k, kmaxOnCells1[k]:] = np.nan
            for k in range(len(kmaxOnCells2)):
                dzOnCells2[k, kmaxOnCells2[k]:] = np.nan

            # Interpolate values onto edges
            dzOnEdges = np.nanmean(np.array([dzOnCells1, dzOnCells2]), axis=0)

            for j in range(nTransects):
                start = int(nTransectStartStop[j])
                stop = int(nTransectStartStop[j+1])
                dx = dvEdge[start:stop]
                dx2d = np.transpose(np.tile(dx, (nLevels, 1)))
                dz = dzOnEdges[start:stop, :]

                normalVel = vel[start:stop, :] * edgeSigns[j, start:stop, np.newaxis]
                maskOnEdges = np.isnan(dz)
                normalVel[maskOnEdges] = np.nan
                dx2d[maskOnEdges] = np.nan
                dArea = dx2d * dz
                #transectLength = np.nansum(dx2d, axis=0)
                #transectArea = np.nansum(np.nansum(dArea)) 

                vol[i, j] =   np.nansum(np.nansum( normalVel * dArea ))

            # ** Compute net surface fluxes and SSH tendency for each region **
            ds = xr.open_dataset(modelfile)
            ds = ds[varlist]
            for j in range(nRegions):
                print('regionName=', regions[j])
                dsMask = dsRegionMask.isel(nRegions=j)
                cellMask = dsMask.regionCellMasks == 1
                localArea = areaCell.where(cellMask, drop=True)

                fluxes = (localArea*ds.where(cellMask, drop=True)).sum(dim='nCells')

                evapFlux[i, j] = fluxes[[var['mpasName'] for var in regionVariables if var['name']=='evap'][0]].values
                rainFlux[i, j] = fluxes[[var['mpasName'] for var in regionVariables if var['name']=='rain'][0]].values
                snowFlux[i, j] = fluxes[[var['mpasName'] for var in regionVariables if var['name']=='snow'][0]].values
                riverRunoffFlux[i, j] = fluxes[[var['mpasName'] for var in regionVariables if var['name']=='riverRunoff'][0]].values
                iceRunoffFlux[i, j] = fluxes[[var['mpasName'] for var in regionVariables if var['name']=='iceRunoff'][0]].values
                seaiceFlux[i, j] = fluxes[[var['mpasName'] for var in regionVariables if var['name']=='seaIceFreshwater'][0]].values
                ssh[i, j] = fluxes[[var['mpasName'] for var in regionVariables if var['name']=='ssh'][0]].values

            i = i+1

    vol = m3ps_to_Sv*vol
    evapFlux = 1/rho0*m3ps_to_Sv*evapFlux
    rainFlux = 1/rho0*m3ps_to_Sv*rainFlux
    snowFlux = 1/rho0*m3ps_to_Sv*snowFlux
    riverRunoffFlux = 1/rho0*m3ps_to_Sv*riverRunoffFlux
    iceRunoffFlux = 1/rho0*m3ps_to_Sv*iceRunoffFlux
    seaiceFlux = 1/rho0*m3ps_to_Sv*seaiceFlux
    sshTend = 1/dt*m3ps_to_Sv*np.diff(ssh, n=1, axis=0, prepend=np.nan)

    # Save to file
    ncid = Dataset(outfile, mode='w', clobber=True, format='NETCDF3_CLASSIC')
    ncid.createDimension('Time', None)
    ncid.createDimension('nTransects', nTransects)
    ncid.createDimension('nRegions', nRegions)
    ncid.createDimension('StrLen', 64)

    transectNames = ncid.createVariable('TransectNames', 'c', ('nTransects', 'StrLen'))
    regionNames = ncid.createVariable('RegionNames', 'c', ('nRegions', 'StrLen'))
    times = ncid.createVariable('Time', 'f8', 'Time')
    volVar = ncid.createVariable('vol', 'f8', ('Time', 'nTransects'))
    evapVar = ncid.createVariable('evapFlux', 'f8', ('Time', 'nRegions'))
    rainVar = ncid.createVariable('rainFlux', 'f8', ('Time', 'nRegions'))
    snowVar = ncid.createVariable('snowFlux', 'f8', ('Time', 'nRegions'))
    lrunoffVar = ncid.createVariable('riverRunoffFlux', 'f8', ('Time', 'nRegions'))
    srunoffVar = ncid.createVariable('iceRunoffFlux', 'f8', ('Time', 'nRegions'))
    icefreshVar = ncid.createVariable('seaiceFlux', 'f8', ('Time', 'nRegions'))
    sshTendVar = ncid.createVariable('sshTend', 'f8', ('Time', 'nRegions'))

    volVar.units = 'Sv'
    evapVar.units = 'Sv'
    rainVar.units = 'Sv'
    snowVar.units = 'Sv'
    lrunoffVar.units = 'Sv'
    srunoffVar.units = 'Sv'
    icefreshVar.units = 'Sv'
    sshTendVar.units = 'Sv'

    volVar.description = 'Volume transport across transect'
    evapVar.description = 'Volume change due to region integrated evaporation'
    rainVar.description = 'Volume change due to region integrated rain precipitation'
    snowVar.description = 'Volume change due to region integrated snow precipitation'
    lrunoffVar.description = 'Volume change due to region integrated liquid runoff'
    srunoffVar.description = 'Volume change due to region integrated solid runoff'
    icefreshVar.description = 'Volume change due to region integrated sea-ice freshwater'
    sshTendVar.description = 'Volume change due to SSH tendency'

    times[:] = t
    volVar[:, :] = vol
    evapVar[:, :] = evapFlux
    rainVar[:, :] = rainFlux
    snowVar[:, :] = snowFlux
    lrunoffVar[:, :] = riverRunoffFlux
    srunoffVar[:, :] = iceRunoffFlux
    icefreshVar[:, :] = seaiceFlux
    sshTendVar[:, :] = sshTend
    for j in range(nTransects):
        nLetters = len(transectList[j])
        transectNames[j, :nLetters] = transectList[j]
    for j in range(nRegions):
        nLetters = len(regions[j])
        regionNames[j, :nLetters] = regions[j]
    ncid.close()
else:
    print(f'File {outfile} already exists. Plotting only...')

# Read in previously computed volume budget quantities
ncid = Dataset(outfile, mode='r')
t = ncid.variables['Time'][:]
vol = ncid.variables['vol'][:, :]
evap = ncid.variables['evapFlux'][:, :]
rain = ncid.variables['rainFlux'][:, :]
snow = ncid.variables['snowFlux'][:, :]
lrunoff = ncid.variables['riverRunoffFlux'][:, :]
srunoff = ncid.variables['iceRunoffFlux'][:, :]
seaiceFW = ncid.variables['seaiceFlux'][:, :]
sshTend = ncid.variables['sshTend'][:, :]
ncid.close()
tot = evap + rain + snow + lrunoff + srunoff + seaiceFW + sshTend

if os.path.exists(transectfeaturefile):
    fctransect = read_feature_collection(transectfeaturefile)
else:
    raise IOError('No transect feature file found')
if os.path.exists(regionfeaturefile):
    fcregion = read_feature_collection(regionfeaturefile)
else:
    raise IOError('No region feature file found')

figdpi = 300
figsize = (8, 4)
for j in range(nTransects):
    if platform.python_version()[0]=='3':
        searchString = transectList[j][2:]
    else:
        searchString = transectList[j]
    transectName_forfigfile = searchString.replace(" ", "")

    fc = FeatureCollection()
    for feature in fctransect.features:
        if feature['properties']['name'] == searchString:
            fc.add_feature(feature)
            break

    vol_runavg = pd.Series.rolling(pd.DataFrame(vol[:, j]), 12, center=True).mean()
    figfile = f'{figdir}/volTransport_{transectName_forfigfile}_{casename}_years{year1:04d}-{year2:04d}.png'
    fig = plt.figure(figsize=figsize, dpi=figdpi)
    ax = fig.add_subplot()
    ax.plot(t, vol[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax.plot(t, vol_runavg, 'k', linewidth=3)
    ax.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.grid(color='k', linestyle=':', linewidth = 0.5)
    ax.set_title(f'mean={np.nanmean(vol[:, j]):6.4f} $\pm$ {np.nanstd(vol[:, j]):6.4f})', \
                    fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volume transport (Sv)', fontsize=12, fontweight='bold')

    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Transect = {searchString}\nrunname = {casename}', fontsize=14, fontweight='bold', y=1.2)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

figsize = (16, 16)
res = np.zeros((nTime, nRegions))
for j in range(nRegions):
    regionName = regions[j]
    regionName_forfigfile = regionName.replace(" ", "")

    if regionName=='Greater Arctic':
        res[:, j] = tot[:, j] + vol[:, 0] + vol[:, 1] + vol[:, 2] + vol[:, 3] + vol[:, 4]
    elif regionName=='North Atlantic subpolar gyre':
        res[:, j] = tot[:, j] + vol[:, 1] + vol[:, 2] + vol[:, 3] + vol[:, 4] + vol[:, 5]
    elif regionName=='North Atlantic subtropical gyre':
        res[:, j] = tot[:, j] + vol[:, 5] + vol[:, 6]
    elif regionName=='Atlantic tropical':
        res[:, j] = tot[:, j] + vol[:, 6] + vol[:, 7]
    else:
        print('Invalid region name: res field not calculated')

    fc = FeatureCollection()
    for feature in fcregion.features:
        if feature['properties']['name'] == regionName:
            fc.add_feature(feature)
            break

    evap_runavg = pd.Series.rolling(pd.DataFrame(evap[:, j]), 12, center=True).mean()
    rain_runavg = pd.Series.rolling(pd.DataFrame(rain[:, j]), 12, center=True).mean()
    snow_runavg = pd.Series.rolling(pd.DataFrame(snow[:, j]), 12, center=True).mean()
    lrunoff_runavg = pd.Series.rolling(pd.DataFrame(lrunoff[:, j]), 12, center=True).mean()
    srunoff_runavg = pd.Series.rolling(pd.DataFrame(srunoff[:, j]), 12, center=True).mean()
    seaiceFW_runavg = pd.Series.rolling(pd.DataFrame(seaiceFW[:, j]), 12, center=True).mean()
    sshTend_runavg = pd.Series.rolling(pd.DataFrame(sshTend[:, j]), 12, center=True).mean()
    figfile = f'{figdir}/volBudget_{regionName_forfigfile}_{casename}_years{year1:04d}-{year2:04d}.png'
    fig, ax = plt.subplots(4, 2, figsize=figsize)
    ax[0, 0].plot(t, evap[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[0, 0].plot(t, evap_runavg, 'k', linewidth=3)
    ax[0, 1].plot(t, rain[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[0, 1].plot(t, rain_runavg, 'k', linewidth=3)
    ax[1, 0].plot(t, snow[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[1, 0].plot(t, snow_runavg, 'k', linewidth=3)
    ax[1, 1].plot(t, lrunoff[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[1, 1].plot(t, lrunoff_runavg, 'k', linewidth=3)
    ax[2, 0].plot(t, srunoff[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[2, 0].plot(t, srunoff_runavg, 'k', linewidth=3)
    ax[2, 1].plot(t, seaiceFW[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[2, 1].plot(t, seaiceFW_runavg, 'k', linewidth=3)
    ax[3, 0].plot(t, sshTend[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[3, 0].plot(t, sshTend_runavg, 'k', linewidth=3)
    ax[3, 1].plot(t, tot[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[3, 1].plot(t, tot_runavg, 'k', linewidth=3)

    ax[0, 0].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[0, 1].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[1, 0].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[1, 1].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[2, 0].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[2, 1].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[3, 0].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[3, 1].plot(t, np.zeros_like(t), 'k', linewidth=1)

    ax[0, 0].autoscale(enable=True, axis='x', tight=True)
    ax[0, 1].autoscale(enable=True, axis='x', tight=True)
    ax[1, 0].autoscale(enable=True, axis='x', tight=True)
    ax[1, 1].autoscale(enable=True, axis='x', tight=True)
    ax[2, 0].autoscale(enable=True, axis='x', tight=True)
    ax[2, 1].autoscale(enable=True, axis='x', tight=True)
    ax[3, 0].autoscale(enable=True, axis='x', tight=True)
    ax[3, 1].autoscale(enable=True, axis='x', tight=True)

    ax[0, 0].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[0, 1].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[1, 0].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[1, 1].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[2, 0].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[2, 1].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[3, 0].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[3, 1].grid(color='k', linestyle=':', linewidth = 0.5)

    ax[0, 0].set_title(f'mean={np.nanmean(evap[:, j]):6.4f} $\pm$ {np.nanstd(evap[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[0, 1].set_title(f'mean={np.nanmean(rain[:, j]):6.4f} $\pm$ {np.nanstd(rain[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[1, 0].set_title(f'mean={np.nanmean(snow[:, j]):6.4f} $\pm$ {np.nanstd(snow[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[1, 1].set_title(f'mean={np.nanmean(lrunoff[:, j]):6.4f} $\pm$ {np.nanstd(lrunoff[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[2, 0].set_title(f'mean={np.nanmean(srunoff[:, j]):6.4f} $\pm$ {np.nanstd(srunoff[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[2, 1].set_title(f'mean={np.nanmean(seaiceFW[:, j]):6.4f} $\pm$ {np.nanstd(seaiceFW[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[3, 0].set_title(f'mean={np.nanmean(sshTend[:, j]):6.4f} $\pm$ {np.nanstd(sshTend[:, j]):6.4f})', \
                       fontsize=16, fontweight='bold')
    ax[3, 1].set_title(f'mean={np.nanmean(tot[:, j]):.2e} $\pm$ {np.nanstd(tot[:, j]):.2e})', \
                       fontsize=16, fontweight='bold')

    ax[3, 0].set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax[3, 1].set_xlabel('Time (Years)', fontsize=12, fontweight='bold')

    ax[0, 0].set_ylabel('Evap flux (Sv)', fontsize=12, fontweight='bold')
    ax[0, 1].set_ylabel('Rain flux (Sv)', fontsize=12, fontweight='bold')
    ax[1, 0].set_ylabel('Snow flux (Sv)', fontsize=12, fontweight='bold')
    ax[1, 1].set_ylabel('River runoff flux (Sv)', fontsize=12, fontweight='bold')
    ax[2, 0].set_ylabel('Ice runoff flux (Sv)', fontsize=12, fontweight='bold')
    ax[2, 1].set_ylabel('Sea ice FW flux (Sv)', fontsize=12, fontweight='bold')
    ax[3, 0].set_ylabel('SSH tendency (Sv)', fontsize=12, fontweight='bold')
    ax[3, 1].set_ylabel('Sum of all terms (Sv)', fontsize=12, fontweight='bold')

    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Region = {regionName}, res={np.nanmean(res[:, j]):.2e}\nrunname = {casename}', \
                 fontsize=14, fontweight='bold', y=1.025)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
