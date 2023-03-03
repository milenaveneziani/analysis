#!/usr/bin/env python
"""
Name: compute_FovFaz.py
Author: Milena Veneziani

Computes Fov (meridional freshwater flux due to overturning circulation)
and Faz (meridional freshwater flux due to the azonal or gyre circulation
component) as defined in de Vries and Weber (2005), their Eqs. 2,3.
Also see section 2.2 in Mecking et al (2017).

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


def get_mask_short_names(mask):
    # This is the right way to handle transects defined in the main transect file mask:
    shortnames = [str(aname.values)[:str(aname.values).find(',')].strip()
                  for aname in mask.transectNames]
    #shortnames = mask.transectNames.values
    mask['shortNames'] = xr.DataArray(shortnames, dims='nTransects')
    mask = mask.set_index(nTransects=['transectNames', 'shortNames'])
    return mask


# Choose years
year1 = 1
year2 = 1
years = range(year1, year1+1)

meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
maskfile = '/global/project/projectdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections.nc'
casenameFull = '20221201.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.lat-dep-bd-submeso.cori-knl'
casename = 'fullyRRM_lat-dep-bd-submeso'
modeldir = f'/global/cscratch1/sd/milena/e3sm_scratch/cori-knl/{casenameFull}/run'
#
#meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oRRS18to6v3/oRRS18to6v3.171116.nc'
#maskfile = '/global/project/projectdirs/e3sm/diagnostics/mpas_analysis/region_masks/RRS18to6v3_atlantic34S.nc'
#casenameFull = 'theta.20180906.branch_noCNT.A_WCYCL1950S_CMIP6_HR.ne120_oRRS18v3_ICG'
#casename = 'E3SM-HR'
#modeldir = f'/global/cscratch1/sd/milena/E3SM_simulations/{casenameFull}/run'
#
#meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/EC30to60E2r2/ocean.EC30to60E2r2.210210.nc'
#maskfile = '/global/project/projectdirs/e3sm/milena/mpas-region_masks/EC30to60E2r2_atlantic34S.nc'
#casenameFull = 'v2.LR.piControl'
#casename = 'v2.LR.piControl'
#modeldir = f'/global/cscratch1/sd//dcomeau/e3sm_scratch/cori-knl/{casenameFull}/archive/ocn/hist'
#

sZero = 35.0
m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups

use_fixedSref = False
if use_fixedSref:
    outfile = f'volFovFaz_{casename}_sref35_years{year1:04d}-{year2:04d}.nc'
else:
    outfile = f'volFovFaz_{casename}_years{year1:04d}-{year2:04d}.nc'

transectName = 'all'
#transectName = 'Atlantic zonal 34S'
figdir = f'./FovFaz/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

nTime = 12*len(years)

# Read in transect information
mask = get_mask_short_names(xr.open_dataset(maskfile))
if transectName=='all':
    transectList = mask.shortNames[:].values
else:
    transectList = transectName.split(',')
    if platform.python_version()[0]=='3':
        for i in range(len(transectList)):
            transectList[i] = "b'" + transectList[i]
nTransects = len(transectList)
maxEdges = mask.dims['maxEdgesInTransect']
print('Computing/plotting time series for these transects: ', transectList)

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
kmax = np.min(maxLevelCell[cellsOnEdge-1], axis=1)
edgeSigns = np.zeros((nTransects, len(edgesToRead)))
for j in range(nTransects):
    edgeSigns[j, :] = mask.sel(nEdges=edgesToRead, shortNames=transectList[j]).squeeze().transectEdgeMaskSigns.values

# Compute transports if outfile does not exist
if not os.path.exists(outfile):
    vTransect = np.zeros((nTime, nTransects))
    sTransect = np.zeros((nTime, nTransects))
    vol = np.zeros((nTime, nTransects))
    Fov = np.zeros((nTime, nTransects))
    Faz = np.zeros((nTime, nTransects))
    t = np.zeros(nTime)
    i = 0
    for year in years:
        print(f'Year = {year:04d}')
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
            saltOnCells1 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge[:, 0]-1, :]
            saltOnCells2 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge[:, 1]-1, :]
            dzOnCells1 = ncid.variables['timeMonthly_avg_layerThickness'][0, cellsOnEdge[:, 0]-1, :]
            dzOnCells2 = ncid.variables['timeMonthly_avg_layerThickness'][0, cellsOnEdge[:, 1]-1, :]
            t[i] = ncid.variables['timeMonthly_avg_daysSinceStartOfSim'][:]/365.
            ncid.close()

            # Mask values that fall on land
            saltOnCells1[cellsOnEdge[:, 0]==0, :] = np.nan
            saltOnCells2[cellsOnEdge[:, 1]==0, :] = np.nan
            dzOnCells1[cellsOnEdge[:, 0]==0, :] = np.nan
            dzOnCells2[cellsOnEdge[:, 1]==0, :] = np.nan
            # Interpolate values onto edges
            saltOnEdges = np.nanmean(np.array([saltOnCells1, saltOnCells2]), axis=0)
            dzOnEdges = np.nanmean(np.array([dzOnCells1, dzOnCells2]), axis=0)

            # Compute volume transport, Fov, and Faz for each transect
            for j in range(nTransects):
                start = int(nTransectStartStop[j])
                stop = int(nTransectStartStop[j+1])
                kmaxTransect = kmax[start:stop]
                dx = dvEdge[start:stop]
                dz = dzOnEdges[start:stop, :]
                dArea = dx[:, np.newaxis] * dz
                transectLength = np.nansum(dx)
                transectArea = np.nansum(np.nansum(dArea)) 

                normalVel = vel[start:stop, :] * edgeSigns[j, start:stop, np.newaxis]
                salt = saltOnEdges[start:stop, :]
                # Mask values that fall onto topography
                for k in range(len(kmaxTransect)):
                    normalVel[k, kmaxTransect[k]:] = np.nan
                    salt[k, kmaxTransect[k]:] = np.nan

                vTransect[i, j] = np.nansum(np.nansum(normalVel * dArea)) / transectArea
                sTransect[i, j] = np.nansum(np.nansum(salt * dArea)) / transectArea
                if use_fixedSref:
                    sRef = sZero
                else:
                    sRef = sTransect[i, j]

                # Each variable is decomposed into a transect averaged part, a zonal averaged
                # part and an azonal residual. For example, for velocity:
                # v = v_transect + v_zonalAvg + v_azonal
                vres = normalVel - vTransect[i, j]
                vZonalAvg = np.nansum(normalVel * dx[:, np.newaxis], axis=0) / transectLength
                sZonalAvg = np.nansum(salt * dx[:, np.newaxis], axis=0) / transectLength
                vAzonal = vres - vZonalAvg[np.newaxis, :]
                sAzonal = salt - sZonalAvg[np.newaxis, :]

                vol[i, j] =   np.nansum(np.nansum(normalVel * dArea))
                Fov[i, j] = - np.nansum(np.nansum(vres * dArea, axis=0) * (sZonalAvg-sRef)) / sRef
                Faz[i, j] = - np.nansum(np.nansum(vAzonal * (sAzonal-sRef) * dArea)) / sRef

            i = i+1

    vol = m3ps_to_Sv*vol
    Fov = m3ps_to_Sv*Fov
    Faz = m3ps_to_Sv*Faz

    # Save to file
    ncid = Dataset(outfile, mode='w', clobber=True, format='NETCDF3_CLASSIC')
    ncid.createDimension('Time', None)
    ncid.createDimension('nTransects', nTransects)
    ncid.createDimension('StrLen', 64)

    transectNames = ncid.createVariable('TransectNames', 'c', ('nTransects', 'StrLen'))
    times = ncid.createVariable('Time', 'f8', 'Time')
    vTransectVar = ncid.createVariable('vTransect', 'f8', ('Time', 'nTransects'))
    sTransectVar = ncid.createVariable('sTransect', 'f8', ('Time', 'nTransects'))
    volVar = ncid.createVariable('vol', 'f8', ('Time', 'nTransects'))
    FovVar = ncid.createVariable('Fov', 'f8', ('Time', 'nTransects'))
    FazVar = ncid.createVariable('Faz', 'f8', ('Time', 'nTransects'))

    vTransectVar.units = 'm/s'
    sTransectVar.units = 'psu'
    volVar.units = 'Sv'
    FovVar.units = 'Sv'
    FazVar.units = 'Sv'

    vTransectVar.description = 'Cross-transect averaged velocity'
    sTransectVar.description = 'Transect averaged salinity'
    volVar.description = 'Volume transport across transect'
    FovVar.description = 'Meridional Freshwater transport due to overturning circulation'
    FazVar.description = 'Meridional Freshwater transport due to azonal (gyre) circulation'

    times[:] = t
    vTransectVar[:, :] = vTransect
    sTransectVar[:, :] = sTransect
    volVar[:, :] = vol
    FovVar[:, :] = Fov
    FazVar[:, :] = Faz
    for j in range(nTransects):
        nLetters = len(transectList[j])
        transectNames[j, :nLetters] = transectList[j]
    ncid.close()
else:
    print(f'File {outfile} already exists. Plotting only...')

# Read in previously computed transport quantities
ncid = Dataset(outfile, mode='r')
t = ncid.variables['Time'][:]
vol = ncid.variables['vol'][:, :]
Fov = ncid.variables['Fov'][:, :]
Faz = ncid.variables['Faz'][:, :]
vTransect = ncid.variables['vTransect'][:, :]
sTransect = ncid.variables['sTransect'][:, :]
ncid.close()
vol_runavg = pd.Series.rolling(pd.DataFrame(vol), 12, center=True).mean()
Fov_runavg = pd.Series.rolling(pd.DataFrame(Fov), 12, center=True).mean()
Faz_runavg = pd.Series.rolling(pd.DataFrame(Faz), 12, center=True).mean()
vTransect_runavg = pd.Series.rolling(pd.DataFrame(vTransect), 12, center=True).mean()
sTransect_runavg = pd.Series.rolling(pd.DataFrame(sTransect), 12, center=True).mean()

# Define some dictionaries for transect plotting
FovObsDict = {'Atlantic zonal 34S':[-0.2, -0.1]}

figsize = (8, 20)
figdpi = 300
for j in range(nTransects):
    if platform.python_version()[0]=='3':
        searchString = transectList[j][2:]
    else:
        searchString = transectList[j]
    transectName_forfigfile = searchString.replace(" ", "")

    if searchString in FovObsDict:
        Fovbounds = FovObsDict[searchString]
    else:
        Fovbounds = None

    # Plot volume transport, Fov, and Faz
    if use_fixedSref:
        figfile = f'{figdir}/volFovFaz_{transectName_forfigfile}_{casename}_sref35.png'
    else:
        figfile = f'{figdir}/volFovFaz_{transectName_forfigfile}_{casename}.png'
    fig, ax = plt.subplots(5, 1, figsize=figsize)
    ax[0].plot(t, Fov[:, j], 'k', linewidth=2)
    ax[1].plot(t, Faz[:, j], 'k', linewidth=2)
    ax[2].plot(t, vol[:, j], 'k', linewidth=2)
    ax[3].plot(t, vTransect[:, j], 'k', linewidth=2)
    ax[4].plot(t, sTransect[:, j], 'k', linewidth=2)

    ax[0].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[1].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[2].plot(t, np.zeros_like(t), 'k', linewidth=1)

    ax[0].autoscale(enable=True, axis='x', tight=True)
    ax[1].autoscale(enable=True, axis='x', tight=True)
    ax[2].autoscale(enable=True, axis='x', tight=True)
    ax[3].autoscale(enable=True, axis='x', tight=True)
    ax[4].autoscale(enable=True, axis='x', tight=True)

    ax[0].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[1].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[2].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[3].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[4].grid(color='k', linestyle=':', linewidth = 0.5)

    ax[0].set_title(f'mean={np.nanmean(Fov[:, j]):5.2f} $\pm$ {np.nanstd(Fov[:, j]):5.2f})', \
                    fontsize=16, fontweight='bold')
    ax[1].set_title(f'mean={np.nanmean(Faz[:, j]):5.2f} $\pm$ {np.nanstd(Faz[:, j]):5.2f})', \
                    fontsize=16, fontweight='bold')
    ax[2].set_title(f'mean={np.nanmean(vol[:, j]):5.2f} $\pm$ {np.nanstd(vol[:, j]):5.2f})', \
                    fontsize=16, fontweight='bold')
    ax[3].set_title(f'mean={np.nanmean(vTransect[:, j]):.2e} $\pm$ {np.nanstd(vTransect[:, j]):.2e})', \
                    fontsize=16, fontweight='bold')
    ax[4].set_title(f'mean={np.nanmean(sTransect[:, j]):5.2f} $\pm$ {np.nanstd(sTransect[:, j]):5.2f})', \
                    fontsize=16, fontweight='bold')
    if Fovbounds is not None:
        ax[0].fill_between(t, np.full_like(t, Fovbounds[0]), np.full_like(t, Fovbounds[1]), alpha=0.3)

    ax[0].set_ylabel('Fov (Sv)', fontsize=12, fontweight='bold')
    ax[1].set_ylabel('Faz (Sv)', fontsize=12, fontweight='bold')
    ax[2].set_ylabel('Volume transport (Sv)', fontsize=12, fontweight='bold')
    ax[3].set_ylabel('Avg cross-transect velocity (m/s)', fontsize=12, fontweight='bold')
    ax[4].set_ylabel('Avg transect salinity (psu)', fontsize=12, fontweight='bold')
    ax[4].set_xlabel('Time (Years)', fontsize=12, fontweight='bold')

    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Transect = {searchString}\nrunname = {casename}', fontsize=14, fontweight='bold', y=1.025)
    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
