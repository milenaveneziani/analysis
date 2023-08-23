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
year1 = 500
year2 = 599
years = range(year1, year2+1)

# Settings for onyx:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#maskfile = '/p/work/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20230307.nc'
#casenameFull = 'E3SMv2.1B60to10rA02'
#casename = 'E3SMv2.1B60to10rA02'
#modeldir = f'/p/work/osinski/archive/{casenameFull}/ocn/hist'
#featurefile = '/p/work/milena/mpas-region_masks/atlanticZonal_sections20230307.geojson'

# Settings for anvil/chrysalis:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#maskfile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/EC30to60E2r2_atlanticZonal_sections20230307.nc'
#casenameFull = 'v2_1.LR.piControl'
#casename = 'v2_1.LR.piControl'
#modeldir = f'/lcrc/group/e3sm/ac.golaz/E3SMv2_1/{casenameFull}/archive/ocn/hist'
#featurefile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/atlanticZonal_sections20230307.geojson'

# Settings for nersc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/EC30to60E2r2_atlanticZonal_sections20230307.nc'
#casenameFull = '20220715.submeso.piControl.ne30pg2_EC30to60E2r2.chrysalis'
#casename = '20220715.submeso.piControl.ne30pg2_EC30to60E2r2'
#modeldir = f'/global/cfs/cdirs/m4259/E3SMv2_1/{casenameFull}/ocn/hist'
#featurefile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/atlanticZonal_sections20230307.geojson'
#
#meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20230307.nc'
#casenameFull = 'E3SMv2.1B60to10rA02'
#casename = 'E3SMv2.1B60to10rA02'
#modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{casenameFull}/ocn/hist'
#featurefile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/atlanticZonal_sections20230307.geojson'
#
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/SOwISC12to60E2r4/mpaso.SOwISC12to60E2r4.rstFromG-anvil.210203.nc'
maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/SOwISC12to60E2r4_atlanticZonal_sections20230307.nc'
casenameFull = '20221116.CRYO1950.ne30pg2_SOwISC12to60E2r4.N2Dependent.submeso'
casename = 'SORRMv2.1.1950control'
modeldir = f'/pscratch/sd/a/abarthel/data/E3SMv2.1/{casenameFull}/archive/ocn/hist'
casenameFull = f'{casenameFull}.chrysalis'
featurefile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/atlanticZonal_sections20230307.geojson'
#

sZero = 35.0
m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups

use_fixedSref = False
use_fixeddz = False

if use_fixedSref:
    outfile = f'volFovFaz_{casename}_sref35_years{year1:04d}-{year2:04d}.nc'
else:
    outfile = f'volFovFaz_{casename}_years{year1:04d}-{year2:04d}.nc'

transectName = 'all'
# Last time I tried it, this wasn't working:
#transectName = 'Atlantic zonal 34S, Atlantic zonal 27.2N, Atlantic zonal OSNAP East'
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
print(f'Computing/plotting time series for these transects: {transectList}\n')

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
edgeSigns = np.zeros((nTransects, len(edgesToRead)))
for j in range(nTransects):
    edgeSigns[j, :] = mask.sel(nEdges=edgesToRead, shortNames=transectList[j]).squeeze().transectEdgeMaskSigns.values
nLevels = mesh.dims['nVertLevels']
if use_fixeddz:
    refBottom = mesh.refBottomDepth.values
    dz = np.zeros(nLevels)
    dz[0] = refBottom[0]
    for k in range(1, nLevels):
        dz[k] = refBottom[k] - refBottom[k-1]

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
            saltOnCells1 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge[:, 0]-1, :]
            saltOnCells2 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge[:, 1]-1, :]
            if not use_fixeddz:
                dzOnCells1 = ncid.variables['timeMonthly_avg_layerThickness'][0, cellsOnEdge[:, 0]-1, :]
                dzOnCells2 = ncid.variables['timeMonthly_avg_layerThickness'][0, cellsOnEdge[:, 1]-1, :]
            t[i] = ncid.variables['timeMonthly_avg_daysSinceStartOfSim'][:]/365.
            ncid.close()

            # Mask values that fall on land
            landmask1 = cellsOnEdge[:, 0]==0
            landmask2 = cellsOnEdge[:, 1]==0
            saltOnCells1[landmask1, :] = np.nan
            saltOnCells2[landmask2, :] = np.nan
            # Mask values that fall onto topography
            for k in range(len(kmaxOnCells1)):
                saltOnCells1[k, kmaxOnCells1[k]:] = np.nan
            for k in range(len(kmaxOnCells2)):
                saltOnCells2[k, kmaxOnCells2[k]:] = np.nan
            if np.any(saltOnCells1[np.logical_or(saltOnCells1> 1e15, saltOnCells1<-1e15)]) or \
               np.any(saltOnCells2[np.logical_or(saltOnCells2> 1e15, saltOnCells2<-1e15)]):
                print('WARNING: something is wrong with land and/or topography masking!')
            if not use_fixeddz:
                dzOnCells1[landmask1, :] = np.nan
                dzOnCells2[landmask2, :] = np.nan
                for k in range(len(kmaxOnCells1)):
                    dzOnCells1[k, kmaxOnCells1[k]:] = np.nan
                for k in range(len(kmaxOnCells2)):
                    dzOnCells2[k, kmaxOnCells2[k]:] = np.nan

            # Interpolate values onto edges
            saltOnEdges = np.nanmean(np.array([saltOnCells1, saltOnCells2]), axis=0)
            if not use_fixeddz:
                dzOnEdges = np.nanmean(np.array([dzOnCells1, dzOnCells2]), axis=0)

            # Compute volume transport, Fov, and Faz for each transect
            for j in range(nTransects):
                start = int(nTransectStartStop[j])
                stop = int(nTransectStartStop[j+1])
                dx = dvEdge[start:stop]
                dx2d = np.transpose(np.tile(dx, (nLevels, 1)))
                if not use_fixeddz:
                    dz = dzOnEdges[start:stop, :]

                normalVel = vel[start:stop, :] * edgeSigns[j, start:stop, np.newaxis]
                salt = saltOnEdges[start:stop, :]
                maskOnEdges = np.isnan(salt)
                normalVel[maskOnEdges] = np.nan
                dx2d[maskOnEdges] = np.nan
                if use_fixeddz:
                    dArea = dx2d * dz[np.newaxis, :]
                else:
                    dArea = dx2d * dz
                transectLength = np.nansum(dx2d, axis=0)
                transectArea = np.nansum(np.nansum(dArea)) 

                # Each variable is decomposed into a transect averaged part, a zonal averaged
                # part, and an azonal residual. For example, for velocity:
                # v = v_transect + v_zonalAvg + v_azonal
                vTransect[i, j] = np.nansum(np.nansum(normalVel * dArea)) / transectArea
                sTransect[i, j] = np.nansum(np.nansum(salt * dArea)) / transectArea
                if use_fixedSref:
                    sRef = sZero
                else:
                    sRef = sTransect[i, j]
                vZonalAvg = np.nansum(normalVel * dx2d, axis=0) / transectLength
                sZonalAvg = np.nansum(salt * dx2d, axis=0) / transectLength
                vAzonal = normalVel - vZonalAvg[np.newaxis, :]
                sAzonal = salt - sZonalAvg[np.newaxis, :]

                vol[i, j] =   np.nansum(np.nansum( normalVel * dArea ))
                # From Eq. 2,3 in Mecking et al. 2017 (note that in Eq. 2 sRef is missing):
                if use_fixeddz:
                    Fov[i, j] = - np.nansum( (vZonalAvg - vTransect[i, j]) * (sZonalAvg-sRef) * transectLength * dz ) / sRef
                else:
                    Fov[i, j] = - np.nansum( (vZonalAvg - vTransect[i, j]) * (sZonalAvg-sRef) * transectLength * np.nanmean(dz, axis=0) ) / sRef
                Faz[i, j] = - np.nansum(np.nansum( vAzonal * sAzonal * dArea )) / sRef

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

# Define some dictionaries for transect plotting
FovObsDict = {'Atlantic zonal 34S':[-0.2, -0.1]}

if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found')

figsize = (8, 20)
figdpi = 300
for j in range(nTransects):
    if platform.python_version()[0]=='3':
        searchString = transectList[j][2:]
    else:
        searchString = transectList[j]
    transectName_forfigfile = searchString.replace(" ", "")

    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == searchString:
            fc.add_feature(feature)
            break

    if searchString in FovObsDict:
        Fovbounds = FovObsDict[searchString]
    else:
        Fovbounds = None

    vol_runavg = pd.Series.rolling(pd.DataFrame(vol[:, j]), 12, center=True).mean()
    Fov_runavg = pd.Series.rolling(pd.DataFrame(Fov[:, j]), 12, center=True).mean()
    Faz_runavg = pd.Series.rolling(pd.DataFrame(Faz[:, j]), 12, center=True).mean()
    vTransect_runavg = pd.Series.rolling(pd.DataFrame(vTransect[:, j]), 12, center=True).mean()
    sTransect_runavg = pd.Series.rolling(pd.DataFrame(sTransect[:, j]), 12, center=True).mean()

    # Plot volume transport, Fov, and Faz
    if use_fixedSref:
        figfile = f'{figdir}/volFovFaz_{transectName_forfigfile}_{casename}_sref35_years{year1:04d}-{year2:04d}.png'
    else:
        figfile = f'{figdir}/volFovFaz_{transectName_forfigfile}_{casename}_years{year1:04d}-{year2:04d}.png'
    fig, ax = plt.subplots(5, 1, figsize=figsize)
    ax[0].plot(t, Fov[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[1].plot(t, Faz[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[2].plot(t, vol[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[3].plot(t, vTransect[:, j], 'k', alpha=0.5, linewidth=1.5)
    ax[4].plot(t, sTransect[:, j], 'k', alpha=0.5, linewidth=1.5)

    ax[0].plot(t, Fov_runavg, 'k', linewidth=3)
    ax[1].plot(t, Faz_runavg, 'k', linewidth=3)
    ax[2].plot(t, vol_runavg, 'k', linewidth=3)
    ax[3].plot(t, vTransect_runavg, 'k', linewidth=3)
    ax[4].plot(t, sTransect_runavg, 'k', linewidth=3)

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
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
