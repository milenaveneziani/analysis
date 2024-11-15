#!/usr/bin/env python
"""
Name: compute_FHovFHaz.py
Author: Milena Veneziani

Computes Fov, Hov (meridional freshwater and heat flux due to overturning circulation)
and Faz, Haz (meridional freshwater and heat flux due to the azonal or gyre circulation
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

from common_functions import add_inset
from geometric_features import FeatureCollection, read_feature_collection
from mpas_analysis.shared.io.utility import decode_strings
from mpas_analysis.shared.io import write_netcdf_with_fill


def get_mask_short_names(mask):
    # This is the right way to handle transects defined in the main transect file mask:
    shortnames = [str(aname.values)[:str(aname.values).find(',')].strip()
                  for aname in mask.transectNames]
    mask['shortNames'] = xr.DataArray(shortnames, dims='nTransects')
    mask = mask.set_index(nTransects=['transectNames', 'shortNames'])
    return mask


# Settings for erdc.hpc.mil
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
maskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20240910.nc'
featurefile = '/p/home/milena/mpas-region_masks/atlanticZonal_sections20240910.geojson'
outfile0 = 'atlanticZonalSectionsFHovFHaz'
#maskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_arcticSections20220916.nc'
#featurefile = '/p/home/milena/mpas-region_masks/arcticSections20210323.geojson'
#outfile0 = 'arcticSectionsFHovFHaz'
#casenameFull = 'E3SMv2.1B60to10rA02'
#casename = 'E3SMv2.1B60to10rA02'
#modeldir = f'/p/cwfs/milena/{casenameFull}/archive/ocn/hist'
casenameFull = 'E3SMv2.1B60to10rA07'
casename = 'E3SMv2.1B60to10rA07'
modeldir = f'/p/work/milena/{casenameFull}/archive/ocn/hist'

# Settings for anvil/chrysalis:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#maskfile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/EC30to60E2r2_atlanticZonal_sections20230307.nc'
#featurefile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/atlanticZonal_sections20230307.geojson'
#outfile0 = 'atlanticZonalSectionsFHovFHaz'
#casenameFull = 'v2_1.LR.piControl'
#casename = 'v2_1.LR.piControl'
#modeldir = f'/lcrc/group/e3sm/ac.golaz/E3SMv2_1/{casenameFull}/archive/ocn/hist'
#
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/RRSwISC6to18E3r5/mpaso.RRSwISC6to18E3r5.20240327.nc'
#maskfile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/RRSwISC6to18E3r5_atlanticZonal_sections20230307.nc'
#featurefile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/atlanticZonal_sections20230307.geojson'
#outfile0 = 'atlanticZonalSectionsFHovFHaz'
#casenameFull = '20240729.HRr5-test12.chrysalis'
#casename = '20240729.HRr5-test12'
#modeldir = f'/lcrc/group/e3sm2/ac.jwolfe/E3SMv3_dev/{casenameFull}/archive/ocn/hist'

# Settings for nersc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/EC30to60E2r2_atlanticZonal_sections20230307.nc'
#featurefile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/atlanticZonal_sections20230307.geojson'
#outfile0 = 'atlanticZonalSectionsFHovFHaz'
#casenameFull = '20220715.submeso.piControl.ne30pg2_EC30to60E2r2.chrysalis'
#casename = '20220715.submeso.piControl.ne30pg2_EC30to60E2r2'
#modeldir = f'/global/cfs/cdirs/m4259/E3SMv2_1/{casenameFull}/ocn/hist'
#
#meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20230307.nc'
#featurefile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/atlanticZonal_sections20230307.geojson'
#outfile0 = 'atlanticZonalSectionsFHovFHaz'
#casenameFull = 'E3SMv2.1B60to10rA02'
#casename = 'E3SMv2.1B60to10rA02'
#modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{casenameFull}/ocn/hist'
#
#meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/SOwISC12to60E2r4/mpaso.SOwISC12to60E2r4.rstFromG-anvil.210203.nc'
#maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/SOwISC12to60E2r4_atlanticZonal_sections20230307.nc'
#featurefile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/atlanticZonal_sections20230307.geojson'
#outfile0 = 'atlanticZonalSectionsFHovFHaz'
#casenameFull = '20221116.CRYO1950.ne30pg2_SOwISC12to60E2r4.N2Dependent.submeso'
#casename = 'SORRMv2.1.1950control'
#modeldir = f'/pscratch/sd/a/abarthel/data/E3SMv2.1/{casenameFull}/archive/ocn/hist'
#casenameFull = f'{casenameFull}.chrysalis'
#

# Choose years
year1 = 1
#year2 = 386 # rA02
year2 = 246 # rA07
years = range(year1, year2+1)

sZero = 34.8
tZero = 0.0
rhoRef = 1027.0 # kg/m^3
cp = 3.987*1e3 # J/(kg*degK)
m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
W_to_TW = 1e-12

use_fixedSref = True
use_fixedTref = True

if use_fixedSref:
    Fdescription = f'{sZero:4.1f}'
    if use_fixedTref:
        outfile0 = f'{outfile0}_{casename}_sref{sZero:4.1f}_tref{tZero:1.0f}'
    else:
        outfile0 = f'{outfile0}_{casename}_sref{sZero:4.1f}'
else:
    Fdescription = 'sTransect'
    if use_fixedTref:
        outfile0 = f'{outfile0}_{casename}_tref{tZero:1.0f}'
    else:
        outfile0 = f'{outfile0}_{casename}'
if use_fixedTref:
    Hdescription = f'{tZero:4.1f}'
else:
    Hdescription = 'tTransect'

figdir = f'./FHovFHaz/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)
outdir = f'./FHovFHaz_data/{casename}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found')
######################################################

# Read in transect information
dsMask = get_mask_short_names(xr.open_dataset(maskfile))
transectNames = decode_strings(dsMask.transectNames)
transectList = dsMask.shortNames.values
nTransects = len(transectList)
maxEdges = dsMask.dims['maxEdgesInTransect']
print(f'\nComputing/plotting time series for these transects: {transectNames}\n')

# Create a list of edges and total edges in each transect
nEdgesInTransect = np.zeros(nTransects)
edgeVals = np.zeros((nTransects, maxEdges))
for i in range(nTransects):
    amask = dsMask.sel(shortNames=transectList[i]).squeeze()
    transectEdges = amask.transectEdgeGlobalIDs.values
    inds = np.where(transectEdges > 0)[0]
    nEdgesInTransect[i] = len(inds)
    transectEdges = transectEdges[inds]
    edgeVals[i, :len(inds)] = np.asarray(transectEdges-1, dtype='i')
nEdgesInTransect = np.asarray(nEdgesInTransect, dtype='i')
edgesToRead = edgeVals[0, :nEdgesInTransect[0]]
for i in range(1, nTransects):
    edgesToRead = np.hstack([edgesToRead, edgeVals[i, :nEdgesInTransect[i]]])
edgesToRead = np.asarray(edgesToRead, dtype='i')

# Create a list with the start and stop for transect bounds
nTransectStartStop = np.zeros(nTransects+1)
for i in range(1, nTransects+1):
    nTransectStartStop[i] = nTransectStartStop[i-1] + nEdgesInTransect[i-1]

# Read in relevant mesh information
dsMesh = xr.open_dataset(meshfile)
dvEdge = dsMesh.dvEdge.sel(nEdges=edgesToRead)
coe0 = dsMesh.cellsOnEdge.isel(TWO=0, nEdges=edgesToRead)
coe1 = dsMesh.cellsOnEdge.isel(TWO=1, nEdges=edgesToRead)
# Build land-sea mask and topomask for the two cells surrounding each edge of the transect 
landmask1 = ~(coe0==0)
landmask2 = ~(coe1==0)
# convert to python indexing
coe0 = coe0 - 1
coe1 = coe1 - 1
#
maxLevelCell = dsMesh.maxLevelCell
nVertLevels = dsMesh.sizes['nVertLevels']
vertIndex = xr.DataArray(data=np.arange(nVertLevels), dims=('nVertLevels',))
#vertIndex = xr.DataArray.from_dict({'dims': ('nVertLevels',), 'data': np.arange(nVertLevels)})
depthmask = (vertIndex < maxLevelCell).transpose('nCells', 'nVertLevels')
depthmask1 = depthmask.isel(nCells=coe0)
depthmask2 = depthmask.isel(nCells=coe1)
#
edgeSigns = np.zeros((nTransects, len(edgesToRead)))
for i in range(nTransects):
    edgeSigns[i, :] = dsMask.sel(nEdges=edgesToRead, shortNames=transectList[i]).squeeze().transectEdgeMaskSigns.values
    # WARNING: The following is a quick hack valid only for the arcticSections mask file!
    # I will need to change the geojson files to make *all* transects go from south to north
    # or west to east, so that I can have the correct edgeSigns for all of them.
    if transectNames[i]!='Bering Strait' and transectNames[i]!='Hudson Bay-Labrador Sea':
        edgeSigns[i, :] = -edgeSigns[i, :]
edgeSigns = xr.DataArray(data=edgeSigns, dims=('nTransect', 'nEdges'))
refBottom = dsMesh.refBottomDepth
#latmean = 180.0/np.pi * dsMesh.latEdge.sel(nEdges=edgesToRead).mean()
#lonmean = 180.0/np.pi * dsMesh.lonEdge.sel(nEdges=edgesToRead).mean()
#pressure = gsw.p_from_z(-refBottom, latmean)

kyear = 0
for year in years:
    kyear = kyear + 1
    print(f'Year = {year:04d} ({kyear} out of {len(years)} years total)')

    outfile = f'{outdir}/{outfile0}_year{year:04d}.nc'
    # Compute transports if outfile does not exist
    if not os.path.exists(outfile):
        dsOut = []
        for month in range(1, 13):
            print(f'  Month= {month:02d}')
            modelfile = f'{modeldir}/{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

            dsIn = xr.open_dataset(modelfile, decode_times=False)
            dsOutMonthly = xr.Dataset()

            if 'timeMonthly_avg_normalTransportVelocity' in dsIn.keys():
                vel = dsIn.timeMonthly_avg_normalTransportVelocity.isel(Time=0, nEdges=edgesToRead)
            elif 'timeMonthly_avg_normalVelocity' in dsIn.keys():
                vel = dsIn.timeMonthly_avg_normalVelocity.isel(Time=0, nEdges=edgesToRead)
                if 'timeMonthly_avg_normalGMBolusVelocity' in dsIn.keys():
                    vel = vel + dsIn.timeMonthly_avg_normalGMBolusVelocity.isel(Time=0, nEdges=edgesToRead)
                if 'timeMonthly_avg_normalMLEvelocity' in dsIn.keys():
                    vel = vel + dsIn.timeMonthly_avg_normalMLEvelocity.isel(Time=0, nEdges=edgesToRead)
            else:
                raise KeyError('no appropriate normalVelocity variable found')
            # Note that the following is incorrect when coe is zero (cell straddling the
            # transect edge is on land), but that is OK because the value will be masked
            # during land-sea masking below
            tempOnCells1 = dsIn.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe0)
            tempOnCells2 = dsIn.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe1)
            saltOnCells1 = dsIn.timeMonthly_avg_activeTracers_salinity.isel(Time=0, nCells=coe0)
            saltOnCells2 = dsIn.timeMonthly_avg_activeTracers_salinity.isel(Time=0, nCells=coe1)

            # Mask values that fall on land
            tempOnCells1 = tempOnCells1.where(landmask1, drop=False)
            tempOnCells2 = tempOnCells2.where(landmask2, drop=False)
            saltOnCells1 = saltOnCells1.where(landmask1, drop=False)
            saltOnCells2 = saltOnCells2.where(landmask2, drop=False)
            # Mask values that fall onto topography
            tempOnCells1 = tempOnCells1.where(depthmask1, drop=False)
            tempOnCells2 = tempOnCells2.where(depthmask2, drop=False)
            saltOnCells1 = saltOnCells1.where(depthmask1, drop=False)
            saltOnCells2 = saltOnCells2.where(depthmask2, drop=False)
            # The following should *not* happen at this point:
            if np.any(tempOnCells1.values[np.logical_or(tempOnCells1.values> 1e15, tempOnCells1.values<-1e15)]) or \
               np.any(tempOnCells2.values[np.logical_or(tempOnCells2.values> 1e15, tempOnCells2.values<-1e15)]):
                print('WARNING: something is wrong with land and/or topography masking!')
            if np.any(saltOnCells1.values[np.logical_or(saltOnCells1.values> 1e15, saltOnCells1.values<-1e15)]) or \
               np.any(saltOnCells2.values[np.logical_or(saltOnCells2.values> 1e15, saltOnCells2.values<-1e15)]):
                print('WARNING: something is wrong with land and/or topography masking!')

            dzOnCells1 = dsIn.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe0)
            dzOnCells2 = dsIn.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe1)
            dzOnCells1 = dzOnCells1.where(landmask1, drop=False)
            dzOnCells2 = dzOnCells2.where(landmask2, drop=False)
            dzOnCells1 = dzOnCells1.where(depthmask1, drop=False)
            dzOnCells2 = dzOnCells2.where(depthmask2, drop=False)

            # Interpolate values onto edges
            tempOnEdges = np.nanmean(np.array([tempOnCells1.values, tempOnCells2.values]), axis=0)
            saltOnEdges = np.nanmean(np.array([saltOnCells1.values, saltOnCells2.values]), axis=0)
            dzOnEdges = np.nanmean(np.array([dzOnCells1.values, dzOnCells2.values]), axis=0)
            # The following doesn't do a proper nansum.. (and couldn't find anything online
            # about nansumming two *separate* xarray datasets):
            #tempOnEdges = 0.5 * (tempOnCells1 + tempOnCells2)
            #saltOnEdges = 0.5 * (saltOnCells1 + saltOnCells2)
            #dzOnEdges = 0.5 * (dzOnCells1 + dzOnCells2)
            tempOnEdges = xr.DataArray(data=tempOnEdges, dims=('nEdges', 'nVertLevels'), name='tempOnEdges')
            saltOnEdges = xr.DataArray(data=saltOnEdges, dims=('nEdges', 'nVertLevels'), name='saltOnEdges')
            dzOnEdges = xr.DataArray(data=dzOnEdges, dims=('nEdges', 'nVertLevels'), name='dzOnEdges')

            # Compute volume transport, Fov, Hov, Faz, and Haz for each transect
            #  Initialize to nan's (note that np.empty does *not* work properly)
            vTransect = np.nan*np.ones(nTransects)
            sTransect = np.nan*np.ones(nTransects)
            tTransect = np.nan*np.ones(nTransects)
            vol = np.nan*np.ones(nTransects)
            Fov = np.nan*np.ones(nTransects)
            Faz = np.nan*np.ones(nTransects)
            Ftot = np.nan*np.ones(nTransects)
            Hov = np.nan*np.ones(nTransects)
            Haz = np.nan*np.ones(nTransects)
            Htot = np.nan*np.ones(nTransects)
            for i in range(nTransects):
                start = int(nTransectStartStop[i])
                stop = int(nTransectStartStop[i+1])

                normalVel = vel.isel(nEdges=range(start, stop)) * edgeSigns.isel(nTransect=i, nEdges=range(start, stop))
                temp = tempOnEdges.isel(nEdges=range(start, stop))
                salt = saltOnEdges.isel(nEdges=range(start, stop))
                maskOnEdges = salt.notnull()
                normalVel = normalVel.where(maskOnEdges, drop=False)

                dx = dvEdge.isel(nEdges=range(start, stop)).expand_dims({'nVertLevels':nVertLevels}, axis=1)
                dx = dx.where(maskOnEdges, drop=False)
                dz = dzOnEdges.isel(nEdges=range(start, stop))
                dArea = dx * dz
                transectArea = dArea.sum(dim='nEdges').sum(dim='nVertLevels')
                transectLength = dx.sum(dim='nEdges')

                # Each variable is decomposed into a transect averaged part, a zonal averaged
                # part, and an azonal residual. For example, for velocity:
                # v = v_transect + v_zonalAvg + v_azonal
                vTransect[i] = (normalVel * dArea).sum(dim='nEdges').sum(dim='nVertLevels') / transectArea
                tTransect[i] = (temp * dArea).sum(dim='nEdges').sum(dim='nVertLevels') / transectArea
                sTransect[i] = (salt * dArea).sum(dim='nEdges').sum(dim='nVertLevels') / transectArea
                vZonalAvg = (normalVel * dx).sum(dim='nEdges') / transectLength
                tZonalAvg = (temp * dx).sum(dim='nEdges') / transectLength
                sZonalAvg = (salt * dx).sum(dim='nEdges') / transectLength
                vAzonal = normalVel - vZonalAvg
                tAzonal = temp - tZonalAvg
                sAzonal = salt - sZonalAvg

                if use_fixedSref:
                    sRef = sZero
                else:
                    sRef = sTransect[i]
                if use_fixedTref:
                    tRef = tZero
                else:
                    tRef = tTransect[i]

                vol[i] = (normalVel * dArea).sum(dim='nEdges').sum(dim='nVertLevels')
                # From Eq. 2,3 in Mecking et al. 2017 (note that in Eq. 2 sRef is missing):
                Fov[i]  = - ( (vZonalAvg - vTransect[i]) * (sZonalAvg-sRef) * transectLength * dz.mean(dim='nEdges') ).sum(dim='nVertLevels') / sRef
                Faz[i]  = - ( vAzonal * sAzonal * dArea ).sum(dim='nEdges').sum(dim='nVertLevels') / sRef
                Ftot[i] = - ( normalVel * (salt-sRef) * dArea ).sum(dim='nEdges').sum(dim='nVertLevels') / sRef
                Hov[i]  = ( (vZonalAvg - vTransect[i]) * (tZonalAvg-tRef) * transectLength * dz.mean(dim='nEdges') ).sum(dim='nVertLevels')
                Haz[i]  = ( vAzonal * tAzonal * dArea ).sum(dim='nEdges').sum(dim='nVertLevels')
                Htot[i] = ( normalVel * (temp-tRef) * dArea ).sum(dim='nEdges').sum(dim='nVertLevels')

            dsOutMonthly['vTransect'] = xr.DataArray(
                    data=vTransect,
                    dims=('nTransects', ),
                    attrs=dict(description='Cross-transect averaged velocity', units='m/s', )
                    )
            dsOutMonthly['tTransect'] = xr.DataArray(
                    data=tTransect,
                    dims=('nTransects', ),
                    attrs=dict(description='Transect averaged temperature', units='degree C', )
                    )
            dsOutMonthly['sTransect'] = xr.DataArray(
                    data=sTransect,
                    dims=('nTransects', ),
                    attrs=dict(description='Transect averaged salinity', units='psu', )
                    )
            dsOutMonthly['vol'] = xr.DataArray(
                    data=m3ps_to_Sv * vol,
                    dims=('nTransects', ),
                    attrs=dict(description='Net volume transport across transect', units='Sv', )
                    )
            dsOutMonthly['Fov'] = xr.DataArray(
                    data=m3ps_to_Sv * Fov,
                    dims=('nTransects', ),
                    attrs=dict(description=f'Meridional Freshwater transport due to overturning circulation (Sref={Fdescription})', units='Sv', )
                    )
            dsOutMonthly['Faz'] = xr.DataArray(
                    data=m3ps_to_Sv * Faz,
                    dims=('nTransects', ),
                    attrs=dict(description=f'Meridional Freshwater transport due to azonal (gyre) circulation (Sref={Fdescription})', units='Sv', )
                    )
            dsOutMonthly['Ftot'] = xr.DataArray(
                    data=m3ps_to_Sv * Ftot,
                    dims=('nTransects', ),
                    attrs=dict(description=f'Net Meridional Freshwater transport (Sref={Fdescription})', units='Sv', )
                    )
            dsOutMonthly['Hov'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * Hov,
                    dims=('nTransects', ),
                    attrs=dict(description=f'Meridional Heat transport due to overturning circulation (Tref={Hdescription})', units='TW', )
                    )
            dsOutMonthly['Haz'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * Haz,
                    dims=('nTransects', ),
                    attrs=dict(description=f'Meridional Heat transport due to azonal (gyre) circulation (Tref={Hdescription})', units='TW', )
                    )
            dsOutMonthly['Htot'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * Htot,
                    dims=('nTransects', ),
                    attrs=dict(description=f'Net Meridional Heat transport (Tref={Hdescription})', units='TW', )
                    )
            dsOutMonthly['transectNames'] = xr.DataArray(data=transectNames, dims=('nTransects', ))
            dsOutMonthly['Time'] = xr.DataArray(
                    data=[dsIn.timeMonthly_avg_daysSinceStartOfSim.isel(Time=0)/365.],
                    dims=('Time', ),
                    attrs=dict(description='days since start of simulation (assumes 365-day year)',
                               units='days', )
                    )
            dsOut.append(dsOutMonthly)

        dsOut = xr.concat(dsOut, dim='Time')
        write_netcdf_with_fill(dsOut, outfile)
    else:
        print(f'  Outfile for year {year} already exists. Proceed...')

print(f'\nPlotting...')
# Read in previously computed transport quantities
infiles = []
for year in years:
    infiles.append(f'{outdir}/{outfile0}_year{year:04d}.nc')
dsIn = xr.open_mfdataset(infiles, decode_times=False)
t = dsIn['Time'].values
vol = dsIn['vol'].values
Fov = dsIn['Fov'].values
Faz = dsIn['Faz'].values
Ftot = dsIn['Ftot'].values
Hov = dsIn['Hov'].values
Haz = dsIn['Haz'].values
Htot = dsIn['Htot'].values
vTransect = dsIn['vTransect'].values
sTransect = dsIn['sTransect'].values
tTransect = dsIn['tTransect'].values

# Define some dictionaries for transect plotting
FovObsDict = {'Atlantic zonal 34S':[-0.2, -0.1]}
labelDict = {'Drake Passage':'drake', 'Tasmania-Ant':'tasmania', 'Africa-Ant':'africaAnt', 'Antilles Inflow':'antilles', \
             'Mona Passage':'monaPassage', 'Windward Passage':'windwardPassage', 'Florida-Cuba':'floridaCuba', \
             'Florida-Bahamas':'floridaBahamas', 'Indonesian Throughflow':'indonesia', 'Agulhas':'agulhas', \
             'Mozambique Channel':'mozambique', 'Bering Strait':'beringStrait', 'Lancaster Sound':'lancasterSound', \
             'Fram Strait':'framStrait', 'Robeson Channel':'robeson', 'Davis Strait':'davisStrait', 'Barents Sea Opening':'barentsSea', \
             'Nares Strait':'naresStrait', 'Denmark Strait':'denmarkStrait', 'Iceland-Faroe-Scotland':'icelandFaroeScotland'}

figsize = (16, 16)
figdpi = 300
for i in range(nTransects):
    transectName = transectNames[i]

    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == transectName:
            fc.add_feature(feature)

    if transectName in labelDict:
        transectName_forfigfile = labelDict[transectName]
    else:
        transectName_forfigfile = transectName.replace(" ", "")

    if transectName in FovObsDict:
        Fovbounds = FovObsDict[transectName]
    else:
        Fovbounds = None

    vol_runavg = pd.Series.rolling(pd.DataFrame(vol[:, i]), 12, center=True).mean()
    Fov_runavg = pd.Series.rolling(pd.DataFrame(Fov[:, i]), 12, center=True).mean()
    Faz_runavg = pd.Series.rolling(pd.DataFrame(Faz[:, i]), 12, center=True).mean()
    Ftot_runavg = pd.Series.rolling(pd.DataFrame(Ftot[:, i]), 12, center=True).mean()
    Hov_runavg = pd.Series.rolling(pd.DataFrame(Hov[:, i]), 12, center=True).mean()
    Haz_runavg = pd.Series.rolling(pd.DataFrame(Haz[:, i]), 12, center=True).mean()
    Htot_runavg = pd.Series.rolling(pd.DataFrame(Htot[:, i]), 12, center=True).mean()
    vTransect_runavg = pd.Series.rolling(pd.DataFrame(vTransect[:, i]), 12, center=True).mean()
    tTransect_runavg = pd.Series.rolling(pd.DataFrame(tTransect[:, i]), 12, center=True).mean()
    sTransect_runavg = pd.Series.rolling(pd.DataFrame(sTransect[:, i]), 12, center=True).mean()

    figfile = f'{figdir}/{outfile0}_{transectName_forfigfile}_years{year1:04d}-{year2:04d}.png'
    fig = plt.figure(figsize=figsize)
    # Plot volume transport
    ax1 = plt.subplot(321)
    ax1.plot(t, vol[:, i], 'k', alpha=0.5, linewidth=1.5)
    ax1.plot(t, vol_runavg, 'k', linewidth=3)
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.grid(color='k', linestyle=':', linewidth = 0.5)
    ax1.set_title(f'mean={np.nanmean(vol[:, i]):5.2f} $\pm$ {np.nanstd(vol[:, i]):5.2f}', \
                    fontsize=16, fontweight='bold')
    ax1.set_ylabel('Volume transport (Sv)', fontsize=12, fontweight='bold')

    # Plot cross-transect velocity
    ax2 = plt.subplot(322)
    ax2.plot(t, 100*vTransect[:, i], 'k', alpha=0.5, linewidth=1.5)
    ax2.plot(t, 100*vTransect_runavg, 'k', linewidth=3)
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.grid(color='k', linestyle=':', linewidth = 0.5)
    #ax2.set_title(f'mean={np.nanmean(vTransect[:, i]):.2e} $\pm$ {np.nanstd(vTransect[:, i]):.2e}', \
    ax2.set_title(f'mean={np.nanmean(100*vTransect[:, i]):5.2f} $\pm$ {np.nanstd(100*vTransect[:, i]):5.2f}', \
                    fontsize=16, fontweight='bold')
    ax2.set_ylabel('Avg cross-transect velocity (cm/s)', fontsize=12, fontweight='bold')

    # Plot transect temperature
    ax3 = plt.subplot(323)
    ax3.plot(t, tTransect[:, i], 'k', alpha=0.5, linewidth=1.5)
    ax3.plot(t, tTransect_runavg, 'k', linewidth=3)
    ax3.autoscale(enable=True, axis='x', tight=True)
    ax3.grid(color='k', linestyle=':', linewidth = 0.5)
    ax3.set_title(f'mean={np.nanmean(tTransect[:, i]):5.2f} $\pm$ {np.nanstd(tTransect[:, i]):5.2f}', \
                    fontsize=16, fontweight='bold')
    ax3.set_ylabel('Avg transect temperature (C)', fontsize=12, fontweight='bold')

    # Plot transect salinity
    ax4 = plt.subplot(324)
    ax4.plot(t, sTransect[:, i], 'k', alpha=0.5, linewidth=1.5)
    ax4.plot(t, sTransect_runavg, 'k', linewidth=3)
    ax4.autoscale(enable=True, axis='x', tight=True)
    ax4.grid(color='k', linestyle=':', linewidth = 0.5)
    ax4.set_title(f'mean={np.nanmean(sTransect[:, i]):5.2f} $\pm$ {np.nanstd(sTransect[:, i]):5.2f}', \
                    fontsize=16, fontweight='bold')
    ax4.set_ylabel('Avg transect salinity (psu)', fontsize=12, fontweight='bold')

    # Plot Fov, Faz, and Ftot
    ax5 = plt.subplot(325)
    ax5.plot(t, Ftot[:, i], 'k', alpha=0.5, linewidth=1.2)
    ax5.plot(t, Ftot_runavg, 'k', linewidth=2.5, label=f'Ftot ({np.nanmean(Ftot[:, i]):5.2f} $\pm$ {np.nanstd(Ftot[:, i]):5.2f})')
    ax5.plot(t, Fov[:, i], 'firebrick', alpha=0.5, linewidth=1.2)
    ax5.plot(t, Fov_runavg, 'firebrick', linewidth=2.5, label=f'Fov ({np.nanmean(Fov[:, i]):5.2f} $\pm$ {np.nanstd(Fov[:, i]):5.2f})')
    ax5.plot(t, Faz[:, i], 'dodgerblue', alpha=0.5, linewidth=1.2)
    ax5.plot(t, Faz_runavg, 'dodgerblue', linewidth=2.5, label=f'Faz ({np.nanmean(Faz[:, i]):5.2f} $\pm$ {np.nanstd(Faz[:, i]):5.2f})')
    ax5.plot(t, np.zeros_like(t), 'k', linewidth=0.8)
    ax5.autoscale(enable=True, axis='x', tight=True)
    ax5.grid(color='k', linestyle=':', linewidth = 0.5)
    #if Fovbounds is not None:
    #    ax5.fill_between(t, np.full_like(t, Fovbounds[0]), np.full_like(t, Fovbounds[1]), alpha=0.3)
    ax5.set_ylabel(f'FW transports (Sref={Fdescription}, Sv)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax5.legend(prop={'size':16, 'weight':'bold'})

    ax6 = plt.subplot(326)
    ax6.plot(t, Htot[:, i], 'k', alpha=0.5, linewidth=1.2)
    ax6.plot(t, Htot_runavg, 'k', linewidth=2.5, label=f'Htot ({np.nanmean(Htot[:, i]):5.2f} $\pm$ {np.nanstd(Htot[:, i]):5.2f})')
    ax6.plot(t, Hov[:, i], 'firebrick', alpha=0.5, linewidth=1.2)
    ax6.plot(t, Hov_runavg, 'firebrick', linewidth=2.5, label=f'Hov ({np.nanmean(Hov[:, i]):5.2f} $\pm$ {np.nanstd(Hov[:, i]):5.2f})')
    ax6.plot(t, Haz[:, i], 'dodgerblue', alpha=0.5, linewidth=1.2)
    ax6.plot(t, Haz_runavg, 'dodgerblue', linewidth=2.5, label=f'Haz ({np.nanmean(Haz[:, i]):5.2f} $\pm$ {np.nanstd(Haz[:, i]):5.2f})')
    ax6.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax6.autoscale(enable=True, axis='x', tight=True)
    ax6.grid(color='k', linestyle=':', linewidth = 0.5)
    ax6.set_ylabel(f'Heat transports (Tref={Hdescription}, TW)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax6.legend(prop={'size':16, 'weight':'bold'})

    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Transect = {transectName}\nrunname = {casename}', fontsize=14, fontweight='bold', y=1.045)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=-0.5, ybuffer=-1.65)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
