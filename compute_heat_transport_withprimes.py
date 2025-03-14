#!/usr/bin/env python
"""
Name: compute_transects.py
Author: Phillip J. Wolfram, Mark Petersen, Luke Van Roekel, Milena Veneziani

Computes transport through sections.

"""

# ensure plots are rendered on ICC
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import gsw

from common_functions import add_inset
from geometric_features import FeatureCollection, read_feature_collection
from mpas_analysis.shared.io.utility import decode_strings
from mpas_analysis.shared.io import write_netcdf_with_fill


def get_mask_short_names(mask):
    # This is the right way to handle transects defined in the main transect file mask:
    shortnames = [str(aname.values)[:str(aname.values).find(',')].strip()
                  for aname in mask.transectNames]
    #shortnames = mask.transectNames.values
    mask['shortNames'] = xr.DataArray(shortnames, dims='nTransects')
    mask = mask.set_index(nTransects=['transectNames', 'shortNames'])
    return mask


# Settings for nersc
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20240910.nc'
#featurefile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/atlanticZonal_sections20240910.geojson'
#outfile0 = 'atlanticZonalSectionsTransports'
maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arcticSections20220916.nc'
featurefile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/arcticSections20210323.geojson'
outfile0 = 'arcticSectionsTransportsWithPrimes'
casenameFull = 'E3SMv2.1B60to10rA02'
casename = 'E3SMv2.1B60to10rA02'
modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{casenameFull}/archive/ocn/singleVarFiles'
climodir = '/pscratch/sd/w/wilbert/E3SM-Arctic/E3SMv2.1B60to10rA02'

# Choose years
year1 = 1
year2 = 386 # rA02
#year2 = 5
#year2 = 246 # rA07
years = range(year1, year2+1)

use_fixeddz = False

figdir = f'./transports/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)
outdir = f'./transports_data/{casename}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found')

saltRef = 34.8
rhoRef = 1027.0 # kg/m^3
cp = 3.987*1e3 # J/(kg*degK) 
m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
m3ps_to_mSv = 1e-3 # m^3/s flux to 10^-3 Sverdrups (mSv)
m3ps_to_km3py = 1e-9*86400*365.25  # m^3/s FW flux to km^3/year
W_to_TW = 1e-12
###################################################################

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
latmean = 180.0/np.pi * dsMesh.latEdge.sel(nEdges=edgesToRead).mean()
lonmean = 180.0/np.pi * dsMesh.lonEdge.sel(nEdges=edgesToRead).mean()
pressure = gsw.p_from_z(-refBottom, latmean)
if use_fixeddz:
    dz = xr.concat([refBottom.isel(nVertLevels=0), refBottom.diff('nVertLevels')], dim='nVertLevels')

# Read in long term mean
climofile = f'{climodir}/activeTracers_temperature/Climo/activeTracers_temperature.E3SMv2.1B60to10rA02.mpaso.hist.am.timeSeriesStatsMonthly.Climo.nc'
dsIn = xr.open_dataset(climofile, decode_times=False)
tmeanOnCells1 = dsIn.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe0)
tmeanOnCells2 = dsIn.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe1)
tmeanOnCells1 = tmeanOnCells1.where(landmask1, drop=False)
tmeanOnCells2 = tmeanOnCells2.where(landmask2, drop=False)
tmeanOnCells1 = tmeanOnCells1.where(depthmask1, drop=False)
tmeanOnCells2 = tmeanOnCells2.where(depthmask2, drop=False)
if np.any(tmeanOnCells1.values[np.logical_or(tmeanOnCells1.values> 1e15, tmeanOnCells1.values<-1e15)]) or \
   np.any(tmeanOnCells2.values[np.logical_or(tmeanOnCells2.values> 1e15, tmeanOnCells2.values<-1e15)]):
    print('WARNING: something is wrong with land and/or topography masking!')
tmean = np.nanmean(np.array([tmeanOnCells1.values, tmeanOnCells2.values]), axis=0)
tmean = xr.DataArray(data=tmean, dims=('nEdges', 'nVertLevels'), name='tmean')

climofile = f'{climodir}/normalVelocity/Climo/normalVelocity.E3SMv2.1B60to10rA02.mpaso.hist.am.timeSeriesStatsMonthly.Climo.nc'
dsIn = xr.open_dataset(climofile, decode_times=False)
velmean = dsIn.timeMonthly_avg_normalVelocity.isel(Time=0, nEdges=edgesToRead)

kyear = 0
for year in years:
    kyear = kyear + 1
    print(f'Year = {year:04d} ({kyear} out of {len(years)} years total)')

    outfile = f'{outdir}/{outfile0}_{casename}_year{year:04d}.nc'
    # Compute transports if outfile does not exist
    if not os.path.exists(outfile):
        dsOut = []
        for month in range(1, 13):
            print(f'   Month= {month:02d}')
            tempfile = f'{modeldir}/activeTracers_temperature/activeTracers_temperature.{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
            velfile = f'{modeldir}/normalVelocity/normalVelocity.{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
            dzfile = f'{modeldir}/layerThickness/layerThickness.{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

            dsOutMonthly = xr.Dataset()
            #if 'timeMonthly_avg_normalTransportVelocity' in dsIn.keys():
            #    vel = dsIn.timeMonthly_avg_normalTransportVelocity.isel(Time=0, nEdges=edgesToRead)
            #elif 'timeMonthly_avg_normalVelocity' in dsIn.keys():
            #    vel = dsIn.timeMonthly_avg_normalVelocity.isel(Time=0, nEdges=edgesToRead)
            #    if 'timeMonthly_avg_normalGMBolusVelocity' in dsIn.keys():
            #        vel = vel + dsIn.timeMonthly_avg_normalGMBolusVelocity.isel(Time=0, nEdges=edgesToRead)
            #    if 'timeMonthly_avg_normalMLEvelocity' in dsIn.keys():
            #        vel = vel + dsIn.timeMonthly_avg_normalMLEvelocity.isel(Time=0, nEdges=edgesToRead)
            #else:
            #    raise KeyError('no appropriate normalVelocity variable found')
            dsIn = xr.open_dataset(velfile, decode_times=False)
            vel = dsIn.timeMonthly_avg_normalVelocity.isel(Time=0, nEdges=edgesToRead)
            # Note that the following is incorrect when coe is zero (cell straddling the
            # transect edge is on land), but that is OK because the value will be masked
            # during land-sea masking below
            dsIn = xr.open_dataset(tempfile, decode_times=False)
            tempOnCells1 = dsIn.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe0)
            tempOnCells2 = dsIn.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe1)

            # Mask values that fall on land
            tempOnCells1 = tempOnCells1.where(landmask1, drop=False)
            tempOnCells2 = tempOnCells2.where(landmask2, drop=False)
            # Mask values that fall onto topography
            tempOnCells1 = tempOnCells1.where(depthmask1, drop=False)
            tempOnCells2 = tempOnCells2.where(depthmask2, drop=False)
            # The following should *not* happen at this point:
            if np.any(tempOnCells1.values[np.logical_or(tempOnCells1.values> 1e15, tempOnCells1.values<-1e15)]) or \
               np.any(tempOnCells2.values[np.logical_or(tempOnCells2.values> 1e15, tempOnCells2.values<-1e15)]):
                print('WARNING: something is wrong with land and/or topography masking!')

            if not use_fixeddz:
                dsIn = xr.open_dataset(dzfile, decode_times=False)
                dzOnCells1 = dsIn.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe0)
                dzOnCells2 = dsIn.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe1)
                dzOnCells1 = dzOnCells1.where(landmask1, drop=False)
                dzOnCells2 = dzOnCells2.where(landmask2, drop=False)
                dzOnCells1 = dzOnCells1.where(depthmask1, drop=False)
                dzOnCells2 = dzOnCells2.where(depthmask2, drop=False)

            # Interpolate values onto edges
            tempOnEdges = np.nanmean(np.array([tempOnCells1.values, tempOnCells2.values]), axis=0)
            if not use_fixeddz:
                dzOnEdges = np.nanmean(np.array([dzOnCells1.values, dzOnCells2.values]), axis=0)
            tempOnEdges = xr.DataArray(data=tempOnEdges, dims=('nEdges', 'nVertLevels'), name='tempOnEdges')
            dzOnEdges = xr.DataArray(data=dzOnEdges, dims=('nEdges', 'nVertLevels'), name='dzOnEdges')

            # Compute transports for each transect
            v_t           = np.zeros(nTransects)
            vprime_tprime = np.zeros(nTransects)
            vmean_tprime  = np.zeros(nTransects)
            vprime_tmean  = np.zeros(nTransects)
            for i in range(nTransects):
                start = int(nTransectStartStop[i])
                stop = int(nTransectStartStop[i+1])

                normalVel = vel.isel(nEdges=range(start, stop)) * edgeSigns.isel(nTransect=i, nEdges=range(start, stop))
                normalVelmean = velmean.isel(nEdges=range(start, stop)) * edgeSigns.isel(nTransect=i, nEdges=range(start, stop))
                vprime = normalVel - normalVelmean
                tprime = tempOnEdges.isel(nEdges=range(start, stop)) - tmean.isel(nEdges=range(start, stop))
                maskOnEdges = tprime.notnull()
                normalVel = normalVel.where(maskOnEdges, drop=False)
                normalVelmean = normalVelmean.where(maskOnEdges, drop=False)
                vprime = vprime.where(maskOnEdges, drop=False)

                if use_fixeddz:
                    dArea = dvEdge.isel(nEdges=range(start, stop)) * dz
                else:
                    dArea = dvEdge.isel(nEdges=range(start, stop)) * dzOnEdges.isel(nEdges=range(start, stop))
                area_transect = dArea.sum(dim='nEdges').sum(dim='nVertLevels')

                v_t[i]           = (tempOnEdges.isel(nEdges=range(start, stop)) * normalVel * dArea).sum(dim='nEdges').sum(dim='nVertLevels')
                vprime_tprime[i] = (tprime * vprime * dArea).sum(dim='nEdges').sum(dim='nVertLevels')
                vmean_tprime[i]  = (tprime * normalVelmean * dArea).sum(dim='nEdges').sum(dim='nVertLevels')
                vprime_tmean[i]  = (tmean.isel(nEdges=range(start, stop)) * vprime * dArea).sum(dim='nEdges').sum(dim='nVertLevels')

            dsOutMonthly['vT'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * v_t,
                    dims=('nTransects', ),
                    attrs=dict(description='Net heat transport (wrt 0degC) across transect due to total v, T', units='TW', )
                    )
            dsOutMonthly['vprimeTprime'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * vprime_tprime,
                    dims=('nTransects', ),
                    attrs=dict(description='Net heat transport across transect due to vprime, Tprime', units='TW', )
                    )
            dsOutMonthly['vmeanTprime'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * vmean_tprime,
                    dims=('nTransects', ),
                    attrs=dict(description='Net heat transport across transect due to vmean, Tprime', units='TW', )
                    )
            dsOutMonthly['vprimeTmean'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * vprime_tmean,
                    dims=('nTransects', ),
                    attrs=dict(description='Net heat transport across transect due to vprime, Tmean', units='TW', )
                    )
            dsOutMonthly['transectNames'] = xr.DataArray(data=transectNames, dims=('nTransects', ))
            dsOutMonthly['Time'] = xr.DataArray(
                    #data=[dsIn.timeMonthly_avg_daysSinceStartOfSim.isel(Time=0)/365.], 
                    data=[dsIn.Time.isel(Time=0)/365.], 
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
    infiles.append(f'{outdir}/{outfile0}_{casename}_year{year:04d}.nc')
dsIn = xr.open_mfdataset(infiles, decode_times=False)
t = dsIn['Time'].values
vT = dsIn['vT'].values
vprimeTprime = dsIn['vprimeTprime'].values
vmeanTprime = dsIn['vmeanTprime'].values
vprimeTmean = dsIn['vprimeTmean'].values

labelDict = {'Drake Passage':'drake', 'Tasmania-Ant':'tasmania', 'Africa-Ant':'africaAnt', 'Antilles Inflow':'antilles', \
             'Mona Passage':'monaPassage', 'Windward Passage':'windwardPassage', 'Florida-Cuba':'floridaCuba', \
             'Florida-Bahamas':'floridaBahamas', 'Indonesian Throughflow':'indonesia', 'Agulhas':'agulhas', \
             'Mozambique Channel':'mozambique', 'Bering Strait':'beringStrait', 'Lancaster Sound':'lancasterSound', \
             'Fram Strait':'framStrait', 'Robeson Channel':'robeson', 'Davis Strait':'davisStrait', 'Barents Sea Opening':'barentsSea', \
             'Nares Strait':'naresStrait', 'Denmark Strait':'denmarkStrait', 'Iceland-Faroe-Scotland':'icelandFaroeScotland'}

figsize = (16, 10)
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

    vT_runavg = pd.Series.rolling(pd.DataFrame(vT[:, i]), 5*12, center=True).mean()
    vprimeTprime_runavg = pd.Series.rolling(pd.DataFrame(vprimeTprime[:, i]), 5*12, center=True).mean()
    vmeanTprime_runavg = pd.Series.rolling(pd.DataFrame(vmeanTprime[:, i]), 5*12, center=True).mean()
    vprimeTmean_runavg = pd.Series.rolling(pd.DataFrame(vprimeTmean[:, i]), 5*12, center=True).mean()

    # Plot total vT
    figfile = f'{figdir}/transportsWithPrimes_{transectName_forfigfile}_{casename}.png'
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(221)
    ax1.plot(t, vT[:,i], 'k', linewidth=2, label=f'net vT ({np.nanmean(vT[:,i]):5.2f} $\pm$ {np.nanstd(vT[:,i]):5.2f})')
    ax1.plot(t, vT_runavg, 'b', linewidth=2, label='5-year run-avg')
    ax1.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax1.grid(color='k', linestyle=':', linewidth = 0.5)
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel('v*T (TW)', fontsize=12, fontweight='bold')
    ax1.legend()

    # Plot vprimeTprime
    ax2 = plt.subplot(222)
    ax2.plot(t, vprimeTprime[:,i], 'k', linewidth=2, label=f'net $v^\prime*T^\prime$ ({np.nanmean(vprimeTprime[:,i]):5.2f} $\pm$ {np.nanstd(vprimeTprime[:,i]):5.2f})')
    ax2.plot(t, vprimeTprime_runavg, 'b', linewidth=2, label='5-year run-avg')
    ax2.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax2.grid(color='k', linestyle=':', linewidth = 0.5)
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_ylabel('$v^\prime*T^\prime$ (TW)', fontsize=12, fontweight='bold')
    ax2.legend()

    # Plot Heat Transport wrt Tref=TfreezingPoint
    ax3 = plt.subplot(223)
    ltxt = '$\overline{v}*T^\prime$'
    ax3.plot(t, vmeanTprime[:,i], 'k', linewidth=2, label=f'net {ltxt} ({np.nanmean(vmeanTprime[:,i]):5.2f} $\pm$ {np.nanstd(vmeanTprime[:,i]):5.2f})')
    ax3.plot(t, vmeanTprime_runavg, 'b', linewidth=2, label='5-year run-avg')
    ax3.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax3.grid(color='k', linestyle=':', linewidth = 0.5)
    ax3.autoscale(enable=True, axis='x', tight=True)
    ax3.set_ylabel('$\overline{v}*T^\prime$ (TW)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax3.legend()

    # Plot transect mean temperature
    ax4 = plt.subplot(224)
    ltxt = '$v^\prime*\overline{T}$'
    ax4.plot(t, vprimeTmean[:,i], 'k', linewidth=2, label=f'net {ltxt} ({np.nanmean(vprimeTmean[:,i]):5.2f} $\pm$ {np.nanstd(vprimeTmean[:,i]):5.2f})')
    ax4.plot(t, vprimeTmean_runavg, 'b', linewidth=2, label='5-year run-avg')
    ax4.grid(color='k', linestyle=':', linewidth = 0.5)
    ax4.autoscale(enable=True, axis='x', tight=True)
    ax4.set_ylabel('$v^\prime*\overline{T}$ (TW)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax4.legend()

    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Transect = {transectName}\nrunname = {casename}', fontsize=14, fontweight='bold', y=1.045)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=-0.5, ybuffer=-1.65)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
