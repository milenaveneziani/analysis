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
import scipy
import cftime
import time

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

def highpass_filter(data, fsampling, fcutoff, filter_order=5):
    """
    Applies a Butterworth high-pass filter to the input data.

    Input:
        data: the input signal as a numpy array, type=float.
        fsampling: the sampling frequency (units should be consistent with fcutoff), type=float.
        fcutoff: the cutoff frequency (units should be consistent with fsampling), type=float.
        filter_order: the order of the filter (higher order means a sharper cutoff), type=integer.

    Returns:
        The filtered data as a numpy array, type=float.
    """

    # Normalized cutoff frequency (fc/(fs/2), where fs/2 is the Nyquist frequency)
    Wn = fcutoff/(fsampling/2)

    # Design the filter coefficients (second-order sections (sos) format is recommended)
    sos = scipy.signal.butter(N=filter_order, Wn=Wn, btype='highpass', output='sos')

    # Apply the filter forward and backward for zero-phase filtering
    filtered_data = scipy.signal.sosfiltfilt(sos, data)

    return filtered_data


runname = 'E3SMv2.1B60to10rA02'
#runname = 'E3SM-Arcticv2.1_historical0101'

meshname = 'ARRM10to60E2r1' # still need to change meshfile below accordingly
#maskname = 'arcticSections20210323'
#maskname = 'atlanticZonal_sections20240910'
maskname = 'arctic_subarctic_transects4transports20250918'

# *** Settings for nersc
meshdir = f'/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/{meshname}'
maskdir = '/global/cfs/cdirs/m1199/milena/mpas-region_masks'
maindir = '/global/cfs/cdirs/m1199/e3sm-arrm-simulations'
indir0 = f'{maindir}/{runname}/archive'
isShortTermArchive = False
isSingleVarFiles = True # True for E3SMv2.1B60to10rA02

# *** Settings for erdc.hpc.mil
#meshdir = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/{meshname}'
#maskdir = '/p/home/milena/mpas-region_masks'
#maindir = '/p/global/milena'
#indir0 = f'{maindir}/{runname}/archive'
#indir0 = f'/p/global/apcraig/archive/{runname}' # for E3SMv2.1B60to10rA07
#isShortTermArchive = True
#isSingleVarFiles = False

# More directory/filename settings
meshfile = f'{meshdir}/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
maskfile = f'{maskdir}/{meshname}_{maskname}.nc'
featurefile = f'{maskdir}/{maskname}.geojson'
outfile3d0 = f'{maskname}Transect3dData'
outfile0 = f'{maskname}TransportsWithPrimesFromHPfilter'
#
if (isShortTermArchive  and isSingleVarFiles) or \
   ((not isShortTermArchive) and (not isSingleVarFiles)):
    raise ValueError('Variables isShortTermArchive and isSingleVarFiles cannot be both True or both False')
if isSingleVarFiles:
    indir = f'{indir0}/ocn/singleVarFiles'
else:
    if isShortTermArchive:
        indir = f'{indir0}/ocn/hist'
    else:
        indir = indir0
#
figdir = f'./transports/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)
#
outdir = f'./transports_data/{runname}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
#
if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found')

# Choose years
yearStart = 1
yearEnd = 386 # rA02
#yearEnd = 246 # rA07
#yearStart = 1950
#yearEnd = 2014
years = range(yearStart, yearEnd+1)

use_fixeddz = False # this should be True only if layerThickness variable is missing

# Define constants
saltRef = 34.8
rhoRef = 1026.0 # kg/m^3 (config_density0)
cp = 3.996*1e3 # J/(kg*degK) (as defined in mpas_constants)
m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
m3ps_to_mSv = 1e-3 # m^3/s flux to 10^-3 Sverdrups (mSv)
m3ps_to_km3py = 1e-9*86400*365.25  # m^3/s FW flux to km^3/year
W_to_TW = 1e-12
referenceDate = '0001-01-01'
calendar = 'noleap' # not needed at the moment
###################################################################

# Read in transect information
dsMask = get_mask_short_names(xr.open_dataset(maskfile))
transectNames = decode_strings(dsMask.transectNames)
transectList = dsMask.shortNames.values
nTransects = len(transectList)
maxEdges = dsMask.sizes['maxEdgesInTransect']
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
edgeSigns = xr.DataArray(data=edgeSigns, dims=('nTransect', 'nEdges'))
# The following is only needed for pressure and if use_fixeddz=True;
# pressure is only used to compute the freezing temperature
refBottom = dsMesh.refBottomDepth
latmean = 180.0/np.pi * dsMesh.latEdge.sel(nEdges=edgesToRead).mean()
lonmean = 180.0/np.pi * dsMesh.lonEdge.sel(nEdges=edgesToRead).mean()
pressure = gsw.p_from_z(-refBottom, latmean)
if use_fixeddz:
    dz = xr.concat([refBottom.isel(nVertLevels=0), refBottom.diff('nVertLevels')], dim='nVertLevels')

outfile3d = f'{outdir}/{outfile3d0}_{runname}_years{yearStart:04d}-{yearEnd:04d}.nc'
outfile3d_filtered = f'{outdir}/{outfile3d0}Filtered_{runname}_years{yearStart:04d}-{yearEnd:04d}.nc'
outfile = f'{outdir}/{outfile0}_{runname}_years{yearStart:04d}-{yearEnd:04d}new.nc'

# Compute 3d fields for each transect if outfile3d does not exist
if not os.path.exists(outfile3d):
    print('Computing vdArea and temp for each transect...')
    kyear = 0
    dsOut = []
    for year in years:
        kyear = kyear + 1
        print(f'Year = {year:04d} ({kyear} out of {len(years)} years total)')
        t0 = time.time()

        for month in range(1, 13):

            if isSingleVarFiles:
                infile = f'{indir}/normalVelocity/normalVelocity.{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                dsInVel = xr.open_dataset(infile, decode_times=False)
                infile = f'{indir}/activeTracers_temperature/activeTracers_temperature.{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                dsInTemp = xr.open_dataset(infile, decode_times=False)
                infile = f'{indir}/activeTracers_salinity/activeTracers_salinity.{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                dsInSalt = xr.open_dataset(infile, decode_times=False)
                infile = f'{indir}/layerThickness/layerThickness.{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                dsInLayerThick = xr.open_dataset(infile, decode_times=False)
            else:
                infile = f'{indir}/{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                dsInVel = xr.open_dataset(infile, decode_times=False)
                dsInTemp = dsInVel
                dsInSalt = dsInVel
                dsInLayerThick = dsInVel

            # Read in velocities, temperature and salinity
            # Just use the resolved normalVelocity
            vel = dsInVel.timeMonthly_avg_normalVelocity.isel(Time=0, nEdges=edgesToRead)
            # Note that the following is incorrect when coe is zero (cell straddling the
            # transect edge is on land), but that is OK because the value will be masked
            # during land-sea masking below
            tempOnCells1 = dsInTemp.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe0)
            tempOnCells2 = dsInTemp.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe1)
            saltOnCells1 = dsInSalt.timeMonthly_avg_activeTracers_salinity.isel(Time=0, nCells=coe0)
            saltOnCells2 = dsInSalt.timeMonthly_avg_activeTracers_salinity.isel(Time=0, nCells=coe1)

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

            if not use_fixeddz:
                dzOnCells1 = dsInLayerThick.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe0)
                dzOnCells2 = dsInLayerThick.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe1)
                dzOnCells1 = dzOnCells1.where(landmask1, drop=False)
                dzOnCells2 = dzOnCells2.where(landmask2, drop=False)
                dzOnCells1 = dzOnCells1.where(depthmask1, drop=False)
                dzOnCells2 = dzOnCells2.where(depthmask2, drop=False)

            # Interpolate values onto edges
            tempOnEdges = np.nanmean(np.array([tempOnCells1.values, tempOnCells2.values]), axis=0)
            saltOnEdges = np.nanmean(np.array([saltOnCells1.values, saltOnCells2.values]), axis=0)
            # The following doesn't do a proper nansum.. (and couldn't find anything online
            # about nansumming two *separate* xarray datasets):
            #tempOnEdges = 0.5 * (tempOnCells1 + tempOnCells2)
            #saltOnEdges = 0.5 * (saltOnCells1 + saltOnCells2)
            if not use_fixeddz:
                dzOnEdges = np.nanmean(np.array([dzOnCells1.values, dzOnCells2.values]), axis=0)
                #dzOnEdges = 0.5 * (dzOnCells1 + dzOnCells2)
            tempOnEdges = xr.DataArray(data=tempOnEdges, dims=('nEdges', 'nVertLevels'), name='tempOnEdges')
            saltOnEdges = xr.DataArray(data=saltOnEdges, dims=('nEdges', 'nVertLevels'), name='saltOnEdges')
            dzOnEdges = xr.DataArray(data=dzOnEdges, dims=('nEdges', 'nVertLevels'), name='dzOnEdges')

            # Compute freezing temperature
            SA = gsw.SA_from_SP(saltOnEdges, pressure, lonmean, latmean)
            CTfp = gsw.CT_freezing(SA, pressure, 0.)
            Tfp = gsw.pt_from_CT(SA, CTfp)

            # Compute transports for each transect
            dsOutMonthly = []
            for i in range(nTransects):
                dsOutTransect = xr.Dataset()
                start = int(nTransectStartStop[i])
                stop = int(nTransectStartStop[i+1])

                if use_fixeddz:
                    dArea = dvEdge.isel(nEdges=range(start, stop)) * dz
                else:
                    dArea = dvEdge.isel(nEdges=range(start, stop)) * dzOnEdges.isel(nEdges=range(start, stop))
                #tfreezing = Tfp.isel(nEdges=range(start, stop))
                #tfreezing = tfreezing.expand_dims(dim='nTransects')

                normalVel = vel.isel(nEdges=range(start, stop)) * edgeSigns.isel(nTransect=i, nEdges=range(start, stop))
                normalVel = normalVel.expand_dims(dim='nTransects')
                temp = tempOnEdges.isel(nEdges=range(start, stop))
                temp = temp.expand_dims(dim='nTransects')
                #temp_wrtTfp = temp - tfreezing
                #salt = saltOnEdges.isel(nEdges=range(start, stop))
                #salt = salt.expand_dims(dim='nTransects')
                maskOnEdges = temp.notnull()
                normalVel = normalVel.where(maskOnEdges, drop=False)
                vdArea = normalVel * dArea

                dsOutTransect['normalVelocity_times_darea'] = xr.DataArray(
                    data=vdArea,
                    dims=('nTransects', 'nEdges', 'nVertLevels', ),
                    attrs=dict(description='normalVelocity times dvEdge times layerthickness', units='m^3/s', )
                    )
                dsOutTransect['temp'] = xr.DataArray(
                    data=temp,
                    dims=('nTransects', 'nEdges', 'nVertLevels', ),
                    attrs=dict(description='temperature', units='degree C', )
                    )
                #dsOutTransect['temp_wrtTfp'] = xr.DataArray(
                #    data=temp_wrtTfp,
                #    dims=('nTransects', 'nEdges', 'nVertLevels', ),
                #    attrs=dict(description='temperature minus local Tfreezing point', units='degree C', )
                #    )
                #dsOutTransect['salt'] = xr.DataArray(
                #    data=salt,
                #    dims=('nTransects', 'nEdges', 'nVertLevels', ),
                #    attrs=dict(description='salinity', units='psu', )
                #    )

                dsOutTransect = dsOutTransect.pad(nEdges=(0, maxEdges-nEdgesInTransect[i]))
                dsOutMonthly.append(dsOutTransect)

            dsOutMonthly = xr.concat(dsOutMonthly, dim='nTransects')
            dsOut.append(dsOutMonthly)
        print(f'\n   time taken (seconds)={time.time()-t0}')

    dsOut = xr.concat(dsOut, dim='Time')
    dsOut['transectNames'] = xr.DataArray(data=transectNames, dims=('nTransects', ))
    dsOut['nEdgesInTransect'] = xr.DataArray(
            data=nEdgesInTransect,
            dims=('nTransects', ),
            attrs=dict(description='total number of edges for each transect', )
            )
    write_netcdf_with_fill(dsOut, outfile3d)
else:
    print(f'  Outfile {outfile3d} already exists. Proceed...')
    dsOut = xr.open_dataset(outfile3d)

vdArea = dsOut.normalVelocity_times_darea
temp = dsOut.temp
#temp_wrtTfp = dsOut.temp_wrtTfp
#salt = dsOut.salt
nEdgesmax = dsOut.nEdgesInTransect.values
t = cftime.date2num(np.hstack(dsOut['Time']), f'days since {referenceDate}') # days

fs = 12 # sampling frequency
fc = 1.2 # cutoff frequency
filter_order = 18
# Computed high-pass filter for each point on the transect (and each
# vertical level) if outfile3d_filtered does not exist
if not os.path.exists(outfile3d_filtered):
    dsOut = []
    for i in range(nTransects):
        transectName = transectNames[i]
        print(f'\nFiltering data for transect {transectName}...')
        t0 = time.time()
        vdArea_filtered = np.empty((len(t), maxEdges, nVertLevels))
        temp_filtered = np.empty((len(t), maxEdges, nVertLevels))
        dsOutTransect = xr.Dataset()
        for n in range(nEdgesmax[i]):
            for k in range(nVertLevels):
                vdArea_filtered[:, n, k] = highpass_filter(np.squeeze(vdArea.values[:, i, n, k]), fs, fc, filter_order=filter_order)
                temp_filtered[:, n, k] = highpass_filter(np.squeeze(temp.values[:, i, n, k]), fs, fc, filter_order=filter_order)
                #temp_wrtTfp_filtered[:, n, k] = highpass_filter(np.squeeze(temp_wrtTfp.values[:, i, n, k]), fs, fc, filter_order=filter_order)
                #salt_filtered[:, n, k] = highpass_filter(np.squeeze(salt.values[:, i, n, k]), fs, fc, filter_order=filter_order)
        vdArea_filtered = np.expand_dims(vdArea_filtered, axis=1)
        temp_filtered = np.expand_dims(temp_filtered, axis=1)
        dsOutTransect['normalVelocity_times_dareaHP'] = xr.DataArray(
                data=vdArea_filtered,
                dims=('Time', 'nTransects', 'nEdges', 'nVertLevels', ),
                attrs=dict(description=f'normalVelocity times dvEdge times layerthickness, high-pass filtered with Butterworth order {filter_order}', units='m^3/s', )
                )
        dsOutTransect['tempHP'] = xr.DataArray(
                data=temp_filtered,
                dims=('Time', 'nTransects', 'nEdges', 'nVertLevels', ),
                attrs=dict(description=f'temperature, high-pass filtered with Butterworth order {filter_order}', units='degree C', )
                )
        dsOut.append(dsOutTransect)
        print(f'   time taken (seconds)={time.time()-t0}')
    dsOut = xr.concat(dsOut, dim='nTransects')
    dsOut['transectNames'] = xr.DataArray(data=transectNames, dims=('nTransects', ))
    dsOut['nEdgesInTransect'] = xr.DataArray(
            data=nEdgesInTransect,
            dims=('nTransects', ),
            attrs=dict(description='total number of edges for each transect', )
            )
    write_netcdf_with_fill(dsOut, outfile3d_filtered)
else:
    print(f'  Outfile {outfile3d_filtered} already exists. Proceed...')
    dsOut = xr.open_dataset(outfile3d_filtered)

# Finally, compute the different components of the heat transport
if not os.path.exists(outfile):
    print('\nCompute the different components of the transports...')
    t0 = time.time()
    vdArea_filtered = dsOut.normalVelocity_times_dareaHP
    temp_filtered = dsOut.tempHP
    #temp_wrtTfp_filtered = dsOut.temp_wrtTfpHP
    #salt_filtered = dsOut.saltHP

    vdArea_longterm = vdArea - vdArea_filtered
    temp_longterm = temp - temp_filtered
    #temp_wrtTfp_longterm = temp_wrtTfp - temp_wrtTfp_filtered
    #salt_longterm = salt - salt_filtered

    factor = W_to_TW * rhoRef * cp
    v_t           = factor * (vdArea * temp).sum(dim='nEdges').sum(dim='nVertLevels')
    vprime_tprime = factor * (vdArea_filtered * temp_filtered).sum(dim='nEdges').sum(dim='nVertLevels')
    vmean_tprime  = factor * (vdArea_longterm * temp_filtered).sum(dim='nEdges').sum(dim='nVertLevels')
    vprime_tmean  = factor * (vdArea_filtered * temp_longterm).sum(dim='nEdges').sum(dim='nVertLevels')
    vmean_tmean   = factor * (vdArea_longterm * temp_longterm).sum(dim='nEdges').sum(dim='nVertLevels')
    vol      = m3ps_to_Sv * vdArea.sum(dim='nEdges').sum(dim='nVertLevels')
    volprime = m3ps_to_Sv * vdArea_filtered.sum(dim='nEdges').sum(dim='nVertLevels')
    volmean  = m3ps_to_Sv * vdArea_longterm.sum(dim='nEdges').sum(dim='nVertLevels')

    dsOut = xr.Dataset()
    dsOut['vT'] = xr.DataArray(
            data=v_t,
            dims=('Time', 'nTransects', ),
            attrs=dict(description='Net heat transport (wrt 0degC) across transect due to total v, T', units='TW', )
            )
    dsOut['vprimeTprime'] = xr.DataArray(
            data=vprime_tprime,
            dims=('Time', 'nTransects', ),
            attrs=dict(description=f'Net heat transport (wrt 0degC) across transect due to vprime, Tprime, where the primes are computed with high-pass Butterworth filter order {filter_order}', units='TW', )
            )
    dsOut['vmeanTmean'] = xr.DataArray(
            data=vmean_tmean,
            dims=('Time', 'nTransects', ),
            attrs=dict(description='Net heat transport (wrt 0degC) across transect due to vmean, Tmean, where the means are computed wrt the filtered fields', units='TW', )
            )
    dsOut['vprimeTmean'] = xr.DataArray(
            data=vprime_tmean,
            dims=('Time', 'nTransects', ),
            attrs=dict(description='Net heat transport (wrt 0degC) across transect due to vprime, Tmean', units='TW', )
            )
    dsOut['vmeanTprime'] = xr.DataArray(
            data=vmean_tprime,
            dims=('Time', 'nTransects', ),
            attrs=dict(description='Net heat transport (wrt 0degC) across transect due to vmean, Tprime', units='TW', )
            )
    dsOut['vol'] = xr.DataArray(
            data=vol,
            dims=('Time', 'nTransects', ),
            attrs=dict(description='Net volume transport across transect due to total v', units='Sv', )
            )
    dsOut['volprime'] = xr.DataArray(
            data=volprime,
            dims=('Time', 'nTransects', ),
            attrs=dict(description='Net volume transport across transect due to vprime', units='Sv', )
            )
    dsOut['volmean'] = xr.DataArray(
            data=volmean,
            dims=('Time', 'nTransects', ),
            attrs=dict(description='Net volume transport across transect due to vmean', units='Sv', )
            )
    dsOut['Time'] = xr.DataArray(
            data=t,
            dims=('Time', ),
            attrs=dict(description=f'Time (days since {referenceDate})', units='days', calendar='noleap')
            )
    dsOut['transectNames'] = xr.DataArray(data=transectNames, dims=('nTransects', ))
    write_netcdf_with_fill(dsOut, outfile)
    print(f'   time taken (seconds)={time.time()-t0}')
else:
    print(f'  Outfile {outfile} already exists. Proceed...')
    dsIn = xr.open_dataset(outfile, decode_times=False)
    v_t = dsIn.vT
    vmean_tmean = dsIn.vmeanTmean
    vprime_tprime = dsIn.vprimeTprime
    vprime_tmean = dsIn.vprimeTmean
    vmean_tprime = dsIn.vmeanTprime
    vol = dsIn.vol
    volprime = dsIn.volprime
    volmean = dsIn.volmean

print('\nPlotting...')
t0 = time.time()
t = t/365. # for no leap calendar this is OK
figsize = (24, 10)
#figsize = (12, 24)
figdpi = 300
labelDict = {'Drake Passage':'drake', 'Tasmania-Ant':'tasmania', 'Africa-Ant':'africaAnt', 'Antilles Inflow':'antilles', \
             'Mona Passage':'monaPassage', 'Windward Passage':'windwardPassage', 'Florida-Cuba':'floridaCuba', \
             'Florida-Bahamas':'floridaBahamas', 'Indonesian Throughflow':'indonesia', 'Agulhas':'agulhas', \
             'Mozambique Channel':'mozambique', 'Bering Strait':'beringStrait', 'Lancaster Sound':'lancasterSound', \
             'Fram Strait':'framStrait', 'Robeson Channel':'robeson', 'Davis Strait':'davisStrait', 'Barents Sea Opening':'barentsSeaOpening', \
             'Nares Strait':'naresStrait', 'Denmark Strait':'denmarkStrait', 'Iceland-Faroe-Scotland':'icelandFaroeScotland'}
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

    figfile = f'{figdir}/transportsWithPrimesFromHPfilter_{transectName_forfigfile}_{runname}.png'
    fig = plt.figure(figsize=figsize)
    #
    ax = plt.subplot(221)
    #ax = plt.subplot(711)
    ax.plot(t, v_t.isel(nTransects=i), 'k', linewidth=2, label='v*T')
    ax.plot(t, vprime_tprime.isel(nTransects=i), 'b', linewidth=2, label='vprime*Tprime')
    ax.plot(t, vmean_tprime.isel(nTransects=i), 'r', linewidth=2, label='vmean*Tprime')
    ax.plot(t, vprime_tmean.isel(nTransects=i), 'salmon', linewidth=2, label='vprime*Tmean')
    ax.plot(t, vmean_tmean.isel(nTransects=i), 'green', linewidth=2, label='vmean*Tmean')
    ax.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax.grid(color='k', linestyle=':', linewidth = 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylabel(r'Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')
    ax.legend()
    #
    ax = plt.subplot(222)
    ax.plot(t, vol.isel(nTransects=i), 'k', linewidth=2, label='v*T')
    ax.plot(t, volprime.isel(nTransects=i), 'b', linewidth=2, label='vprime*Tprime')
    ax.plot(t, volmean.isel(nTransects=i), 'green', linewidth=2, label='vmean*Tprime')
    ax.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax.grid(color='k', linestyle=':', linewidth = 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylabel('Net volume transport (Sv)', fontsize=12, fontweight='bold')
    ax.legend()
    #
    trange = range(0, 72) # first 6 years
    ax = plt.subplot(223)
    #ax = plt.subplot(713)
    ax.plot(t[trange], v_t.isel(Time=trange, nTransects=i), 'k', linewidth=2, label='v*T')
    ax.plot(t[trange], vprime_tprime.isel(Time=trange, nTransects=i), 'b', linewidth=2, label='vprime*Tprime')
    ax.plot(t[trange], vmean_tprime.isel(Time=trange, nTransects=i), 'r', linewidth=2, label='vmean*Tprime')
    ax.plot(t[trange], vprime_tmean.isel(Time=trange, nTransects=i), 'salmon', linewidth=2, label='vprime*Tmean')
    ax.plot(t[trange], vmean_tmean.isel(Time=trange, nTransects=i), 'green', linewidth=2, label='vmean*Tmean')
    ax.plot(t[trange], np.zeros_like(t[trange]), 'k', linewidth=1)
    ax.grid(color='k', linestyle=':', linewidth = 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')
    ax.legend()
    #
    ax = plt.subplot(224)
    ax.plot(t[trange], vol.isel(Time=trange, nTransects=i), 'k', linewidth=2, label='v*T')
    ax.plot(t[trange], volprime.isel(Time=trange, nTransects=i), 'b', linewidth=2, label='vprime*Tprime')
    ax.plot(t[trange], volmean.isel(Time=trange, nTransects=i), 'green', linewidth=2, label='vmean*Tprime')
    ax.plot(t[trange], np.zeros_like(t[trange]), 'k', linewidth=1)
    ax.grid(color='k', linestyle=':', linewidth = 0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Net volume transport (Sv)', fontsize=12, fontweight='bold')
    ax.legend()
    #
    #ntime = len(t)
    #dtime = int(ntime/5)
    #trange = range(0, dtime)
    #ax = plt.subplot(713)
    #ax.plot(t[trange], v_t.isel(Time=trange, nTransects=i), 'k', linewidth=2, label='v*T')
    #ax.plot(t[trange], vprime_tprime.isel(Time=trange, nTransects=i), 'b', linewidth=2, label='vprime*Tprime')
    #ax.plot(t[trange], vmean_tprime.isel(Time=trange, nTransects=i), 'r', linewidth=2, label='vmean*Tprime')
    #ax.plot(t[trange], vprime_tmean.isel(Time=trange, nTransects=i), 'salmon', linewidth=2, label='vprime*Tmean')
    #ax.plot(t[trange], vmean_tmean.isel(Time=trange, nTransects=i), 'green', linewidth=2, label='vmean*Tmean')
    #ax.plot(t[trange], np.zeros_like(t[trange]), 'k', linewidth=1)
    #ax.grid(color='k', linestyle=':', linewidth = 0.5)
    #ax.autoscale(enable=True, axis='x', tight=True)
    #ax.set_ylabel(r'Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')
    #ax.legend()
    #
    #trange = range(dtime, 2*dtime)
    #ax = plt.subplot(714)
    #ax.plot(t[trange], v_t.isel(Time=trange, nTransects=i), 'k', linewidth=2, label='v*T')
    #ax.plot(t[trange], vprime_tprime.isel(Time=trange, nTransects=i), 'b', linewidth=2, label='vprime*Tprime')
    #ax.plot(t[trange], vmean_tprime.isel(Time=trange, nTransects=i), 'r', linewidth=2, label='vmean*Tprime')
    #ax.plot(t[trange], vprime_tmean.isel(Time=trange, nTransects=i), 'salmon', linewidth=2, label='vprime*Tmean')
    #ax.plot(t[trange], vmean_tmean.isel(Time=trange, nTransects=i), 'green', linewidth=2, label='vmean*Tmean')
    #ax.plot(t[trange], np.zeros_like(t[trange]), 'k', linewidth=1)
    #ax.grid(color='k', linestyle=':', linewidth = 0.5)
    #ax.autoscale(enable=True, axis='x', tight=True)
    #ax.set_ylabel(r'Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')
    #ax.legend()
    #
    #trange = range(2*dtime, 3*dtime)
    #ax = plt.subplot(715)
    #ax.plot(t[trange], v_t.isel(Time=trange, nTransects=i), 'k', linewidth=2, label='v*T')
    #ax.plot(t[trange], vprime_tprime.isel(Time=trange, nTransects=i), 'b', linewidth=2, label='vprime*Tprime')
    #ax.plot(t[trange], vmean_tprime.isel(Time=trange, nTransects=i), 'r', linewidth=2, label='vmean*Tprime')
    #ax.plot(t[trange], vprime_tmean.isel(Time=trange, nTransects=i), 'salmon', linewidth=2, label='vprime*Tmean')
    #ax.plot(t[trange], vmean_tmean.isel(Time=trange, nTransects=i), 'green', linewidth=2, label='vmean*Tmean')
    #ax.plot(t[trange], np.zeros_like(t[trange]), 'k', linewidth=1)
    #ax.grid(color='k', linestyle=':', linewidth = 0.5)
    #ax.autoscale(enable=True, axis='x', tight=True)
    #ax.set_ylabel(r'Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')
    #ax.legend()
    #
    #trange = range(3*dtime, 4*dtime)
    #ax = plt.subplot(716)
    #ax.plot(t[trange], v_t.isel(Time=trange, nTransects=i), 'k', linewidth=2, label='v*T')
    #ax.plot(t[trange], vprime_tprime.isel(Time=trange, nTransects=i), 'b', linewidth=2, label='vprime*Tprime')
    #ax.plot(t[trange], vmean_tprime.isel(Time=trange, nTransects=i), 'r', linewidth=2, label='vmean*Tprime')
    #ax.plot(t[trange], vprime_tmean.isel(Time=trange, nTransects=i), 'salmon', linewidth=2, label='vprime*Tmean')
    #ax.plot(t[trange], vmean_tmean.isel(Time=trange, nTransects=i), 'green', linewidth=2, label='vmean*Tmean')
    #ax.plot(t[trange], np.zeros_like(t[trange]), 'k', linewidth=1)
    #ax.grid(color='k', linestyle=':', linewidth = 0.5)
    #ax.autoscale(enable=True, axis='x', tight=True)
    #ax.set_ylabel(r'Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')
    #ax.legend()
    #
    #trange = range(4*dtime, 5*dtime)
    #ax = plt.subplot(717)
    #ax.plot(t[trange], v_t.isel(Time=trange, nTransects=i), 'k', linewidth=2, label='v*T')
    #ax.plot(t[trange], vprime_tprime.isel(Time=trange, nTransects=i), 'b', linewidth=2, label='vprime*Tprime')
    #ax.plot(t[trange], vmean_tprime.isel(Time=trange, nTransects=i), 'r', linewidth=2, label='vmean*Tprime')
    #ax.plot(t[trange], vprime_tmean.isel(Time=trange, nTransects=i), 'salmon', linewidth=2, label='vprime*Tmean')
    #ax.plot(t[trange], vmean_tmean.isel(Time=trange, nTransects=i), 'green', linewidth=2, label='vmean*Tmean')
    #ax.plot(t[trange], np.zeros_like(t[trange]), 'k', linewidth=1)
    #ax.grid(color='k', linestyle=':', linewidth = 0.5)
    #ax.autoscale(enable=True, axis='x', tight=True)
    #ax.set_xlabel('Time (years)', fontsize=12, fontweight='bold')
    #ax.set_ylabel(r'Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')

    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Transect = {transectName}\nrunname = {runname}', fontsize=14, fontweight='bold', y=1.045)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=-0.5, ybuffer=-1.65)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
print(f'   time taken (seconds)={time.time()-t0}')
