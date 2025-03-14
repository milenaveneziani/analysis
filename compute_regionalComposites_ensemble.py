#
# This script does two things: 1) identifies years of anomalously high and low
# 'timeseriesVar' previously computed as monthly averages over a specific region.
# Before step 2, 'timeseriesVar' is also (optionally) detrended and seasonally
# averaged; 2) computes composites of a number of variables (native MPAS fields 
# or processed quantities such as the barotropic streamfunction or depth-averaged
# fields) based on the years identified in step 1.
#
# This particular version deals with ensembles of simulations, and steps 1,2
# are performed across ensemble members.
#

from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import subprocess
from subprocess import call
import xarray as xr
import numpy as np
import netCDF4
import gsw
from scipy.signal import detrend
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from mpas_analysis.ocean.utility import compute_zmid
from barotropicStreamfunction import compute_barotropic_streamfunction_vertex


#matplotlib.use('TkAgg')
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
plt.rc('font', weight='bold')

# Settings for nersc
meshFile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
ensembleName = 'E3SM-Arcticv2.1_historical'
ensembleMemberNames = ['0101', '0151', '0201', '0251', '0301']
# Directories where fields for step 2) are stored:
maindir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations'
postprocmaindir = maindir
isShortTermArchive = True # if True 'archive/{modelComp}/hist' will be affixed to maindir later on

startSimYear = 1950
startYear = 1950
endYear = 2014
years = np.arange(startYear, endYear + 1)
calendar = 'gregorian'
referenceDate = '0001-01-01'

# Main variable with respect to which composites are calculated
#timeseriesVar = 'iceArea'
#timeseriesUnits = 'km$^2$'
timeseriesVar = 'maxMLD'
timeseriesUnits = 'm'

# Timeseries detrending is on by default. View raw timeseries
# first and then decide whether to detrend or not
#use_detrend = True
use_detrend = False # for maxMLD
view_timeseries = False

# Months over which timeseriesVar is averaged before 
# computing the composites
climoMonths = [1, 2, 3, 4]
titleClimoMonths = 'JFMA'

# Information for region over which timeseriesVar is averaged
# before computing the composites
regionGroup = 'Arctic Regions'
groupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')
# one region at a time, for now:
region = 'Greenland Sea'
#region = 'Norwegian Sea'
regionNameShort = region[0].lower() + region[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')

# Fields relevant for step 2):
# Choose either variables in timeSeriesStatsMonthly
# or variables in timeSeriesStatsMonthlyMax (2d only) or
# ice variables (2d only)
#
#   Ocean variables
modelComp = 'ocn'
modelName = 'mpaso'
mpasFile = 'timeSeriesStatsMonthly'
variables = [
#             {'name': 'velocityZonalDepthAvg',
#              'mpas': 'timeMonthly_avg_velocityZonal'},
#             {'name': 'velocityMeridionalDepthAvg',
#              'mpas': 'timeMonthly_avg_velocityMeridional'},
#             {'name': 'velocityZonal',
#              'mpas': 'timeMonthly_avg_velocityZonal'},
#             {'name': 'velocityMeridional',
#              'mpas': 'timeMonthly_avg_velocityMeridional'},
#             {'name': 'activeTracers_temperature',
#              'mpas': 'timeMonthly_avg_activeTracers_temperature'},
#             {'name': 'activeTracers_salinity',
#              'mpas': 'timeMonthly_avg_activeTracers_salinity'},
             {'name': 'activeTracers_temperatureDepthAvg',
              'mpas': 'timeMonthly_avg_activeTracers_temperature'},
             {'name': 'activeTracers_salinityDepthAvg',
             'mpas': 'timeMonthly_avg_activeTracers_salinity'},
#             {'name': 'dThreshMLD',
#              'mpas': 'timeMonthly_avg_dThreshMLD'},
#             {'name': 'windStressZonal',
#              'mpas': 'timeMonthly_avg_windStressZonal'},
#             {'name': 'windStressMeridional',
#              'mpas': 'timeMonthly_avg_windStressMeridional'},
#             {'name': 'spiciness',
#              'mpas': None},
#             {'name': 'barotropicStreamfunction',
#              'mpas': 'barotropicStreamfunction'},
#             {'name': 'surfaceBuoyancyForcing',
#              'mpas': 'timeMonthly_avg_surfaceBuoyancyForcing'},
#             {'name': 'shortWaveHeatFlux',
#              'mpas': 'timeMonthly_avg_shortWaveHeatFlux'},
#             {'name': 'longWaveHeatFluxUp',
#              'mpas': 'timeMonthly_avg_longWaveHeatFluxUp'},
#             {'name': 'longWaveHeatFluxDown',
#              'mpas': 'timeMonthly_avg_longWaveHeatFluxDown'},
#             {'name': 'seaIceFreshWaterFlux',
#              'mpas': 'timeMonthly_avg_seaIceFreshWaterFlux'},
#             {'name': 'sensibleHeatFlux',
#              'mpas': 'timeMonthly_avg_sensibleHeatFlux'},
             {'name': 'latentHeatFlux',
              'mpas': 'timeMonthly_avg_latentHeatFlux'}
             ]
#
#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'mpas': 'timeMonthlyMax_max_dThreshMLD'}
#            ]
#   Sea ice variables
#modelComp = 'ice'
#modelName = 'mpassi'
#mpasFile = 'timeSeriesStatsMonthly'
#variables = [
#             {'name': 'iceArea',
#              'mpas': 'timeMonthly_avg_iceAreaCell'},
#             {'name': 'iceVolume',
#              'mpas': 'timeMonthly_avg_iceVolumeCell'},
#             {'name': 'iceDivergence',
#              'mpas': 'timeMonthly_avg_divergence'},
#             {'name': 'uVelocityGeo',
#              'mpas': 'timeMonthly_avg_uVelocityGeo'},
#             {'name': 'vVelocityGeo',
#              'mpas': 'timeMonthly_avg_vVelocityGeo'}
#            ]
#   Atmosphere variables
#modelComp = 'atm'
#modelName = 'eam'

# For depthAvg variables, choose zmin,zmax values over which to average
# Note: for now, it is easier to do this for each depth range
#zmin = -100.
#zmax = 10.
#zmin = -600.
#zmax = -100.
zmin = -8000.
zmax = -600.

outdir = f'./composites_{timeseriesVar}based_data/{ensembleName}'
figdir = f'./composites_{timeseriesVar}based/{ensembleName}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
if not os.path.isdir(figdir):
    os.makedirs(figdir)
########################################################################

# The following is only relevant for depthAvg variables, the
# barotropic streamfunction, and for gsw-derived variables
dsMesh = xr.open_dataset(meshFile)
depth = dsMesh.bottomDepth
lat = 180.0/np.pi*dsMesh.latCell
lon = 180.0/np.pi*dsMesh.lonCell
maxLevelCell = dsMesh.maxLevelCell
pressure = gsw.p_from_z(-depth, lat)

#####
##### STEP 1
#####
##### Compute high and low values of regionally (and seasonally) averaged timeseriesVar
#####

print(f'\nIdentify years of low/high {timeseriesVar} based on seasonal values averaged in the {region}\n')
nEnsembles = len(ensembleMemberNames)
timeseries_seasonal = np.zeros((nEnsembles, len(years)))
timeseries_seasonal_monthly = np.zeros((nEnsembles, len(years), len(climoMonths)))
for nEns in range(nEnsembles):
    ensembleMemberName = ensembleMemberNames[nEns]
    timeseriesDir = f'./timeseries_data/{ensembleName}{ensembleMemberName}/{timeseriesVar}'
    timeseriesFiles = []
    for year in years:
        if timeseriesVar=='maxMLD':
            timeseriesFiles.append(f'{timeseriesDir}/{groupName}_max_year{year:04d}.nc')
        else:
            timeseriesFiles.append(f'{timeseriesDir}/{groupName}_year{year:04d}.nc')
    ds = xr.open_mfdataset(timeseriesFiles, combine='nested',
                           concat_dim='Time', decode_times=False)
    regionNames = ds.regionNames[0].values
    regionIndex = np.where(regionNames==region)[0]

    timeseries = np.squeeze(ds[timeseriesVar].isel(nRegions=regionIndex).values)
    if use_detrend is True:
        timeseries = detrend(timeseries, type='linear')
    if view_timeseries is True:
        delta = timeseries - timeseries_detrend
        plt.plot(ds.Time.values, timeseries, 'k')
        plt.plot(ds.Time.values, delta, 'b')
        plt.plot(ds.Time.values, timeseries_detrend, 'r')
        plt.grid(alpha=0.75)
        plt.show()

    # Compute seasonal averages
    datetimes = netCDF4.num2date(ds.Time, f'days since {referenceDate}', calendar=calendar)
    timeyears = []
    timemonths = []
    for date in datetimes.flat:
        timeyears.append(date.year)
        timemonths.append(date.month)
    monthmask = [i for i, x in enumerate(timemonths) if x in set(climoMonths)]

    for iy, year in enumerate(years):
        yearmask = [i for i, x in enumerate(timeyears) if x==year]
        mask = np.intersect1d(yearmask, monthmask)
        if np.size(mask)==0:
            raise ValueError('Something is wrong with time mask')
        timeseries_seasonal[nEns, iy] = np.nanmean(timeseries[mask])
        timeseries_seasonal_monthly[nEns, iy, :] = timeseries[mask]

timeseries_flat = timeseries_seasonal.flatten()
print('quantile 0 (and min value) =', np.quantile(timeseries_flat, 0))
print('quantile 1 =', np.quantile(timeseries_flat, 0.25))
print('quantile 2 (and median) =', np.quantile(timeseries_flat, 0.5))
print('quantile 3 =', np.quantile(timeseries_flat, 0.75))
print('quantile 4 (and max value) =', np.quantile(timeseries_flat, 1))
print('mean = ', np.mean(timeseries_flat))
print('std = ', np.std(timeseries_flat))
# this works only for normally distributed fields:
#timeseries_std = np.std(timeseries_flat)
#timeseries1 = np.min(timeseries_flat) + 1.5*timeseries_std
#timeseries2 = np.max(timeseries_flat) - 1.5*timeseries_std
timeseries1 = np.quantile(timeseries_flat, 0.15)
timeseries2 = np.quantile(timeseries_flat, 0.85)
#timeseries1 = np.quantile(timeseries_flat, 0.25) # first quartile
#timeseries2 = np.quantile(timeseries_flat, 0.75) # third quartile
print('timeseries1 = ', timeseries1, 'timeseries2 = ', timeseries2)

# Make histogram plot
plt.figure(figsize=[10, 8], dpi=150)
ax = plt.subplot()
n, bins, patches = plt.hist(timeseries_flat, bins=12, color='#607c8e', alpha=0.7, rwidth=0.9)
ax.set_xticks(bins)
ax.set_xticklabels(np.int16(bins))
ax.axvspan(np.min(timeseries_flat), np.quantile(timeseries_flat, 0.15), alpha=0.3, color='salmon')
#ax.axvspan(np.min(timeseries_flat), np.quantile(timeseries_flat, 0.25), alpha=0.3, color='salmon')
ax.axvspan(np.quantile(timeseries_flat, 0.85), np.max(timeseries_flat), alpha=0.3, color='salmon')
#ax.axvspan(np.quantile(timeseries_flat, 0.75), np.max(timeseries_flat), alpha=0.3, color='salmon')
ax.set_xlim(np.min(timeseries_flat), np.max(timeseries_flat))
ax.set_xlabel(f'{titleClimoMonths}-avg {timeseriesVar} [{timeseriesUnits}]', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('# of years', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(f'Distribution of {timeseriesVar} in the {region}', fontsize=18, fontweight='bold', pad=15)
ax.yaxis.set_minor_locator(MultipleLocator(5))
plt.grid(axis='y', alpha=0.75)
#plt.grid(axis='y', which='both', alpha=0.75)
plt.savefig(f'{figdir}/{timeseriesVar}hist_{regionNameShort}.png', bbox_inches='tight')
plt.close()

timeseries_monthly_flat = timeseries_seasonal_monthly.flatten()
plt.figure(figsize=[10, 8], dpi=150)
ax = plt.subplot()
n, bins, patches = plt.hist(timeseries_monthly_flat, bins=12, color='#607c8e', alpha=0.7, rwidth=0.9)
ax.set_xticks(bins)
ax.set_xticklabels(np.int16(bins))
ax.axvspan(np.min(timeseries_monthly_flat), np.quantile(timeseries_monthly_flat, 0.15), alpha=0.3, color='salmon')
#ax.axvspan(np.min(timeseries_monthly_flat), np.quantile(timeseries_monthly_flat, 0.25), alpha=0.3, color='salmon')
ax.axvspan(np.quantile(timeseries_monthly_flat, 0.85), np.max(timeseries_monthly_flat), alpha=0.3, color='salmon')
#ax.axvspan(np.quantile(timeseries_monthly_flat, 0.75), np.max(timeseries_monthly_flat), alpha=0.3, color='salmon')
ax.set_xlim(np.min(timeseries_monthly_flat), np.max(timeseries_monthly_flat))
ax.set_xlabel(f'{titleClimoMonths} monthly {timeseriesVar} [{timeseriesUnits}]', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('# of years', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(f'Distribution of {timeseriesVar} in the {region}', fontsize=18, fontweight='bold', pad=15)
ax.yaxis.set_minor_locator(MultipleLocator(5))
plt.grid(axis='y', alpha=0.75)
#plt.grid(axis='y', which='both', alpha=0.75)
plt.savefig(f'{figdir}/{timeseriesVar}monthlyhist_{regionNameShort}.png', bbox_inches='tight')
plt.close()

conditionLow = np.nan*np.ones((nEnsembles, len(years)))
conditionHigh = np.nan*np.ones((nEnsembles, len(years)))
conditionMed = np.nan*np.ones((nEnsembles, len(years)))
for nEns in range(nEnsembles): 
    conditionLow[nEns, :]  = np.less(timeseries_seasonal[nEns, :], timeseries1)
    conditionHigh[nEns, :] = np.greater_equal(timeseries_seasonal[nEns, :], timeseries2)
    conditionMed[nEns, :]  = np.logical_and(timeseries_seasonal[nEns, :]>=timeseries1,
                                            timeseries_seasonal[nEns, :]<timeseries2)
years2d = np.tile(years, (nEnsembles, 1))
years_low  = np.int32(years2d*conditionLow)
years_high = np.int32(years2d*conditionHigh)
years_med  = np.int32(years2d*conditionMed)

# Save this information to ascii files
with open(f'{outdir}/years_{timeseriesVar}low_{regionNameShort}.dat', 'w') as outfile:
    outfile.write(f'Years associated with low {timeseriesVar} in the {region} for each ensemble member\n')
    for nEns in range(nEnsembles):
        outfile.write(f'\nEnsemble member: {ensembleName}{ensembleMemberNames[nEns]}\n')
        np.savetxt(outfile, years_low[nEns, np.nonzero(years_low[nEns, :])][0], fmt='%5d', delimiter=' ')
with open(f'{outdir}/years_{timeseriesVar}high_{regionNameShort}.dat', 'w') as outfile:
    outfile.write(f'Years associated with high {timeseriesVar} in the {region} for each ensemble member\n')
    for nEns in range(nEnsembles):
        outfile.write(f'\nEnsemble member: {ensembleName}{ensembleMemberNames[nEns]}\n')
        np.savetxt(outfile, years_high[nEns, np.nonzero(years_high[nEns, :])][0], fmt='%5d', delimiter=' ')
#####
##### STEP 2
#####
##### Compute monthly climatologies associated with the composites
##### computed in STEP 1
#####

for im in range(1, 13):
#for im in range(3, 4):
    print(f'   climatological month: {im}')
    for var in variables:
        varname = var['name']
        print(f'    var: {varname}')
        if modelName == 'mpaso' or modelName == 'mpassi':
            varmpasname = var['mpas']

        if varname=='velocityZonalDepthAvg' or varname=='velocityMeridionalDepthAvg' or \
           varname=='activeTracers_temperatureDepthAvg' or varname=='activeTracers_salinityDepthAvg':
            outfileLow  = f'{outdir}/{varname}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{timeseriesVar}low_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'
            outfileHigh = f'{outdir}/{varname}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{timeseriesVar}high_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'
        else:
            outfileLow  = f'{outdir}/{varname}_{timeseriesVar}low_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'
            outfileHigh = f'{outdir}/{varname}_{timeseriesVar}high_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'

        if not os.path.isfile(outfileLow):
            print(f'\nComposite file {outfileLow} does not exist. Creating it with ncea...')
            infiles = []
            for nEns in range(nEnsembles):
                runName = f'{ensembleName}{ensembleMemberNames[nEns]}'
                if isShortTermArchive:
                    rundir = f'{maindir}/{runName}/archive/{modelComp}/hist'
                    # The following is only relevant for post-processed variables (such as depthAvg fields)
                    postprocdir = f'{postprocmaindir}/{runName}/archive/{modelComp}/postproc'
                else:
                    rundir = f'{maindir}/{runName}/run'
                    # The following is only relevant for post-processed variables (such as depthAvg fields)
                    postprocdir = f'{postprocmaindir}/{runName}/run'
                if not os.path.isdir(postprocdir):
                    os.makedirs(postprocdir)
                yLow = years_low[nEns, np.nonzero(years_low[nEns, :])][0]
                if np.size(yLow)!=0:
                    for k in range(len(yLow)):
                        iy = yLow[k]
                        if im > np.max(climoMonths) and iy != startSimYear:
                            iy = iy-1  # pick months *preceding* the climoMonths period of each year
                        if modelComp == 'atm':
                            datafile = f'{rundir}/{runName}.{modelName}.h0.{int(iy):04d}-{int(im):02d}.nc'
                        else:
                            datafile = f'{rundir}/{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                        # Check if file exists
                        if not os.path.isfile(datafile):
                            raise SystemExit(f'File {datafile} not found. Exiting...\n')

                        # Compute complex variables before making composites
                        if varname=='velocityZonalDepthAvg' or varname=='velocityMeridionalDepthAvg' or \
                           varname=='activeTracers_temperatureDepthAvg' or varname=='activeTracers_salinityDepthAvg':
                            layerThickness = xr.open_dataset(datafile).timeMonthly_avg_layerThickness
                            fld = xr.open_dataset(datafile)[varmpasname]

                            # Compute post-processed field and write to file if datafile does not exist
                            datafile = f'{postprocdir}/{varname}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            if not os.path.isfile(datafile):
                                # Depth-masked zmin-zmax layer thickness
                                zMid = compute_zmid(depth, maxLevelCell, layerThickness)
                                depthMask = np.logical_and(zMid >= zmin, zMid <= zmax)
                                layerThickness = layerThickness.where(depthMask, drop=False)
                                layerDepth = layerThickness.sum(dim='nVertLevels')
                                fld = fld.where(depthMask, drop=False)
                                fld = (fld * layerThickness).sum(dim='nVertLevels')/layerDepth
                                dsOut = xr.Dataset()
                                dsOut[varmpasname] = fld
                                dsOut.to_netcdf(datafile)
                        elif varname=='spiciness':
                            temp = xr.open_dataset(datafile)['timeMonthly_avg_activeTracers_temperature']
                            salt = xr.open_dataset(datafile)['timeMonthly_avg_activeTracers_salinity']

                            # Compute post-processed field and write to file if datafile does not exist
                            datafile = f'{postprocdir}/{varname}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            if not os.path.isfile(datafile):
                                SA = gsw.SA_from_SP(salt, pressure, lon, lat)
                                CT = gsw.CT_from_pt(SA, temp)
                                spiciness0 = gsw.spiciness0(SA, CT)
                                #spiciness1 = gsw.spiciness1(SA, CT)
                                #spiciness2 = gsw.spiciness2(SA, CT)
                                print(f'*** Low composite, datafile={datafile}')
                                dsOut = xr.Dataset()
                                dsOut['spiciness0'] = spiciness0
                                dsOut['spiciness0'].attrs['long_name'] = 'Spiciness computed wrt sea level pressure through gsw package'
                                dsOut['spiciness0'].attrs['units'] = 'kg/m^3'
                                dsOut.to_netcdf(datafile)
                                #dsOut = xr.Dataset()
                                #dsOut['spiciness1'] = spiciness1
                                #dsOut['spiciness1'].attrs['long_name'] = 'Spiciness computed wrt 1000 dbar pressure through gsw package'
                                #dsOut['spiciness1'].attrs['units'] = 'kg/m^3'
                                #dsOut.to_netcdf(datafile, mode='a')
                                #dsOut = xr.Dataset()
                                #dsOut['spiciness2'] = spiciness2
                                #dsOut['spiciness2'].attrs['long_name'] = 'Spiciness computed wrt 2000 dbar pressure through gsw package'
                                #dsOut['spiciness2'].attrs['units'] = 'kg/m^3'
                                #dsOut.to_netcdf(datafile, mode='a')
                        elif varname=='barotropicStreamfunction':
                            dsIn = xr.open_dataset(datafile)

                            # Compute post-processed field and write to file if datafile does not exist
                            datafile = f'{postprocdir}/{varname}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            if not os.path.isfile(datafile):
                                print(f'*** Low composite, datafile={datafile}')
                                min_lat = -45.0
                                min_depth = -10000.0
                                max_depth = 10.0
                                fld = compute_barotropic_streamfunction_vertex(dsMesh, dsIn, min_lat, min_depth, max_depth)
                                dsOut = xr.Dataset()
                                dsOut['barotropicStreamfunction'] = fld
                                dsOut['barotropicStreamfunction'].attrs['long_name'] = 'Barotropic streamfunction'
                                dsOut['barotropicStreamfunction'].attrs['units'] = 'Sv'
                                dsOut.to_netcdf(datafile)

                        infiles.append(datafile)
            if varname=='spiciness':
                args = ['ncea', '-O', '-v', 'spiciness0']
                #args = ['ncea', '-O', '-v', 'spiciness0,spiciness1,spiciness2']
            else:
                args = ['ncea', '-O', '-v', varmpasname]
            args.extend(infiles)
            args.append(outfileLow)
            subprocess.check_call(args)

        if not os.path.isfile(outfileHigh):
            print(f'\nComposite file {outfileHigh} does not exist. Creating it with ncea...')
            infiles = []
            for nEns in range(nEnsembles):
                runName = f'{ensembleName}{ensembleMemberNames[nEns]}'
                if isShortTermArchive:
                    rundir = f'{maindir}/{runName}/archive/{modelComp}/hist'
                    # The following is only relevant for post-processed variables (such as depthAvg fields)
                    postprocdir = f'{postprocmaindir}/{runName}/archive/{modelComp}/postproc'
                else:
                    rundir = f'{maindir}/{runName}/run'
                    # The following is only relevant for post-processed variables (such as depthAvg fields)
                    postprocdir = f'{postprocmaindir}/{runName}/run'
                if not os.path.isdir(postprocdir):
                    os.makedirs(postprocdir)
                yHigh = years_high[nEns, np.nonzero(years_high[nEns, :])][0]
                if np.size(yHigh)!=0:
                    for k in range(len(yHigh)):
                        iy = yHigh[k]
                        if im > np.max(climoMonths) and iy != startSimYear:
                            iy = iy-1  # pick months *preceding* the climoMonths period of each year
                        if modelComp == 'atm':
                            datafile = f'{rundir}/{runName}.{modelName}.h0.{int(iy):04d}-{int(im):02d}.nc'
                        else:
                            datafile = f'{rundir}/{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                        # Check if file exists
                        if not os.path.isfile(datafile):
                            raise SystemExit(f'File {datafile} not found. Exiting...\n')
                        # Compute complex variables before making composites
                        if varname=='velocityZonalDepthAvg' or varname=='velocityMeridionalDepthAvg' or \
                           varname=='activeTracers_temperatureDepthAvg' or varname=='activeTracers_salinityDepthAvg':
                            layerThickness = xr.open_dataset(datafile).timeMonthly_avg_layerThickness
                            fld = xr.open_dataset(datafile)[varmpasname]

                            # Compute post-processed field and write to file if datafile does not exist
                            datafile = f'{postprocdir}/{varname}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            if not os.path.isfile(datafile):
                                zMid = compute_zmid(depth, maxLevelCell, layerThickness)
                                # Depth-masked zmin-zmax layer thickness
                                depthMask = np.logical_and(zMid >= zmin, zMid <= zmax)
                                layerThickness = layerThickness.where(depthMask, drop=False)
                                layerDepth = layerThickness.sum(dim='nVertLevels')
                                fld = fld.where(depthMask, drop=False)
                                fld = (fld * layerThickness).sum(dim='nVertLevels')/layerDepth
                                dsOut = xr.Dataset()
                                dsOut[varmpasname] = fld
                                dsOut.to_netcdf(datafile)
                        elif varname=='spiciness':
                            temp = xr.open_dataset(datafile)['timeMonthly_avg_activeTracers_temperature']
                            salt = xr.open_dataset(datafile)['timeMonthly_avg_activeTracers_salinity']

                            # Compute post-processed field and write to file if datafile does not exist
                            datafile = f'{postprocdir}/{varname}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            if not os.path.isfile(datafile):
                                SA = gsw.SA_from_SP(salt, pressure, lon, lat)
                                CT = gsw.CT_from_pt(SA, temp)
                                spiciness0 = gsw.spiciness0(SA, CT)
                                #spiciness1 = gsw.spiciness1(SA, CT)
                                #spiciness2 = gsw.spiciness2(SA, CT)
                                print(f'*** High composite, datafile={datafile}')
                                dsOut = xr.Dataset()
                                dsOut['spiciness0'] = spiciness0
                                dsOut['spiciness0'].attrs['long_name'] = 'Spiciness computed wrt sea level pressure through gsw package'
                                dsOut['spiciness0'].attrs['units'] = 'kg/m^3'
                                dsOut.to_netcdf(datafile)
                                #dsOut = xr.Dataset()
                                #dsOut['spiciness1'] = spiciness1
                                #dsOut['spiciness1'].attrs['long_name'] = 'Spiciness computed wrt 1000 dbar pressure through gsw package'
                                #dsOut['spiciness1'].attrs['units'] = 'kg/m^3'
                                #dsOut.to_netcdf(datafile, mode='a')
                                #dsOut = xr.Dataset()
                                #dsOut['spiciness2'] = spiciness2
                                #dsOut['spiciness2'].attrs['long_name'] = 'Spiciness computed wrt 2000 dbar pressure through gsw package'
                                #dsOut['spiciness2'].attrs['units'] = 'kg/m^3'
                                #dsOut.to_netcdf(datafile, mode='a')
                        elif varname=='barotropicStreamfunction':
                            dsIn = xr.open_dataset(datafile)

                            # Compute post-processed field and write to file if datafile does not exist
                            datafile = f'{postprocdir}/{varname}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            if not os.path.isfile(datafile):
                                print(f'*** High composite, datafile={datafile}')
                                min_lat = -45.0
                                min_depth = -10000.0
                                max_depth = 10.0
                                fld = compute_barotropic_streamfunction_vertex(dsMesh, dsIn, min_lat, min_depth, max_depth)
                                dsOut = xr.Dataset()
                                dsOut['barotropicStreamfunction'] = fld
                                dsOut['barotropicStreamfunction'].attrs['long_name'] = 'Barotropic streamfunction'
                                dsOut['barotropicStreamfunction'].attrs['units'] = 'Sv'
                                dsOut.to_netcdf(datafile)

                        infiles.append(datafile)
            if varname=='spiciness':
                args = ['ncea', '-O', '-v', 'spiciness0']
                #args = ['ncea', '-O', '-v', 'spiciness0,spiciness1,spiciness2']
            else:
                args = ['ncea', '-O', '-v', varmpasname]
            args.extend(infiles)
            args.append(outfileHigh)
            subprocess.check_call(args)
