#
# This script does two things: 1) identifies years of anomalously high and low
# convection in a specific region, based on seasonal maximum mixed layer depth
# whose monthly values have been computed previously (and stored in maxMLDdir);
# 2) computes composites of a number of variables (native MPAS fields or
# processed quantities such as depth-averaged fields) based on the years
# identified in 1).
# This particular version deals with ensembles of simulations, and 1) and 2)
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
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


from mpas_analysis.ocean.utility import compute_zmid


matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
plt.rc('font', weight='bold')

startSimYear = 1950
startYear = 1950
endYear = 2014
years = np.arange(startYear, endYear + 1)
calendar = 'gregorian'
referenceDate = '0001-01-01'

# Settings for nersc
meshFile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
ensembleName = 'E3SM-Arcticv2.1_historical'
ensembleMemberNames = ['0101', '0151', '0201', '0251', '0301']
# Directories where fields for step 2) are stored:
maindir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations'
postprocmaindir = maindir
isShortTermArchive = True # if True 'archive/{modelComp}/hist' will be affixed to maindir later on

outdir = f'./composites_maxMLDbased_data/{ensembleName}'
figdir = f'./composites_maxMLDbased/{ensembleName}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
if not os.path.isdir(figdir):
    os.makedirs(figdir)

regionGroup = 'Arctic Regions'
groupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')
# one region at a time, for now:
region = 'Greenland Sea'
#regions = ['Greenland Sea', 'Norwegian Sea']

climoMonths = [1, 2, 3, 4] # JFMA
titleClimoMonths = 'JFMA'

# Fields relevant for step 2):
# Choose either variables in timeSeriesStatsMonthly
# or variables in timeSeriesStatsMonthlyMax (2d only) or
# ice variables (2d only)
#
#   Ocean variables
modelComp = 'ocn'
modelName = 'mpaso'
#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'mpas': 'timeMonthlyMax_max_dThreshMLD'}
#            ]
#
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
#             {'name': 'activeTracers_temperatureDepthAvg',
#              'mpas': 'timeMonthly_avg_activeTracers_temperature'},
#             {'name': 'activeTracers_salinityDepthAvg',
#              'mpas': 'timeMonthly_avg_activeTracers_salinity'}
              #'mpas': 'timeMonthly_avg_activeTracers_salinity'},
#             {'name': 'dThreshMLD',
#              'mpas': 'timeMonthly_avg_dThreshMLD'},
#             {'name': 'windStressZonal',
#              'mpas': 'timeMonthly_avg_windStressZonal'},
#             {'name': 'windStressMeridional',
#              'mpas': 'timeMonthly_avg_windStressMeridional'},
#             {'name': 'sensibleHeatFlux',
#              'mpas': 'timeMonthly_avg_sensibleHeatFlux'},
             {'name': 'spiciness',
              'mpas': None}
             ]
#             #{'name': 'surfaceBuoyancyForcing',
#             # 'mpas': 'timeMonthly_avg_surfaceBuoyancyForcing'}
#             #{'name': 'latentHeatFlux',
#             # 'mpas': 'timeMonthly_avg_latentHeatFlux'}
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
#zmin = -50.
#zmax = 0.
zmin = -600.
zmax = -100.
#zmin = -600.
#zmax = 0.
# The following is only relevant for depthAvg variables
# and for gsw-derived variables
dsMesh = xr.open_dataset(meshFile)
z = dsMesh.refBottomDepth
lat = 180.0/np.pi*dsMesh.latCell
lon = 180.0/np.pi*dsMesh.lonCell
maxLevelCell = dsMesh.maxLevelCell
pressure = gsw.p_from_z(-z, lat)

#####
##### STEP 1 #####
#####

# Identify high-convection and low-convection years based on
# previously computed regional averages of monthly maxMLD fields
print(f'\nIdentify years of low/high convection based on seasonal maxMLD in the {region}\n')
nEnsembles = len(ensembleMemberNames)
maxMLD_seasonal = np.zeros((nEnsembles, len(years)))
maxMLD_seasonal_monthly = np.zeros((nEnsembles, len(years), len(climoMonths)))
regionNameShort = region[0].lower() + region[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
for nEns in range(nEnsembles):
    ensembleMemberName = ensembleMemberNames[nEns]
    maxMLDdir = f'./timeseries_data/{ensembleName}{ensembleMemberName}/maxMLD'
    timeSeriesFiles = []
    for year in years:
        timeSeriesFiles.append(f'{maxMLDdir}/{groupName}_max_year{year:04d}.nc')
    dsIn = xr.open_mfdataset(timeSeriesFiles, combine='nested',
                             concat_dim='Time', decode_times=False)
    regionNames = dsIn.regionNames[0].values
    regionIndex = np.where(regionNames==region)[0]

    datetimes = netCDF4.num2date(dsIn.Time, f'days since {referenceDate}', calendar=calendar)
    timeyears = []
    for date in datetimes.flat:
        timeyears.append(date.year)

    maxMLD = np.squeeze(dsIn.maxMLD.isel(nRegions=regionIndex).values)
    for iy, year in enumerate(years):
        yearmask = [i for i, x in enumerate(timeyears) if x==year]
        dsIn_yearly = dsIn.isel(Time=yearmask)
        datetimes = netCDF4.num2date(dsIn_yearly.Time, f'days since {referenceDate}', calendar=calendar)
        timemonths = []
        for date in datetimes.flat:
            timemonths.append(date.month)
        monthmask = [i for i, x in enumerate(timemonths) if x in set(climoMonths)]
        maxMLD_seasonal[nEns, iy] = dsIn_yearly.maxMLD.isel(Time=monthmask, nRegions=regionIndex).mean().values
        maxMLD_seasonal_monthly[nEns, iy, :] = dsIn_yearly.maxMLD.isel(Time=monthmask, nRegions=regionIndex).values

mld_flat = maxMLD_seasonal.flatten()
print('quantile 0 =', np.quantile(mld_flat, 0), '  min = ', np.min(mld_flat))
print('quantile 1 =', np.quantile(mld_flat, 0.25))
print('quantile 2 =', np.quantile(mld_flat, 0.5), '  median = ', np.median(mld_flat))
print('quantile 3 =', np.quantile(mld_flat, 0.75))
print('quantile 4 =', np.quantile(mld_flat, 1), '  max = ', np.max(mld_flat))
print('mean = ', np.mean(mld_flat))
print('std = ', np.std(mld_flat))
# this works only for normally distributed fields:
#maxMLDstd = np.std(mld_flat)
#mld1 = np.min(mld_flat) + 1.5*maxMLDstd
#mld2 = np.max(mld_flat) - 1.5*maxMLDstd
mld1 = np.quantile(mld_flat, 0.15)
mld2 = np.quantile(mld_flat, 0.85)
#mld1 = np.quantile(mld_flat, 0.25) # first quartile
#mld2 = np.quantile(mld_flat, 0.75) # third quartile
print('mld1 = ', mld1, 'mdl2 = ', mld2)

# Make histogram plot
plt.figure(figsize=[10, 8], dpi=150)
ax = plt.subplot()
n, bins, patches = plt.hist(mld_flat, bins=12, color='#607c8e', alpha=0.7, rwidth=0.9)
ax.set_xticks(bins)
ax.set_xticklabels(np.int16(bins))
ax.axvspan(np.min(mld_flat), np.quantile(mld_flat, 0.15), alpha=0.3, color='salmon')
#ax.axvspan(np.min(mld_flat), np.quantile(mld_flat, 0.25), alpha=0.3, color='salmon')
ax.axvspan(np.quantile(mld_flat, 0.85), np.max(mld_flat), alpha=0.3, color='salmon')
#ax.axvspan(np.quantile(mld_flat, 0.75), np.max(mld_flat), alpha=0.3, color='salmon')
ax.set_xlim(np.min(mld_flat), np.max(mld_flat))
ax.set_xlabel(f'{titleClimoMonths}-avg maxMLD [m]', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('# of years', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(f'Distribution of maxMLD in the {region}', fontsize=18, fontweight='bold', pad=15)
ax.yaxis.set_minor_locator(MultipleLocator(5))
plt.grid(axis='y', alpha=0.75)
#plt.grid(axis='y', which='both', alpha=0.75)
plt.savefig(f'{figdir}/maxMLDhist_{regionNameShort}.png', bbox_inches='tight')
plt.close()

mld_monthly_flat = maxMLD_seasonal_monthly.flatten()
plt.figure(figsize=[10, 8], dpi=150)
ax = plt.subplot()
n, bins, patches = plt.hist(mld_monthly_flat, bins=12, color='#607c8e', alpha=0.7, rwidth=0.9)
ax.set_xticks(bins)
ax.set_xticklabels(np.int16(bins))
ax.axvspan(np.min(mld_monthly_flat), np.quantile(mld_monthly_flat, 0.15), alpha=0.3, color='salmon')
#ax.axvspan(np.min(mld_monthly_flat), np.quantile(mld_monthly_flat, 0.25), alpha=0.3, color='salmon')
ax.axvspan(np.quantile(mld_monthly_flat, 0.85), np.max(mld_monthly_flat), alpha=0.3, color='salmon')
#ax.axvspan(np.quantile(mld_monthly_flat, 0.75), np.max(mld_monthly_flat), alpha=0.3, color='salmon')
ax.set_xlim(np.min(mld_monthly_flat), np.max(mld_monthly_flat))
ax.set_xlabel(f'{titleClimoMonths} monthly maxMLD [m]', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('# of years', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title(f'Distribution of maxMLD in the {region}', fontsize=18, fontweight='bold', pad=15)
ax.yaxis.set_minor_locator(MultipleLocator(5))
plt.grid(axis='y', alpha=0.75)
#plt.grid(axis='y', which='both', alpha=0.75)
plt.savefig(f'{figdir}/maxMLDmonthlyhist_{regionNameShort}.png', bbox_inches='tight')
plt.close()

conditionLow = np.nan*np.ones((nEnsembles, len(years)))
conditionHigh = np.nan*np.ones((nEnsembles, len(years)))
conditionMed = np.nan*np.ones((nEnsembles, len(years)))
for nEns in range(nEnsembles): 
    conditionLow[nEns, :]  = np.less(maxMLD_seasonal[nEns, :], mld1)
    conditionHigh[nEns, :] = np.greater_equal(maxMLD_seasonal[nEns, :], mld2)
    conditionMed[nEns, :]  = np.logical_and(maxMLD_seasonal[nEns, :]>=mld1, maxMLD_seasonal[nEns, :]<mld2)
years2d = np.tile(years, (nEnsembles, 1))
years_low  = np.int32(years2d*conditionLow)
years_high = np.int32(years2d*conditionHigh)
years_med  = np.int32(years2d*conditionMed)
#print(years_low)
#print(years_high)
#print(years_med)

# Save this information to ascii files
with open(f'{outdir}/years_maxMLDlow.dat', 'w') as outfile:
    outfile.write(f'Years associated with low-convection in the {region} for each ensemble member\n')
    for nEns in range(nEnsembles):
        outfile.write(f'\nEnsemble member: {ensembleName}{ensembleMemberNames[nEns]}\n')
        np.savetxt(outfile, years_low[nEns, np.nonzero(years_low[nEns, :])][0], fmt='%5d', delimiter=' ')
with open(f'{outdir}/years_maxMLDhigh.dat', 'w') as outfile:
    outfile.write(f'Years associated with high-convection in the {region} for each ensemble member\n')
    for nEns in range(nEnsembles):
        outfile.write(f'\nEnsemble member: {ensembleName}{ensembleMemberNames[nEns]}\n')
        np.savetxt(outfile, years_high[nEns, np.nonzero(years_high[nEns, :])][0], fmt='%5d', delimiter=' ')

#####
##### STEP 2 #####
#####

# Compute monthly climatologies associated with these composites
for im in range(1, 13):
    print(f'   climatological month: {im}')
    for var in variables:
        varname = var['name']
        print(f'    var: {varname}')
        if modelName == 'mpaso' or modelName == 'mpassi':
            varmpasname = var['mpas']

        if varname=='velocityZonalDepthAvg' or varname=='velocityMeridionalDepthAvg' or \
           varname=='activeTracers_temperatureDepthAvg' or varname=='activeTracers_salinityDepthAvg':
            outfileLow  = f'{outdir}/{varname}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDlow_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'
            outfileHigh = f'{outdir}/{varname}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDhigh_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'
        else:
            outfileLow  = f'{outdir}/{varname}_maxMLDlow_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'
            outfileHigh = f'{outdir}/{varname}_maxMLDhigh_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'

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
                            zMid = compute_zmid(z, maxLevelCell, layerThickness)
                            fld = xr.open_dataset(datafile)[varmpasname]

                            # Depth-masked zmin-zmax layer thickness
                            depthMask = np.logical_and(zMid >= zmin, zMid <= zmax)
                            layerThickness = layerThickness.where(depthMask, drop=False)
                            layerDepth = layerThickness.sum(dim='nVertLevels')

                            fld = fld.where(depthMask, drop=False)
                            fld = (fld * layerThickness).sum(dim='nVertLevels')/layerDepth

                            # Write to post-processed datafile
                            datafile = f'{postprocdir}/{varname}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            dsOut = xr.Dataset()
                            dsOut[varmpasname] = fld
                            dsOut.to_netcdf(datafile)
                        elif varname=='spiciness':
                            temp = xr.open_dataset(datafile)['timeMonthly_avg_activeTracers_temperature']
                            salt = xr.open_dataset(datafile)['timeMonthly_avg_activeTracers_salinity']
                            SA = gsw.SA_from_SP(salt, pressure, lon, lat)
                            CT = gsw.CT_from_pt(SA, temp)
                            spiciness0 = gsw.spiciness0(SA, CT)
                            #spiciness1 = gsw.spiciness1(SA, CT)
                            #spiciness2 = gsw.spiciness2(SA, CT)

                            # Write to post-processed datafile
                            datafile = f'{postprocdir}/{varname}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            print(f'*** LR composite, datafile={datafile}')
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
                            zMid = compute_zmid(z, maxLevelCell, layerThickness)
                            fld = xr.open_dataset(datafile)[varmpasname]

                            # Depth-masked zmin-zmax layer thickness
                            depthMask = np.logical_and(zMid >= zmin, zMid <= zmax)
                            layerThickness = layerThickness.where(depthMask, drop=False)
                            layerDepth = layerThickness.sum(dim='nVertLevels')

                            fld = fld.where(depthMask, drop=False)
                            fld = (fld * layerThickness).sum(dim='nVertLevels')/layerDepth

                            # Write to post-processed datafile
                            datafile = f'{postprocdir}/{varname}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            dsOut = xr.Dataset()
                            dsOut[varmpasname] = fld
                            dsOut.to_netcdf(datafile)
                        elif varname=='spiciness':
                            temp = xr.open_dataset(datafile)['timeMonthly_avg_activeTracers_temperature']
                            salt = xr.open_dataset(datafile)['timeMonthly_avg_activeTracers_salinity']
                            SA = gsw.SA_from_SP(salt, pressure, lon, lat)
                            CT = gsw.CT_from_pt(SA, temp)
                            spiciness0 = gsw.spiciness0(SA, CT)
                            #spiciness1 = gsw.spiciness1(SA, CT)
                            #spiciness2 = gsw.spiciness2(SA, CT)

                            # Write to post-processed datafile
                            datafile = f'{postprocdir}/{varname}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            print(f'*** HR composite, datafile={datafile}')
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

                        infiles.append(datafile)
            if varname=='spiciness':
                args = ['ncea', '-O', '-v', 'spiciness0']
                #args = ['ncea', '-O', '-v', 'spiciness0,spiciness1,spiciness2']
            else:
                args = ['ncea', '-O', '-v', varmpasname]
            args.extend(infiles)
            args.append(outfileHigh)
            subprocess.check_call(args)
