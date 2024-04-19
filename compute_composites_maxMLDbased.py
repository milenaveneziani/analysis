from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import subprocess
from subprocess import call
import xarray as xr
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


from mpas_analysis.ocean.utility import compute_zmid


matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
plt.rc('font', weight='bold')

#startSimYear = 1950
#startYear = [1950]
#endYear = [2014]
startSimYear = 1
startYear = [1, 245]
endYear = [140, 386]
years = np.arange(startYear[0], endYear[0] + 1)
for iy in range(1, np.size(startYear)):
    years = np.append(years, np.arange(startYear[iy], endYear[iy] + 1))
calendar = 'gregorian'
referenceDate = '0001-01-01'

# Settings for nersc
#meshFile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runName = 'E3SM-Arcticv2.1_historical0151'
##runName = 'E3SMv2.1B60to10rA02'
#rundir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runName}'
#postprocmaindir = rundir
## Note: the following two variables cannot be both True
#isShortTermArchive = True # if True 'archive/{modelComp}/hist' will be affixed to rundir later on
#isSingleVarFiles = False # if True 'archive/{modelComp}/singleVarFiles' will be affixed to rundir later on

# Settings for erdc.hpc.mil
meshFile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runName = 'E3SMv2.1B60to10rA02'
#rundir = f'/p/archive/osinski/E3SM/{runName}'
rundir = f'/p/work/milena/archive/{runName}'
postprocmaindir = f'/p/work/milena/archive/{runName}'
# Note: the following two variables cannot be both True
isShortTermArchive = True # if True '{modelComp}/hist' will be affixed to rundir later on
isSingleVarFiles = False # if True '{modelComp}/singleVarFiles' will be affixed to rundir later on
 
maxMLDdir = f'./timeseries_data/{runName}/maxMLD'
outdir0 = f'./composites_maxMLDbased_data/{runName}'
figdir0 = f'./composites_maxMLDbased/{runName}'
outdir = f'Years{startYear[0]}-{endYear[0]}'
figdir = f'Years{startYear[0]}-{endYear[0]}'
for iy in range(1, np.size(startYear)):
    outdir = f'{outdir}_{startYear[iy]}-{endYear[iy]}'
    figdir = f'{figdir}_{startYear[iy]}-{endYear[iy]}'
outdir = f'{outdir0}/{outdir}'
figdir = f'{figdir0}/{figdir}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
if not os.path.isdir(figdir):
    os.makedirs(figdir)

regionGroup = 'Arctic Regions'
groupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')
regions = ['Greenland Sea', 'Norwegian Sea']

climoMonths = [1, 2, 3, 4] # JFMA
titleClimoMonths = 'JFMA'

# Choose either variables in timeSeriesStatsMonthly
# or variables in timeSeriesStatsMonthlyMax (2d only) or
# ice variables (2d only)
#
#   Ocean variables
#modelComp = 'ocn'
#modelName = 'mpaso'
#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'mpas': 'timeMonthlyMax_max_dThreshMLD'}
#            ]
#
#mpasFile = 'timeSeriesStatsMonthly'
#variables = [
#             #{'name': 'velocityZonalDepthAvg',
#             # 'mpas': 'timeMonthly_avg_velocityZonal'},
#             #{'name': 'velocityMeridionalDepthAvg',
#             # 'mpas': 'timeMonthly_avg_velocityMeridional'},
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
#              'mpas': 'timeMonthly_avg_activeTracers_salinity'},
#             {'name': 'dThreshMLD',
#              'mpas': 'timeMonthly_avg_dThreshMLD'},
#             {'name': 'sensibleHeatFlux',
#              'mpas': 'timeMonthly_avg_sensibleHeatFlux'}
#             ]
             #{'name': 'surfaceBuoyancyForcing',
             # 'mpas': 'timeMonthly_avg_surfaceBuoyancyForcing'}
             #{'name': 'latentHeatFlux',
             # 'mpas': 'timeMonthly_avg_latentHeatFlux'}
#   Sea ice variables
modelComp = 'ice'
modelName = 'mpassi'
mpasFile = 'timeSeriesStatsMonthly'
variables = [
             {'name': 'iceArea',
              'mpas': 'timeMonthly_avg_iceAreaCell'},
             {'name': 'iceVolume',
              'mpas': 'timeMonthly_avg_iceVolumeCell'}
            ]
#   Atmosphere variables
#modelComp = 'atm'
#modelName = 'eam'

if isShortTermArchive:
    #rundir = f'{rundir}/archive/{modelComp}/hist'
    rundir = f'{rundir}/{modelComp}/hist'
if isSingleVarFiles:
    rundir = f'{rundir}/archive/{modelComp}/singleVarFiles'
# The following is only relevant for post-processed variables (such as depthAvg fields)
if postprocmaindir==rundir:
    postprocdir = f'{postprocmaindir}/archive/{modelComp}/postproc'
else:
    postprocdir = f'{postprocmaindir}/{modelComp}/postproc'
if not os.path.isdir(postprocdir):
    os.makedirs(postprocdir)

# For depthAvg variables, choose zmin,zmax values over which to average
# Note: for now, it is easier to do this for each depth range
#zmins = [-100., -600., -8000., -8000.]
#zmaxs = [0., -100., -600., 0.]
#zmin = -100.
#zmax = 0.
zmin = -600.
zmax = -100.
# The following is only relevant for depthAvg variables
dsMesh = xr.open_dataset(meshFile)
z = dsMesh.refBottomDepth
maxLevelCell = dsMesh.maxLevelCell

# Identify high-convection and low-convection years based on
# previously computed regional averages of JFMA maxMLD fields
timeSeriesFiles = []
for year in years:
    timeSeriesFiles.append(f'{maxMLDdir}/{groupName}_max_year{year:04d}.nc')
dsIn = xr.open_mfdataset(timeSeriesFiles, combine='nested',
                         concat_dim='Time', decode_times=False)
regionNames = dsIn.regionNames[0].values

datetimes = netCDF4.num2date(dsIn.Time, f'days since {referenceDate}', calendar=calendar)
timeyears = []
for date in datetimes.flat:
    timeyears.append(date.year)

for regionName in regions:
    print(f'\nIdentify years of low/high convection based on maxMLD for region: {regionName}')
    regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')
    regionIndex = np.where(regionNames==regionName)[0]

    maxMLD = np.squeeze(dsIn.maxMLD.isel(nRegions=regionIndex).values)
    maxMLD_seasonal = np.zeros(len(years))
    for iy, year in enumerate(years):
        yearmask = [i for i, x in enumerate(timeyears) if x==year]
        dsIn_yearly = dsIn.isel(Time=yearmask)
        datetimes = netCDF4.num2date(dsIn_yearly.Time, f'days since {referenceDate}', calendar=calendar)
        timemonths = []
        for date in datetimes.flat:
            timemonths.append(date.month)
        monthmask = [i for i, x in enumerate(timemonths) if x in set(climoMonths)]
        maxMLD_seasonal[iy] = dsIn_yearly.maxMLD.isel(Time=monthmask, nRegions=regionIndex).mean().values

    #ax = plt.subplot(3, 1, 1)
    #n, bins, patches = plt.hist(maxMLD_seasonal, bins=10, color='#607c8e', alpha=0.7, rwidth=0.9)
    #ax.set_xticks(bins)
    #ax.set_xticklabels(np.int16(bins))
    #plt.grid(axis='y', alpha=0.75)
    #plt.xlabel(f'{titleClimoMonths}-avg maxMLD')
    #plt.ylabel('# of years')
    #plt.title(f'{regionName}')
    #ax = plt.subplot(3, 1, 2)
    #n, bins, patches = plt.hist(maxMLD, bins=10, color='#607c8e', alpha=0.7, rwidth=0.9)
    #ax.set_xticks(bins)
    #ax.set_xticklabels(np.int16(bins))
    #plt.grid(axis='y', alpha=0.75)
    #plt.xlabel(f'{titleClimoMonths} (monthly) maxMLD')
    #plt.ylabel('# of months')
    #ax = plt.subplot(3, 1, 3)
    #plt.plot(years, maxMLD_seasonal, linewidth=2)
    #plt.xlabel('years')
    #plt.ylabel(f'{titleClimoMonths}-avg maxMLD')
    #plt.savefig(f'{figdir}/maxMLDhist_{regionNameShort}.png', dpi='figure', bbox_inches='tight', pad_inches=0.1)
    #plt.close()

    maxMLDstd = np.std(maxMLD_seasonal)
    mld1 = np.min(maxMLD_seasonal) + maxMLDstd
    mld2 = np.max(maxMLD_seasonal) - maxMLDstd
    #print(mld1, mld2)
    #print(np.min(maxMLD_seasonal) + 1.5*maxMLDstd, np.max(maxMLD_seasonal) - 1.5*maxMLDstd)
    #print(np.min(maxMLD_seasonal) + 2*maxMLDstd, np.max(maxMLD_seasonal) - 2*maxMLDstd)
    conditionLow  = np.where(maxMLD_seasonal<mld1)
    conditionHigh = np.where(maxMLD_seasonal>=mld2)
    conditionMed  = np.logical_and(maxMLD_seasonal>=mld1, maxMLD_seasonal<mld2)
    #dbinRange = bins[-1]-bins[0]
    #conditionLow  = np.where(maxMLD_seasonal<bins[0]+0.2*dbinRange)
    #conditionHigh = np.where(maxMLD_seasonal>=bins[-1]-0.2*dbinRange)
    #conditionMed  = np.logical_and(maxMLD_seasonal>=bins[0]+0.2*dbinRange, maxMLD_seasonal<bins[-1]-0.2*dbinRange)

    years_low  = years[conditionLow]
    years_high = years[conditionHigh]
    years_med  = years[conditionMed]
    #print('years_low=', np.size(years_low))
    #print('years_low1.5=', np.size(years[np.where(maxMLD_seasonal<np.min(maxMLD_seasonal) + 1.5*maxMLDstd)]))
    #print('years_low2=', np.size(years[np.where(maxMLD_seasonal<np.min(maxMLD_seasonal) + 2*maxMLDstd)]))
    #print('years_high=', years_high)
    #print('years_high1.5=', years[np.where(maxMLD_seasonal>=np.max(maxMLD_seasonal) - 1.5*maxMLDstd)])
    #print('years_high2=', years[np.where(maxMLD_seasonal>=np.max(maxMLD_seasonal) - 2*maxMLDstd)])

    # Save this information to ascii files
    np.savetxt(f'{outdir}/years_maxMLDlow.dat', years_low, fmt='%5d', delimiter=' ')
    np.savetxt(f'{outdir}/years_maxMLDhigh.dat', years_high, fmt='%5d', delimiter=' ')

    # Make better histogram plot
    plt.figure(figsize=[10, 8], dpi=150)
    ax = plt.subplot()
    n, bins, patches = plt.hist(maxMLD_seasonal, bins=12, color='#607c8e', alpha=0.7, rwidth=0.9)
    ax.set_xticks(bins)
    ax.set_xticklabels(np.int16(bins))
    ax.axvspan(np.min(maxMLD_seasonal), mld1, alpha=0.3, color='salmon')
    ax.axvspan(mld2, np.max(maxMLD_seasonal), alpha=0.3, color='salmon')
    ax.set_xlim(np.min(maxMLD_seasonal), np.max(maxMLD_seasonal))
    ax.set_xlabel(f'{titleClimoMonths}-avg maxMLD [m]', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('# of years', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title(f'Distribution of maxMLD in the {regionName}', fontsize=18, fontweight='bold', pad=15)
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    plt.grid(axis='y', alpha=0.75)
    #plt.grid(axis='y', which='both', alpha=0.75)
    plt.savefig(f'{figdir}/maxMLDhist_{regionNameShort}.png', bbox_inches='tight')
    plt.close()

    # Now compute monthly climatologies associated with these composites and plot them
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
                #outfileMed = f'{outdir}/{varname}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDmed_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'
            else:
                outfileLow  = f'{outdir}/{varname}_maxMLDlow_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'
                outfileHigh = f'{outdir}/{varname}_maxMLDhigh_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'
                #outfileMed = f'{outdir}/{varname}_maxMLDmed_{titleClimoMonths}_{regionNameShort}_M{im:02d}.nc'

            if not os.path.isfile(outfileLow):
                print(f'\nComposite file {outfileLow} does not exist. Creating it with ncea...')
                infiles = []
                for k in range(len(years_low)):
                    iy = years_low[k]
                    if im > np.max(climoMonths) and iy != startSimYear:
                        iy = iy-1  # pick months *preceding* the climoMonths period of each year
                    if modelComp == 'atm':
                        if isSingleVarFiles:
                            datafile = f'{rundir}/{varname}.{runName}.{modelName}.h0.{int(iy):04d}-{int(im):02d}.nc'
                        else:
                            datafile = f'{rundir}/{runName}.{modelName}.h0.{int(iy):04d}-{int(im):02d}.nc'
                    else:
                        if isSingleVarFiles:
                            datafile = f'{rundir}/{varname}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            if varname=='velocityZonalDepthAvg' or varname=='velocityMeridionalDepthAvg' or \
                               varname=='activeTracers_temperatureDepthAvg' or varname=='activeTracers_salinityDepthAvg':
                                thicknessfile = f'{rundir}/layerThickness.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                        else:
                            datafile = f'{rundir}/{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            if varname=='velocityZonalDepthAvg' or varname=='velocityMeridionalDepthAvg' or \
                               varname=='activeTracers_temperatureDepthAvg' or varname=='activeTracers_salinityDepthAvg':
                                thicknessfile = datafile
                    # Check if file exists
                    if not os.path.isfile(datafile):
                        raise SystemExit(f'File {datafile} not found. Exiting...\n')
                    # Compute complex variables before making composites
                    if varname=='velocityZonalDepthAvg' or varname=='velocityMeridionalDepthAvg' or \
                       varname=='activeTracers_temperatureDepthAvg' or varname=='activeTracers_salinityDepthAvg':
                        layerThickness = xr.open_dataset(thicknessfile).timeMonthly_avg_layerThickness
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

                    infiles.append(datafile)
                args = ['ncea', '-O', '-v', varmpasname]
                args.extend(infiles)
                args.append(outfileLow)
                subprocess.check_call(args)
            if not os.path.isfile(outfileHigh):
                print(f'\nComposite file {outfileHigh} does not exist. Creating it with ncea...')
                infiles = []
                for k in range(len(years_high)):
                    iy = years_high[k]
                    if im > np.max(climoMonths) and iy != startSimYear:
                        iy = iy-1  # pick months *preceding* the climoMonths period of each year
                    if modelComp == 'atm':
                        if isSingleVarFiles:
                            datafile = f'{rundir}/{varname}.{runName}.{modelName}.h0.{int(iy):04d}-{int(im):02d}.nc'
                        else:
                            datafile = f'{rundir}/{runName}.{modelName}.h0.{int(iy):04d}-{int(im):02d}.nc'
                    else:
                        if isSingleVarFiles:
                            datafile = f'{rundir}/{varname}.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            if varname=='velocityZonalDepthAvg' or varname=='velocityMeridionalDepthAvg' or \
                               varname=='activeTracers_temperatureDepthAvg' or varname=='activeTracers_salinityDepthAvg':
                                thicknessfile = f'{rundir}/layerThickness.{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                        else:
                            datafile = f'{rundir}/{runName}.{modelName}.hist.am.{mpasFile}.{int(iy):04d}-{int(im):02d}-01.nc'
                            if varname=='velocityZonalDepthAvg' or varname=='velocityMeridionalDepthAvg' or \
                               varname=='activeTracers_temperatureDepthAvg' or varname=='activeTracers_salinityDepthAvg':
                                thicknessfile = datafile
                    # Check if file exists
                    if not os.path.isfile(datafile):
                        raise SystemExit(f'File {datafile} not found. Exiting...\n')
                    # Compute complex variables before making composites
                    if varname=='velocityZonalDepthAvg' or varname=='velocityMeridionalDepthAvg' or \
                       varname=='activeTracers_temperatureDepthAvg' or varname=='activeTracers_salinityDepthAvg':
                        layerThickness = xr.open_dataset(thicknessfile).timeMonthly_avg_layerThickness
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

                    infiles.append(datafile)
                args = ['ncea', '-O', '-v', varmpasname]
                args.extend(infiles)
                args.append(outfileHigh)
                subprocess.check_call(args)
