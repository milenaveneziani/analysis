from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from common_functions import timeseries_analysis_plot

startYear = 1950
#startYear = 2013
endYear = 2014
#startYear = 1
#endYear = 1
#endYear = 50
#endYear = 246 # rA07
#endYear = 386 # rA02
years = range(startYear, endYear + 1)
calendar = 'gregorian_noleap'

# Settings for nersc
runName = 'E3SM-Arcticv2.1_historical0301'
runNameShort = 'E3SMv2.1-Arctic-historical0301'
rundir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runName}'
isShortTermArchive = True
if isShortTermArchive:
    if runName=='E3SMv2.1B60to10rA07':
        rundir = f'{rundir}/atm/hist'
    else:
        rundir = f'{rundir}/archive/atm/hist'
 
# Quantities needed for plotting only
#movingAverageMonths = 12 # use this for monthly fields only
movingAverageMonths = 1
monthsToPlot = range(1, 13)

#regionName = 'Beaufort Sea larger' # Siobhan's version
#lonmin = 180
#lonmax = -115 + 360
#latmin = 66
#latmax = 82
#regionName = 'Nordic Seas west'
#lonmin = -20 + 360 # approximately Greenland coast
#lonmax = 360
#latmin = 69 # only includes half of the Iceland Sea
#latmax = 79 # Fram Strait
regionName = 'Nordic Seas east'
lonmin = 0
lonmax = 17.5 # includes central Norwegian coast
latmin = 62.5
latmax = 79 # Fram Strait

atmfile = 'h1'
variables = [
             {'name': 'U10',
              'title': 'wind magnitude at 10 m',
              'units': 'm/s',
              'factor': 1},
             {'name': 'UBOT',
              'title': 'zonal wind at lowest model level',
              'units': 'm/s',
              'factor': 1},
             {'name': 'VBOT',
              'title': 'meridional wind at lowest model level',
              'units': 'm/s',
              'factor': 1},
            ]

outdir = f'./timeseries_data/{runName}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
figdir = f'./timeseries/{runName}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '')
if atmfile=='h0': # monthly averages 
    outfile = f'{regionNameShort}_atm_'
if atmfile=='h1': # daily averages
    outfile = f'{regionNameShort}_atm_daily_'
if atmfile=='h2': # 6-hourly averages
    outfile = f'{regionNameShort}_atm_6hourly_'

for var in variables:
    varname = var['name']
    varfactor = var['factor']
    varunits = var['units']
    vartitle = var['title']

    outdirvar = f'{outdir}/{varname}'
    if not os.path.isdir(outdirvar):
        os.makedirs(outdirvar) 

    print('')
    for year in years:

        timeSeriesFile = f'{outdirvar}/{outfile}year{year:04d}.nc'

        if not os.path.exists(timeSeriesFile):
            print(f'Processing variable = {vartitle},  year={year}')
            # Load in yearly data set for chosen variable
            #datasets = []
            #for month in range(1, 13):
            #    inputFile = glob.glob(f'{rundir}/{runName}.eam.{atmfile}.{year:04d}-{month:02d}*.nc')[0]
            #    if not os.path.exists(inputFile):
            #        raise IOError(f'Input file: {inputFile} not found')

            #    ds = xr.open_dataset(inputFile, decode_cf=True, decode_times=False, lock=False)
            #    dsSlice = ds[varname]
            #    lon = ds['lon']
            #    lat = ds['lat']
            #    area = ds['area']
            #    regionMask = np.logical_and(lon>=lonmin, lon<=lonmax)
            #    regionMask = np.logical_and(lat>=latmin, lat<=latmax)
            #    localArea = area.where(regionMask, drop=True)
            #    regionalArea = localArea.sum()
            #    datasets.append(dsSlice)
            ## combine data sets into a single data set
            #dsIn = xr.concat(datasets, 'time')
            #dsOut = (localArea*dsIn.where(regionMask, drop=True)).sum(dim='ncol') / regionalArea

            inputFiles = glob.glob(f'{rundir}/{runName}.eam.{atmfile}.{year:04d}-*.nc')
            ds = xr.open_mfdataset(inputFiles, decode_cf=True, decode_times=False, lock=False)
            #fld = ds[[varname]]
            fld = ds[varname]
            lon = ds['lon'].isel(time=0, drop=True) # remove time dimension
            lat = ds['lat'].isel(time=0, drop=True) # remove time dimension
            area = ds[['area']].isel(time=0, drop=True) # remove time dimension
            time = ds['time']
            # This is needed for not getting into the "RuntimeError: 
            # Resource temporarily unavailable" problem!:
            fld.load()
            lon.load()
            lat.load()
            area.load()
            time.load()
            localArea = (area.where(lon>=lonmin)).where(lon<=lonmax)
            localArea = (localArea.where(lat>=latmin)).where(lat<=latmax)
            regionalArea = localArea.sum()
            #dsIn = xr.Dataset(data_vars={varname: (('time', 'ncol'), fld.values),
            #                             'Time': (('time'), time.values),},
            dsIn = xr.Dataset(data_vars={varname: (('time', 'ncol'), fld.values)},
                              coords={'lon': (('ncol'), lon.values),
                                      'lat': (('ncol'), lat.values)},
                              )
            fld_masked = (dsIn[varname].where(lon>=lonmin)).where(lon<=lonmax)
            fld_masked = (fld_masked.where(lat>=latmin)).where(lat<=latmax)

            dsOut = (localArea*fld_masked).sum(dim='ncol') / regionalArea
            dsOut = varfactor * dsOut
            dsOut = dsOut.rename_vars({'area': varname})
            dsOut[varname].attrs['units'] = varunits
            dsOut[varname].attrs['description'] = vartitle
            dsOut['time'] = time.values
            dsOut['time'].attrs['units'] = time.attrs['units']
            dsOut['time'].attrs['long_name'] = time.attrs['long_name']
            dsOut['time'].attrs['calendar'] = time.attrs['calendar']
            dsOut['regionName'] = regionName
            dsOut.to_netcdf(timeSeriesFile)
        else:
            print(f'Time series file already exists for {varname} and year {year}. Skipping it...')

    # Time series calculated ==> make plots
    print(f'\n  now plot {varname} for each region\n')
    timeSeriesFiles = []
    for year in years:
        timeSeriesFile = f'{outdirvar}/{outfile}year{year:04d}.nc'
        timeSeriesFiles.append(timeSeriesFile)

    dsIn = xr.open_mfdataset(timeSeriesFiles, combine='nested',
                             concat_dim='time', decode_times=False)

    field = [dsIn[varname]]
    xLabel = 'Time (yr)'
    yLabel = f'{vartitle} ({varunits})'
    lineColors = ['k']
    lineWidths = [2.5]
    legendText = [runNameShort]
    if movingAverageMonths==1:
        if atmfile=='h0':
            title = f'Monthly {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'
        if atmfile=='h1':
            title = f'Daily {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'
        if atmfile=='h2':
            title = f'6-hourly {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'
    else:
        movingAverageYears = movingAverageMonths/12
        title = f'{movingAverageYears}-year running mean {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'

    if atmfile=='h0':
        figFileName = f'{figdir}/{regionNameShort}_{varname}_years{years[0]}-{years[-1]}.png'
    if atmfile=='h1':
        figFileName = f'{figdir}/{regionNameShort}_{varname}_years{years[0]}-{years[-1]}_daily.png'
    if atmfile=='h2':
        figFileName = f'{figdir}/{regionNameShort}_{varname}_years{years[0]}-{years[-1]}_6hourly.png'

    fig = timeseries_analysis_plot(field, movingAverageMonths,
                                   title, xLabel, yLabel,
                                   calendar=calendar,
                                   timevarname='time',
                                   lineColors=lineColors,
                                   lineWidths=lineWidths,
                                   legendText=legendText)

    plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
    plt.close()
