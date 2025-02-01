from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from scipy.signal import detrend

from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf_with_fill
#from mpas_analysis.shared.io import open_mpas_dataset, write_netcdf
from mpas_analysis.shared.io.utility import get_files_year_month, decode_strings

from geometric_features import FeatureCollection, read_feature_collection

from common_functions import timeseries_analysis_plot, add_inset, days_to_datetime

# Settings for nersc
regionMaskDir = '/global/cfs/cdirs/m1199/milena/mpas-region_masks'
ensembleName = 'E3SM-Arcticv2.1_historical'
ensembleMemberNames = ['0101', '0151', '0201', '0251', '0301']
colors = ['mediumblue', 'dodgerblue', 'deepskyblue', 'lightseagreen', 'green']
 
startYear = 1950
endYear = 2014
calendar = 'gregorian'

# Settings for regional time series
tsdir = './timeseries_data'
varname = 'maxMLD'
regionGroup = 'Arctic Regions'
regionName = 'Greenland Sea'
#regionGroup = 'arctic_atlantic_budget_regions_new20240408'
#regionGroup = 'OceanOHC Regions'
#regionGroup = 'Antarctic Regions'
transectName = None
groupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')

# Settings for transect time series
#tsdir = './transports_data'
#varname = ''
#regionName = None
#transectName = ''

figdir = f'./timeseries/{ensembleName}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

movingAverageMonths = 1
monthsToPlot = [1, 2, 3, 4] # JFMA only (movingAverageMonths is changed to 1 later on)
titleMonthsToPlot = 'JFMA'

startDate = f'{startYear:04d}-01-01_00:00:00'
endDate = f'{endYear:04d}-12-31_23:59:59'
years = range(startYear, endYear + 1)

featureFile = f'{regionMaskDir}/{groupName}.geojson'
if os.path.exists(featureFile):
    fcAll = read_feature_collection(featureFile)
else:
    raise IOError('No feature file found for this region group')

figsize = (15, 5)
figdpi = 150
fontsize_smallLabels = 10
fontsize_labels = 16
fontsize_titles = 18
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
##############################################################

nEnsembles = len(ensembleMemberNames)
timeseries_seasonal = np.zeros((nEnsembles, len(years)))
for nEns in range(nEnsembles):
    ensembleMemberName = ensembleMemberNames[nEns]
    if regionName is not None:
        timeseriesDir = f'{tsdir}/{ensembleName}{ensembleMemberName}/{timeseriesVar}'
        timeseriesFiles = []
        for year in years:
            if varname=='maxMLD':
                timeseriesFiles.append(f'{timeseriesDir}/{groupName}_max_year{year:04d}.nc')
            else:
                timeseriesFiles.append(f'{timeseriesDir}/{groupName}_year{year:04d}.nc')
        ds = xr.open_mfdataset(timeseriesFiles, combine='nested',
                               concat_dim='Time', decode_times=False)
        regionNames = ds.regionNames[0].values
        regionIndex = np.where(regionNames==regionName)[0]
        timeseries = np.squeeze(ds[timeseriesVar].isel(nRegions=regionIndex).values)
    elif transectName is not None:
    else:
        raise ValueError('Both regionName and transectName are None')

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
    ensembleMean = np.nanmean(timeseries_seasonal, axis=0)
    ensembleTrend = ensembleMean - detrend(ensembleMean, type='linear')

    plt.plot(ds.Time.values, timeseries, colors[nEns], linewidth=1.5)
    plt.plot(ds.Time.values, ensembleTrend, 'k', linewidth=2.5)
    plt.grid(alpha=0.75)

******************************************************
            xLabel = 'Time (yr)'
            yLabel = f'{vartitle} ({varunits})'
            lineColors = ['k']
            lineWidths = [2.5]
            legendText = [runNameShort]
            if titleMonthsToPlot is None:
                title = f'1-year running mean {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'
            else:
                title = f'{titleMonthsToPlot} {vartitle} in {regionName} region\n{np.nanmean(field):5.2f} $\pm$ {np.nanstd(field):5.2f} {varunits}'
            figFileName = f'{figdir}/{regionNameShort}_{varname}_years{years[0]}-{years[-1]}.png'

            fig = timeseries_analysis_plot(field, movingAverageMonths,
                                           title, xLabel, yLabel,
                                           calendar=calendar,
                                           lineColors=lineColors,
                                           lineWidths=lineWidths,
                                           legendText=legendText)

            # do this before the inset because otherwise it moves the inset
            # and cartopy doesn't play too well with tight_layout anyway
            plt.tight_layout()

            if regionName!='Global':
                add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)

            plt.savefig(figFileName, dpi='figure', bbox_inches='tight', pad_inches=0.1)
            plt.close()
