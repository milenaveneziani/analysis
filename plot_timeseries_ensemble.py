from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from scipy.signal import detrend
import matplotlib.pyplot as plt

from geometric_features import read_feature_collection

from common_functions import add_inset

# Settings for nersc
regionMaskDir = '/global/cfs/cdirs/m1199/milena/mpas-region_masks'
ensembleName = 'E3SM-Arcticv2.1_historical'
ensembleMemberNames = ['0101', '0151', '0201', '0251', '0301']
colors = ['mediumblue', 'dodgerblue', 'deepskyblue', 'lightseagreen', 'green']
 
startYear = 1950
endYear = 2014
calendar = 'gregorian'

# Settings for regional time series
#tsdir = './timeseries_data'
#varname = 'maxMLD'
#vartitle = 'Max MLD'
#varunits = '[m]'
#varname = 'iceArea'
#vartitle = 'ice area'
#varunits = '[km$^2$]'
#regionName = 'Greenland Sea'
#regionGroup = 'Arctic Regions' # defines feature filename, as well as regional ts filenames
#regionGroupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')
#transectName = None

# Settings for transect time series
tsdir = './transports_data'
varname = 'FWTransportSref'
vartitle = 'FW (Sref=34.8 psu) transport'
varunits = '[mSv]'
transectName = 'Fram Strait'
transectGroup = 'Arctic Sections' # defines feature filename, as well as transport ts filenames
transectGroupName = transectGroup[0].lower() + transectGroup[1:].replace(' ', '')
regionName = None

figdir = f'./timeseries/{ensembleName}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

# This only makes sense if climoMonths is None:
movingAverageMonths = 12 # number of months over which to compute running average
#climoMonths = None
#climoMonths = [1, 2, 3, 4] # JFMA only
#titleClimoMonths = 'JFMA-avg'
climoMonths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # Annual average
titleClimoMonths = 'ANN-avg'

startDate = f'{startYear:04d}-01-01_00:00:00'
endDate = f'{endYear:04d}-12-31_23:59:59'
years = range(startYear, endYear + 1)
referenceDate = '0001-01-01'
calendar = 'gregorian'

#featureFile = f'{regionMaskDir}/{regionGroupName}.geojson'
#if os.path.exists(featureFile):
#    fcAll = read_feature_collection(featureFile)
#else:
#    raise IOError('No feature file found for this region group')

figsize = (15, 5)
figdpi = 150
fontsize_smallLabels = 16
fontsize_labels = 20
fontsize_titles = 22
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
##############################################################

if regionName is not None:
    regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '')
    figfile = f'{figdir}/{varname}_{regionNameShort}_years{years[0]}-{years[-1]}.png'
    if climoMonths is not None:
        figtitle = f'{vartitle} ({titleClimoMonths}) in {regionName} region'
    else:
        figtitle = f'{vartitle} ({int(movingAverageMonths/12)}-year running avg) in {regionName} region'
elif transectName is not None:
    transectNameShort = transectName[0].lower() + transectName[1:].replace(' ', '')
    figfile = f'{figdir}/{varname}_{transectNameShort}_years{years[0]}-{years[-1]}.png'
    if climoMonths is not None:
        figtitle = f'{vartitle} ({titleClimoMonths}) across {transectName}'
    else:
        figtitle = f'{vartitle} ({int(movingAverageMonths/12)}-year running avg) across {transectName}'
else:
    raise ValueError('Both regionName and transectName are None')

fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
ax.yaxis.get_offset_text().set_weight('bold')
ax.set_xlabel('Time (yr)', fontsize=fontsize_labels, fontweight='bold')
ax.set_ylabel(f'{varname} {varunits}', fontsize=fontsize_labels, fontweight='bold')
ax.set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
ax.set_xlim(years[0], years[-1])
plt.grid(alpha=0.75)

nEnsembles = len(ensembleMemberNames)
timeseries_seasonal = np.zeros((nEnsembles, len(years)))
# This is only used if climoMonths is None:
timeseries_ensemble = np.zeros((nEnsembles, len(years)*12))
for nEns in range(nEnsembles):
    ensembleMemberName = ensembleMemberNames[nEns]
    if regionName is not None:
        timeseriesDir = f'{tsdir}/{ensembleName}{ensembleMemberName}/{varname}'
        timeseriesFiles = []
        for year in years:
            if varname=='maxMLD':
                timeseriesFiles.append(f'{timeseriesDir}/{regionGroupName}_max_year{year:04d}.nc')
            else:
                timeseriesFiles.append(f'{timeseriesDir}/{regionGroupName}_year{year:04d}.nc')
        ds = xr.open_mfdataset(timeseriesFiles, combine='nested',
                               concat_dim='Time', decode_times=False)
        regionNames = ds.regionNames[0].values
        regionIndex = np.where(regionNames==regionName)[0]
        dsvar = ds[varname].isel(nRegions=regionIndex)
    elif transectName is not None:
        timeseriesDir = f'{tsdir}/{ensembleName}{ensembleMemberName}'
        timeseriesFiles = []
        for year in years:
            timeseriesFiles.append(f'{timeseriesDir}/{transectGroupName}Transports_{ensembleName}{ensembleMemberName}_year{year:04d}.nc')
        ds = xr.open_mfdataset(timeseriesFiles, combine='nested',
                               concat_dim='Time', decode_times=False)
        transectNames = ds.transectNames[0].values
        transectIndex = np.where(transectNames==transectName)[0]
        dsvar = ds[varname].isel(nTransects=transectIndex)

    if climoMonths is not None:
        if len(climoMonths)==12:
            # Compute and plot seasonal averages
            timeseries_seasonal[nEns, :] = np.squeeze(dsvar.groupby_bins('Time', len(years)).mean().rename({'Time_bins': 'Time'}).values)
        else:
            # Compute and plot seasonal averages
            #  (note: this approach only works for the regional
            #   time series, not for the transport time series)
            if regionName is not None:
                timeseries = np.squeeze(dsvar.values)
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
            else:
                raise RuntimeError('Seasonal averages not yet supported for transport time series')
        plt.plot(years, timeseries_seasonal[nEns, :], colors[nEns], linewidth=1.5, label=ensembleMemberName)
    else:
        if movingAverageMonths!=1:
            window = int(movingAverageMonths)
            timeseries_ensemble[nEns, :] = pd.Series(timeseries).rolling(window, center=True).mean()
        else:
            timeseries_ensemble[nEns, :] = timeseries

        plt.plot(ds.Time.values, timeseries_ensemble[nEns, :], colors[nEns], linewidth=1.5, label=ensembleMemberName)

#if climoMonths is not None:
#    ensembleMean = np.nanmean(timeseries_seasonal, axis=0)
#    ensembleTrend = ensembleMean - detrend(ensembleMean, type='linear')
#    plt.plot(years, ensembleTrend, 'k', linewidth=2.5, label='ensemble mean')
#else:
#    ensembleMean = np.nanmean(timeseries_ensemble, axis=0)
#    ensembleTrend = ensembleMean - detrend(ensembleMean, type='linear')
#    plt.plot(ds.Time.values, ensembleTrend, 'k', linewidth=2.5, label='ensemble mean')

ax.legend(prop=legend_properties)

# do this before the inset because otherwise it moves the inset
# and cartopy doesn't play too well with tight_layout anyway
plt.tight_layout()

#if regionName!='Global':
#    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)

plt.savefig(figfile, dpi='figure', bbox_inches='tight', pad_inches=0.1)
plt.close()
