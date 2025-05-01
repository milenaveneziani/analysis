from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from scipy.signal import detrend
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')

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
regionName = 'Greenland Sea'
#regionName = 'Norwegian Sea'
regionGroup = 'Arctic Regions' # defines feature filename, as well as regional ts filenames
regionGroupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')

# Settings for transect time series
transectName = 'Iceland-Faroe-Scotland'
transectGroup = 'Arctic Sections' # defines feature filename, as well as transport ts filenames
transectGroupName = transectGroup[0].lower() + transectGroup[1:].replace(' ', '')

figdir = f'./timeseries/{ensembleName}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

startDate = f'{startYear:04d}-01-01_00:00:00'
endDate = f'{endYear:04d}-12-31_23:59:59'
years = range(startYear, endYear + 1)
referenceDate = '0001-01-01'
calendar = 'gregorian'

figsize = (14, 7)
figdpi = 150
fontsize_smallLabels = 10
fontsize_labels = 12
fontsize_titles = 14
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
##############################################################

climoMonths = [1, 2, 3, 4] # JFMA only
regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '')
transectNameShort = transectName[0].lower() + transectName[1:].replace(' ', '')

figfile = f'{figdir}/spice0{transectNameShort}_maxMLD_corr_nordicSeaPaper.png'
figtitle = f'Lagged correlations between spiciness0 across\n {transectName} and GS maxMLD (JFMA-avg values)'
indirRegion  = './timeseries_data'
indirTransect = './transports_data'

#lags = np.arange(-10, 11, dtype=np.int16)
lags = np.arange(0, 11, dtype=np.int16)

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
ax.set_xlabel('Lag (yr)', fontsize=fontsize_labels, fontweight='bold')
#ax.set_xlabel('Time (yr)', fontsize=fontsize_labels, fontweight='bold')
#ax.set_ylabel('spice []', fontsize=fontsize_labels, fontweight='bold')
ax.set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
ax.set_xticks(lags)
original_labels = [str(label) for label in ax.get_xticks()]
labels_of_interest = [str(i) for i in lags[0::2]]
new_labels = [label if label in labels_of_interest else "" for label in original_labels]
ax.set_xticklabels(new_labels)
ax.set_xlim(lags[0], lags[-1])
ax.axhline(y=0, color='k', linestyle='-')

nEnsembles = len(ensembleMemberNames)
maxMLD_seasonal  = np.zeros((nEnsembles, len(years)))
spice_seasonal = np.zeros((nEnsembles, len(years)))
#corr = np.empty((nEnsembles, len(lags)))
#pval = np.empty((nEnsembles, len(lags)))
for nEns in range(nEnsembles):
    ensembleMemberName = ensembleMemberNames[nEns]
    timeseriesDir1 = f'{indirRegion}/{ensembleName}{ensembleMemberName}/maxMLD'
    timeseriesDir2 = f'{indirTransect}/{ensembleName}{ensembleMemberName}'
    timeseriesFiles1 = []
    timeseriesFiles2 = []
    for year in years:
        timeseriesFiles1.append(f'{timeseriesDir1}/{regionGroupName}_max_year{year:04d}.nc')
        timeseriesFiles2.append(f'{timeseriesDir2}/{transectGroupName}Transports_z0000-0500_{ensembleName}{ensembleMemberName}_year{year:04d}.nc')
    ds1 = xr.open_mfdataset(timeseriesFiles1, combine='nested',
                            concat_dim='Time', decode_times=False)
    ds2 = xr.open_mfdataset(timeseriesFiles2, combine='nested',
                            concat_dim='Time', decode_times=False)
    regionNames = ds1.regionNames[0].values
    regionIndex = np.where(regionNames==regionName)[0]
    dsvar1 = ds1['maxMLD'].isel(nRegions=regionIndex)
    transectNames = ds2.transectNames[0].values
    transectIndex = np.where(transectNames==transectName)[0]
    dsvar2 = ds2['spiceTransect'].isel(nTransects=transectIndex)

    # Compute and plot seasonal averages
    #  (note: this approach only works for the regional
    #   time series, not for the transport time series)
    maxMLD = np.squeeze(dsvar1.values)
    spice = np.squeeze(dsvar2.values)
    datetimes = netCDF4.num2date(ds1.Time, f'days since {referenceDate}', calendar=calendar)
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
        maxMLD_seasonal[nEns, iy] = np.nanmean(maxMLD[mask])
        spice_seasonal[nEns, iy] = np.nanmean(spice[mask])

    corr = []
    pval = []
    #conf = []
    # for debugging:
    #print(spice_seasonal[nEns, :])
    #print(maxMLD_seasonal[nEns, :])
    for lag in lags:
        if lag<0:
            # for debugging:
            #print(spice_seasonal[nEns, -1:-lag-1:-1])
            #print(maxMLD_seasonal[nEns, -1+lag::-1])
            # correlate flipped a(t) with flipped b(t-tau), considering that python does 
            # *not* include last element of array (-lag-1 for spice_seasonal, for example):
            pearson_corr = stats.pearsonr(spice_seasonal[nEns, -1:-lag-1:-1], maxMLD_seasonal[nEns, -1+lag::-1])
        if lag==0:
            # correlate a(t) with b(t)
            pearson_corr = stats.pearsonr(spice_seasonal[nEns, :], maxMLD_seasonal[nEns, :])
        if lag>0:
            # for debugging:
            #print(spice_seasonal[nEns, 0:-lag])
            #print(maxMLD_seasonal[nEns, lag::])
            # correlate a(t) with v(t+tau)
            pearson_corr = stats.pearsonr(spice_seasonal[nEns, 0:-lag], maxMLD_seasonal[nEns, lag::])
        corr = np.append(corr, pearson_corr.statistic)
        pval = np.append(pval, pearson_corr.pvalue)
        #conf = np.append(conf, pearson_corr.confidence_interval(confidence_level=0.95))
    #print(corr)
    #print(pval)
    sigValues = np.where(pval < .01) # choose pvalue<1%
    insigValues = np.where(pval >= .01)
    #ax.plot(years, spice_seasonal[nEns, :], colors[nEns], marker='o', linewidth=1.5, label=f'{ensembleMemberName}, r={corr.statistic:5.2f}')
    #ax.scatter(maxMLD_seasonal[nEns, :], spice_seasonal[nEns, :], s=5, c=colors[nEns], marker='o', label=ensembleMemberName)
    ax.plot(lags, corr, colors[nEns], linewidth=1, label=ensembleMemberName)
    ax.scatter(lags[sigValues], corr[sigValues], s=20, c=colors[nEns], marker='o')
    ax.scatter(lags[insigValues], corr[insigValues], s=20, c=colors[nEns], marker='o', alpha=0.3)

ax.grid(visible=True, which='both')
ax.legend(prop=legend_properties)
fig.savefig(figfile, dpi='figure', bbox_inches='tight', pad_inches=0.1)
#plt.show()
