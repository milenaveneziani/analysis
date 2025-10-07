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

# Settings for regional time series 1 (maxMLD)
regionName1 = 'Greenland Sea'
regionGroup1 = 'Arctic Regions'
regionGroupName1 = regionGroup1[0].lower() + regionGroup1[1:].replace(' ', '')

# Settings for regional time series 2 (winds)
regionName2 = 'Nordic Seas west'
regionGroup2 = 'nordicSeaswest' # defines feature filename, as well as regional ts filenames
regionGroupName2 = regionGroup2[0].lower() + regionGroup2[1:].replace(' ', '')

figdir = f'./timeseries/{ensembleName}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

startDate = f'{startYear:04d}-01-01_00:00:00'
endDate = f'{endYear:04d}-12-31_23:59:59'
years = range(startYear, endYear + 1)
referenceDate1 = '0001-01-01' # for MPAS fields
referenceDate2 = '1950-01-01' # for atm fields
calendar = 'gregorian'

figsize = (14, 7)
figdpi = 150
fontsize_smallLabels = 10
fontsize_labels = 10
fontsize_titles = 12
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
##############################################################

climoMonths = [1, 2, 3, 4] # JFMA only (for regional maxMLD)
regionNameShort1 = regionName1[0].lower() + regionName1[1:].replace(' ', '')
regionNameShort2 = regionName2[0].lower() + regionName2[1:].replace(' ', '')

figfile = f'{figdir}/VBOT{regionNameShort2}_maxMLD{regionNameShort1}_corr_nordicSeaPaper.png'
figtitle = f'Lagged correlations between JFMA {regionName1} VBOT and {regionName2} maxMLD'
indirRegion  = './timeseries_data'

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
vbot_seasonal = np.zeros((nEnsembles, len(years)))
u10_seasonal = np.zeros((nEnsembles, len(years)))
#corr = np.empty((nEnsembles, len(lags)))
#pval = np.empty((nEnsembles, len(lags)))
for nEns in np.arange(nEnsembles):
    ensembleMemberName = ensembleMemberNames[nEns]
    timeseriesDir1 = f'{indirRegion}/{ensembleName}{ensembleMemberName}/maxMLD'
    timeseriesDir2 = f'{indirRegion}/{ensembleName}{ensembleMemberName}/VBOT'
    timeseriesDir3 = f'{indirRegion}/{ensembleName}{ensembleMemberName}/U10'
    timeseriesFiles1 = []
    timeseriesFiles2 = []
    timeseriesFiles3 = []
    for year in years:
        timeseriesFiles1.append(f'{timeseriesDir1}/{regionGroupName1}_max_year{year:04d}.nc')
        timeseriesFiles2.append(f'{timeseriesDir2}/{regionGroupName2}_atm_daily_year{year:04d}.nc')
        timeseriesFiles3.append(f'{timeseriesDir3}/{regionGroupName2}_atm_daily_year{year:04d}.nc')
    ds1 = xr.open_mfdataset(timeseriesFiles1, combine='nested',
                            concat_dim='Time', decode_times=False)
    ds2 = xr.open_mfdataset(timeseriesFiles2, combine='nested',
                            concat_dim='time', decode_times=False)
    ds3 = xr.open_mfdataset(timeseriesFiles3, combine='nested',
                            concat_dim='time', decode_times=False)
    regionNames = ds1.regionNames[0].values
    regionIndex = np.where(regionNames==regionName1)[0]
    dsvar1 = ds1['maxMLD'].isel(nRegions=regionIndex)
    dsvar2 = ds2['VBOT']
    dsvar3 = ds3['U10']

    # Compute and plot seasonal averages
    #  (note: this approach only works for the regional
    #   time series, not for the transport time series)
    maxMLD = np.squeeze(dsvar1.values)
    vbot = np.squeeze(dsvar2.values)
    u10 = np.squeeze(dsvar3.values)
    datetimes_ocn = netCDF4.num2date(ds1.Time, f'days since {referenceDate1}', calendar=calendar)
    datetimes_atm = netCDF4.num2date(ds2.time, f'days since {referenceDate2}', calendar=calendar)
    timeyears_ocn = []
    timemonths_ocn = []
    for date in datetimes_ocn.flat:
        timeyears_ocn.append(date.year)
        timemonths_ocn.append(date.month)
    timeyears_atm = []
    timemonths_atm = []
    for date in datetimes_atm.flat:
        timeyears_atm.append(date.year)
        timemonths_atm.append(date.month)
    #annualmask = [i for i, x in enumerate(timemonths) if x in set(range(1, 13))] # all months
    monthmask_ocn = [i for i, x in enumerate(timemonths_ocn) if x in set(climoMonths)] # winter months only
    monthmask_atm = [i for i, x in enumerate(timemonths_atm) if x in set(climoMonths)] # winter months only

    for iy, year in enumerate(years):
        yearmask_ocn = [i for i, x in enumerate(timeyears_ocn) if x==year]
        yearmask_atm = [i for i, x in enumerate(timeyears_atm) if x==year]
        mask_ocn = np.intersect1d(yearmask_ocn, monthmask_ocn)
        mask_atm = np.intersect1d(yearmask_atm, monthmask_atm)
        if np.size(mask_ocn)==0:
           raise ValueError('Something is wrong with time mask for MPAS fields')
        if np.size(mask_atm)==0:
           raise ValueError('Something is wrong with time mask for atm fields')
        #print('year=  ', year)
        #print(mask_atm)
        #print(vbot[mask_atm])
        #print(np.nanmean(vbot[mask_atm]))
        #print('')
        maxMLD_seasonal[nEns, iy] = np.nanmean(maxMLD[mask_ocn])
        vbot_seasonal[nEns, iy] = np.nanmean(vbot[mask_atm])
        u10_seasonal[nEns, iy] = np.nanmean(u10[mask_atm])

    corr = []
    pval = []
    #conf = []
    for lag in lags:
        if lag<0:
            # correlate flipped a(t) with flipped b(t-tau), considering that python does 
            # *not* include last element of array (-lag-1 for spice_annual, for example):
            pearson_corr = stats.pearsonr(vbot_seasonal[nEns, -1:-lag-1:-1], maxMLD_seasonal[nEns, -1+lag::-1])
        if lag==0:
            # correlate a(t) with b(t)
            pearson_corr = stats.pearsonr(vbot_seasonal[nEns, :], maxMLD_seasonal[nEns, :])
        if lag>0:
            # correlate a(t) with v(t+tau)
            pearson_corr = stats.pearsonr(vbot_seasonal[nEns, 0:-lag], maxMLD_seasonal[nEns, lag::])
        corr = np.append(corr, pearson_corr.statistic)
        pval = np.append(pval, pearson_corr.pvalue)
        #conf = np.append(conf, pearson_corr.confidence_interval(confidence_level=0.95))

    #print(pval)
    sigValues = np.where(pval < .01) # choose pvalue<1%
    insigValues = np.where(pval >= .01)
    ax.plot(lags, corr, colors[nEns], linewidth=1, label=ensembleMemberName)
    ax.scatter(lags[sigValues], corr[sigValues], s=20, c=colors[nEns], marker='o')
    ax.scatter(lags[insigValues], corr[insigValues], s=20, c=colors[nEns], marker='o', alpha=0.3)

ax.grid(visible=True, which='both')
ax.legend(prop=legend_properties)
fig.savefig(figfile, dpi='figure', bbox_inches='tight', pad_inches=0.1)

########################################################################

maxMLD_flat = maxMLD_seasonal.flatten()
maxMLDLC = np.quantile(maxMLD_flat, 0.15)
maxMLDHC = np.quantile(maxMLD_flat, 0.85)
vbot_flat = vbot_seasonal.flatten()
vbotLow = np.quantile(vbot_flat, 0.15)
vbotHigh= np.quantile(vbot_flat, 0.85)
u10_flat = u10_seasonal.flatten()
u10Low = np.quantile(u10_flat, 0.15)
u10High= np.quantile(u10_flat, 0.85)
indMLDLow = np.less_equal(maxMLD_flat, maxMLDLC)
indMLDHigh=np.greater_equal(maxMLD_flat, maxMLDHC)
indvbotLow = np.less_equal(vbot_flat, vbotLow)
indvbotHigh= np.greater_equal(vbot_flat, vbotHigh)
indu10Low = np.less_equal(u10_flat, u10Low)
indu10High= np.greater_equal(u10_flat, u10High)

percentage = 100 * np.size(np.where(np.logical_and(indvbotLow, indMLDHigh))) / np.size(np.where(indMLDHigh))
print(f'\nPercentage of HC years for the {regionName1} also associated with high northerly winds for the {regionName2}: {percentage}')
percentage = 100 * np.size(np.where(np.logical_and(indvbotHigh,  indMLDLow))) / np.size(np.where(indMLDLow))
print(f'Percentage of LC years for the {regionName1} also associated with high southerly winds for the {regionName2}: {percentage}\n')

percentage = 100 * np.size(np.where(np.logical_and(indu10High, indMLDHigh))) / np.size(np.where(indMLDHigh))
print(f'Percentage of HC years for the {regionName1} also associated with high wind amplitude for the {regionName2}: {percentage}')
percentage = 100 * np.size(np.where(np.logical_and(indu10Low,  indMLDLow))) / np.size(np.where(indMLDLow))
print(f'Percentage of LC years for the {regionName1} also associated with low wind amplitude for the {regionName2}: {percentage}\n')

figdpi = 300
figsize = (20, 5)
# Make scatter plot for a range of lags
for lag in np.arange(0, 7, dtype=np.int16):
    figfile = f'{figdir}/{regionNameShort2}_maxMLD{regionNameShort1}_scatterLag{lag:02d}.png'
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    for i in np.arange(2):
        for tick in ax[i].xaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        for tick in ax[i].yaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        ax[i].yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
        ax[i].yaxis.get_offset_text().set_weight('bold')
    ax[0].set_ylabel('JFMA maxMLD', fontsize=fontsize_labels, fontweight='bold')
    ax[0].set_xlabel('JFMA meridional wind (VBOT)', fontsize=fontsize_labels, fontweight='bold')
    figtitle = f'lag={lag:02d}-year'
    ax[0].set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
    ax[1].set_xlabel('JFMA wind amplitude at 10 m', fontsize=fontsize_labels, fontweight='bold')
    figtitle = f'lag={lag:02d}-year'
    ax[1].set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')

    if lag==0:
        fld0 = vbot_seasonal.flatten()
        fld1 = u10_seasonal.flatten()
        mld = maxMLD_seasonal.flatten()
    if lag>0:
        fld0 = vbot_seasonal[:, 0:-lag].flatten()
        fld1 = u10_seasonal[:, 0:-lag].flatten()
        mld = maxMLD_seasonal[:, lag::].flatten()
    if lag<0:
        fld0 = vbot_seasonal[:, -1:-lag-1:-1].flatten()
        fld1 = u10_seasonal[:, -1:-lag-1:-1].flatten()
        mld = maxMLD_seasonal[:, -1+lag::-1].flatten()

    ind_maxMLDLC = np.less_equal(mld, maxMLDLC)
    ind_maxMLDHC = np.greater_equal(mld, maxMLDHC)

    pearson_corr = stats.pearsonr(fld0, mld)
    corr = pearson_corr.statistic
    pval = pearson_corr.pvalue
    #print(corr, pval)
    ax[0].scatter(fld0, mld, s=20, c='k', marker='d', label=f'r={corr:5.2f}')
    ax[0].scatter(fld0[ind_maxMLDLC], mld[ind_maxMLDLC], s=20, c='b', marker='d')
    ax[0].scatter(fld0[ind_maxMLDHC], mld[ind_maxMLDHC], s=20, c='r', marker='d')
    ax[0].grid(visible=True, which='both')
    ax[0].legend(prop=legend_properties)

    pearson_corr = stats.pearsonr(fld1, mld)
    corr = pearson_corr.statistic
    pval = pearson_corr.pvalue
    #print(corr, pval)
    ax[1].scatter(fld1, mld, s=20, c='k', marker='d', label=f'r={corr:5.2f}')
    ax[1].scatter(fld1[ind_maxMLDLC], mld[ind_maxMLDLC], s=20, c='b', marker='d')
    ax[1].scatter(fld1[ind_maxMLDHC], mld[ind_maxMLDHC], s=20, c='r', marker='d')
    ax[1].grid(visible=True, which='both')
    ax[1].legend(prop=legend_properties)

    fig.suptitle(f'Scatter plot between JFMA-maxMLD in the {regionName1} and\n wind field quantities in the {regionName2}', fontsize=12, fontweight='bold', y=1.04)
    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight', pad_inches=0.1)

#plt.show()
