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
#regionName = 'Greenland Sea'
#regionName = 'Norwegian Sea'
#regionGroup = 'Arctic Regions' # defines feature filename, as well as regional ts filenames
regionName = 'Greenland Sea Interior'
regionName = 'Norwegian Sea new'
regionGroup = 'ginSeas_new' # defines feature filename, as well as regional ts filenames
regionGroupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')

# Settings for transect time series
transectName = 'Iceland-Faroe-Scotland'
transectGroup = 'Arctic Subarctic Sections' # defines feature filename, as well as transport ts filenames
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
fontsize_labels = 10
fontsize_titles = 12
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
##############################################################

climoMonths = [1, 2, 3, 4] # JFMA only (for regional maxMLD)
regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '')
transectNameShort = transectName[0].lower() + transectName[1:].replace(' ', '')

figfile = f'{figdir}/spice0{transectNameShort}_maxMLD{regionNameShort}_corr_nordicSeaPaper.png'
figtitle = f'Lagged correlations between annual spiciness0 across\n {transectName} and {regionName} JFMA maxMLD'
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
spice_annual = np.zeros((nEnsembles, len(years)))
volT_annual = np.zeros((nEnsembles, len(years)))
heatT_annual = np.zeros((nEnsembles, len(years)))
fwT_annual = np.zeros((nEnsembles, len(years)))
#corr = np.empty((nEnsembles, len(lags)))
#pval = np.empty((nEnsembles, len(lags)))
for nEns in np.arange(nEnsembles):
    ensembleMemberName = ensembleMemberNames[nEns]
    timeseriesDir1 = f'{indirRegion}/{ensembleName}{ensembleMemberName}/maxMLD'
    timeseriesDir2 = f'{indirTransect}/{ensembleName}{ensembleMemberName}'
    timeseriesFiles1 = []
    timeseriesFiles2 = []
    for year in years:
        timeseriesFiles1.append(f'{timeseriesDir1}/{regionGroupName}_max_year{year:04d}.nc')
        timeseriesFiles2.append(f'{timeseriesDir2}/{transectGroupName}Transports_z0000-0400_{ensembleName}{ensembleMemberName}_year{year:04d}.nc')
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
    dsvar3 = ds2['volTransport'].isel(nTransects=transectIndex)
    dsvar4 = ds2['heatTransport'].isel(nTransects=transectIndex)
    dsvar5 = ds2['FWTransportSref'].isel(nTransects=transectIndex)

    # Compute and plot seasonal averages
    #  (note: this approach only works for the regional
    #   time series, not for the transport time series)
    maxMLD = np.squeeze(dsvar1.values)
    spice = np.squeeze(dsvar2.values)
    volT = np.squeeze(dsvar3.values)
    heatT = np.squeeze(dsvar4.values)
    fwT = np.squeeze(dsvar5.values)
    datetimes = netCDF4.num2date(ds1.Time, f'days since {referenceDate}', calendar=calendar)
    timeyears = []
    timemonths = []
    for date in datetimes.flat:
        timeyears.append(date.year)
        timemonths.append(date.month)
    monthmask = [i for i, x in enumerate(timemonths) if x in set(climoMonths)] # winter months only
    annualmask = [i for i, x in enumerate(timemonths) if x in set(range(1, 13))] # all months

    for iy, year in enumerate(years):
        yearmask = [i for i, x in enumerate(timeyears) if x==year]
        mask1 = np.intersect1d(yearmask, monthmask)
        mask2 = np.intersect1d(yearmask, annualmask)
        if np.size(mask1)==0:
           raise ValueError('Something is wrong with time mask1')
        if np.size(mask2)==0:
           raise ValueError('Something is wrong with time mask2')
        maxMLD_seasonal[nEns, iy] = np.nanmean(maxMLD[mask1])
        spice_annual[nEns, iy] = np.nanmean(spice[mask2])
        volT_annual[nEns, iy] = np.nanmean(volT[mask2])
        heatT_annual[nEns, iy] = np.nanmean(heatT[mask2])
        fwT_annual[nEns, iy] = np.nanmean(fwT[mask2])

    corr = []
    pval = []
    #conf = []
    # for debugging:
    #print(spice_seasonal[nEns, :])
    #print(maxMLD_seasonal[nEns, :])
    for lag in lags:
        if lag<0:
            # for debugging:
            #print(spice_annual[nEns, -1:-lag-1:-1])
            #print(maxMLD_seasonal[nEns, -1+lag::-1])
            # correlate flipped a(t) with flipped b(t-tau), considering that python does 
            # *not* include last element of array (-lag-1 for spice_annual, for example):
            pearson_corr = stats.pearsonr(spice_annual[nEns, -1:-lag-1:-1], maxMLD_seasonal[nEns, -1+lag::-1])
        if lag==0:
            # correlate a(t) with b(t)
            pearson_corr = stats.pearsonr(spice_annual[nEns, :], maxMLD_seasonal[nEns, :])
        if lag>0:
            # for debugging:
            #print(spice_annual[nEns, 0:-lag])
            #print(maxMLD_annual[nEns, lag::])
            # correlate a(t) with v(t+tau)
            pearson_corr = stats.pearsonr(spice_annual[nEns, 0:-lag], maxMLD_seasonal[nEns, lag::])
        corr = np.append(corr, pearson_corr.statistic)
        pval = np.append(pval, pearson_corr.pvalue)
        #conf = np.append(conf, pearson_corr.confidence_interval(confidence_level=0.95))

    #print(corr)
    #print(pval)
    sigValues = np.where(pval < .01) # choose pvalue<1%
    insigValues = np.where(pval >= .01)
    #ax.plot(years, spice_annual[nEns, :], colors[nEns], marker='o', linewidth=1.5, label=f'{ensembleMemberName}, r={corr.statistic:5.2f}')
    #ax.scatter(maxMLD_seasonal[nEns, :], spice_annual[nEns, :], s=5, c=colors[nEns], marker='o', label=ensembleMemberName)
    ax.plot(lags, corr, colors[nEns], linewidth=1, label=ensembleMemberName)
    ax.scatter(lags[sigValues], corr[sigValues], s=20, c=colors[nEns], marker='o')
    ax.scatter(lags[insigValues], corr[insigValues], s=20, c=colors[nEns], marker='o', alpha=0.3)

ax.grid(visible=True, which='both')
ax.legend(prop=legend_properties)
fig.savefig(figfile, dpi='figure', bbox_inches='tight', pad_inches=0.1)

########################################################################

fwT_annual = -fwT_annual # convert FW transport to salt water transport (makes more sense in the Iceland-Faroe-Scotland channel)
maxMLD_flat = maxMLD_seasonal.flatten()
maxMLDLC = np.quantile(maxMLD_flat, 0.15)
maxMLDHC = np.quantile(maxMLD_flat, 0.85)
spice_flat = spice_annual.flatten()
spiceLow = np.quantile(spice_flat, 0.15)
spiceHigh= np.quantile(spice_flat, 0.85)
volT_flat = volT_annual.flatten()
volTLow = np.quantile(volT_flat, 0.15)
volTHigh= np.quantile(volT_flat, 0.85)
heatT_flat = heatT_annual.flatten()
heatTLow = np.quantile(heatT_flat, 0.15)
heatTHigh= np.quantile(heatT_flat, 0.85)
fwT_flat = fwT_annual.flatten()
fwTLow = np.quantile(fwT_flat, 0.15)
fwTHigh= np.quantile(fwT_flat, 0.85)
indMLDLow = np.less_equal(maxMLD_flat, maxMLDLC)
indMLDHigh=np.greater_equal(maxMLD_flat, maxMLDHC)
indvolTLow = np.less_equal(volT_flat, volTLow)
indvolTHigh= np.greater_equal(volT_flat, volTHigh)
indheatTLow = np.less_equal(heatT_flat, heatTLow)
indheatTHigh= np.greater_equal(heatT_flat, heatTHigh)
indfwTLow = np.less_equal(fwT_flat, fwTLow)
indfwTHigh= np.greater_equal(fwT_flat, fwTHigh)

percentage = 100 * np.size(np.where(np.logical_and(indvolTHigh, indMLDHigh))) / np.size(np.where(indMLDHigh))
print(f'\nPercentage of HC years for the {regionName} also associated with high volume transport across {transectName}: {percentage}')
percentage = 100 * np.size(np.where(np.logical_and(indvolTLow,  indMLDLow))) / np.size(np.where(indMLDLow))
print(f'Percentage of LC years for the {regionName} also associated with low volume transport across {transectName}: {percentage}\n')

percentage = 100 * np.size(np.where(np.logical_and(indheatTHigh, indMLDHigh))) / np.size(np.where(indMLDHigh))
print(f'Percentage of HC years for the {regionName} also associated with high heat transport across {transectName}: {percentage}')
percentage = 100 * np.size(np.where(np.logical_and(indheatTLow,  indMLDLow))) / np.size(np.where(indMLDLow))
print(f'Percentage of LC years for the {regionName} also associated with low heat transport across {transectName}: {percentage}\n')
percentage = 100 * np.size(np.where(np.logical_and(indfwTHigh, indMLDHigh))) / np.size(np.where(indMLDHigh))

print(f'Percentage of HC years for the {regionName} also associated with high saltwater transport across {transectName}: {percentage}')
percentage = 100 * np.size(np.where(np.logical_and(indfwTLow,  indMLDLow))) / np.size(np.where(indMLDLow))
print(f'Percentage of LC years for the {regionName} also associated with low saltwater transport across {transectName}: {percentage}\n')

percentage = 100 * np.size(np.where(np.logical_and(np.logical_and(indvolTHigh, indMLDHigh), indheatTHigh))) / np.size(np.where(indMLDHigh))
print(f'Percentage of HC years for the {regionName} also associated with high volume and heat transport across {transectName}: {percentage}')
percentage = 100 * np.size(np.where(np.logical_and(np.logical_and(indvolTHigh, indMLDHigh), indfwTHigh))) / np.size(np.where(indMLDHigh))
print(f'Percentage of HC years for the {regionName} also associated with high volume and saltwater transport across {transectName}: {percentage}\n')

figdpi = 300
figsize = (20, 5)
# Make scatter plot for a range of lags
for lag in np.arange(0, 7, dtype=np.int16):
    # First do one scatter plot for *all* ensemble members 
    figfile = f'{figdir}/{transectNameShort}_maxMLD{regionNameShort}_scatterLag{lag:02d}.png'
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    for i in np.arange(4):
        for tick in ax[i].xaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        for tick in ax[i].yaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        ax[i].yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
        ax[i].yaxis.get_offset_text().set_weight('bold')
    ax[0].set_ylabel('JFMA maxMLD', fontsize=fontsize_labels, fontweight='bold')
    ax[0].set_xlabel('Annual spice0', fontsize=fontsize_labels, fontweight='bold')
    figtitle = f'lag={lag:02d}-year'
    ax[0].set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
    ax[1].set_xlabel('Annual net volume transport', fontsize=fontsize_labels, fontweight='bold')
    figtitle = f'lag={lag:02d}-year'
    ax[1].set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
    ax[2].set_xlabel('Annual net heat trasnport (Tref=0)', fontsize=fontsize_labels, fontweight='bold')
    figtitle = f'lag={lag:02d}-year'
    ax[2].set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
    ax[3].set_xlabel('Annual net saltwater transport (Sref=34.8)', fontsize=fontsize_labels, fontweight='bold')
    figtitle = f'lag={lag:02d}-year'
    ax[3].set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')

    if lag==0:
        fld0 = spice_annual.flatten()
        fld1 = volT_annual.flatten()
        fld2 = heatT_annual.flatten()
        fld3 = fwT_annual.flatten()
        mld = maxMLD_seasonal.flatten()
    if lag>0:
        fld0 = spice_annual[:, 0:-lag].flatten()
        fld1 = volT_annual[:, 0:-lag].flatten()
        fld2 = heatT_annual[:, 0:-lag].flatten()
        fld3 = fwT_annual[:, 0:-lag].flatten()
        mld = maxMLD_seasonal[:, lag::].flatten()
    if lag<0:
        fld0 = spice_annual[:, -1:-lag-1:-1].flatten()
        fld1 = volT_annual[:, -1:-lag-1:-1].flatten()
        fld2 = heatT_annual[:, -1:-lag-1:-1].flatten()
        fld3 = fwT_annual[:, -1:-lag-1:-1].flatten()
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

    pearson_corr = stats.pearsonr(fld2, mld)
    corr = pearson_corr.statistic
    pval = pearson_corr.pvalue
    #print(corr, pval)
    ax[2].scatter(fld2, mld, s=20, c='k', marker='d', label=f'r={corr:5.2f}')
    ax[2].scatter(fld2[ind_maxMLDLC], mld[ind_maxMLDLC], s=20, c='b', marker='d')
    ax[2].scatter(fld2[ind_maxMLDHC], mld[ind_maxMLDHC], s=20, c='r', marker='d')
    ax[2].grid(visible=True, which='both')
    ax[2].legend(prop=legend_properties)

    pearson_corr = stats.pearsonr(fld3, mld)
    corr = pearson_corr.statistic
    pval = pearson_corr.pvalue
    #print(corr, pval)
    ax[3].scatter(fld3, mld, s=20, c='k', marker='d', label=f'r={corr:5.2f}')
    ax[3].scatter(fld3[ind_maxMLDLC], mld[ind_maxMLDLC], s=20, c='b', marker='d')
    ax[3].scatter(fld3[ind_maxMLDHC], mld[ind_maxMLDHC], s=20, c='r', marker='d')
    ax[3].grid(visible=True, which='both')
    ax[3].legend(prop=legend_properties)

    fig.suptitle(f'Scatter plot between JFMA-maxMLD in the {regionName} and\n 0-400 m integrated quantities across the {transectName} transect', fontsize=12, fontweight='bold', y=1.04)
    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight', pad_inches=0.1)

    # Then do one scatter plot for each ensemble member 
    #for nEns in range(nEnsembles):
    #    ensembleMemberName = ensembleMemberNames[nEns]
    #    figfile = f'{figdir}{ensembleMemberName}/spice0{transectNameShort}_maxMLD{regionNameShort}_scatterLag{lag:02d}.png'
    #    figtitle = f'Upper 400 m {transectName} spice vs \n{regionName} maxMLD (Lag={lag:02d}-year, ensemble={ensembleMemberName})'
    #    fig = plt.figure(figsize=(5, 5), dpi=figdpi)
    #    ax = fig.add_subplot()
    #    for tick in ax.xaxis.get_ticklabels():
    #        tick.set_fontsize(fontsize_smallLabels)
    #        tick.set_weight('bold')
    #    for tick in ax.yaxis.get_ticklabels():
    #        tick.set_fontsize(fontsize_smallLabels)
    #        tick.set_weight('bold')
    #    ax.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
    #    ax.yaxis.get_offset_text().set_weight('bold')
    #    ax.set_xlabel('JFMA spice', fontsize=fontsize_labels, fontweight='bold')
    #    ax.set_ylabel('JFMA maxMLD', fontsize=fontsize_labels, fontweight='bold')
    #    ax.set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
    # 
    #    if lag==0:
    #        fld1 = spice_annual[nEns, :]
    #        fld2 = maxMLD_seasonal[nEns, :]
    #    if lag>0:
    #        fld1 = spice_annual[nEns, 0:-lag]
    #        fld2 = maxMLD_seasonal[nEns, lag::]
    #    if lag<0:
    #        fld1 = spice_annual[nEns -1:-lag-1:-1]
    #        fld2 = maxMLD_seasonal[nEns -1+lag::-1]
    #
    #    pearson_corr = stats.pearsonr(fld1, fld2)
    #    corr = pearson_corr.statistic
    #    pval = pearson_corr.pvalue
    #    ax.scatter(fld1, fld2, s=20, c='k', marker='d', label=f'r={corr:5.2f}')
    #    ax.grid(visible=True, which='both')
    #    ax.legend(prop=legend_properties)
    #    fig.savefig(figfile, dpi='figure', bbox_inches='tight', pad_inches=0.1)

#plt.show()
