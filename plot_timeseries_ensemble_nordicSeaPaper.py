from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from scipy.signal import detrend
from scipy import stats
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
regionName = 'Greenland Sea'
#regionName = 'Norwegian Sea'
regionGroup = 'Arctic Regions' # defines feature filename, as well as regional ts filenames
regionGroupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')

# Settings for transect time series
transectName = 'Fram Strait'
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
fontsize_smallLabels = 14
fontsize_labels = 20
fontsize_titles = 22
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
##############################################################

climoMonths = [1, 2, 3, 4] # JFMA only
regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '')
transectNameShort = transectName[0].lower() + transectName[1:].replace(' ', '')

figfile1 = f'{figdir}/maxMLD_{regionNameShort}_nordicSeaPaper.png'
figfile2 = f'{figdir}/iceArea_{regionNameShort}_nordicSeaPaper.png'
figfile3 = f'{figdir}/FWTransportSref_{transectNameShort}_nordicSeaPaper.png'
figfile4 = f'{figdir}/totalHeatFlux_{regionNameShort}_nordicSeaPaper.png'
figfile5 = f'{figdir}/surfaceBuoyancyForcing_{regionNameShort}_nordicSeaPaper.png'
figtitle1 = f'Max MLD (JFMA-avg) in {regionName} region'
figtitle2 = f'Ice area (JFMA-avg) in {regionName} region'
figtitle3 = f'FW (Sref=34.8 psu) transport (ANN-avg) across {transectName}'
figtitle4 = f'Total heat flux (JFMA-avg) in {regionName} region'
figtitle5 = f'Surface buoyancy flux (JFMA-avg) in {regionName} region'
indirRegion  = './timeseries_data'
indirTransect = './transports_data'

fig1 = plt.figure(figsize=figsize, dpi=figdpi)
ax1 = fig1.add_subplot()
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax1.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
ax1.yaxis.get_offset_text().set_weight('bold')
ax1.set_xlabel('Time (yr)', fontsize=fontsize_labels, fontweight='bold')
ax1.set_ylabel('maxMLD [m]', fontsize=fontsize_labels, fontweight='bold')
ax1.set_title(figtitle1, fontsize=fontsize_titles, fontweight='bold')
ax1.set_xlim(years[0], years[-1])
ax1.grid(visible=True, which='both', alpha=0.75)

fig2 = plt.figure(figsize=figsize, dpi=figdpi)
ax2 = fig2.add_subplot()
for tick in ax2.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax2.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax2.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
ax2.yaxis.get_offset_text().set_weight('bold')
ax2.set_xlabel('Time (yr)', fontsize=fontsize_labels, fontweight='bold')
ax2.set_ylabel('iceArea [x10$^5$ km$^2$]', fontsize=fontsize_labels, fontweight='bold')
ax2.set_title(figtitle2, fontsize=fontsize_titles, fontweight='bold')
ax2.set_xlim(years[0], years[-1])
ax2.grid(visible=True, which='both', alpha=0.75)

fig3 = plt.figure(figsize=figsize, dpi=figdpi)
ax3 = fig3.add_subplot()
for tick in ax3.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax3.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax3.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
ax3.yaxis.get_offset_text().set_weight('bold')
ax3.set_xlabel('Time (yr)', fontsize=fontsize_labels, fontweight='bold')
ax3.set_ylabel('FWTransportSref [mSv]', fontsize=fontsize_labels, fontweight='bold')
ax3.set_title(figtitle3, fontsize=fontsize_titles, fontweight='bold')
ax3.set_xlim(years[0], years[-1])
ax3.grid(visible=True, which='both', alpha=0.75)

fig4 = plt.figure(figsize=figsize, dpi=figdpi)
ax4 = fig4.add_subplot()
for tick in ax4.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax4.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax4.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
ax4.yaxis.get_offset_text().set_weight('bold')
ax4.set_xlabel('Time (yr)', fontsize=fontsize_labels, fontweight='bold')
ax4.set_ylabel('Heat flux [W/m$^2$]', fontsize=fontsize_labels, fontweight='bold')
ax4.set_title(figtitle4, fontsize=fontsize_titles, fontweight='bold')
ax4.set_xlim(years[0], years[-1])
ax4.grid(visible=True, which='both', alpha=0.75)

fig5 = plt.figure(figsize=figsize, dpi=figdpi)
ax5 = fig5.add_subplot()
for tick in ax5.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax5.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax5.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
ax5.yaxis.get_offset_text().set_weight('bold')
ax5.set_xlabel('Time (yr)', fontsize=fontsize_labels, fontweight='bold')
ax5.set_ylabel('Sfc buoyancy flux [x10$^{-8}$ m$^2$/s$^3$]', fontsize=fontsize_labels, fontweight='bold')
ax5.set_title(figtitle5, fontsize=fontsize_titles, fontweight='bold')
ax5.set_xlim(years[0], years[-1])
ax5.grid(visible=True, which='both', alpha=0.75)

nEnsembles = len(ensembleMemberNames)
maxMLD_seasonal  = np.zeros((nEnsembles, len(years)))
iceArea_seasonal = np.zeros((nEnsembles, len(years)))
heatFlux_seasonal = np.zeros((nEnsembles, len(years)))
buoyFlux_seasonal = np.zeros((nEnsembles, len(years)))
FW_seasonal = np.zeros((nEnsembles, len(years)))
for nEns in range(nEnsembles):
    ensembleMemberName = ensembleMemberNames[nEns]
    timeseriesDir1 = f'{indirRegion}/{ensembleName}{ensembleMemberName}/maxMLD'
    timeseriesDir2 = f'{indirRegion}/{ensembleName}{ensembleMemberName}/iceArea'
    timeseriesDir3 = f'{indirTransect}/{ensembleName}{ensembleMemberName}'
    timeseriesDir4 = f'{indirRegion}/{ensembleName}{ensembleMemberName}'
    timeseriesDir5 = f'{indirRegion}/{ensembleName}{ensembleMemberName}/surfaceBuoyancyForcing'
    timeseriesFiles1 = []
    timeseriesFiles2 = []
    timeseriesFiles3 = []
    timeseriesFiles4sens = []
    timeseriesFiles4late = []
    timeseriesFiles4LWup = []
    timeseriesFiles4LWdo = []
    timeseriesFiles4SW   = []
    timeseriesFiles5 = []
    for year in years:
        timeseriesFiles1.append(f'{timeseriesDir1}/{regionGroupName}_max_year{year:04d}.nc')
        timeseriesFiles2.append(f'{timeseriesDir2}/{regionGroupName}_year{year:04d}.nc')
        timeseriesFiles3.append(f'{timeseriesDir3}/{transectGroupName}Transports_{ensembleName}{ensembleMemberName}_year{year:04d}.nc')
        timeseriesFiles4sens.append(f'{timeseriesDir4}/sensibleHeatFlux/{regionGroupName}_year{year:04d}.nc')
        timeseriesFiles4late.append(f'{timeseriesDir4}/latentHeatFlux/{regionGroupName}_year{year:04d}.nc')
        timeseriesFiles4LWup.append(f'{timeseriesDir4}/longWaveHeatFluxUp/{regionGroupName}_year{year:04d}.nc')
        timeseriesFiles4LWdo.append(f'{timeseriesDir4}/longWaveHeatFluxDown/{regionGroupName}_year{year:04d}.nc')
        timeseriesFiles4SW.append(f'{timeseriesDir4}/shortWaveHeatFlux/{regionGroupName}_year{year:04d}.nc')
        timeseriesFiles5.append(f'{timeseriesDir5}/{regionGroupName}_year{year:04d}.nc')
    ds1 = xr.open_mfdataset(timeseriesFiles1, combine='nested',
                            concat_dim='Time', decode_times=False)
    ds2 = xr.open_mfdataset(timeseriesFiles2, combine='nested',
                            concat_dim='Time', decode_times=False)
    ds3 = xr.open_mfdataset(timeseriesFiles3, combine='nested',
                            concat_dim='Time', decode_times=False)
    ds4sens = xr.open_mfdataset(timeseriesFiles4sens, combine='nested',
                                concat_dim='Time', decode_times=False)
    ds4late = xr.open_mfdataset(timeseriesFiles4late, combine='nested',
                                concat_dim='Time', decode_times=False)
    ds4LWup = xr.open_mfdataset(timeseriesFiles4LWup, combine='nested',
                                concat_dim='Time', decode_times=False)
    ds4LWdo = xr.open_mfdataset(timeseriesFiles4LWdo, combine='nested',
                                concat_dim='Time', decode_times=False)
    ds4SW   = xr.open_mfdataset(timeseriesFiles4SW, combine='nested',
                                concat_dim='Time', decode_times=False)
    ds5 = xr.open_mfdataset(timeseriesFiles5, combine='nested',
                            concat_dim='Time', decode_times=False)
    regionNames = ds1.regionNames[0].values
    regionIndex = np.where(regionNames==regionName)[0]
    dsvar1 = ds1['maxMLD'].isel(nRegions=regionIndex)
    dsvar2 = ds2['iceArea'].isel(nRegions=regionIndex)
    transectNames = ds3.transectNames[0].values
    transectIndex = np.where(transectNames==transectName)[0]
    dsvar3 = ds3['FWTransportSref'].isel(nTransects=transectIndex)
    dsvar4 = ds4sens['sensibleHeatFlux'].isel(nRegions=regionIndex)     +\
             ds4late['latentHeatFlux'].isel(nRegions=regionIndex)       +\
             ds4LWup['longWaveHeatFluxUp'].isel(nRegions=regionIndex)   +\
             ds4LWdo['longWaveHeatFluxDown'].isel(nRegions=regionIndex) +\
             ds4SW['shortWaveHeatFlux'].isel(nRegions=regionIndex)
    dsvar5 = ds5['surfaceBuoyancyForcing'].isel(nRegions=regionIndex)

    # Compute and plot seasonal averages
    #  (note: this approach only works for the regional
    #   time series, not for the transport time series)
    maxMLD = np.squeeze(dsvar1.values)
    iceArea = np.squeeze(dsvar2.values)
    heatFlux = np.squeeze(dsvar4.values)
    buoyFlux = np.squeeze(dsvar5.values)
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
        iceArea_seasonal[nEns, iy] = np.nanmean(iceArea[mask])
        heatFlux_seasonal[nEns, iy] = np.nanmean(heatFlux[mask])
        buoyFlux_seasonal[nEns, iy] = np.nanmean(buoyFlux[mask])

    corr1 = stats.pearsonr(maxMLD_seasonal[nEns, :], iceArea_seasonal[nEns, :])
    #print(corr.statistic)
    #print(corr.confidence_interval(confidence_level=0.95))
    corr2 = stats.pearsonr(maxMLD_seasonal[nEns, :], heatFlux_seasonal[nEns, :])
    corr3 = stats.pearsonr(maxMLD_seasonal[nEns, :], buoyFlux_seasonal[nEns, :])

    ax1.plot(years, maxMLD_seasonal[nEns, :], colors[nEns], marker='o', linewidth=1.5, label=ensembleMemberName)
    ax2.plot(years, 1e-5*iceArea_seasonal[nEns, :], colors[nEns], marker='o', linewidth=1.5, label=f'{ensembleMemberName}, r={corr1.statistic:5.2f}')
    ax4.plot(years, heatFlux_seasonal[nEns, :], colors[nEns], marker='o', linewidth=1.5, label=f'{ensembleMemberName}, r={corr2.statistic:5.2f}')
    ax5.plot(years, 1e8*buoyFlux_seasonal[nEns, :], colors[nEns], marker='o', linewidth=1.5, label=f'{ensembleMemberName}, r={corr3.statistic:5.2f}')

    # Compute and plot annual averages
    FW_seasonal[nEns, :] = np.squeeze(dsvar3.groupby_bins('Time', len(years)).mean().rename({'Time_bins': 'Time'}).values)
    corr = stats.pearsonr(maxMLD_seasonal[nEns, 1:-1], FW_seasonal[nEns, 0:-2]) # correlation with previous year FW transport
    #print(corr.statistic)
    #print(corr.confidence_interval(confidence_level=0.95))
    ax3.plot(years, FW_seasonal[nEns, :], colors[nEns], marker='o', linewidth=1.5, label=f'{ensembleMemberName}, r={corr.statistic:5.2f}')

maxMLD_flat = maxMLD_seasonal.flatten()
maxMLDLC = np.quantile(maxMLD_flat, 0.15)
maxMLDHC = np.quantile(maxMLD_flat, 0.85)
#print('maxMLDLC, maxMLDHC = ', maxMLDLC, maxMLDHC)

ax1.axhspan(maxMLDLC, maxMLDHC, alpha=0.3, color='grey')

ax1.legend(prop=legend_properties)
ax2.legend(prop=legend_properties)
ax3.legend(prop=legend_properties)
ax4.legend(prop=legend_properties)
ax5.legend(prop=legend_properties)

fig1.savefig(figfile1, dpi='figure', bbox_inches='tight', pad_inches=0.1)
fig2.savefig(figfile2, dpi='figure', bbox_inches='tight', pad_inches=0.1)
fig3.savefig(figfile3, dpi='figure', bbox_inches='tight', pad_inches=0.1)
fig4.savefig(figfile4, dpi='figure', bbox_inches='tight', pad_inches=0.1)
fig5.savefig(figfile5, dpi='figure', bbox_inches='tight', pad_inches=0.1)
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)
plt.close(fig5)

#if climoMonths is not None:
#    ensembleMean = np.nanmean(timeseries_seasonal, axis=0)
#    ensembleTrend = ensembleMean - detrend(ensembleMean, type='linear')
#    plt.plot(years, ensembleTrend, 'k', linewidth=2.5, label='ensemble mean')
#else:
#    ensembleMean = np.nanmean(timeseries_ensemble, axis=0)
#    ensembleTrend = ensembleMean - detrend(ensembleMean, type='linear')
#    plt.plot(ds.Time.values, ensembleTrend, 'k', linewidth=2.5, label='ensemble mean')
