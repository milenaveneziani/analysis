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
runNameControl = 'E3SMv2.1B60to10rA02'
runName = 'E3SMv2.1B60to10rA07'
#colors = ['mediumblue', 'dodgerblue', 'deepskyblue', 'lightseagreen', 'green']
 
startYear = 1
endYear = 50
years = range(startYear, endYear + 1)

movingAverageYears = 1 # number of years over which to compute running average

# Settings for regional time series
tsdir = './timeseries_data'
varname = 'sensibleHeatFlux'
vartitle = 'Sensible heat flux'
#varname = 'latentHeatFlux'
#vartitle = 'Latent heat flux'
varunits = '[W/m$^2$]'
#varname = 'evaporationFlux'
#vartitle = 'Evaporation flux'
#varname = 'rainFlux'
#vartitle = 'Rain flux'
#varname = 'snowFlux'
#vartitle = 'Snow flux'
#varname = 'riverRunoffFlux'
#vartitle = 'River runoff flux'
#varname = 'iceRunoffFlux'
#vartitle = 'Ice runoff flux'
#varname = 'seaIceFreshWaterFlux'
#vartitle = 'sea ice FW flux'
#varunits = '[kg m$^-2$ s$^-1$]'
#varname = 'surfaceBuoyancyForcing'
#vartitle = 'sfc buoyancy forcing'
#varunits = '[m$^2$ s$^{-3}$]'
#varname = 'maxMLD'
#vartitle = 'Max MLD'
#varunits = '[m]'
#varname = 'iceArea'
#vartitle = 'ice area'
#varunits = '[km$^2$]'
#regionName = 'Greenland Sea'
#regionName = 'Norwegian Sea'
#regionName = 'Labrador Sea'
#regionName = 'Irminger Sea'
#regionGroup = 'Arctic Regions' # defines feature filename, as well as regional ts filenames
#regionName = 'North Atlantic subpolar gyre'
regionName = 'North Atlantic subtropical gyre'
#regionName = 'Atlantic tropical'
#regionName = 'South Atlantic subtropical gyre'
#regionGroup = 'arctic_atlantic_budget_regions_new20240408'
#regionName = 'Southern Ocean Atlantic Sector'
#regionName = 'Southern Ocean Basin'
#regionGroup = 'oceanSubBasins20210315'
regionGroupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')
transectName = None

# Settings for transect time series
#tsdir = './transports_data'
#varname = 'FWTransportSref'
#vartitle = 'FW (Sref=34.8 psu) transport'
#varunits = '[mSv]'
#transectName = 'Fram Strait'
#transectGroup = 'Arctic Sections' # defines feature filename, as well as transport ts filenames
#transectGroupName = transectGroup[0].lower() + transectGroup[1:].replace(' ', '')
#regionName = None

figdir = f'./figs4AMOCpaper/{runName}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

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
    figtitle = f'{vartitle} in {regionName} region'
elif transectName is not None:
    transectNameShort = transectName[0].lower() + transectName[1:].replace(' ', '')
    figfile = f'{figdir}/{varname}_{transectNameShort}_years{years[0]}-{years[-1]}.png'
    figtitle = f'{vartitle} across {transectName}'
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

if regionName is not None:
    timeseriesDirControl = f'{tsdir}/{runNameControl}/{varname}'
    timeseriesDir = f'{tsdir}/{runName}/{varname}'
    timeseriesFilesControl = []
    timeseriesFiles = []
    for year in years:
        if varname=='maxMLD':
            timeseriesFilesControl.append(f'{timeseriesDirControl}/{regionGroupName}_max_year{year:04d}.nc')
            timeseriesFiles.append(f'{timeseriesDir}/{regionGroupName}_max_year{year:04d}.nc')
        else:
            timeseriesFilesControl.append(f'{timeseriesDirControl}/{regionGroupName}_year{year:04d}.nc')
            timeseriesFiles.append(f'{timeseriesDir}/{regionGroupName}_year{year:04d}.nc')
    dsControl = xr.open_mfdataset(timeseriesFilesControl, combine='nested',
                                  concat_dim='Time', decode_times=False)
    ds = xr.open_mfdataset(timeseriesFiles, combine='nested',
                           concat_dim='Time', decode_times=False)
    regionNames = ds.regionNames[0].values
    regionIndex = np.where(regionNames==regionName)[0]
    dsvarControl = dsControl[varname].isel(nRegions=regionIndex)
    dsvar = ds[varname].isel(nRegions=regionIndex)
elif transectName is not None:
    timeseriesDirControl = f'{tsdir}/{runNameControl}'
    timeseriesDir = f'{tsdir}/{ensembleName}{ensembleMemberName}'
    timeseriesFilesControl = []
    timeseriesFiles = []
    for year in years:
        timeseriesFilesControl.append(f'{timeseriesDirControl}/{transectGroupName}Transports_{runNameControl}_year{year:04d}.nc')
        timeseriesFiles.append(f'{timeseriesDir}/{transectGroupName}Transports_{runName}_year{year:04d}.nc')
    dsControl = xr.open_mfdataset(timeseriesFilesControl, combine='nested',
                                  concat_dim='Time', decode_times=False)
    ds = xr.open_mfdataset(timeseriesFiles, combine='nested',
                           concat_dim='Time', decode_times=False)
    transectNames = ds.transectNames[0].values
    transectIndex = np.where(transectNames==transectName)[0]
    dsvarControl = dsControl[varname].isel(nTransects=transectIndex)
    dsvar = ds[varname].isel(nTransects=transectIndex)

window = int(movingAverageYears*12)
timeseriesControl = np.squeeze(dsvarControl.values)
timeseriesControl_runavg = pd.Series(timeseriesControl).rolling(window, center=True).mean()
timeseries = np.squeeze(dsvar.values)
timeseries_runavg = pd.Series(timeseries).rolling(window, center=True).mean()
#meanControl = np.nanmean(timeseriesControl)
#stdControl = np.nanstd(timeseriesControl)

plt.plot(ds.Time.values/365, timeseries, 'grey', linewidth=1.2, label='monthly')
plt.plot(ds.Time.values/365, timeseries_runavg, 'k', linewidth=2, label=f'{movingAverageYears:d}-year run-avg')
plt.plot(dsControl.Time.values/365, timeseriesControl_runavg, 'salmon', linewidth=2, label=f'control {movingAverageYears:d}-year run-avg')
#plt.axhline(y=np.nanmean(timeseries), color='k', label='mean')
#plt.axhline(y=np.nanmean(timeseriesControl), color='salmon', label='control mean')
#plt.axhspan(meanControl-stdControl, meanControl+stdControl, alpha=0.3, color='salmon', label='control range')
plt.axvline(x=8, color='mediumturquoise')
plt.axvline(x=34, color='deepskyblue')
plt.axvline(x=38, color='blue')
#plt.axvline(x=8, color='paleturquoise', label='recovery start')
#plt.axvline(x=34, color='deepskyblue', label='first crossing')
#plt.axvline(x=38, color='blue', label='recovery end')

ax.legend(prop=legend_properties)

# do this before the inset because otherwise it moves the inset
# and cartopy doesn't play too well with tight_layout anyway
plt.tight_layout()

#if regionName!='Global':
#    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)

plt.savefig(figfile, dpi='figure', bbox_inches='tight', pad_inches=0.1)
plt.close()
