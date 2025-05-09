from __future__ import absolute_import, division, print_function, \
    unicode_literals

import os
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from scipy.signal import detrend
import matplotlib.pyplot as plt

from geometric_features import FeatureCollection, read_feature_collection

from common_functions import add_inset

# Settings for nersc
regionMaskDir = '/global/cfs/cdirs/m1199/milena/mpas-region_masks'
runNameControl = 'E3SMv2.1B60to10rA02'
runNameRecovery = 'E3SMv2.1B60to10rA07'
runNameCollapse = 'E3SMv2.1G60to10_01'
#colors = ['mediumblue', 'dodgerblue', 'deepskyblue', 'lightseagreen', 'green']
 
startYear = 1
endYear = 50
years = range(startYear, endYear + 1)

movingAverageYears = 1 # number of years over which to compute running average

# Settings for regional time series
tsdir = '/global/cfs/cdirs/m4259/milena/AMOCpaper/timeseries_data'
variables = [
#             {'name': 'maxMLD',
#              'title': 'Max MLD',
#              'units': 'm'},
#             {'name': 'iceArea',
#              'title': 'Ice area',
#              'units': 'km$^2$'},
#             {'name': 'iceVolume',
#              'title': 'Ice volume',
#              'units': 'km$^3$'},
#             {'name': 'sensibleHeatFlux',
#              'title': 'Sensible heat flux',
#              'units': 'W/m$^2$'},
#             {'name': 'latentHeatFlux',
#              'title': 'Latent heat flux',
#              'units': 'W/m$^2$'},
#             {'name': 'evaporationFlux',
#              'title': 'Evaporation flux',
#              'units': 'kg m$^-2$ s$^-1$'},
#             {'name': 'rainFlux',
#              'title': 'Rain flux',
#              'units': 'kg m$^-2$ s$^-1$'},
#             {'name': 'snowFlux',
#              'title': 'Snow flux',
#              'units': 'kg m$^-2$ s$^-1$'},
#             {'name': 'riverRunoffFlux',
#              'title': 'River runoff flux',
#              'units': 'kg m$^-2$ s$^-1$'},
#             {'name': 'iceRunoffFlux',
#              'title': 'Ice runoff flux',
#              'units': 'kg m$^-2$ s$^-1$'},
#             {'name': 'seaIceFreshWaterFlux',
#              'title': 'Sea ice FW flux',
#              'units': 'kg m$^-2$ s$^-1$'},
#             {'name': 'surfaceBuoyancyForcing',
#              'title': 'Surface buoyancy forcing',
#              'units': 'm$^2$ s$^{-3}$'},
             {'name': 'totalHeatFlux',
              'title': 'Total heat flux (Sens+Lat+SWNet+LWNet)',
              'units': 'W/m$^2$'},
             {'name': 'totalFWFlux',
              'title': 'Total FW flux (E-P+runoff+seaiceFW)',
              'units': 'kg m$^-2$ s$^-1$'},
#             {'name': 'temperature',
#              'title': 'SST',
#              'units': '$^\circ$C'},
#             {'name': 'salinity',
#              'title': 'SSS',
#              'units': 'psu'}
            ]

#regionName = 'Greenland Sea'
#regionName = 'Norwegian Sea'
#regionName = 'Labrador Sea'
regionName = 'Irminger Sea'
regionGroup = 'Arctic Regions' # defines feature filename, as well as regional ts filenames
#regionName = 'North Atlantic subpolar gyre'
#regionName = 'North Atlantic subtropical gyre'
#regionName = 'Atlantic tropical'
#regionName = 'South Atlantic subtropical gyre'
#regionName = 'Greater Arctic'
#regionGroup = 'arctic_atlantic_budget_regions_new20240408'
#regionName = 'Southern Ocean Atlantic Sector'
#regionName = 'Southern Ocean Basin'
#regionGroup = 'oceanSubBasins20210315'
regionGroupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')
transectName = None

# Settings for transect time series
#tsdir = './transports_data'
#variables = [
#             {'name': 'FWTransportSref',
#              'title': 'FW (Sref=34.8 psu) transport',
#              'units': 'mSv'},
#            ]
#transectName = 'Fram Strait'
#transectGroup = 'Arctic Sections' # defines feature filename, as well as transport ts filenames
#transectGroupName = transectGroup[0].lower() + transectGroup[1:].replace(' ', '')
#regionName = None

figdir = f'./figs4AMOCpaper/{runNameRecovery}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figsize = (15, 5)
figdpi = 150
fontsize_smallLabels = 16
fontsize_labels = 20
fontsize_titles = 22
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
##############################################################

if regionName is not None:
    regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '')
    featureFile = f'{regionMaskDir}/{regionGroupName}.geojson'
    print(featureFile)
    if os.path.exists(featureFile):
        fcAll = read_feature_collection(featureFile)
    else:
        raise IOError('No feature file found for this region group')
    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == regionName:
            fc.add_feature(feature)
            break
elif transectName is not None:
    transectNameShort = transectName[0].lower() + transectName[1:].replace(' ', '')
    featureFile = f'{regionMaskDir}/{transectGroupName}.geojson'
    if os.path.exists(featureFile):
        fcAll = read_feature_collection(featureFile)
    else:
        raise IOError('No feature file found for this transect group')
    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == transectName:
            fc.add_feature(feature)
            break
else:
    raise ValueError('Both regionName and transectName are None')

for var in variables:
    varname = var['name']
    print(f'    var: {varname}')
    vartitle = var['title']
    varunits = var['units']

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
    ax.set_ylabel(varunits, fontsize=fontsize_labels, fontweight='bold')
    ax.set_xlim(years[0], years[-1])
    plt.grid(alpha=0.75)

    if regionName is not None:
        figfile = f'{figdir}/{varname}_{regionNameShort}_years{years[0]}-{years[-1]}.png'
        figtitle = f'{vartitle} in {regionName} region'
        if varname=='temperature' or varname=='salinity':
            timeseriesDirControl = f'{tsdir}/{runNameControl}'
            timeseriesDirRecovery = f'{tsdir}/{runNameRecovery}'
            timeseriesDirCollapse = f'{tsdir}/{runNameCollapse}'
        else:
            timeseriesDirControl = f'{tsdir}/{runNameControl}/{varname}'
            timeseriesDirRecovery = f'{tsdir}/{runNameRecovery}/{varname}'
            timeseriesDirCollapse = f'{tsdir}/{runNameCollapse}/{varname}'
        timeseriesFilesControl = []
        timeseriesFilesRecovery = []
        timeseriesFilesCollapse = []
        for year in years:
            if varname=='maxMLD':
                timeseriesFilesControl.append(f'{timeseriesDirControl}/{regionGroupName}_max_year{year:04d}.nc')
                timeseriesFilesRecovery.append(f'{timeseriesDirRecovery}/{regionGroupName}_max_year{year:04d}.nc')
                timeseriesFilesCollapse.append(f'{timeseriesDirCollapse}/{regionGroupName}_max_year{year:04d}.nc')
            else:
                if varname=='temperature' or varname=='salinity':
                    timeseriesFilesControl.append(f'{timeseriesDirControl}/{regionGroupName}_depth0000_year{year:04d}.nc')
                    timeseriesFilesRecovery.append(f'{timeseriesDirRecovery}/{regionGroupName}_depth0000_year{year:04d}.nc')
                    timeseriesFilesCollapse.append(f'{timeseriesDirCollapse}/{regionGroupName}_depth0000_year{year:04d}.nc')
                else:
                    timeseriesFilesControl.append(f'{timeseriesDirControl}/{regionGroupName}_year{year:04d}.nc')
                    timeseriesFilesRecovery.append(f'{timeseriesDirRecovery}/{regionGroupName}_year{year:04d}.nc')
                    timeseriesFilesCollapse.append(f'{timeseriesDirCollapse}/{regionGroupName}_year{year:04d}.nc')
        dsControl = xr.open_mfdataset(timeseriesFilesControl, combine='nested',
                                      concat_dim='Time', decode_times=False)
        dsRecovery = xr.open_mfdataset(timeseriesFilesRecovery, combine='nested',
                                       concat_dim='Time', decode_times=False)
        dsCollapse = xr.open_mfdataset(timeseriesFilesCollapse, combine='nested',
                                       concat_dim='Time', decode_times=False)
        regionNames = dsControl.regionNames[0].values
        regionIndex = np.where(regionNames==regionName)[0]
        dsvarControl = dsControl[varname].isel(nRegions=regionIndex)
        dsvarRecovery = dsRecovery[varname].isel(nRegions=regionIndex)
        dsvarCollapse = dsCollapse[varname].isel(nRegions=regionIndex)
    elif transectName is not None:
        figfile = f'{figdir}/{varname}_{transectNameShort}_years{years[0]}-{years[-1]}.png'
        figtitle = f'{vartitle} across {transectName}'
        timeseriesDirControl = f'{tsdir}/{runNameControl}'
        timeseriesDirRecovery = f'{tsdir}/{runNameRecovery}'
        timeseriesDirCollapse = f'{tsdir}/{runNameCollapse}'
        timeseriesFilesControl = []
        timeseriesFilesRecovery = []
        timeseriesFilesCollapse = []
        for year in years:
            timeseriesFilesControl.append(f'{timeseriesDirControl}/{transectGroupName}Transports_{runNameControl}_year{year:04d}.nc')
            timeseriesFilesRecovery.append(f'{timeseriesDirRecovery}/{transectGroupName}Transports_{runNameRecovery}_year{year:04d}.nc')
            timeseriesFilesCollapse.append(f'{timeseriesDirCollapse}/{transectGroupName}Transports_{runNameCollapse}_year{year:04d}.nc')
        dsControl = xr.open_mfdataset(timeseriesFilesControl, combine='nested',
                                      concat_dim='Time', decode_times=False)
        dsRecovery = xr.open_mfdataset(timeseriesFilesRecovery, combine='nested',
                                       concat_dim='Time', decode_times=False)
        dsCollapse = xr.open_mfdataset(timeseriesFilesCollapse, combine='nested',
                                       concat_dim='Time', decode_times=False)
        transectNames = dsControl.transectNames[0].values
        transectIndex = np.where(transectNames==transectName)[0]
        dsvarControl = dsControl[varname].isel(nTransects=transectIndex)
        dsvarRecovery = dsRecovery[varname].isel(nTransects=transectIndex)
        dsvarCollapse = dsCollapse[varname].isel(nTransects=transectIndex)

    window = int(movingAverageYears*12)
    timeseriesControl = np.squeeze(dsvarControl.values)
    timeseriesControl_runavg = pd.Series(timeseriesControl).rolling(window, center=True).mean()
    timeseriesRecovery = np.squeeze(dsvarRecovery.values)
    timeseriesRecovery_runavg = pd.Series(timeseriesRecovery).rolling(window, center=True).mean()
    timeseriesCollapse = np.squeeze(dsvarCollapse.values)
    timeseriesCollapse_runavg = pd.Series(timeseriesCollapse).rolling(window, center=True).mean()
    #meanControl = np.nanmean(timeseriesControl)
    #stdControl = np.nanstd(timeseriesControl)

    plt.plot(dsControl.Time.values/365, timeseriesControl, 'grey', linewidth=1.2)
    plt.plot(dsControl.Time.values/365, timeseriesControl_runavg, 'k', linewidth=2, label=f'control {movingAverageYears:d}-year run-avg')
    plt.plot(dsRecovery.Time.values/365, timeseriesRecovery, 'salmon', linewidth=1.2)
    plt.plot(dsRecovery.Time.values/365, timeseriesRecovery_runavg, 'r', linewidth=2, label=f'recovery {movingAverageYears:d}-year run-avg')
    plt.plot(dsCollapse.Time.values/365, timeseriesCollapse, 'lightgreen', linewidth=1.2)
    plt.plot(dsCollapse.Time.values/365, timeseriesCollapse_runavg, 'green', linewidth=2, label=f'collapse {movingAverageYears:d}-year run-avg')
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

    if regionName!='Global':
        add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)

    ax.set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
    plt.savefig(figfile, dpi='figure', bbox_inches='tight', pad_inches=0.1)
    plt.close()
