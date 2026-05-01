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

# Settings for lanl
regionMaskDir = '/users/milena/mpas-region_masks'
runName = 'E3SM-Arcticv3.1_1950control'
tsdir = f'./timeseries_data/{runName}'
colorIceCats = ['mediumblue', 'dodgerblue', 'deepskyblue', 'lightseagreen', 'green']
 
startYear = 120
endYear = 120
years = range(startYear, endYear + 1)

# Number of data points to use in the moving average. For any frequency
# file, NmovingAverage = 1 means no moving average is applied. For monthly
# fields, NmovingAverage = 12 means 1-year running averages. Etcetera.
NmovingAverage = 1

variables = [
             {'name': 'iceArea',
              'title': 'Integrated ice area',
              'fileType': '_daily',
              'units': 'km$^2$'},
             {'name': 'iceVolume',
              'title': 'Integrated ice volume',
              'fileType': '_daily',
              'units': 'km$^3$'},
             {'name': 'iceAreaCategory',
              'title': 'Integrated ice area per category',
              'fileType': '_daily',
              'units': 'km$^2$'},
             {'name': 'iceVolumeCategory',
              'title': 'Integrated ice volume per category',
              'fileType': '_daily',
              'units': 'km$^3$'},
             {'name': 'icePressure',
              'title': 'Ice pressure',
              'fileType': '_daily',
              'units': 'N m$^{-1}$'},
             {'name': 'levelIceArea',
              'title': 'Integrated level-ice area per category',
              'fileType': '_daily',
              'units': 'km$^2$'},
             {'name': 'levelIceVolume',
              'title': 'Integrated level-ice volume per category',
              'fileType': '_daily',
              'units': 'km$^3$'},
             {'name': 'ridgedIceArea',
              'title': 'Integrated ridged-ice area per category',
              'fileType': '_daily',
              'units': 'km$^2$'},
             {'name': 'ridgedIceVolume',
              'title': 'Integrated ridged-ice volume per category',
              'fileType': '_daily',
              'units': 'km$^3$'},
             {'name': 'ridgedIceAreaAverage',
              'title': 'Integrated ridged-ice area',
              'fileType': '_daily',
              'units': 'km$^2$'},
             {'name': 'ridgedIceVolumeAverage',
              'title': 'Integrated ridged-ice volume',
              'fileType': '_daily',
              'units': 'km$^3$'},
             {'name': 'ridgeConvergence',
              'title': 'Normalized energy dissipation due to convergence',
              'fileType': '_daily',
              'units': 's$^{-1}$'},
             {'name': 'ridgeShear',
              'title': 'Normalized energy dissipation due to shear',
              'fileType': '_daily',
              'units': 's$^{-1}$'},
             #{'name': 'maxMLD',
             # 'title': 'Max MLD',
             # 'fileType': '_max',
             # 'units': 'm'},
             #{'name': 'temperature',
             # 'title': 'SST',
             # 'fileType': '_depth0000',
             # 'units': '$^\circ$C'},
             #{'name': 'salinity',
             # 'title': 'SSS',
             # 'fileType': '_depth0000',
             # 'units': 'psu'},
             #{'name': 'totalHeatFlux',
             # 'title': 'Total heat flux (Sens+Lat+SWNet+LWNet)',
             # 'fileType': '',
             # 'units': 'W/m$^2$'},
            ]

regionName = 'Beaufort Gyre'
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

figdir = f'./timeseries/{runName}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figsize = (15, 5)
figdpi = 150
fontsize_smallLabels = 12
fontsize_labels = 20
fontsize_titles = 22
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
##############################################################

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

for var in variables:
    varname = var['name']
    print(f'    var: {varname}')
    vartitle = var['title']
    varunits = var['units']
    fileType = var['fileType']

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
    plt.grid(alpha=0.75)

    figfile = f'{figdir}/{varname}_{regionNameShort}_years{years[0]}-{years[-1]}.png'
    figtitle = f'{vartitle} in {regionName} region'
    timeseriesFiles = []
    timeseriesFiles1 = []
    timeseriesFiles2 = []
    for year in years:
        if varname=='ridgedIceArea' or varname=='ridgedIceVolume':
            if varname=='ridgedIceArea':
                timeseriesFiles1.append(f'{tsdir}/iceAreaCategory/{regionGroupName}{fileType}_year{year:04d}.nc')
                timeseriesFiles2.append(f'{tsdir}/levelIceArea/{regionGroupName}{fileType}_year{year:04d}.nc')
            else:
                timeseriesFiles1.append(f'{tsdir}/iceVolumeCategory/{regionGroupName}{fileType}_year{year:04d}.nc')
                timeseriesFiles2.append(f'{tsdir}/levelIceVolume/{regionGroupName}{fileType}_year{year:04d}.nc')
            ds1 = xr.open_mfdataset(timeseriesFiles1, combine='nested',
                                    concat_dim='Time', decode_times=False)
            ds2 = xr.open_mfdataset(timeseriesFiles2, combine='nested',
                                    concat_dim='Time', decode_times=False)
            if varname=='ridgedIceArea':
                ds = ds1['iceAreaCategory'] - ds2['levelIceArea']
            else:
                ds = ds1['iceVolumeCategory'] - ds2['levelIceVolume']
            ds = ds.rename(varname)
            regionNames = ds1.regionNames[0].values
            regionIndex = np.where(regionNames==regionName)[0]
            dsvar = ds.isel(nRegions=regionIndex)
        else:
            timeseriesFiles.append(f'{tsdir}/{varname}/{regionGroupName}{fileType}_year{year:04d}.nc')
            ds = xr.open_mfdataset(timeseriesFiles, combine='nested',
                                   concat_dim='Time', decode_times=False)
            regionNames = ds.regionNames[0].values
            regionIndex = np.where(regionNames==regionName)[0]
            dsvar = ds[varname].isel(nRegions=regionIndex)

    t = ds['Time'].to_numpy()/365 # from days to years
    timeseries = np.squeeze(dsvar.to_numpy())
    if varname=='iceAreaCategory' or varname=='iceVolumeCategory' or \
       varname=='levelIceArea' or varname=='levelIceVolume' or \
       varname=='ridgedIceArea' or varname=='ridgedIceVolume':
        # These are ice category variables, so we need to plot one line per category (and no running average)
        ncats = np.shape(timeseries)[1]
        if len(colorIceCats)!=ncats:
            raise ValueError('length of colorIceCats should be the same as the number of ice categories')
        for ncat in range(ncats):
            plt.plot(t, timeseries[:, ncat], color=colorIceCats[ncat], linewidth=2, label=f'ice cat {ncat+1:d}')
        ax.legend(prop=legend_properties)
    else:
        #mean = np.nanmean(timeseries)
        #std = np.nanstd(timeseries)
        if NmovingAverage!=1:
            window = int(NmovingAverage)
            timeseries_runavg = pd.Series(timeseriesControl).rolling(window, center=True).mean()
            plt.plot(t, timeseries, 'grey', linewidth=1.2)
            plt.plot(t, timeseries_runavg, 'k', linewidth=2, label=f'{NmovingAverage:d}-point run-avg')
        else:
            plt.plot(t, timeseries, 'k', linewidth=2)

    ax.set_xlim(t[0], t[-1])

    # do this before the inset because otherwise it moves the inset
    # and cartopy doesn't play too well with tight_layout anyway
    plt.tight_layout()

    if regionName!='Global':
        add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.2, ybuffer=-1)

    ax.set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
    plt.savefig(figfile, dpi='figure', bbox_inches='tight', pad_inches=0.1)
    plt.close()
