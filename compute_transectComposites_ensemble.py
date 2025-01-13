#
# This script identifies years of anomalously high and low spiciness0
# across specific transects, where the anomalies are computed by 1) removing
# the trend from monthly timeseries and 2) focusing on the SON period.
# This particular version deals with ensembles of simulations, and 1) and 2)
# are performed across ensemble members.
#

from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import xarray as xr
import numpy as np
import netCDF4
from scipy.signal import detrend
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


matplotlib.use('TkAgg')
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
plt.rc('font', weight='bold')

startSimYear = 1950
startYear = 1950
endYear = 2014
years = np.arange(startYear, endYear + 1)
calendar = '365_day'
referenceDate = '1950-01-01'
#referenceDate = '0001-01-01'

# Settings for nersc
ensembleName = 'E3SM-Arcticv2.1_historical'
ensembleMemberNames = ['0101', '0151', '0201', '0251', '0301']

outdir = f'./composites_spice0based_data/{ensembleName}'
figdir = f'./composites_spice0based/{ensembleName}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
if not os.path.isdir(figdir):
    os.makedirs(figdir)

#transectGroup = 'arcticSections'
#transectName = 'Iceland-Faroe-Scotland'
transectGroup = 'atlanticZonalSections'
transectName = 'Atlantic zonal OSNAP East'
#transectName = 'Atlantic zonal 45N'

transectNameShort = transectName.replace(' ', '').replace('-', '')

climoMonths = [9, 10, 11] # SON
titleClimoMonths = 'SON'
####################################################################

nEnsembles = len(ensembleMemberNames)
spice_seasonal = np.zeros((nEnsembles, len(years)))
spice_seasonal_monthly = np.zeros((nEnsembles, len(years), len(climoMonths)))
for nEns in range(nEnsembles):
    ensembleMemberName = ensembleMemberNames[nEns]
    transportDir = f'./transports_data/{ensembleName}{ensembleMemberName}'
    timeSeriesFiles = []
    for year in years:
        timeSeriesFiles.append(f'{transportDir}/{transectGroup}Transports_z0000-0500_{ensembleName}{ensembleMemberName}_year{year:04d}.nc')
    dsIn = xr.open_mfdataset(timeSeriesFiles, combine='nested',
                             concat_dim='Time', decode_times=False)
    transectNames = dsIn.transectNames[0].values
    transectIndex = np.where(transectNames==transectName)[0]

    datetimes = netCDF4.num2date(365.*dsIn.Time, f'days since {referenceDate}', calendar=calendar)
    timeyears = []
    timemonths = []
    for date in datetimes.flat:
        timeyears.append(date.year)
        timemonths.append(date.month)
    monthmask = [i for i, x in enumerate(timemonths) if x in set(climoMonths)]

    spice = np.squeeze(dsIn.spiceTransect.isel(nTransects=transectIndex).values)
    spice_detrend = detrend(spice, type='linear')
    plt.plot(dsIn.Time.values, spice, 'k')
    plt.plot(dsIn.Time.values, spice-spice_detrend, 'b')
    plt.plot(dsIn.Time.values, spice_detrend, 'r')
    plt.show()
    for iy, year in enumerate(years):
        yearmask = [i for i, x in enumerate(timeyears) if x==year]
        mask = np.intersect1d(yearmask, monthmask)
        if np.size(mask)==0:
            raise ValueError('Something is wrong with time mask')
        spice_seasonal[nEns, iy] = np.nanmean(spice_detrend[mask])
        spice_seasonal_monthly[nEns, iy, :] = spice_detrend[mask]

spice_flat = spice_seasonal.flatten()
print('quantile 0 =', np.quantile(spice_flat, 0), '  min = ', np.min(spice_flat))
print('quantile 1 =', np.quantile(spice_flat, 0.25))
print('quantile 2 =', np.quantile(spice_flat, 0.5), '  median = ', np.median(spice_flat))
print('quantile 3 =', np.quantile(spice_flat, 0.75))
print('quantile 4 =', np.quantile(spice_flat, 1), '  max = ', np.max(spice_flat))
print('mean = ', np.mean(spice_flat))
print('std = ', np.std(spice_flat))
# this works only for normally distributed fields:
#spicestd = np.std(spice_flat)
#spice1 = np.min(spice_flat) + 1.5*spicestd
#spice2 = np.max(spice_flat) - 1.5*spicestd
spice1 = np.quantile(spice_flat, 0.15)
spice2 = np.quantile(spice_flat, 0.85)
#spice1 = np.quantile(spice_flat, 0.25) # first quartile
#spice2 = np.quantile(spice_flat, 0.75) # third quartile
print('spice1 = ', spice1, 'spice2 = ', spice2)

# Make histogram plot
plt.figure(figsize=[10, 8], dpi=150)
ax = plt.subplot()
n, bins, patches = plt.hist(spice_flat, bins=12, color='#607c8e', alpha=0.7, rwidth=0.9)
ax.set_xticks(bins)
ax.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
ax.axvspan(np.min(spice_flat), spice1, alpha=0.3, color='salmon')
#ax.axvspan(np.min(spice_flat), np.quantile(spice_flat, 0.25), alpha=0.3, color='salmon')
ax.axvspan(spice2, np.max(spice_flat), alpha=0.3, color='salmon')
#ax.axvspan(np.quantile(spice_flat, 0.75), np.max(spice_flat), alpha=0.3, color='salmon')
ax.set_xlim(np.min(spice_flat), np.max(spice_flat))
ax.set_xlabel(f'{titleClimoMonths}-avg spiciness0 [Kg/m$^3$]', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('# of years', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title(f'Distribution of spiciness0 across the {transectName} transect', fontsize=14, fontweight='bold', pad=15)
ax.yaxis.set_minor_locator(MultipleLocator(5))
plt.grid(axis='y', alpha=0.75)
#plt.grid(axis='y', which='both', alpha=0.75)
plt.savefig(f'{figdir}/spicehist_{transectNameShort}.png', bbox_inches='tight')
plt.close()

spice_monthly_flat = spice_seasonal_monthly.flatten()
plt.figure(figsize=[10, 8], dpi=150)
ax = plt.subplot()
n, bins, patches = plt.hist(spice_monthly_flat, bins=12, color='#607c8e', alpha=0.7, rwidth=0.9)
ax.set_xticks(bins)
ax.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
ax.axvspan(np.min(spice_monthly_flat), np.quantile(spice_monthly_flat, 0.15), alpha=0.3, color='salmon')
#ax.axvspan(np.min(spice_monthly_flat), np.quantile(spice_monthly_flat, 0.25), alpha=0.3, color='salmon')
ax.axvspan(np.quantile(spice_monthly_flat, 0.85), np.max(spice_monthly_flat), alpha=0.3, color='salmon')
#ax.axvspan(np.quantile(spice_monthly_flat, 0.75), np.max(spice_monthly_flat), alpha=0.3, color='salmon')
ax.set_xlim(np.min(spice_monthly_flat), np.max(spice_monthly_flat))
ax.set_xlabel(f'{titleClimoMonths} monthly spiciness0 [Kg/m$^3$]', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('# of years', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title(f'Distribution of spiciness0 across the {transectName} transect', fontsize=14, fontweight='bold', pad=15)
ax.yaxis.set_minor_locator(MultipleLocator(5))
plt.grid(axis='y', alpha=0.75)
#plt.grid(axis='y', which='both', alpha=0.75)
plt.savefig(f'{figdir}/spicemonthlyhist_{transectNameShort}.png', bbox_inches='tight')
plt.close()

conditionLow = np.nan*np.ones((nEnsembles, len(years)))
conditionHigh = np.nan*np.ones((nEnsembles, len(years)))
conditionMed = np.nan*np.ones((nEnsembles, len(years)))
for nEns in range(nEnsembles): 
    conditionLow[nEns, :]  = np.less(spice_seasonal[nEns, :], spice1)
    conditionHigh[nEns, :] = np.greater_equal(spice_seasonal[nEns, :], spice2)
    conditionMed[nEns, :]  = np.logical_and(spice_seasonal[nEns, :]>=spice1, spice_seasonal[nEns, :]<spice2)
years2d = np.tile(years, (nEnsembles, 1))
years_low  = np.int32(years2d*conditionLow)
years_high = np.int32(years2d*conditionHigh)
years_med  = np.int32(years2d*conditionMed)
#print(years_low)
#print(years_high)
#print(years_med)

# Save this information to ascii files
with open(f'{outdir}/years_spicelow_{transectNameShort}.dat', 'w') as outfile:
    outfile.write(f'Years associated with low upper 500m spiciness0 across the {transectName} transect for each ensemble member\n')
    for nEns in range(nEnsembles):
        outfile.write(f'\nEnsemble member: {ensembleName}{ensembleMemberNames[nEns]}\n')
        np.savetxt(outfile, years_low[nEns, np.nonzero(years_low[nEns, :])][0], fmt='%5d', delimiter=' ')
with open(f'{outdir}/years_spicehigh_{transectNameShort}.dat', 'w') as outfile:
    outfile.write(f'Years associated with high upper 500m spiciness0 across the {transectName} transect for each ensemble member\n')
    for nEns in range(nEnsembles):
        outfile.write(f'\nEnsemble member: {ensembleName}{ensembleMemberNames[nEns]}\n')
        np.savetxt(outfile, years_high[nEns, np.nonzero(years_high[nEns, :])][0], fmt='%5d', delimiter=' ')
