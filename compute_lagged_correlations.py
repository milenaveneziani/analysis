from __future__ import absolute_import, division, print_function, \
    unicode_literals
    
import os
import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
import cftime
from scipy.signal import detrend
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests as grangertest
import matplotlib.pyplot as plt

from mpas_analysis.shared.io.utility import decode_strings


################################
def _compute_annualMeans_fromMonthly(time_monthly, fld_monthly, calendar, daysInYear, referenceDate):

    weights = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])/daysInYear

    datetimes = netCDF4.num2date(time_monthly, f'days since {referenceDate}', calendar=calendar)
    t_years = np.empty(np.shape(time_monthly), dtype=np.int64)
    for i, date in enumerate(datetimes.flat):
        t_years[i] = date.year
    years = range(1, np.int32(np.shape(time_monthly)[0]/12)+1)
    time_annual = np.empty(np.shape(years), dtype=np.float64)
    fld_annual = np.empty(np.shape(years), dtype=np.float64)
    k = 0
    for year in years:
        indexYears = np.where(t_years==year)[0]
        time_annual[k] = np.mean(time_monthly[indexYears]/daysInYear) # from days to years
        fld_annual[k] = np.sum(weights * fld_monthly[indexYears])
        k = k + 1
    return time_annual, fld_annual


# Settings for nersc
mocdir = '/global/cfs/cdirs/m4259/milena/AMOCpaper/moc_timeseries'
#runname = 'E3SMv2.1B60to10rA02'
#mocfile = f'{mocdir}/{runname}/mocTimeSeries_0001-0386.nc'
runname = 'E3SMv2.1G60to10_01'
mocfile = f'{mocdir}/{runname}/mocTimeSeries_0001-0059.nc'
tsdir = f'./timeseries_data/{runname}'
tsfile_header = 'arctic_atlantic_budget_regions_20260827_z0000-0800_anomalieswrtYear1'
transportsdir = f'./transports_data/{runname}'
transportfile_header = f'atlanticZonalSectionsTransports_z0000-0800_{runname}'

mocLat = 45.0

# max pvalue for which results will be considered significant
pvalMax =  0.01 # choose pvalue<1%
#pvalMax =  0.05 # choose pvalue<5%

regionName = 'North Atlantic subpolar gyre'
regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '')

yearStart = 1
yearEnd = 50

calendar = 'noleap'
referenceDate = '0001-01-01 00:00:00'
daysInYear = 365.0

figdir = f'./timeseries/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

computeAnnual = True
#computeAnnual = False
if computeAnnual is True:
    maxLag = 15 # index lag ==> 5 means 5 years for annual data
    xlabelText = 'Lag (year)'
    tsfigfile = f'{figdir}/crosscorrMOC{np.int64(mocLat)}_fields{regionNameShort}_annual.png'
    figfile45N = f'{figdir}/crosscorrMOC{np.int64(mocLat)}_transports45N_annual.png'
    trfigfile = f'{figdir}/crosscorrFields{regionNameShort}_bryFields_annual.png'
else:
    maxLag = 60 # index lag ==> 60 means 5 years for monthly data
    xlabelText = 'Lag (months)'
    tsfigfile = f'{figdir}/crosscorrMOC{np.int64(mocLat)}_fields{regionNameShort}_monthly.png'
    figfile45N = f'{figdir}/crosscorrMOC{np.int64(mocLat)}_transports45N_monthly.png'
    trfigfile = f'{figdir}/crosscorrFields{regionNameShort}_bryFields_monthly.png'
lags = np.arange(-maxLag, maxLag+1, dtype=np.int16)

figdpi = 150
fontsize_smallLabels = 9
fontsize_labels = 10
fontsize_titles = 12
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
#################################################################

# Read in MOC at desired latitude
#moc = xr.open_dataset(mocfile)
#mocTime_monthly = cftime.date2num(mocTime_monthly, f'days since {referenceDate}')
moc = xr.open_dataset(mocfile, decode_times=False)
mocTime_monthly = moc['Time'].values
moc_monthly = moc['mocAtlantic'].sel(lat=mocLat, method='nearest').max(dim='depth').values
if computeAnnual is True:
    [mocTime_annual, moc_annual] = _compute_annualMeans_fromMonthly(mocTime_monthly, moc_monthly,
                                                                    calendar, daysInYear, referenceDate)
    mocTime = mocTime_annual[yearStart-1:yearEnd]
    moc_timeseries = moc_annual[yearStart-1:yearEnd]
else:
    mocTime = mocTime_monthly[(yearStart-1)*12:yearEnd*12]
    moc_timeseries = moc_monthly[(yearStart-1)*12:yearEnd*12]

# Read in regional timeseries
years = range(yearStart, yearEnd + 1)
timeSeriesFiles = []
for year in years:
    timeSeriesFile = f'{tsdir}/{tsfile_header}_year{year:04d}.nc'
    timeSeriesFiles.append(timeSeriesFile)
regionNames = xr.open_dataset(timeSeriesFile)['regionNames']
regionNames = decode_strings(regionNames)
regionIndex = regionNames.index(regionName)
dsIn  = xr.open_mfdataset(timeSeriesFiles, combine='nested', data_vars='all',
                          concat_dim='Time', decode_times=False).isel(nRegions=regionIndex)
tsTime_monthly = dsIn['Time'].values
rho_monthly = dsIn['gswPotentialDensity'].values
rhoAlpha_monthly = dsIn['alphaPotentialDensity'].values
rhoBeta_monthly = dsIn['betaPotentialDensity'].values
temp_monthly = dsIn['temperature'].values
salt_monthly = dsIn['salinity'].values
if computeAnnual is True:
    [tsTime, rho_timeseries] = _compute_annualMeans_fromMonthly(tsTime_monthly, rho_monthly,
                                                                calendar, daysInYear, referenceDate)
    [tsTime, rhoAlpha_timeseries] = _compute_annualMeans_fromMonthly(tsTime_monthly, rhoAlpha_monthly,
                                                                     calendar, daysInYear, referenceDate)
    [tsTime, rhoBeta_timeseries] = _compute_annualMeans_fromMonthly(tsTime_monthly, rhoBeta_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [tsTime, temp_timeseries] = _compute_annualMeans_fromMonthly(tsTime_monthly, temp_monthly,
                                                                 calendar, daysInYear, referenceDate)
    [tsTime, salt_timeseries] = _compute_annualMeans_fromMonthly(tsTime_monthly, salt_monthly,
                                                                 calendar, daysInYear, referenceDate)
else:
    tsTime = tsTime_monthly
    rho_timeseries = rho_monthly
    rhoAlpha_timeseries = rhoAlpha_monthly
    rhoBeta_timeseries = rhoBeta_monthly
    temp_timeseries = temp_monthly
    salt_timeseries = salt_monthly

# Read in transports timeseries
transportsFiles = []
for year in years:
    transportsFile = f'{transportsdir}/{transportfile_header}_year{year:04d}.nc'
    transportsFiles.append(transportsFile)
dsIn  = xr.open_mfdataset(transportsFiles, combine='nested', data_vars='all',
                          concat_dim='Time', decode_times=False)
trTime_monthly = daysInYear*dsIn['Time'].values
transectNames = xr.open_dataset(transportsFile)['transectNames']
transectNames = decode_strings(transectNames)
#
transectIndex = transectNames.index('Atlantic zonal 45N')
volT45N_monthly = dsIn['volTransport'].isel(nTransects=transectIndex).values
temp45N_monthly = dsIn['heatTransportTfp'].isel(nTransects=transectIndex).values
salt45N_monthly = dsIn['FWTransportSref'].isel(nTransects=transectIndex).values
#temp45N_monthly = dsIn['tempTransect'].isel(nTransects=transectIndex).values
#salt45N_monthly = dsIn['saltTransect'].isel(nTransects=transectIndex).values
#
transectIndex = transectNames.index('Davis Strait South')
volTDav_monthly = dsIn['volTransport'].isel(nTransects=transectIndex).values
tempDav_monthly = dsIn['heatTransportTfp'].isel(nTransects=transectIndex).values
saltDav_monthly = dsIn['FWTransportSref'].isel(nTransects=transectIndex).values
#tempDav_monthly = dsIn['tempTransect'].isel(nTransects=transectIndex).values
#saltDav_monthly = dsIn['saltTransect'].isel(nTransects=transectIndex).values
#
transectIndex = transectNames.index('Denmark Strait')
volTDen_monthly = dsIn['volTransport'].isel(nTransects=transectIndex).values
tempDen_monthly = dsIn['heatTransportTfp'].isel(nTransects=transectIndex).values
saltDen_monthly = dsIn['FWTransportSref'].isel(nTransects=transectIndex).values
#tempDen_monthly = dsIn['tempTransect'].isel(nTransects=transectIndex).values
#saltDen_monthly = dsIn['saltTransect'].isel(nTransects=transectIndex).values
#
transectIndex = transectNames.index('Iceland Norway 65N')
volTIce_monthly = dsIn['volTransport'].isel(nTransects=transectIndex).values
tempIce_monthly = dsIn['heatTransportTfp'].isel(nTransects=transectIndex).values
saltIce_monthly = dsIn['FWTransportSref'].isel(nTransects=transectIndex).values
#tempIce_monthly = dsIn['tempTransect'].isel(nTransects=transectIndex).values
#saltIce_monthly = dsIn['saltTransect'].isel(nTransects=transectIndex).values
if computeAnnual is True:
    [trTime, volT45N_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, volT45N_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, temp45N_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, temp45N_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, salt45N_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, salt45N_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, volTDav_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, volTDav_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, tempDav_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, tempDav_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, saltDav_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, saltDav_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, volTDen_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, volTDen_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, tempDen_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, tempDen_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, saltDen_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, saltDen_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, volTIce_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, volTIce_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, tempIce_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, tempIce_monthly,
                                                                    calendar, daysInYear, referenceDate)
    [trTime, saltIce_timeseries] = _compute_annualMeans_fromMonthly(trTime_monthly, saltIce_monthly,
                                                                    calendar, daysInYear, referenceDate)
else:
    trTime = trTime_monthly
    volT45N_timeseries = volT45N_monthly
    temp45N_timeseries = temp45N_monthly
    salt45N_timeseries = salt45N_monthly
    volTDav_timeseries = volTDav_monthly
    tempDav_timeseries = tempDav_monthly
    saltDav_timeseries = saltDav_monthly
    volTDen_timeseries = volTDen_monthly
    tempDen_timeseries = tempDen_monthly
    saltDen_timeseries = saltDen_monthly
    volTIce_timeseries = volTIce_monthly
    tempIce_timeseries = tempIce_monthly
    saltIce_timeseries = saltIce_monthly

# Remove means
#moc_timeseries = moc_timeseries - np.mean(moc_timeseries)
#rho_timeseries = rho_timeseries - np.mean(rho_timeseries)
#rhoAlpha_timeseries = rhoAlpha_timeseries - np.mean(rhoAlpha_timeseries)
#rhoBeta_timeseries = rhoBeta_timeseries - np.mean(rhoBeta_timeseries)
#temp_timeseries = temp_timeseries - np.mean(temp_timeseries)
#salt_timeseries = salt_timeseries - np.mean(salt_timeseries)
#volT45N_timeseries = volT45N_timeseries - np.mean(volT45N_timeseries)
#temp45N_timeseries = temp45N_timeseries - np.mean(temp45N_timeseries)
#salt45N_timeseries = salt45N_timeseries - np.mean(salt45N_timeseries)
#tempDav_timeseries = tempDav_timeseries - np.mean(tempDav_timeseries)
#saltDav_timeseries = saltDav_timeseries - np.mean(saltDav_timeseries)
#tempDen_timeseries = tempDen_timeseries - np.mean(tempDen_timeseries)
#saltDen_timeseries = saltDen_timeseries - np.mean(saltDen_timeseries)
#tempIce_timeseries = tempIce_timeseries - np.mean(tempIce_timeseries)
#saltIce_timeseries = saltIce_timeseries - np.mean(saltIce_timeseries)
# Remove trends
#moc_timeseries = moc_timeseries - detrend(moc_timeseries, type='linear')
#rho_timeseries = rho_timeseries - detrend(rho_timeseries, type='linear')
#rhoAlpha_timeseries = rhoAlpha_timeseries - detrend(rhoAlpha_timeseries, type='linear')
#rhoBeta_timeseries = rhoBeta_timeseries - detrend(rhoBeta_timeseries, type='linear')
#temp_timeseries = temp_timeseries - detrend(temp_timeseries, type='linear')
#salt_timeseries = salt_timeseries - detrend(salt_timeseries, type='linear')
#volT45N_timeseries = volT45N_timeseries - detrend(volT45N_timeseries, type='linear')
#temp45N_timeseries = temp45N_timeseries - detrend(temp45N_timeseries, type='linear')
#salt45N_timeseries = salt45N_timeseries - detrend(salt45N_timeseries, type='linear')
#tempDav_timeseries = tempDav_timeseries - detrend(tempDav_timeseries, type='linear')
#saltDav_timeseries = saltDav_timeseries - detrend(saltDav_timeseries, type='linear')
#tempDen_timeseries = tempDen_timeseries - detrend(tempDen_timeseries, type='linear')
#saltDen_timeseries = saltDen_timeseries - detrend(saltDen_timeseries, type='linear')
#tempIce_timeseries = tempIce_timeseries - detrend(tempIce_timeseries, type='linear')
#saltIce_timeseries = saltIce_timeseries - detrend(saltIce_timeseries, type='linear')

# Define the data array for testing whether the time series in the
# *second* column Granger causes the time series in the *first* column
#
print('\nTesting whether moc45N Granger causes rho-upper 800m in subpolar gyre')
data = [rho_timeseries, moc_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether rho-upper 800m in subpolar gyre Granger causes moc45N')
data = [moc_timeseries, rho_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
#print('\nTesting whether moc45N Granger causes T-upper 800m in subpolar gyre')
#data = [temp_timeseries, moc_timeseries]
print('\nTesting whether moc45N Granger causes rhoAlpha-upper 800m in subpolar gyre')
data = [rhoAlpha_timeseries, moc_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
#print('\nTesting whether T-upper 800m in subpolar gyre Granger causes moc45N')
#data = [moc_timeseries, temp_timeseries]
print('\nTesting whether rhoAlpha-upper 800m in subpolar gyre Granger causes moc45N')
data = [moc_timeseries, rhoAlpha_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
#print('\nTesting whether moc45N Granger causes S-upper 800m in subpolar gyre')
#data = [salt_timeseries, moc_timeseries]
print('\nTesting whether moc45N Granger causes rhoBeta-upper 800m in subpolar gyre')
data = [rhoBeta_timeseries, moc_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
#print('\nTesting whether S-upper 800m in subpolar gyre Granger causes moc45N')
#data = [moc_timeseries, salt_timeseries]
print('\nTesting whether rhoBeta-upper 800m in subpolar gyre Granger causes moc45N')
data = [moc_timeseries, rhoBeta_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether rhoAlpha Granger causes rho-upper 800m in subpolar gyre')
data = [rho_timeseries, rhoAlpha_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether rhoBeta Granger causes rho-upper 800m in subpolar gyre')
data = [rho_timeseries, rhoBeta_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether tflux45N Granger causes T-upper 800m in subpolar gyre')
data = [temp_timeseries, temp45N_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether T-upper 800m in subpolar gyre Granger causes tflux45N')
data = [temp45N_timeseries, temp_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether sflux45N Granger causes S-upper 800m in subpolar gyre')
data = [salt_timeseries, salt45N_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether S-upper 800m in subpolar gyre Granger causes sflux45N')
data = [salt45N_timeseries, salt_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether tfluxDen Granger causes T-upper 800m in subpolar gyre')
data = [temp_timeseries, tempDen_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether T-upper 800m in subpolar gyre Granger causes tfluxDen')
data = [tempDen_timeseries, temp_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether sfluxDen Granger causes S-upper 800m in subpolar gyre')
data = [salt_timeseries, saltDen_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether S-upper 800m in subpolar gyre Granger causes sfluxDen')
data = [saltDen_timeseries, salt_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether tfluxIce Granger causes T-upper 800m in subpolar gyre')
data = [temp_timeseries, tempIce_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether T-upper 800m in subpolar gyre Granger causes tfluxIce')
data = [tempIce_timeseries, temp_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether sfluxIce Granger causes S-upper 800m in subpolar gyre')
data = [salt_timeseries, saltIce_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether S-upper 800m in subpolar gyre Granger causes sfluxIce')
data = [saltIce_timeseries, salt_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether tfluxDav Granger causes T-upper 800m in subpolar gyre')
data = [temp_timeseries, tempDav_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether T-upper 800m in subpolar gyre Granger causes tfluxDav')
data = [tempDav_timeseries, temp_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether sfluxDav Granger causes S-upper 800m in subpolar gyre')
data = [salt_timeseries, saltDav_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#
print('\nTesting whether S-upper 800m in subpolar gyre Granger causes sfluxDav')
data = [saltDav_timeseries, salt_timeseries]
data = np.transpose(data)
test_results = grangertest(data, maxLag, verbose=False)
p_values    = [test_results[i + 1][0]['ssr_ftest'][1] for i in range(maxLag)]
test_values = [test_results[i + 1][0]['ssr_ftest'][0] for i in range(maxLag)]
for lag, p_value in enumerate(p_values):
    if p_value < pvalMax:
        print('lag (years) =', lag+1, 'grangerTest=', test_values[lag], 'pvalue=', p_value) 
#

moc_rhoCorr = np.empty(np.shape(lags), dtype=np.float64)
moc_rhoPval = np.empty(np.shape(lags), dtype=np.float64)
moc_rhoAlphaCorr = np.empty(np.shape(lags), dtype=np.float64)
moc_rhoAlphaPval = np.empty(np.shape(lags), dtype=np.float64)
moc_rhoBetaCorr = np.empty(np.shape(lags), dtype=np.float64)
moc_rhoBetaPval = np.empty(np.shape(lags), dtype=np.float64)
moc_tempCorr = np.empty(np.shape(lags), dtype=np.float64)
moc_tempPval = np.empty(np.shape(lags), dtype=np.float64)
moc_saltCorr = np.empty(np.shape(lags), dtype=np.float64)
moc_saltPval = np.empty(np.shape(lags), dtype=np.float64)
moc_volT45NCorr = np.empty(np.shape(lags), dtype=np.float64)
moc_volT45NPval = np.empty(np.shape(lags), dtype=np.float64)
moc_temp45NCorr = np.empty(np.shape(lags), dtype=np.float64)
moc_temp45NPval = np.empty(np.shape(lags), dtype=np.float64)
moc_salt45NCorr = np.empty(np.shape(lags), dtype=np.float64)
moc_salt45NPval = np.empty(np.shape(lags), dtype=np.float64)
tSubPolar_t45NCorr = np.empty(np.shape(lags), dtype=np.float64)
tSubPolar_t45NPval = np.empty(np.shape(lags), dtype=np.float64)
sSubPolar_s45NCorr = np.empty(np.shape(lags), dtype=np.float64)
sSubPolar_s45NPval = np.empty(np.shape(lags), dtype=np.float64)
tSubPolar_tDavCorr = np.empty(np.shape(lags), dtype=np.float64)
tSubPolar_tDavPval = np.empty(np.shape(lags), dtype=np.float64)
sSubPolar_sDavCorr = np.empty(np.shape(lags), dtype=np.float64)
sSubPolar_sDavPval = np.empty(np.shape(lags), dtype=np.float64)
tSubPolar_tDenCorr = np.empty(np.shape(lags), dtype=np.float64)
tSubPolar_tDenPval = np.empty(np.shape(lags), dtype=np.float64)
sSubPolar_sDenCorr = np.empty(np.shape(lags), dtype=np.float64)
sSubPolar_sDenPval = np.empty(np.shape(lags), dtype=np.float64)
tSubPolar_tIceCorr = np.empty(np.shape(lags), dtype=np.float64)
tSubPolar_tIcePval = np.empty(np.shape(lags), dtype=np.float64)
sSubPolar_sIceCorr = np.empty(np.shape(lags), dtype=np.float64)
sSubPolar_sIcePval = np.empty(np.shape(lags), dtype=np.float64)
k = 0
for lag in lags:
    if lag==0:
        moc = moc_timeseries
        rho = rho_timeseries
        rhoAlpha = rhoAlpha_timeseries
        rhoBeta = rhoBeta_timeseries
        temp = temp_timeseries
        salt = salt_timeseries
        tempSubPolar = temp_timeseries
        saltSubPolar = salt_timeseries
        volT45N = volT45N_timeseries
        temp45N = temp45N_timeseries
        salt45N = salt45N_timeseries
        tempDav = tempDav_timeseries
        saltDav = saltDav_timeseries
        tempDen = tempDen_timeseries
        saltDen = saltDen_timeseries
        tempIce = tempIce_timeseries
        saltIce = saltIce_timeseries
    if lag>0:
        moc = moc_timeseries[:-lag] # leading field
        rho = rho_timeseries[lag:]
        rhoAlpha = rhoAlpha_timeseries[lag:]
        rhoBeta = rhoBeta_timeseries[lag:]
        temp = temp_timeseries[lag:]
        salt = salt_timeseries[lag:]
        tempSubPolar = temp_timeseries[:-lag] # leading field
        saltSubPolar = salt_timeseries[:-lag] # leading field
        volT45N = volT45N_timeseries[lag:]
        temp45N = temp45N_timeseries[lag:]
        salt45N = salt45N_timeseries[lag:]
        tempDav = tempDav_timeseries[lag:]
        saltDav = saltDav_timeseries[lag:]
        tempDen = tempDen_timeseries[lag:]
        saltDen = saltDen_timeseries[lag:]
        tempIce = tempIce_timeseries[lag:]
        saltIce = saltIce_timeseries[lag:]
    if lag<0:
        moc = moc_timeseries[-lag:] # lagging field
        rho = rho_timeseries[:lag]
        rhoAlpha = rhoAlpha_timeseries[:lag]
        rhoBeta = rhoBeta_timeseries[:lag]
        temp = temp_timeseries[:lag]
        salt = salt_timeseries[:lag]
        tempSubPolar = temp_timeseries[-lag:] # lagging field
        saltSubPolar = salt_timeseries[-lag:] # lagging field
        volT45N = volT45N_timeseries[:lag]
        temp45N = temp45N_timeseries[:lag]
        salt45N = salt45N_timeseries[:lag]
        tempDav = tempDav_timeseries[:lag]
        saltDav = saltDav_timeseries[:lag]
        tempDen = tempDen_timeseries[:lag]
        saltDen = saltDen_timeseries[:lag]
        tempIce = tempIce_timeseries[:lag]
        saltIce = saltIce_timeseries[:lag]

    pearson_corr = stats.pearsonr(moc, rho)
    moc_rhoCorr[k] = pearson_corr.statistic
    moc_rhoPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(moc, rhoAlpha)
    moc_rhoAlphaCorr[k] = pearson_corr.statistic
    moc_rhoAlphaPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(moc, rhoBeta)
    moc_rhoBetaCorr[k] = pearson_corr.statistic
    moc_rhoBetaPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(moc, temp)
    moc_tempCorr[k] = pearson_corr.statistic
    moc_tempPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(moc, salt)
    moc_saltCorr[k] = pearson_corr.statistic
    moc_saltPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(moc, volT45N)
    moc_volT45NCorr[k] = pearson_corr.statistic
    moc_volT45NPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(moc, temp45N)
    moc_temp45NCorr[k] = pearson_corr.statistic
    moc_temp45NPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(moc, salt45N)
    moc_salt45NCorr[k] = pearson_corr.statistic
    moc_salt45NPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(tempSubPolar, temp45N)
    tSubPolar_t45NCorr[k] = pearson_corr.statistic
    tSubPolar_t45NPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(saltSubPolar, salt45N)
    sSubPolar_s45NCorr[k] = pearson_corr.statistic
    sSubPolar_s45NPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(tempSubPolar, tempDav)
    tSubPolar_tDavCorr[k] = pearson_corr.statistic
    tSubPolar_tDavPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(saltSubPolar, saltDav)
    sSubPolar_sDavCorr[k] = pearson_corr.statistic
    sSubPolar_sDavPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(tempSubPolar, tempDen)
    tSubPolar_tDenCorr[k] = pearson_corr.statistic
    tSubPolar_tDenPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(saltSubPolar, saltDen)
    sSubPolar_sDenCorr[k] = pearson_corr.statistic
    sSubPolar_sDenPval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(tempSubPolar, tempIce)
    tSubPolar_tIceCorr[k] = pearson_corr.statistic
    tSubPolar_tIcePval[k] = pearson_corr.pvalue
    #
    pearson_corr = stats.pearsonr(saltSubPolar, saltIce)
    sSubPolar_sIceCorr[k] = pearson_corr.statistic
    sSubPolar_sIcePval[k] = pearson_corr.pvalue
    #
    k = k + 1

fig = plt.figure(figsize=(15, 6), dpi=figdpi)
ax = fig.add_subplot()
titleText = f'Lagged cross-correlations between maxMOC at {np.int64(mocLat)} and upper-800m fields in the {regionName}'
titleText = f'{titleText}\n{runname} (positive lag is for maxMOC leading)'
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
ax.yaxis.get_offset_text().set_weight('bold')
ax.set_xlabel(xlabelText, fontsize=fontsize_labels, fontweight='bold')
ax.set_title(titleText, fontsize=fontsize_titles, fontweight='bold')
ax.set_xticks(lags)
original_labels = [str(label) for label in ax.get_xticks()]
labels_of_interest = [str(i) for i in lags[0::2]]
new_labels = [label if label in labels_of_interest else "" for label in original_labels]
ax.set_xticklabels(new_labels)
ax.set_xlim(lags[0], lags[-1])
ax.axhline(y=0, color='k', linestyle='-')
markerSize = 30

ax.plot(lags, moc_rhoCorr, 'k', linewidth=2, label='moc-drho')
sigValues = np.where(moc_rhoPval < pvalMax)
insigValues = np.where(moc_rhoPval >= pvalMax)
ax.scatter(lags[sigValues], moc_rhoCorr[sigValues], s=markerSize, c='k', marker='o')
ax.scatter(lags[insigValues], moc_rhoCorr[insigValues], s=markerSize, c='k', marker='o', alpha=0.3)
#
ax.plot(lags, moc_rhoAlphaCorr, 'r', linewidth=2, label='moc-drhoAlpha')
sigValues = np.where(moc_rhoAlphaPval < pvalMax)
insigValues = np.where(moc_rhoAlphaPval >= pvalMax)
ax.scatter(lags[sigValues], moc_rhoAlphaCorr[sigValues], s=markerSize, c='r', marker='o')
ax.scatter(lags[insigValues], moc_rhoAlphaCorr[insigValues], s=markerSize, c='r', marker='o', alpha=0.3)
#
ax.plot(lags, moc_rhoBetaCorr, 'b', linewidth=2, label='moc-drhoBeta')
sigValues = np.where(moc_rhoBetaPval < pvalMax)
insigValues = np.where(moc_rhoBetaPval >= pvalMax)
ax.scatter(lags[sigValues], moc_rhoBetaCorr[sigValues], s=markerSize, c='b', marker='o')
ax.scatter(lags[insigValues], moc_rhoBetaCorr[insigValues], s=markerSize, c='b', marker='o', alpha=0.3)
#
#ax.plot(lags, moc_tempCorr, 'salmon', linewidth=2, label='moc-dT')
#sigValues = np.where(moc_tempPval < pvalMax)
#insigValues = np.where(moc_tempPval >= pvalMax)
#ax.scatter(lags[sigValues], moc_tempCorr[sigValues], s=markerSize, c='salmon', marker='o')
#ax.scatter(lags[insigValues], moc_tempCorr[insigValues], s=markerSize, c='salmon', marker='o', alpha=0.3)
#
#ax.plot(lags, moc_saltCorr, 'dodgerblue', linewidth=2, label='moc-dS')
#sigValues = np.where(moc_saltPval < pvalMax)
#insigValues = np.where(moc_saltPval >= pvalMax)
#ax.scatter(lags[sigValues], moc_saltCorr[sigValues], s=markerSize, c='dodgerblue', marker='o')
#ax.scatter(lags[insigValues], moc_saltCorr[insigValues], s=markerSize, c='dodgerblue', marker='o', alpha=0.3)

ax.grid(color='gray', linestyle=':', linewidth=0.8, alpha=0.8)
ax.legend(prop=legend_properties)
fig.savefig(tsfigfile, dpi='figure', bbox_inches='tight', pad_inches=0.1)

fig = plt.figure(figsize=(15, 6), dpi=figdpi)
ax = fig.add_subplot()
titleText = f'Lagged cross-correlations between maxMOC at {np.int64(mocLat)} and upper-800m transports at 45N'
titleText = f'{titleText}\n{runname} (positive lag is for maxMOC leading)'
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
ax.yaxis.get_offset_text().set_weight('bold')
ax.set_xlabel(xlabelText, fontsize=fontsize_labels, fontweight='bold')
ax.set_title(titleText, fontsize=fontsize_titles, fontweight='bold')
ax.set_xticks(lags)
original_labels = [str(label) for label in ax.get_xticks()]
labels_of_interest = [str(i) for i in lags[0::2]]
new_labels = [label if label in labels_of_interest else "" for label in original_labels]
ax.set_xticklabels(new_labels)
ax.set_xlim(lags[0], lags[-1])
ax.axhline(y=0, color='k', linestyle='-')
markerSize = 30

ax.plot(lags, moc_volT45NCorr, 'k', linewidth=2, label='moc-volT')
sigValues = np.where(moc_volT45NPval < pvalMax)
insigValues = np.where(moc_volT45NPval >= pvalMax)
ax.scatter(lags[sigValues], moc_volT45NCorr[sigValues], s=markerSize, c='k', marker='o')
ax.scatter(lags[insigValues], moc_volT45NCorr[insigValues], s=markerSize, c='k', marker='o', alpha=0.3)
ax.plot(lags, moc_temp45NCorr, 'salmon', linewidth=2, label='moc-temp')
sigValues = np.where(moc_temp45NPval < pvalMax)
insigValues = np.where(moc_temp45NPval >= pvalMax)
ax.scatter(lags[sigValues], moc_temp45NCorr[sigValues], s=markerSize, c='salmon', marker='o')
ax.scatter(lags[insigValues], moc_temp45NCorr[insigValues], s=markerSize, c='salmon', marker='o', alpha=0.3)
ax.plot(lags, moc_salt45NCorr, 'dodgerblue', linewidth=2, label='moc-salt')
sigValues = np.where(moc_salt45NPval < pvalMax)
insigValues = np.where(moc_salt45NPval >= pvalMax)
ax.scatter(lags[sigValues], moc_salt45NCorr[sigValues], s=markerSize, c='dodgerblue', marker='o')
ax.scatter(lags[insigValues], moc_salt45NCorr[insigValues], s=markerSize, c='dodgerblue', marker='o', alpha=0.3)

ax.grid(color='gray', linestyle=':', linewidth=0.8, alpha=0.8)
ax.legend(prop=legend_properties)
fig.savefig(figfile45N, dpi='figure', bbox_inches='tight', pad_inches=0.1)

fig = plt.figure(figsize=(15, 6), dpi=figdpi)
ax = fig.add_subplot()
titleText = f'Lagged cross-correlations between upper-800m fields in the {regionName} and'
titleText = f'{titleText}\nfluxes at region open boundaries {runname} (positive lag is for reg-avg leading)'
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
ax.yaxis.get_offset_text().set_weight('bold')
ax.set_xlabel(xlabelText, fontsize=fontsize_labels, fontweight='bold')
ax.set_title(titleText, fontsize=fontsize_titles, fontweight='bold')
ax.set_xticks(lags)
original_labels = [str(label) for label in ax.get_xticks()]
labels_of_interest = [str(i) for i in lags[0::2]]
new_labels = [label if label in labels_of_interest else "" for label in original_labels]
ax.set_xticklabels(new_labels)
ax.set_xlim(lags[0], lags[-1])
ax.axhline(y=0, color='k', linestyle='-')
ax.axvline(x=0, color='k', linestyle='-')
markerSize = 30

ax.plot(lags, tSubPolar_t45NCorr, 'r', linewidth=1.2, label='dtemp-tflux45N')
sigValues = np.where(tSubPolar_t45NPval < pvalMax)
insigValues = np.where(tSubPolar_t45NPval >= pvalMax)
ax.scatter(lags[sigValues], tSubPolar_t45NCorr[sigValues], s=markerSize, c='r', marker='o')
ax.scatter(lags[insigValues], tSubPolar_t45NCorr[insigValues], s=markerSize, c='r', marker='o', alpha=0.3)
#
ax.plot(lags, sSubPolar_s45NCorr, 'b', linewidth=1.2, label='dsalt-sflux45N')
sigValues = np.where(sSubPolar_s45NPval < pvalMax)
insigValues = np.where(sSubPolar_s45NPval >= pvalMax)
ax.scatter(lags[sigValues], sSubPolar_s45NCorr[sigValues], s=markerSize, c='b', marker='o')
ax.scatter(lags[insigValues], sSubPolar_s45NCorr[insigValues], s=markerSize, c='b', marker='o', alpha=0.3)
#
ax.plot(lags, tSubPolar_tDavCorr, 'pink', linewidth=1.2, label='dtemp-tfluxDav')
sigValues = np.where(tSubPolar_tDavPval < pvalMax)
insigValues = np.where(tSubPolar_tDavPval >= pvalMax)
ax.scatter(lags[sigValues], tSubPolar_tDavCorr[sigValues], s=markerSize, c='pink', marker='o')
ax.scatter(lags[insigValues], tSubPolar_tDavCorr[insigValues], s=markerSize, c='pink', marker='o', alpha=0.3)
#
ax.plot(lags, sSubPolar_sDavCorr, 'lightskyblue', linewidth=1.2, label='dsalt-sfluxDav')
sigValues = np.where(sSubPolar_sDavPval < pvalMax)
insigValues = np.where(sSubPolar_sDavPval >= pvalMax)
ax.scatter(lags[sigValues], sSubPolar_sDavCorr[sigValues], s=markerSize, c='lightskyblue', marker='o')
ax.scatter(lags[insigValues], sSubPolar_sDavCorr[insigValues], s=markerSize, c='lightskyblue', marker='o', alpha=0.3)
#
ax.plot(lags, tSubPolar_tDenCorr, 'indianred', linewidth=1.2, label='dtemp-tfluxDen')
sigValues = np.where(tSubPolar_tDenPval < pvalMax)
insigValues = np.where(tSubPolar_tDenPval >= pvalMax)
ax.scatter(lags[sigValues], tSubPolar_tDenCorr[sigValues], s=markerSize, c='indianred', marker='o')
ax.scatter(lags[insigValues], tSubPolar_tDenCorr[insigValues], s=markerSize, c='indianred', marker='o', alpha=0.3)
#
ax.plot(lags, sSubPolar_sDenCorr, 'steelblue', linewidth=1.2, label='dsalt-sfluxDen')
sigValues = np.where(sSubPolar_sDenPval < pvalMax)
insigValues = np.where(sSubPolar_sDenPval >= pvalMax)
ax.scatter(lags[sigValues], sSubPolar_sDenCorr[sigValues], s=markerSize, c='steelblue', marker='o')
ax.scatter(lags[insigValues], sSubPolar_sDenCorr[insigValues], s=markerSize, c='steelblue', marker='o', alpha=0.3)
#
ax.plot(lags, tSubPolar_tIceCorr, 'salmon', linewidth=1.2, label='dtemp-tfluxIce')
sigValues = np.where(tSubPolar_tIcePval < pvalMax)
insigValues = np.where(tSubPolar_tIcePval >= pvalMax)
ax.scatter(lags[sigValues], tSubPolar_tIceCorr[sigValues], s=markerSize, c='salmon', marker='o')
ax.scatter(lags[insigValues], tSubPolar_tIceCorr[insigValues], s=markerSize, c='salmon', marker='o', alpha=0.3)
#
ax.plot(lags, sSubPolar_sIceCorr, 'dodgerblue', linewidth=1.2, label='dsalt-sfluxIce')
sigValues = np.where(sSubPolar_sIcePval < pvalMax)
insigValues = np.where(sSubPolar_sIcePval >= pvalMax)
ax.scatter(lags[sigValues], sSubPolar_sIceCorr[sigValues], s=markerSize, c='dodgerblue', marker='o')
ax.scatter(lags[insigValues], sSubPolar_sIceCorr[insigValues], s=markerSize, c='dodgerblue', marker='o', alpha=0.3)

ax.grid(color='gray', linestyle=':', linewidth=0.8, alpha=0.8)
ax.legend(prop=legend_properties)
fig.savefig(trfigfile, dpi='figure', bbox_inches='tight', pad_inches=0.1)
