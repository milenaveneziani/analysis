#
# Plots various MOC time series related stuff
#
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import cftime
import netCDF4
from datetime import datetime
mpl.use('TkAgg')


# Settings for lcrc
#runname = '20240729.HRr5-test12' # timeseries/ will be prepended to this
#mocdir = '/lcrc/group/acme/ac.jwolfe/case_output/20240626.HRr5-test11.chrysalis/timeseries/moc'
#regionalTSdir = '/home/ac.milena/analysis-MPAS/analysis-git-repo/timeseries_data/20240729.HRr5-test12'
#runname = '20240519_icos30_JRAp5_wISC30E3r5' # timeseries/ will be prepended to this
#mocdir = '/lcrc/group/e3sm/ac.fspereira/scratch/anvil/diagnostics/20240519_icos30_JRAp5_wISC30E3r5_001_189/timeseries/moc'
#regionalTSdir = ''

# Settings for chicoma
#runname = '20240726.icFromLRGcase.GMPAS-JRA1p5.TL319_RRSwISC6to18E3r5.chicoma' # timeseries/ will be prepended to this
#mocdir = '/lustre/scratch4/turquoise/milena/E3SMv3/20240726.icFromLRGcase.GMPAS-JRA1p5.TL319_RRSwISC6to18E3r5.chicoma/mpas-analysis/clim_1-10_ts_1-31/timeseries/moc'
#regionalTSdir = '/users/milena/analysis-git-repo/timeseries_data/20240726.icFromLRGcase.GMPAS-JRA1p5.TL319_RRSwISC6to18E3r5.chicoma'

# Settings for nersc
runname1 = 'E3SMv2.1B60to10rA02'
runname2 = 'E3SMv2.1B60to10rA07'
runname3 = 'E3SMv2.1G60to10_01'
mocdir = '/global/cfs/cdirs/m4259/milena/AMOCpaper/moc_timeseries'
mocfile1 = f'{mocdir}/{runname1}/mocTimeSeries_0001-0386.nc'
mocfile2 = f'{mocdir}/{runname2}/mocTimeSeries_0001-0246.nc'
mocfile3 = f'{mocdir}/{runname3}/mocTimeSeries_0001-0059.nc'
regionalTSdir = '/global/cfs/cdirs/m4259/milena/AMOCpaper/timeseries_data'

moccolors = ['black', 'magenta', 'green'] # same length as number of runnames
#moccolors = ['mediumblue', 'dodgerblue', 'teal'] #, 'lightseagreen', 'green'] # same length as number of runnames

regionGroup = 'arctic_atlantic_budget_regions_new20240408'

yearStart = 1
yearEnd = 50

tsvariable = 'iceArea'
#tsvariable = 'iceVolume'
regionName = 'Greater Arctic'
#regionName = 'Nordic Seas'
#regionName = 'North Atlantic subpolar gyre'
regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '')
moclats = [-34, 26, 45, 60, 65]
plotAnnualAvg = True
referenceDate = '0001-01-01 00:00:00'
calendar = 'noleap'
daysInYear = 365.0
# The following is relevant only if plotAnnualAvg = False
movingAverageMonths = 12 # months

npanelsToPlot = 1 # only plots multi-model moc (for each moclats)
# This script needs to be edited in order to plot more than 1 panel (as of Apr 2026)
#npanelsToPlot = 2 # plots multi-model moc (for each moclats) and regional time series
#npanelsToPlot = 3 # plots multi-model moc (for each moclats), regional time series, and depth of max moc (for each moclats)
if npanelsToPlot>3:
    raise SystemExit('Trying to plot more than 3 panels. Exiting...\n')
figsize = (12, npanelsToPlot*8)

figdir = f'./moc_multimodel'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figdpi = 300
fontsize_smallLabels = 14
fontsize_labels = 16
fontsize_titles = 18
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
################

# Read in moc datasets
if os.path.exists(mocfile1):
    dsMOC1 = xr.open_dataset(mocfile1)
else:
    raise IOError(f'MOC file {mocfile1} not found')
if os.path.exists(mocfile2):
    dsMOC2 = xr.open_dataset(mocfile2)
else:
    raise IOError(f'MOC file {mocfile2} not found')
if os.path.exists(mocfile3):
    dsMOC3 = xr.open_dataset(mocfile3)
else:
    raise IOError(f'MOC file {mocfile3} not found')

#time1 = np.arange(len(dsMOC1.Time.values))/12
#time2 = np.arange(len(dsMOC2.Time.values))/12
#time3 = np.arange(len(dsMOC3.Time.values))/12
#time1 = dsMOC1['Time']
#time2 = dsMOC2['Time']
#time3 = dsMOC3['Time']
time1 = cftime.date2num(np.hstack(dsMOC1['Time']), f'days since {referenceDate}')
time2 = cftime.date2num(np.hstack(dsMOC2['Time']), f'days since {referenceDate}')
time3 = cftime.date2num(np.hstack(dsMOC3['Time']), f'days since {referenceDate}')

for nlat in range(len(moclats)):
    lat = moclats[nlat]

    if lat>0:
        legendlabel = f'{lat:d}N'
    elif lat<0:
        legendlabel = f'{np.abs(lat):d}S'
    else:
        legendlabel = f'{lat:d}'

    if npanelsToPlot==1:
        if plotAnnualAvg is True:
            figfile = f'{figdir}/moc{legendlabel}Annual_multimodel.png'
        else:
            figfile = f'{figdir}/moc{legendlabel}_multimodel.png'
    elif npanelsToPlot==2:
        figfile = f'{figdir}/moc{legendlabel}_{tsvariable}_{regionNameShort}_multimodel.png'
    else:
        figfile = f'{figdir}/depthMocLats{legendlabel}.png'
    fig = plt.figure(figsize=figsize, dpi=figdpi)

    if npanelsToPlot==1:
        ax1  = fig.add_subplot()
        ax1.set_xlabel('Years', fontsize=fontsize_labels, fontweight='bold')
    else:
        ax1 = fig.add_subplot(npanelsToPlot, 1, 1)
        ax2 = fig.add_subplot(npanelsToPlot, 1, 2)
        if npanelsToPlot==3:
            ax3 = fig.add_subplot(npanelsToPlot, 1, 3)

    #### First plot max MOC time series for various models ####
    moc1 = dsMOC1.mocAtlantic.sel(lat=lat, method='nearest').max(dim='depth')
    moc2 = dsMOC2.mocAtlantic.sel(lat=lat, method='nearest').max(dim='depth')
    moc3 = dsMOC3.mocAtlantic.sel(lat=lat, method='nearest').max(dim='depth')
    #legendlabel = f'{legendlabel} ({moc.mean():5.2f} $\pm$ {moc.std():5.2f})'
    if plotAnnualAvg is True:
        weights = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])/daysInYear

        datetimes = netCDF4.num2date(time1, f'days since {referenceDate}', calendar=calendar)
        t_years = np.empty(np.shape(time1), dtype=np.int64)
        for i, date in enumerate(datetimes.flat):
            t_years[i] = date.year
        years1 = range(1, 387)
        t1_annual = np.empty(np.shape(years1), dtype=np.float64)
        moc1_annual = np.empty(np.shape(years1), dtype=np.float64)
        k = 0
        for year in years1:
            indexYears = np.where(t_years==year)[0]
            t1_annual[k] = np.mean(time1[indexYears]/daysInYear)
            moc1_annual[k] = np.sum(weights * moc1.values[indexYears])
            k = k + 1

        datetimes = netCDF4.num2date(time2, f'days since {referenceDate}', calendar=calendar)
        t_years = np.empty(np.shape(time2), dtype=np.int64)
        for i, date in enumerate(datetimes.flat):
            t_years[i] = date.year
        years2 = range(1, 247)
        t2_annual = np.empty(np.shape(years2), dtype=np.float64)
        moc2_annual = np.empty(np.shape(years2), dtype=np.float64)
        k = 0
        for year in years2:
            indexYears = np.where(t_years==year)[0]
            t2_annual[k] = np.mean(time2[indexYears]/daysInYear)
            moc2_annual[k] = np.sum(weights * moc2.values[indexYears])
            k = k + 1

        datetimes = netCDF4.num2date(time3, f'days since {referenceDate}', calendar=calendar)
        t_years = np.empty(np.shape(time3), dtype=np.int64)
        for i, date in enumerate(datetimes.flat):
            t_years[i] = date.year
        years3 = range(1, 60)
        t3_annual = np.empty(np.shape(years3), dtype=np.float64)
        moc3_annual = np.empty(np.shape(years3), dtype=np.float64)
        k = 0
        for year in years3:
            indexYears = np.where(t_years==year)[0]
            t3_annual[k] = np.mean(time3[indexYears]/daysInYear)
            moc3_annual[k] = np.sum(weights * moc3.values[indexYears])
            k = k + 1

        ax1.plot(t1_annual, moc1_annual, '-', color=moccolors[0], linewidth=2, label=runname1)
        ax1.plot(t2_annual, moc2_annual, '-', color=moccolors[1], linewidth=2, label=runname2)
        ax1.plot(t3_annual, moc3_annual, '-', color=moccolors[2], linewidth=2, label=runname3)
    else:
        if movingAverageMonths>1:
            window = int(movingAverageMonths)
            moc1_runavg = pd.Series(moc1).rolling(window, center=True).mean()
            moc2_runavg = pd.Series(moc2).rolling(window, center=True).mean()
            moc3_runavg = pd.Series(moc3).rolling(window, center=True).mean()
            ax1.plot(time1, moc1_runavg, '-', color=moccolors[0], linewidth=2, label=runname1)
            ax1.plot(time2, moc2_runavg, '-', color=moccolors[1], linewidth=2, label=runname2)
            ax1.plot(time3, moc3_runavg, '-', color=moccolors[2], linewidth=2, label=runname3)
            #ax1.plot(time1, moc1, '-', color=moccolors[0], alpha=0.5, linewidth=1.2)
            if npanelsToPlot>1:
                ax2.plot(time1, moc1_runavg, '-', color=moccolors[0], linewidth=2, label=runaname1)
                ax2.plot(time2, moc2_runavg, '-', color=moccolors[1], linewidth=2, label=runaname2)
                ax2.plot(time3, moc3_runavg, '-', color=moccolors[2], linewidth=2, label=runaname3)
        else:
            ax1.plot(time1, moc1, '-', color=moccolors[0], linewidth=2, label=runname1)
            ax1.plot(time2, moc2, '-', color=moccolors[1], linewidth=2, label=runname2)
            ax1.plot(time3, moc3, '-', color=moccolors[2], linewidth=2, label=runname3)
            if npanelsToPlot>1:
                ax2.plot(time1, moc1, '-', color=moccolors[0], linewidth=2, label=runname1)
                ax2.plot(time2, moc2, '-', color=moccolors[1], linewidth=2, label=runname2)
                ax2.plot(time3, moc3, '-', color=moccolors[2], linewidth=2, label=runname3)
    ax1.grid(visible=True, which='both')
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    ax1.legend(prop=legend_properties)
    ax1.set_ylabel('Sv', fontsize=fontsize_labels, fontweight='bold')
    ax1.set_title(f'Max AMOC at {legendlabel}', fontsize=fontsize_titles, fontweight='bold')
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlim(0, yearEnd)
    #ax1.set_xlim(0, 50)

    # This below needs to be edited in order to work (point to the right data)
    if npanelsToPlot>1:
        #### Plot max MOC time series for various latitudes plus ice time series ####
        ax2.grid(visible=True, which='both')
        ax2.tick_params(axis='y', labelcolor='dodgerblue')
        for tick in ax2.xaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        for tick in ax2.yaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        ax2.spines['left'].set_color('dodgerblue')
        #ax2.legend(prop=legend_properties, loc='lower right')
        ##ax2.legend(prop=legend_properties)
        if npanelsToPlot==2:
            ax2.set_xlabel('Years', fontsize=fontsize_labels, fontweight='bold')
        ax2.set_ylabel('Sv', fontsize=fontsize_labels, fontweight='bold', color='dodgerblue')
        ax2.autoscale(enable=True, axis='x', tight=True)
        ax2.set_xlim(0, yearEnd)

        axice = ax2.twinx()
        axicecolor = 'firebrick'
        tsfiles = []
        for year in range(yearStart, yearEnd+1):
            tsfile = f'{regionalTSdir}/{tsvariable}/{regionGroup}_year{year:04d}.nc'
            if os.path.exists(tsfile):
                tsfiles.append(tsfile)
            else:
                raise IOError(f'Time series file {tsfile} not found')
        dsTS = xr.open_mfdataset(tsfiles)
        regionNames = dsTS.regionNames.isel(Time=0).values
        regionIndex = np.where(regionNames==regionName)
        ts = dsTS[f'{tsvariable}'].isel(nRegions=regionIndex[0]).squeeze(dim='nRegions')

        if movingAverageMonths>1:
            window = int(movingAverageMonths)
            ts_runavg = pd.Series(ts).rolling(window, center=True).mean()
            axice.plot(time, ts_runavg, '-', color=axicecolor, linewidth=2)
            #axice.plot(time, ts, '-', color=axicecolor, alpha=0.5, linewidth=1.2)
        else:
            axice.plot(time, ts, '-', color=axicecolor, linewidth=2)
        axice.tick_params(axis='y', labelcolor=axicecolor)
        axice.spines['right'].set_color(axicecolor)
        for tick in axice.yaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        axice.set_ylabel(f'{tsvariable} in the {regionName} region', fontsize=fontsize_labels, fontweight='bold', color=axicecolor)

        if npanelsToPlot==3:
            #### Add time series of depth of max MOC for various latitudes ####
            ax3.grid(visible=True, which='both')
            for tick in ax3.xaxis.get_ticklabels():
                tick.set_fontsize(fontsize_smallLabels)
                tick.set_weight('bold')
            for tick in ax3.yaxis.get_ticklabels():
                tick.set_fontsize(fontsize_smallLabels)
                tick.set_weight('bold')
            ax3.legend(prop=legend_properties)
            ax3.set_xlabel('Years', fontsize=fontsize_labels, fontweight='bold')
            ax3.set_ylabel('depth of max MOC (m)', fontsize=fontsize_labels, fontweight='bold')
            ax3.autoscale(enable=True, axis='x', tight=True)
            ax3.set_xlim(0, yearEnd)

    fig.tight_layout(pad=0.5)
    fig.savefig(figfile, bbox_inches='tight')
    plt.close(fig)
