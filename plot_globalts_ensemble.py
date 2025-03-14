#
# Plots time series of MOC at 26N for an ensemble of simulations
#
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import datetime
import netCDF4

#from common_functions import plot_xtick_format, days_to_datetime


def _add_figure_panel(figsize, figdpi, figtitle, figylabel, xlim):

    fontsize_smallLabels = 10
    fontsize_labels = 16
    fontsize_titles = 18
    legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}

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
    ax.set_xlabel('Years', fontsize=fontsize_labels, fontweight='bold')
    ax.set_ylabel(figylabel, fontsize=fontsize_labels, fontweight='bold')
    ax.set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
    ax.set_xlim(xlim[0], xlim[1])
    ax.grid(visible=True, which='both')
    return [fig, ax]


#figdir = 'E3SM-LRv2.1'
figdir = 'E3SM-Arcticv2.1'
#figdir = 'E3SMv2.1B60to10rA02'
maindir = '/global/cfs/cdirs/m1199/e3sm-arrm-simulations'
#maindir = '/global/cfs/cdirs/m1199/E3SMv2.1-LR'
simsToPlot = [
              # {'mocDatadir': f'{maindir}/E3SMv2.1B60to10rA02/mpas-analysis/timeseries/moc',
              #  'iceDatadir': f'{maindir}/E3SMv2.1B60to10rA02/mpas-analysis/timeseries',
              #  'ocnDatadir': f'{maindir}/E3SMv2.1B60to10rA02/globalTS/ocn/glb/ts/monthly/386yr',
              #  'atmDatadir': f'{maindir}/E3SMv2.1B60to10rA02/globalTS/atm/glb/ts/monthly/386yr',
              #  'yearStart': 1,
              #  'yearEnd':386,
              #  'shiftyear': 0,
              #  'label': '1950-control',
              #  'color': 'k',
              #  'linewidth': 1.2,
              #  'alpha': 1}
              ##  'alpha': 1},
               {'mocDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0101/mpas-analysis/timeseries/moc',
                'iceDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0101/mpas-analysis/timeseries',
                'ocnDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0101/globalTS/ocn/glb/ts/monthly/65yr',
                'atmDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0101/globalTS/atm/glb/ts/monthly/65yr',
              # {'mocDatadir': f'{maindir}/v2_1.LR.historical_0101/post/ocn/glb/ts/monthly/5yr',
              #  'iceDatadir': f'{maindir}/v2_1.LR.historical_0101/post/analysis/mpas_analysis/ts_1850-2014_climo_1985-2014/timeseries',
              #  'ocnDatadir': f'{maindir}/v2_1.LR.historical_0101/post/ocn/glb/ts/monthly/165yr',
              #  'atmDatadir': f'{maindir}/v2_1.LR.historical_0101/post/atm/glb/ts/monthly/165yr',
                'yearStart': 1950,
              #  'yearStart': 1850,
                'yearEnd':2014,
              #  'shiftyear': 101,
                'shiftyear': 1950,
              #  'shiftyear': 1850,
                'label': 'hist 101',
                'color': 'mediumblue',
                'linewidth': 1.2,
                'alpha': 0.6},
               {'mocDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0151/mpas-analysis/timeseries/moc',
                'iceDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0151/mpas-analysis/timeseries',
                'ocnDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0151/globalTS/ocn/glb/ts/monthly/65yr',
                'atmDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0151/globalTS/atm/glb/ts/monthly/65yr',
              # {'mocDatadir': f'{maindir}/v2_1.LR.historical_0151/post/ocn/glb/ts/monthly/5yr',
              #  'iceDatadir': f'{maindir}/v2_1.LR.historical_0151/post/analysis/mpas_analysis/ts_1850-2014_climo_1985-2014/timeseries',
              #  'ocnDatadir': f'{maindir}/v2_1.LR.historical_0151/post/ocn/glb/ts/monthly/165yr',
              #  'atmDatadir': f'{maindir}/v2_1.LR.historical_0151/post/atm/glb/ts/monthly/165yr',
                'yearStart': 1950,
              #  'yearStart': 1850,
                'yearEnd':2014,
              #  'shiftyear': 151,
                'shiftyear': 1950,
              #  'shiftyear': 1850,
                'label': 'hist 151',
                'color': 'dodgerblue',
                'linewidth': 1.2,
                'alpha': 0.6},
               {'mocDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0201/mpas-analysis/timeseries/moc',
                'iceDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0201/mpas-analysis/timeseries',
                'ocnDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0201/globalTS/ocn/glb/ts/monthly/65yr',
                'atmDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0201/globalTS/atm/glb/ts/monthly/65yr',
              # {'mocDatadir': f'{maindir}/v2_1.LR.historical_0201/post/ocn/glb/ts/monthly/5yr',
              #  'iceDatadir': f'{maindir}/v2_1.LR.historical_0201/post/analysis/mpas_analysis/ts_1850-2014_climo_1985-2014/timeseries',
              #  'ocnDatadir': f'{maindir}/v2_1.LR.historical_0201/post/ocn/glb/ts/monthly/165yr',
              #  'atmDatadir': f'{maindir}/v2_1.LR.historical_0201/post/atm/glb/ts/monthly/165yr',
              #  'shiftyear': 1850,
              #  'yearStart': 1850,
              ## {'mocDatadir': f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/E3SM-Arctv2.1_60to30cAhis0201/mpas-analysis/timeseries/moc',
              ##  'iceDatadir': f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/E3SM-Arctv2.1_60to30cAhis0201/mpas-analysis/timeseries',
              ##  'ocnDatadir': f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/E3SM-Arctv2.1_60to30cAhis0201/globalTS/ocn/glb/ts/monthly/65yr',
              ##  'atmDatadir': f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/E3SM-Arctv2.1_60to30cAhis0201/globalTS/atm/glb/ts/monthly/65yr',
                'yearStart': 1950,
                'yearEnd':2014,
              #  'shiftyear': 201,
                'shiftyear': 1950,
                'label': 'hist 201',
                'color': 'deepskyblue',
                'linewidth': 1.2,
                'alpha': 0.6},
               {'mocDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0251/mpas_analysis_output/yrs2000-2014/timeseries/moc',
                'iceDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0251/mpas_analysis_output/yrs2000-2014/timeseries',
                'ocnDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0251/globalTS/ocn/glb/ts/monthly/65yr',
                'atmDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0251/globalTS/atm/glb/ts/monthly/65yr',
              # {'mocDatadir': f'{maindir}/v2_1.LR.historical_0251/post/ocn/glb/ts/monthly/5yr',
              #  'iceDatadir': f'{maindir}/v2_1.LR.historical_0251/post/analysis/mpas_analysis/ts_1850-2014_climo_1985-2014/timeseries',
              #  'ocnDatadir': f'{maindir}/v2_1.LR.historical_0251/post/ocn/glb/ts/monthly/165yr',
              #  'atmDatadir': f'{maindir}/v2_1.LR.historical_0251/post/atm/glb/ts/monthly/165yr',
                'yearStart': 1950,
              #  'yearStart': 1850,
                'yearEnd':2014,
              #  'shiftyear': 251,
                'shiftyear': 1950,
              #  'shiftyear': 1850,
                'label': 'hist 251',
                'color': 'lightseagreen',
                'linewidth': 1.2,
                'alpha': 0.6},
               {'mocDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0301/mpas_analysis_output/yrs2000-2014/timeseries/moc',
                'iceDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0301/mpas_analysis_output/yrs2000-2014/timeseries',
                'ocnDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0301/globalTS/ocn/glb/ts/monthly/65yr',
                'atmDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0301/globalTS/atm/glb/ts/monthly/65yr',
              # {'mocDatadir': f'{maindir}/v2_1.LR.historical_0301/post/ocn/glb/ts/monthly/5yr',
              #  'iceDatadir': f'{maindir}/v2_1.LR.historical_0301/post/analysis/mpas_analysis/ts_1850-2014_climo_1985-2014/timeseries',
              #  'ocnDatadir': f'{maindir}/v2_1.LR.historical_0301/post/ocn/glb/ts/monthly/165yr',
              #  'atmDatadir': f'{maindir}/v2_1.LR.historical_0301/post/atm/glb/ts/monthly/165yr',
                'yearStart': 1950,
              #  'yearStart': 1850,
                'yearEnd':2014,
              #  'shiftyear': 301,
                'shiftyear': 1950,
              #  'shiftyear': 1850,
                'label': 'hist 301',
                'color': 'green',
                'linewidth': 1.2,
                'alpha': 0.6}
            ]

plotEnsembleMean = True # Turn this off if plotting a PI or 1950 control on top of the ensemble
movingAverageMonths = 12 # months

obsdir = '/global/cfs/cdirs/m1199/milena/Obs4E3SM-ArcticPaper'
##############################################################
figdir = f'./timeseries/{figdir}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figsize = (15, 5)
figdpi = 150
fontsize_smallLabels = 10
fontsize_labels = 16
fontsize_titles = 18
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
################

figfile_moc = f'{figdir}/moc26.png'
figtitle = f'Max MOC at 26N ({int(movingAverageMonths/12)}-year running avg)'
figylabel = 'Sv'
[fig_moc, ax_moc] = _add_figure_panel(figsize, figdpi, figtitle, figylabel, [1950, 2014])
#[fig_moc, ax_moc] = _add_figure_panel(figsize, figdpi, figtitle, figylabel, [0, 386])

figfile_restom = f'{figdir}/restom.png'
figtitle = f'Top of the atmosphere energy budget ({int(movingAverageMonths/12)}-year running avg)'
figylabel = 'W/m$^2$'
[fig_restom, ax_restom] = _add_figure_panel(figsize, figdpi, figtitle, figylabel, [1950, 2014])
#[fig_restom, ax_restom] = _add_figure_panel(figsize, figdpi, figtitle, figylabel, [0, 386])

figfile_ts = f'{figdir}/ts.png'
figtitle = f'Global surface temperature ({int(movingAverageMonths/12)}-year running avg)'
figylabel = 'K'
[fig_ts, ax_ts] = _add_figure_panel(figsize, figdpi, figtitle, figylabel, [1950, 2014])
#[fig_ts, ax_ts] = _add_figure_panel(figsize, figdpi, figtitle, figylabel, [0, 386])

figfile_ohc = f'{figdir}/ohc.png'
figtitle = f'Global ocean heat content ({int(movingAverageMonths/12)}-year running avg)'
figylabel = 'J'
[fig_ohc, ax_ohc] = _add_figure_panel(figsize, figdpi, figtitle, figylabel, [1950, 2014])
#[fig_ohc, ax_ohc] = _add_figure_panel(figsize, figdpi, figtitle, figylabel, [0, 386])

figfile_ice = f'{figdir}/iceVolume.png'
figtitle = f'Northern Hemisphere integrated ice volume ({int(movingAverageMonths/12)}-year running avg)'
figylabel = 'm$^3$'
[fig_ice, ax_ice] = _add_figure_panel(figsize, figdpi, figtitle, figylabel, [1950, 2015])
#[fig_ice, ax_ice] = _add_figure_panel(figsize, figdpi, figtitle, figylabel, [0, 386])

# Process observations
# moc from RAPID
obsfile = f'{obsdir}/RAPID/moc_transports.nc'
dsObs = xr.open_dataset(obsfile)
yearsAll = dsObs.time.dt.year
years = np.unique(yearsAll)
moc = dsObs.moc_mar_hc10.values
mocObs_yearly = np.zeros(len(years))
for iy, year in enumerate(years):
    mocObs_yearly[iy] = np.nanmean(moc[yearsAll==year])
mocObs_years = years

# restom from CERES-EBAS
obsfile = f'{obsdir}/CERES_EBAF-TOA/CERES_EBAF-TOA_Edition4.1_200003-202203.nc'
dsObs = xr.open_dataset(obsfile)
yearsAll = dsObs.time.dt.year
years = np.unique(yearsAll)
restomObs = dsObs.gtoa_net_all_mon.values
restomObs_yearly = np.zeros(len(years))
for iy, year in enumerate(years):
    restomObs_yearly[iy] = np.nanmean(restomObs[yearsAll==year])
restomObs_years = years

# surface temperature from HadCRUT5
obsfile = f'{obsdir}/HadCRUT5.0/HadCRUT5.0Analysis_gl.txt'
tsObs_annual = 13.974 # from HadCRUT5.0/abs_glnhsh.txt
tsObs_annual = tsObs_annual + 273.15
f = open(obsfile, 'r')
years = []
tsObs_yearly = []
for line in f:
    line = line.split()
    if len(line)==14:
        years.append(np.int16(line[0]))
        tsObs_yearly.append(np.float64(line[13]) + tsObs_annual)
tsObs_years = years

# OHC from NOAA (0-2000m)
obsfile = f'{obsdir}/OHC_NOAA/heat_content_anomaly_0-2000_yearly.nc'
dsObs = xr.open_dataset(obsfile, decode_times=False)
ohcObs_years = np.int16(dsObs.time/12 + 1955) # months since Jan 1955
ohcObs_yearly = 1e22*dsObs.yearl_h22_WO
ohcObs_anomaly = ohcObs_yearly-np.nanmean(ohcObs_yearly)

moc26_ensembleMean = 0.0
restom_ensembleMean = 0.0
ohc_ensembleMean = 0.0
iceVol_ensembleMean = 0.0
ts_ensembleMean = 0.0

for sim in simsToPlot:
    mocindir = sim['mocDatadir']
    iceindir = sim['iceDatadir']
    ocnindir = sim['ocnDatadir']
    atmindir = sim['atmDatadir']
    yearStart = sim['yearStart']
    yearEnd = sim['yearEnd']
    shiftyear = sim['shiftyear']
    legendlabel = sim['label']
    linecolor = sim['color']
    linewidth = sim['linewidth']
    linealpha = sim['alpha']

    infile = f'{mocindir}/mocTimeSeries_{yearStart:04d}-{yearEnd:04d}.nc'
    if os.path.exists(infile):
        dsIn = xr.open_dataset(infile)
    else:
        raise IOError(f'MOC file {infile} not found')
    moc26 = dsIn.mocAtlantic26
    if movingAverageMonths!=1:
        window = int(movingAverageMonths)
        moc26 = pd.Series(moc26).rolling(window, center=True).mean()
    moc26_ensembleMean = moc26_ensembleMean + moc26.values
    kmonths = len(moc26.values)
    time = np.arange(kmonths)/12 + shiftyear
    # The following doesn't work. Honestly, I don't know why python handles times so weirdly..
    #time = dsIn.Time
    #if adjustTime is True:
    #    newtime = []
    #    for t in time:
    #        delta_seconds = 1e-9*(t - time[0]) # ns to s
    #        newtime.append(datetime.datetime(shiftyear, 1, 1) + datetime.timedelta(days=0, seconds=delta_seconds))
    #    time = newtime
    ax_moc.plot(time, moc26, '-', color=linecolor, alpha=linealpha, linewidth=linewidth)
    #ax_moc.plot(time, moc26, '-', color=linecolor, alpha=linealpha, linewidth=linewidth, label=legendlabel)

    infile = f'{atmindir}/FSNT_{yearStart:04d}01_{yearEnd:04d}12.nc'
    if os.path.exists(infile):
        #dsFSNT = xr.open_dataset(infile) # older global time series diag did not include other regions
        dsFSNT = xr.open_dataset(infile).isel(rgn=0)
    else:
        raise IOError(f'FSNT file {infile} not found')
    infile = f'{atmindir}/FLNT_{yearStart:04d}01_{yearEnd:04d}12.nc'
    if os.path.exists(infile):
        #dsFLNT = xr.open_dataset(infile) # older global time series diag did not include other regions
        dsFLNT = xr.open_dataset(infile).isel(rgn=0)
    else:
        raise IOError(f'FLNT file {infile} not found')
    restom = dsFSNT.FSNT - dsFLNT.FLNT
    if movingAverageMonths!=1:
        window = int(movingAverageMonths)
        restom = pd.Series(restom).rolling(window, center=True).mean()
    restom_ensembleMean = restom_ensembleMean + restom.values
    ax_restom.plot(time, restom, '-', color=linecolor, alpha=linealpha, linewidth=linewidth)
    #ax_restom.plot(time, restom, '-', color=linecolor, alpha=linealpha, linewidth=linewidth, label=legendlabel)

    infile = f'{atmindir}/TS_{yearStart:04d}01_{yearEnd:04d}12.nc'
    if os.path.exists(infile):
        #dsTS = xr.open_dataset(infile) # older global time series diag did not include other regions
        dsTS = xr.open_dataset(infile).isel(rgn=0)
    else:
        raise IOError(f'TS file {infile} not found')
    ts = dsTS.TS
    if movingAverageMonths!=1:
        window = int(movingAverageMonths)
        ts = pd.Series(ts).rolling(window, center=True).mean()
    ts_ensembleMean = ts_ensembleMean + ts.values
    ax_ts.plot(time, ts, '-', color=linecolor, alpha=linealpha, linewidth=linewidth)
    #ax_ts.plot(time, ts, '-', color=linecolor, alpha=linealpha, linewidth=linewidth, label=legendlabel)

    infile = f'{ocnindir}/mpaso.glb.{yearStart:04d}01-{yearEnd:04d}12.nc'
    if os.path.exists(infile):
        dsOHC = xr.open_dataset(infile)
    else:
        raise IOError(f'OHC file {infile} not found')
    ohc = dsOHC.ohc
    if movingAverageMonths!=1:
        window = int(movingAverageMonths)
        ohc = pd.Series(ohc).rolling(window, center=True).mean()
    ohc_ensembleMean = ohc_ensembleMean + ohc.values
    ax_ohc.plot(time, ohc, '-', color=linecolor, alpha=linealpha, linewidth=linewidth)
    #ax_ohc.plot(time, ohc, '-', color=linecolor, alpha=linealpha, linewidth=linewidth, label=legendlabel)

    infile = f'{iceindir}/seaIceAreaVolNH.nc'
    if os.path.exists(infile):
        dsIce = xr.open_dataset(infile)
    else:
        raise IOError(f'Sea ice file {infile} not found')
    iceVol = dsIce.iceVolume
    if movingAverageMonths!=1:
        window = int(movingAverageMonths)
        iceVol = pd.Series(iceVol).rolling(window, center=True).mean()
    iceVol_ensembleMean = iceVol_ensembleMean + iceVol.values
    ax_ice.plot(time, iceVol, '-', color=linecolor, alpha=linealpha, linewidth=linewidth, label=legendlabel)

nsims = len(simsToPlot)

moc26_ensembleMean = moc26_ensembleMean/nsims
restom_ensembleMean = restom_ensembleMean/nsims
ts_ensembleMean = ts_ensembleMean/nsims
ohc_ensembleMean = ohc_ensembleMean/nsims
iceVol_ensembleMean = iceVol_ensembleMean/nsims

if plotEnsembleMean is True:
    ax_moc.plot(time, moc26_ensembleMean, '-', color='black', alpha=1, linewidth=2, label='ensemble mean')
ax_moc.plot(mocObs_years, mocObs_yearly, '-', color='salmon', linewidth=2, label='obs (RAPID)')
ax_moc.legend(prop=legend_properties)
#ax_moc.legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1, 0.5))
fig_moc.savefig(figfile_moc, bbox_inches='tight')
plt.close(fig_moc)

if plotEnsembleMean is True:
    ax_restom.plot(time, restom_ensembleMean, '-', color='black', alpha=1, linewidth=2, label='ensemble mean')
ax_restom.plot(restomObs_years[restomObs_years<=2014], restomObs_yearly[restomObs_years<=2014], '-', color='salmon', linewidth=2, label='obs (CERES-EBAS)')
ax_restom.legend(prop=legend_properties)
fig_restom.savefig(figfile_restom, bbox_inches='tight')
plt.close(fig_restom)

if plotEnsembleMean is True:
    ax_ts.plot(time, ts_ensembleMean, '-', color='black', alpha=1, linewidth=2, label='ensemble mean')
ax_ts.plot(tsObs_years, tsObs_yearly, '-', color='salmon', linewidth=2, label='obs (HadCRUT5)')
ax_ts.legend(prop=legend_properties)
fig_ts.savefig(figfile_ts, bbox_inches='tight')
plt.close(fig_ts)

if plotEnsembleMean is True:
    ax_ohc.plot(time, ohc_ensembleMean, '-', color='black', alpha=1, linewidth=2, label='ensemble mean')
ax_ohc.plot(ohcObs_years, ohcObs_anomaly+np.nanmean(ohc_ensembleMean), '-', color='salmon', linewidth=2, label='obs trend (upper 2000 m)')
ax_ohc.legend(prop=legend_properties)
fig_ohc.savefig(figfile_ohc, bbox_inches='tight')
plt.close(fig_ohc)

if plotEnsembleMean is True:
    ax_ice.plot(time, iceVol_ensembleMean, '-', color='black', alpha=1, linewidth=2, label='ensemble mean')
ax_ice.legend(prop=legend_properties)
fig_ice.savefig(figfile_ice, bbox_inches='tight')
plt.close(fig_ice)
