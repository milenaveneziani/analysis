#
# Plots various MOC time series related stuff
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
runname = 'E3SMv2.1B60to10rA02' # timeseries/ will be prepended to this
mocdir = '/global/cfs/cdirs/m1199/e3sm-arrm-simulations/E3SMv2.1B60to10rA02/mpas-analysis/timeseries/moc'
regionalTSdir = '/global/cfs/cdirs/e3sm/milena/analysis-MPAS/analysis-git-repo/timeseries_data/E3SMv2.1B60to10rA02'

regionGroup = 'arctic_atlantic_budget_regions_new20240408'

yearStart = 1
yearEnd = 386

#tsvariable = 'iceArea'
tsvariable = 'iceVolume'
regionName = 'Greater Arctic'
#regionName = 'Nordic Seas'
#regionName = 'North Atlantic subpolar gyre'
regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '')
moclats = [26, 45, 65]
#moccolors = ['slateblue', 'firebrick', 'salmon'] # same length as moclats
#moccolors = ['firebrick', 'salmon', 'sandybrown'] # same length as moclats
moccolors = ['mediumblue', 'dodgerblue', 'teal'] #, 'lightseagreen', 'green'] # same length as moclats
movingAverageMonths = 12 # months

npanelsToPlot = 1
if npanelsToPlot>3:
    raise SystemExit('Trying to plot more than 3 panels. Exiting...\n')
##############################################################

figdir = f'./timeseries/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figdpi = 300
fontsize_smallLabels = 14
fontsize_labels = 16
fontsize_titles = 18
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
################

figsize = (12, npanelsToPlot*8)
if npanelsToPlot==1:
    figfile = f'{figdir}/mocLats.png'
elif npanelsToPlot==2:
    figfile = f'{figdir}/mocLats_{tsvariable}_{regionNameShort}.png'
else:
    figfile = f'{figdir}/depthMocLats.png'
figtitle = runname
fig = plt.figure(figsize=figsize, dpi=figdpi)

if npanelsToPlot==1:
    ax  = fig.add_subplot()
else:
    ax  = fig.add_subplot(npanelsToPlot, 1, 1)
    axice1 = fig.add_subplot(npanelsToPlot, 1, 2)
    if npanelsToPlot==3:
        ax3 = fig.add_subplot(npanelsToPlot, 1, 3)

#### PANEL 1: max MOC time series for various latitudes ####
mocfile = f'{mocdir}/mocTimeSeries_{yearStart:04d}-{yearEnd:04d}.nc'
if os.path.exists(mocfile):
    dsMOC = xr.open_dataset(mocfile)
else:
    raise IOError(f'MOC file {mocfile} not found')

kmonths = len(dsMOC.Time.values)
time = np.arange(kmonths)/12
for nlat in range(len(moclats)):
    color = moccolors[nlat]

    lat = moclats[nlat]
    moc = dsMOC.mocAtlantic.sel(lat=lat, method='nearest').max(dim='depth')

    if lat>0:
        legendlabel = f'{lat:d}N'
    elif lat<0:
        legendlabel = f'{lat:d}S'
    else:
        legendlabel = f'{lat:d}'
    legendlabel = f'{legendlabel} ({moc.mean():5.2f} $\pm$ {moc.std():5.2f})'
    if movingAverageMonths>1:
        window = int(movingAverageMonths)
        moc_runavg = pd.Series(moc).rolling(window, center=True).mean()
        ax.plot(time, moc_runavg, '-', color=color, linewidth=2, label=legendlabel)
        #ax.plot(time, moc, '-', color=color, alpha=0.5, linewidth=1.2)
        if npanelsToPlot>1:
            axice1.plot(time, moc_runavg, '-', color=color, linewidth=2, label=legendlabel)
    else:
        ax.plot(time, moc, '-', color=color, linewidth=2, label=legendlabel)
        if npanelsToPlot>1:
            axice1.plot(time, moc, '-', color=color, linewidth=2, label=legendlabel)
ax.grid(visible=True, which='both')
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax.legend(prop=legend_properties)
if npanelsToPlot==1:
    ax.set_xlabel('Years', fontsize=fontsize_labels, fontweight='bold')
ax.set_ylabel('max MOC (Sv)', fontsize=fontsize_labels, fontweight='bold')
ax.set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_xlim(0, yearEnd)
#ax.set_xlim(0, 30)

if npanelsToPlot>1:
#### PANEL 2: max MOC time series for various latitudes plus ice time series ####
    axice1.grid(visible=True, which='both')
    axice1.tick_params(axis='y', labelcolor='dodgerblue')
    for tick in axice1.xaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    for tick in axice1.yaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    axice1.spines['left'].set_color('dodgerblue')
    #axice1.legend(prop=legend_properties, loc='lower right')
    ##axice1.legend(prop=legend_properties)
    if npanelsToPlot==2:
        axice1.set_xlabel('Years', fontsize=fontsize_labels, fontweight='bold')
    axice1.set_ylabel('max MOC (Sv)', fontsize=fontsize_labels, fontweight='bold', color='dodgerblue')
    axice1.autoscale(enable=True, axis='x', tight=True)
    axice1.set_xlim(0, yearEnd)

    axice2 = axice1.twinx()
    #ax2color = 'tab:blue'
    #ax2color = 'lightseagreen'
    ax2color = 'firebrick'
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
        axice2.plot(time, ts_runavg, '-', color=ax2color, linewidth=2)
        #axice2.plot(time, ts, '-', color=ax2color, alpha=0.5, linewidth=1.2)
    else:
        axice2.plot(time, ts, '-', color=ax2color, linewidth=2)
    axice2.tick_params(axis='y', labelcolor=ax2color)
    axice2.spines['right'].set_color(ax2color)
    for tick in axice2.yaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    axice2.set_ylabel(f'{tsvariable} in the {regionName} region', fontsize=fontsize_labels, fontweight='bold', color=ax2color)

    if npanelsToPlot==3:
#### PANEL 3: time series of depth of max MOC for various latitudes ####
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
