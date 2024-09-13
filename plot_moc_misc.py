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

tsvariable = 'iceArea'
#tsvariable = 'iceVolume'
regionName = 'Greater Arctic'
#regionName = 'Nordic Seas'
#regionName = 'North Atlantic subpolar gyre'
regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '')
moclats = [26, 45, 65]
#moccolors = ['slateblue', 'firebrick', 'salmon'] # same length as moclats
#moccolors = ['firebrick', 'salmon', 'sandybrown'] # same length as moclats
moccolors = ['mediumblue', 'dodgerblue', 'teal'] #, 'lightseagreen', 'green'] # same length as moclats
movingAverageMonths = 12 # months
##############################################################

figdir = f'./timeseries/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figsize = (12, 10)
figdpi = 300
fontsize_smallLabels = 14
fontsize_labels = 16
fontsize_titles = 18
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
################

figfile = f'{figdir}/moc_{tsvariable}_{regionNameShort}.png'
figtitle = runname
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax1 = fig.add_subplot(211)
ax  = fig.add_subplot(212)

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
    if movingAverageMonths!=1:
        window = int(movingAverageMonths)
        moc_runavg = pd.Series(moc).rolling(window, center=True).mean()
        ax1.plot(time, moc_runavg, '-', color=color, linewidth=2, label=legendlabel)
        ax.plot(time, moc_runavg, '-', color=color, linewidth=2, label=legendlabel)
        ax.plot(time, moc, '-', color=color, alpha=0.5, linewidth=1.2)
    else:
        ax1.plot(time, moc, '-', color=color, linewidth=2, label=legendlabel)
        ax.plot(time, moc, '-', color=color, linewidth=2, label=legendlabel)
ax1.grid(visible=True, which='both')
ax1.tick_params(axis='y', labelcolor='dodgerblue')
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax1.spines['left'].set_color('dodgerblue')
#ax1.legend(prop=legend_properties, loc='lower right')
##ax1.legend(prop=legend_properties)
#ax1.set_xlabel('Years', fontsize=fontsize_labels, fontweight='bold')
ax1.set_ylabel('max MOC (Sv)', fontsize=fontsize_labels, fontweight='bold', color='dodgerblue')
ax1.set_title(figtitle, fontsize=fontsize_titles, fontweight='bold')
ax1.autoscale(enable=True, axis='x', tight=True)
ax1.set_xlim(0, yearEnd)

ax.grid(visible=True, which='both')
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
for tick in ax.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax.legend(prop=legend_properties)
ax.set_xlabel('Years', fontsize=fontsize_labels, fontweight='bold')
ax.set_ylabel('max MOC (Sv)', fontsize=fontsize_labels, fontweight='bold')
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_xlim(0, yearEnd)

ax2 = ax1.twinx()
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

if movingAverageMonths!=1:
    window = int(movingAverageMonths)
    ts_runavg = pd.Series(ts).rolling(window, center=True).mean()
    ax2.plot(time, ts_runavg, '-', color=ax2color, linewidth=2)
    #ax2.plot(time, ts, '-', color=ax2color, alpha=0.5, linewidth=1.2)
else:
    ax2.plot(time, ts, '-', color=ax2color, linewidth=2)

ax2.tick_params(axis='y', labelcolor=ax2color)
ax2.spines['right'].set_color(ax2color)
for tick in ax2.yaxis.get_ticklabels():
    tick.set_fontsize(fontsize_smallLabels)
    tick.set_weight('bold')
ax2.set_ylabel(f'{tsvariable} in the {regionName} region', fontsize=fontsize_labels, fontweight='bold', color=ax2color)

fig.tight_layout(pad=0.5)
fig.savefig(figfile, bbox_inches='tight')
plt.close(fig)
