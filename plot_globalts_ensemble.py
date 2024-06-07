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
#import datetime

#from common_functions import plot_xtick_format, days_to_datetime


figdir = 'E3SM-Arcticv2.1'
maindir = '/global/cfs/cdirs/m1199/e3sm-arrm-simulations'
simsToPlot = [
               {'mocDatadir': f'{maindir}/E3SMv2.1B60to10rA02/mpas-analysis/timeseries/moc',
                'mocYears': '0001-0386',
                'shiftyear': 0,
                'label': '1950-control',
                'color': 'k',
                'linewidth': 1.2,
                'alpha': 1},
               {'mocDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0101/mpas-analysis/timeseries/moc',
                'mocYears': '1950-2014',
                'shiftyear': 101,
                'label': 'hist 101',
                'color': 'mediumblue',
                'linewidth': 2,
                'alpha': 0.6},
               {'mocDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0151/mpas-analysis/timeseries/moc',
                'mocYears': '1950-2014',
                'shiftyear': 151,
                'label': 'hist 151',
                'color': 'dodgerblue',
                'linewidth': 2,
                'alpha': 0.6},
               {'mocDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0201/mpas-analysis/timeseries/moc',
                'mocYears': '1950-2014',
                'shiftyear': 201,
                'label': 'hist 201',
                'color': 'deepskyblue',
                'linewidth': 2,
                'alpha': 0.6},
               {'mocDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0251/mpas_analysis_output/yrs2000-2014/timeseries/moc',
                'mocYears': '1950-2014',
                'shiftyear': 251,
                'label': 'hist 251',
                'color': 'lightseagreen',
                'linewidth': 2,
                'alpha': 0.6},
               {'mocDatadir': f'{maindir}/E3SM-Arcticv2.1_historical0301/mpas_analysis_output/yrs2000-2014/timeseries/moc',
                'mocYears': '1950-2014',
                'shiftyear': 301,
                'label': 'hist 301',
                'color': 'teal',
                'linewidth': 2,
                'alpha': 0.6}
            ]

movingAverageMonths = 12 # months
##############################################################
figdir = f'./timeseries/{figdir}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figsize = (15, 5)
figdpi = 150
fontsize_smallLabels = 10
fontsize_labels = 20
fontsize_titles = 22
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
################

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
ax.set_ylabel('Sv', fontsize=fontsize_labels, fontweight='bold')
ax.set_title(f'Max MOC at 26N ({int(movingAverageMonths/12)}-year running avg)', fontsize=fontsize_titles, fontweight='bold')
ax.set_xlim(0, 386)
ax.grid(visible=True, which='both')
figfile = f'{figdir}/moc26.png'

for sim in simsToPlot:
    indir = sim['mocDatadir']
    years = sim['mocYears']
    shiftyear = sim['shiftyear']
    legendlabel = sim['label']
    linecolor = sim['color']
    linewidth = sim['linewidth']
    linealpha = sim['alpha']

    infile = f'{indir}/mocTimeSeries_{years}.nc'
    if os.path.exists(infile):
        dsIn = xr.open_dataset(infile)
    else:
        raise IOError(f'MOC file {infile} not found')

    moc26 = dsIn.mocAtlantic26
    if movingAverageMonths!=1:
        window = int(movingAverageMonths)
        moc26 = pd.Series(moc26).rolling(window, center=True).mean()
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

    ax.plot(time, moc26, '-', color=linecolor, alpha=linealpha, linewidth=linewidth, label=legendlabel)

ax.legend(prop=legend_properties)
#ax.legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1, 0.5))
fig.savefig(figfile, bbox_inches='tight')
plt.close(fig)
