#!/usr/bin/env python
#
# Note from Milena: this script should be combined with
# compute_transects.py to be able to:
# 1) compute mass transport for each year and save it
# 2) read the transports from tile if previously computed
# 3) plot it
# I am taking a shortcut now for lack of time... ==> this
# script should be run after compute_transects because it reads
# the data computed there.
#

# ensure plots are rendered on ICC
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import matplotlib as mpl
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

casename1 = 'E3SM-Arctic-OSI_60to10'
casename2 = 'E3SM60to30'

maxYears = 177
monthsNum4runavg_yearly  = 12 # 1 year
monthsNum4runavg  = 5*12 # 5 years

infile1 = './transectsTransport_{}.nc'.format(casename1)
infile2 = './transectsTransport_{}.nc'.format(casename2)
outfile1 = 'meanTransportsArctic_{}.txt'.format(casename1)
outfile2 = 'meanTransportsArctic_{}.txt'.format(casename2)
f1 = open(outfile1, 'w')
f1.write('TransectName VolTransportNet_mean VolTransportNet_rms VolTransportIn_mean VolTransportIn_rms VolTransportOut_mean VolTransportOut_rms HeatTransportNet_mean HeatTransportNet_rms HeatTransportIn_mean HeatTransportIn_rms HeatTransportOut_mean HeatTransportOut_rms HeatTransportTfpNet_mean HeatTransportTfpNet_rms HeatTransportTfpIn_mean HeatTransportTfpIn_rms HeatTransportTfpOut_mean HeatTransportTfpOut_rms FWTransportNet_mean FWTransportNet_rms FWTransportIn_mean FWTransportIn_rms FWTransportOut_mean FWTransportOut_rms\n')
f2 = open(outfile2, 'w')
f2.write('TransectName VolTransportNet_mean VolTransportNet_rms VolTransportIn_mean VolTransportIn_rms VolTransportOut_mean VolTransportOut_rms HeatTransportNet_mean HeatTransportNet_rms HeatTransportIn_mean HeatTransportIn_rms HeatTransportOut_mean HeatTransportOut_rms HeatTransportTfpNet_mean HeatTransportTfpNet_rms HeatTransportTfpIn_mean HeatTransportTfpIn_rms HeatTransportTfpOut_mean HeatTransportTfpOut_rms FWTransportNet_mean FWTransportNet_rms FWTransportIn_mean FWTransportIn_rms FWTransportOut_mean FWTransportOut_rms\n')

figdir = './transects'
if not os.path.isdir(figdir):
    os.mkdir(figdir)
figsize = (20, 10)
figdpi = 150
fontsize_smallLabels = 18
fontsize_labels = 20
fontsize_titles = 24
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}

# Define some dictionaries for transect plotting
volobsDict = {'Drake Passage':[120, 184], 'Tasmania-Ant':[147, 167], 'Africa-Ant':None, 'Antilles Inflow':[-23.1, -13.7], \
              'Mona Passage':[-3.8, -1.4],'Windward Passage':[-7.2, -6.8], 'Florida-Cuba':[24.7, 35.3], 'Florida-Bahamas':[28.8, 35.4], \
              'Indonesian Throughflow':[-20, -10], 'Agulhas':[-90, -50], 'Mozambique Channel':[-20, -8], \
              'Bering Strait':[0.6, 1.0], 'Lancaster Sound':[-1.0, -0.5], 'Fram Strait':[-4.7, 0.7], \
              'Robeson Channel':None, 'Davis Strait':[-1.1, -2.1], 'Barents Sea Opening':[1.4, 2.6], \
              'Nares Strait':[-1.1, -0.5], 'Denmark Strait':None, 'Iceland-Faroe-Scotland':None}
heatobsDict = {'Drake Passage':None, 'Tasmania-Ant':None, 'Africa-Ant':None, 'Antilles Inflow':None, \
               'Mona Passage':None,'Windward Passage':None, 'Florida-Cuba':None, 'Florida-Bahamas':None, \
               'Indonesian Throughflow':None, 'Agulhas':None, 'Mozambique Channel':None, \
               'Bering Strait':[10, 20], 'Lancaster Sound':None, 'Fram Strait':[30, 42], \
               'Robeson Channel':None, 'Davis Strait':[1, 35], 'Barents Sea Opening':[50, 70], \
               'Nares Strait':None, 'Denmark Strait':None, 'Iceland-Faroe-Scotland':None}
FWobsDict = {'Drake Passage':None, 'Tasmania-Ant':None, 'Africa-Ant':None, 'Antilles Inflow':None, \
             'Mona Passage':None,'Windward Passage':None, 'Florida-Cuba':None, 'Florida-Bahamas':None, \
             'Indonesian Throughflow':None, 'Agulhas':None, 'Mozambique Channel':None, \
             'Bering Strait':[2200, 2800], 'Lancaster Sound':[-1900, -950], 'Fram Strait':[-3188, -2132], \
             'Robeson Channel':None, 'Davis Strait':[-3120, -2740], 'Barents Sea Opening':[-184, 4], \
             'Nares Strait':[-1700, -1000], 'Denmark Strait':None, 'Iceland-Faroe-Scotland':None}
labelDict = {'Drake Passage':'drake', 'Tasmania-Ant':'tasmania', 'Africa-Ant':'africaAnt', 'Antilles Inflow':'antilles', \
             'Mona Passage':'monaPassage', 'Windward Passage':'windwardPassage', 'Florida-Cuba':'floridaCuba', \
             'Florida-Bahamas':'floridaBahamas', 'Indonesian Throughflow':'indonesia', 'Agulhas':'agulhas', \
             'Mozambique Channel':'mozambique', 'Bering Strait':'beringStrait', 'Lancaster Sound':'lancasterSound', \
             'Fram Strait':'framStrait', 'Robeson Channel':'robeson', 'Davis Strait':'davisStrait', 'Barents Sea Opening':'BarentsSea', \
             'Nares Strait':'naresStrait', 'Denmark Strait':'denmarkStrait', 'Iceland-Faroe-Scotland':'icelandFaroeScotland'}

# Read data in from previously compute mass transports
ds1 = xr.open_dataset(infile1)
time1 = ds1.Time.values # Simulated days/365 (will want to change this eventually)
volTransports1 = ds1.volTransport.values # dims=[Time, nTransects], units=Sv
volTransportsIn1 = ds1.volTransportIn.values
volTransportsOut1 = ds1.volTransportOut.values
heatTransports1 = ds1.heatTransport.values
heatTransportsIn1 = ds1.heatTransportIn.values
heatTransportsOut1 = ds1.heatTransportOut.values
heatTransportsTfp1 = ds1.heatTransportTfp.values
heatTransportsTfpIn1 = ds1.heatTransportTfpIn.values
heatTransportsTfpOut1 = ds1.heatTransportTfpOut.values
FWTransports1 = ds1.FWTransport.values
FWTransportsIn1 = ds1.FWTransportIn.values
FWTransportsOut1 = ds1.FWTransportOut.values
nTimes1, nTransects = np.shape(volTransports1)

transectNames = [str(ds1.TransectNames.values[k], encoding='UTF-8')[2:] for k in range(nTransects)]

ds2 = xr.open_dataset(infile2)
time2 = ds2.Time.values # Simulated days/365 (will want to change this eventually)
volTransports2 = ds2.volTransport.values # dims=[Time, nTransects], units=Sv
volTransportsIn2 = ds2.volTransportIn.values
volTransportsOut2 = ds2.volTransportOut.values
heatTransports2 = ds2.heatTransport.values
heatTransportsIn2 = ds2.heatTransportIn.values
heatTransportsOut2 = ds2.heatTransportOut.values
heatTransportsTfp2 = ds2.heatTransportTfp.values
heatTransportsTfpIn2 = ds2.heatTransportTfpIn.values
heatTransportsTfpOut2 = ds2.heatTransportTfpOut.values
FWTransports2 = ds2.FWTransport.values
FWTransportsIn2 = ds2.FWTransportIn.values
FWTransportsOut2 = ds2.FWTransportOut.values
nTimes2, nTransects = np.shape(volTransports2)

for i in range(nTransects):
    volTransports_runavg_yearly1 = pd.Series.rolling(ds1.volTransport[:, i].to_pandas(),
                                                     monthsNum4runavg_yearly, center=True).mean()
    volTransports_runavg_yearly2 = pd.Series.rolling(ds2.volTransport[:, i].to_pandas(),
                                                     monthsNum4runavg_yearly, center=True).mean()
    volTransports_runavg1 = pd.Series.rolling(ds1.volTransport[:, i].to_pandas(),
                                              monthsNum4runavg, center=True).mean()
    volTransports_runavg2 = pd.Series.rolling(ds2.volTransport[:, i].to_pandas(),
                                              monthsNum4runavg, center=True).mean()

    # Plot Volume transport
    bounds = volobsDict[transectNames[i]]
    figfile = '{}/volTransport_{}_{}_{}.png'.format(figdir, labelDict[transectNames[i]],
                                                    casename1, casename2)
    fig =  plt.figure(figsize=figsize, dpi=figdpi)
    ax = fig.add_subplot()
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    ax.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
    ax.yaxis.get_offset_text().set_weight('bold')

    ax.plot(time1, volTransports_runavg_yearly1, color='firebrick', linewidth=3,
            label='E3SM-Arctic-OSI annual mean ({:5.2f} $\pm$ {:5.2f} Sv)'.format(
            np.nanmean(volTransports_runavg_yearly1), np.nanstd(volTransports_runavg_yearly1)))
    ax.plot(time1, volTransports1[:, i], color='salmon', alpha=0.5, linewidth=1.5,
            label='E3SM-Arctic-OSI monthly')
    if transectNames[i] != 'Nares Strait': # skip plotting of Nares Strait for E3SM-LR (strait is closed)
        ax.plot(time2, volTransports_runavg_yearly2, color='k', linewidth=3,
                label='E3SM-LR-OSI annual mean ({:5.2f} $\pm$ {:5.2f} Sv)'.format(
                np.nanmean(volTransports_runavg_yearly2), np.nanstd(volTransports_runavg_yearly2)))
    if bounds is not None:
        fig.gca().fill_between(range(1, maxYears+1),
                               bounds[0]*np.ones_like(range(1, maxYears+1)),
                               bounds[1]*np.ones_like(range(1, maxYears+1)),
                               color='limegreen', alpha=0.3, label='obs variability')
    if np.max(volTransports1[:, i])>0.0 and np.min(volTransports1[:, i])<0:
        ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=59, color='blueviolet', linewidth=5)
    ax.axvline(x=118, color='blueviolet', linewidth=5)
    ax.set_xlabel('Time (Years)', fontsize=fontsize_labels, fontweight='bold')
    ax.set_ylabel('Volume transport (Sv)', fontsize=fontsize_labels, fontweight='bold')
    ax.set_title('Volume transport for {}'.format(transectNames[i]), fontsize=fontsize_titles, fontweight='bold')
    ax.legend(prop=legend_properties)
    ax.set_xlim(1, maxYears)
    #ax.autoscale(enable=True, axis='x', tight=True)
    fig.savefig(figfile, bbox_inches='tight')
    plt.close(fig)

    # Plot Freshwater and Heat transport for Arctic gateways only
    heatbounds = heatobsDict[transectNames[i]]
    FWbounds = FWobsDict[transectNames[i]]
    if transectNames[i]=='Bering Strait' or transectNames[i]=='Lancaster Sound' or \
       transectNames[i]=='Fram Strait' or transectNames[i]=='Davis Strait' or \
       transectNames[i]=='Barents Sea Opening' or transectNames[i]=='Nares Strait':
        heatTransports_runavg_yearly1 = pd.Series.rolling(ds1.heatTransport[:, i].to_pandas(),
                                                          monthsNum4runavg_yearly, center=True).mean()
        heatTransports_runavg_yearly2 = pd.Series.rolling(ds2.heatTransport[:, i].to_pandas(),
                                                          monthsNum4runavg_yearly, center=True).mean()
        heatTransports_runavg1 = pd.Series.rolling(ds1.heatTransport[:, i].to_pandas(),
                                                   monthsNum4runavg, center=True).mean()
        heatTransports_runavg2 = pd.Series.rolling(ds2.heatTransport[:, i].to_pandas(),
                                                   monthsNum4runavg, center=True).mean()
        heatTransportsTfp_runavg_yearly1 = pd.Series.rolling(ds1.heatTransportTfp[:, i].to_pandas(),
                                                             monthsNum4runavg_yearly, center=True).mean()
        heatTransportsTfp_runavg_yearly2 = pd.Series.rolling(ds2.heatTransportTfp[:, i].to_pandas(),
                                                             monthsNum4runavg_yearly, center=True).mean()
        heatTransportsTfp_runavg1 = pd.Series.rolling(ds1.heatTransportTfp[:, i].to_pandas(),
                                                      monthsNum4runavg, center=True).mean()
        heatTransportsTfp_runavg2 = pd.Series.rolling(ds2.heatTransportTfp[:, i].to_pandas(),
                                                      monthsNum4runavg, center=True).mean()
        FWTransports_runavg_yearly1 = pd.Series.rolling(ds1.FWTransport[:, i].to_pandas(),
                                                        monthsNum4runavg_yearly, center=True).mean()
        FWTransports_runavg_yearly2 = pd.Series.rolling(ds2.FWTransport[:, i].to_pandas(),
                                                        monthsNum4runavg_yearly, center=True).mean()
        FWTransports_runavg1 = pd.Series.rolling(ds1.FWTransport[:, i].to_pandas(),
                                                 monthsNum4runavg, center=True).mean()
        FWTransports_runavg2 = pd.Series.rolling(ds2.FWTransport[:, i].to_pandas(),
                                                 monthsNum4runavg, center=True).mean()

        f1.write('%s %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n' % \
                 (labelDict[transectNames[i]], np.nanmean(volTransports1[:, i]), np.nanstd(volTransports1[:, i]), \
                 np.nanmean(volTransportsIn1[:, i]), np.nanstd(volTransportsIn1[:, i]), np.nanmean(volTransportsOut1[:, i]), np.nanstd(volTransportsOut1[:, i]), \
                 np.nanmean(heatTransports1[:, i]), np.nanstd(heatTransports1[:, i]), np.nanmean(heatTransportsIn1[:, i]), \
                 np.nanstd(heatTransportsIn1[:, i]), np.nanmean(heatTransportsOut1[:, i]), np.nanstd(heatTransportsOut1[:, i]), \
                 np.nanmean(heatTransportsTfp1[:, i]), np.nanstd(heatTransportsTfp1[:, i]), np.nanmean(heatTransportsTfpIn1[:, i]), \
                 np.nanstd(heatTransportsTfpIn1[:, i]), np.nanmean(heatTransportsTfpOut1[:, i]), np.nanstd(heatTransportsTfpOut1[:, i]), \
                 np.nanmean(FWTransports1[:, i]), np.nanstd(FWTransports1[:, i]), np.nanmean(FWTransportsIn1[:, i]), \
                 np.nanstd(FWTransportsIn1[:, i]), np.nanmean(FWTransportsOut1[:, i]), np.nanstd(FWTransportsOut1[:, i])))
        f2.write('%s %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n' % \
                 (labelDict[transectNames[i]], np.nanmean(volTransports2[:, i]), np.nanstd(volTransports2[:, i]), \
                 np.nanmean(volTransportsIn2[:, i]), np.nanstd(volTransportsIn2[:, i]), np.nanmean(volTransportsOut2[:, i]), np.nanstd(volTransportsOut2[:, i]), \
                 np.nanmean(heatTransports2[:, i]), np.nanstd(heatTransports2[:, i]), np.nanmean(heatTransportsIn2[:, i]), \
                 np.nanstd(heatTransportsIn2[:, i]), np.nanmean(heatTransportsOut2[:, i]), np.nanstd(heatTransportsOut2[:, i]), \
                 np.nanmean(heatTransportsTfp2[:, i]), np.nanstd(heatTransportsTfp2[:, i]), np.nanmean(heatTransportsTfpIn2[:, i]), \
                 np.nanstd(heatTransportsTfpIn2[:, i]), np.nanmean(heatTransportsTfpOut2[:, i]), np.nanstd(heatTransportsTfpOut2[:, i]), \
                 np.nanmean(FWTransports2[:, i]), np.nanstd(FWTransports2[:, i]), np.nanmean(FWTransportsIn2[:, i]), \
                 np.nanstd(FWTransportsIn2[:, i]), np.nanmean(FWTransportsOut2[:, i]), np.nanstd(FWTransportsOut2[:, i])))

        figfile = '{}/heatTransport_{}_{}_{}.png'.format(figdir, labelDict[transectNames[i]],
                                                         casename1, casename2)
        fig =  plt.figure(figsize=figsize, dpi=figdpi)
        ax = fig.add_subplot()
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        ax.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
        ax.yaxis.get_offset_text().set_weight('bold')

        ax.plot(time1, heatTransports_runavg_yearly1, color='firebrick', linewidth=3,
                label='E3SM-Arctic-OSI annual mean ({:5.2f} $\pm$ {:5.2f} TW)'.format(
                np.nanmean(heatTransports_runavg_yearly1), np.nanstd(heatTransports_runavg_yearly1)))
        ax.plot(time1, heatTransports1[:, i], color='salmon', alpha=0.5, linewidth=1.5,
                label='E3SM-Arctic-OSI monthly')
        if transectNames[i] != 'Nares Strait': # skip plotting of Nares Strait for E3SM-LR (strait is closed)
            ax.plot(time2, heatTransports_runavg_yearly2, color='k', linewidth=3,
                    label='E3SM-LR-OSI annual mean ({:5.2f} $\pm$ {:5.2f} TW)'.format(
                    np.nanmean(heatTransports_runavg_yearly2), np.nanstd(heatTransports_runavg_yearly2)))
        if heatbounds is not None:
            fig.gca().fill_between(range(1, maxYears+1),
                                   heatbounds[0]*np.ones_like(range(1, maxYears+1)),
                                   heatbounds[1]*np.ones_like(range(1, maxYears+1)),
                                   color='limegreen', alpha=0.3, label='obs variability')
        if np.max(heatTransports1[:, i])>0.0 and np.min(heatTransports1[:, i])<0:
            ax.axhline(y=0, color='k', linewidth=1)
        ax.axvline(x=59, color='blueviolet', linewidth=5)
        ax.axvline(x=118, color='blueviolet', linewidth=5)
        ax.set_xlabel('Time (Years)', fontsize=fontsize_labels, fontweight='bold')
        ax.set_ylabel('Heat transport (TW)', fontsize=fontsize_labels, fontweight='bold')
        ax.set_title('Heat transport wrt 0 $^\circ$C for {}'.format(transectNames[i]), fontsize=fontsize_titles, fontweight='bold')
        ax.legend(prop=legend_properties)
        ax.set_xlim(1, maxYears)
        fig.savefig(figfile, bbox_inches='tight')
        plt.close(fig)

        figfile = '{}/heatTransportTfp_{}_{}_{}.png'.format(figdir, labelDict[transectNames[i]],
                                                            casename1, casename2)
        fig =  plt.figure(figsize=figsize, dpi=figdpi)
        ax = fig.add_subplot()
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        ax.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
        ax.yaxis.get_offset_text().set_weight('bold')

        ax.plot(time1, heatTransportsTfp_runavg_yearly1, color='firebrick', linewidth=3,
                label='E3SM-Arctic-OSI annual mean ({:5.2f} $\pm$ {:5.2f} TW)'.format(
                np.nanmean(heatTransportsTfp_runavg_yearly1), np.nanstd(heatTransportsTfp_runavg_yearly1)))
        ax.plot(time1, heatTransportsTfp1[:, i], color='salmon', alpha=0.5, linewidth=1.5,
                label='E3SM-Arctic-OSI monthly')
        if transectNames[i] != 'Nares Strait': # skip plotting of Nares Strait for E3SM-LR (strait is closed)
            ax.plot(time2, heatTransportsTfp_runavg_yearly2, color='k', linewidth=3,
                    label='E3SM-LR-OSI annual mean ({:5.2f} $\pm$ {:5.2f} TW)'.format(
                    np.nanmean(heatTransportsTfp_runavg_yearly2), np.nanstd(heatTransportsTfp_runavg_yearly2)))
        if heatbounds is not None:
            fig.gca().fill_between(range(1, maxYears+1),
                                   heatbounds[0]*np.ones_like(range(1, maxYears+1)),
                                   heatbounds[1]*np.ones_like(range(1, maxYears+1)),
                                   color='limegreen', alpha=0.3, label='obs variability')
        if np.max(heatTransportsTfp1[:, i])>0.0 and np.min(heatTransportsTfp1[:, i])<0:
            ax.axhline(y=0, color='k', linewidth=1)
        ax.axvline(x=59, color='blueviolet', linewidth=5)
        ax.axvline(x=118, color='blueviolet', linewidth=5)
        ax.set_xlabel('Time (Years)', fontsize=fontsize_labels, fontweight='bold')
        ax.set_ylabel('Heat transport (TW)', fontsize=fontsize_labels, fontweight='bold')
        ax.set_title('Heat transport wrt freezing point for {}'.format(transectNames[i]), fontsize=fontsize_titles, fontweight='bold')
        ax.legend(prop=legend_properties)
        ax.set_xlim(1, maxYears)
        fig.savefig(figfile, bbox_inches='tight')
        plt.close(fig)

        figfile = '{}/FWTransport_{}_{}_{}.png'.format(figdir, labelDict[transectNames[i]],
                                                       casename1, casename2)
        fig =  plt.figure(figsize=figsize, dpi=figdpi)
        ax = fig.add_subplot()
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        ax.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
        ax.yaxis.get_offset_text().set_weight('bold')

        ax.plot(time1, FWTransports_runavg_yearly1, color='firebrick', linewidth=3,
                label='E3SM-Arctic-OSI annual mean ({:5.2f} $\pm$ {:5.2f} km$^3$/year)'.format(
                np.nanmean(FWTransports_runavg_yearly1), np.nanstd(FWTransports_runavg_yearly1)))
        ax.plot(time1, FWTransports1[:, i], color='salmon', alpha=0.5, linewidth=1.5,
                label='E3SM-Arctic-OSI monthly')
        #ax.plot(time1, FWTransports_runavg1, color='k', linewidth=3,
        #        label='60to10 5-year mean ({:5.2f} $\pm$ {:5.2f} km$^3$/year)'.format(
        #        np.nanmean(FWTransports_runavg1), np.nanstd(FWTransports_runavg1)))
        #ax.plot(time2, FWTransports_runavg2, color='firebrick', linewidth=3,
        #        label='60to6 5-year mean ({:5.2f} $\pm$ {:5.2f} km$^3$/year)'.format(
        #        np.nanmean(FWTransports_runavg2), np.nanstd(FWTransports_runavg2)))
        if transectNames[i] != 'Nares Strait': # skip plotting of Nares Strait for E3SM-LR (strait is closed)
            ax.plot(time2, FWTransports_runavg_yearly2, color='k', linewidth=3,
                    label='E3SM-LR-OSI annual mean ({:5.2f} $\pm$ {:5.2f} km$^3$/year)'.format(
                    np.nanmean(FWTransports_runavg_yearly2), np.nanstd(FWTransports_runavg_yearly2)))
        if FWbounds is not None:
            fig.gca().fill_between(range(1, maxYears+1),
                                   FWbounds[0]*np.ones_like(range(1, maxYears+1)),
                                   FWbounds[1]*np.ones_like(range(1, maxYears+1)),
                                   color='limegreen', alpha=0.3, label='obs variability')
        if np.max(FWTransports1[:, i])>0.0 and np.min(FWTransports1[:, i])<0:
            ax.axhline(y=0, color='k', linewidth=1)
        ax.axvline(x=59, color='blueviolet', linewidth=5)
        ax.axvline(x=118, color='blueviolet', linewidth=5)
        ax.set_xlabel('Time (Years)', fontsize=fontsize_labels, fontweight='bold')
        ax.set_ylabel('Freshwater transport (km$^3$/year)', fontsize=fontsize_labels, fontweight='bold')
        ax.set_title('Freshwater transport for {}'.format(transectNames[i]), fontsize=fontsize_titles, fontweight='bold')
        ax.legend(prop=legend_properties)
        ax.set_xlim(1, maxYears)
        fig.savefig(figfile, bbox_inches='tight')
        plt.close(fig)
f1.close()
f2.close()
