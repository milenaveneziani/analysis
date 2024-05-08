from __future__ import absolute_import, division, print_function, \
    unicode_literals
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import glob
from mpas_analysis.shared.io.utility import decode_strings
import gsw

seasons = [2, 8]

# Barents Sea
#lonMean = 42.5
#latMean = 75
# Canada Basin
lonMean = -142.5
latMean = 75

figdir = f'./TSprofiles/E3SM-Arcticv2.1_historical'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

outdir = f'./TSprofiles_data/E3SM-Arcticv2.1_historical/0151/reconstructed'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

figsize = (10, 15)
figdpi = 150
fontsize_smallLabels = 18
fontsize_labels = 20
fontsize_titles = 22
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}
###################################################################

for season in seasons:
    # Initialize figure and axis objects
    fig_Tprofile = plt.figure(figsize=figsize, dpi=figdpi)
    ax_Tprofile = fig_Tprofile.add_subplot()
    for tick in ax_Tprofile.xaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    for tick in ax_Tprofile.yaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    ax_Tprofile.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
    ax_Tprofile.yaxis.get_offset_text().set_weight('bold')
    #
    fig_Sprofile = plt.figure(figsize=figsize, dpi=figdpi)
    ax_Sprofile = fig_Sprofile.add_subplot()
    for tick in ax_Sprofile.xaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    for tick in ax_Sprofile.yaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    ax_Sprofile.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
    ax_Sprofile.yaxis.get_offset_text().set_weight('bold')
    #
    fig_Cprofile = plt.figure(figsize=figsize, dpi=figdpi)
    ax_Cprofile = fig_Cprofile.add_subplot()
    for tick in ax_Cprofile.xaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    for tick in ax_Cprofile.yaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    ax_Cprofile.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
    ax_Cprofile.yaxis.get_offset_text().set_weight('bold')
    #Tfigtitle = f'Temperature (Barents Sea)\n{season:02d} - years 2031-2050 (reconstructed)'
    #Tfigfile = f'{figdir}/TprofileBarentsSea_E3SM-Arcticv2.1_historical0151_{season:02d}_years2031-2050_reconstructed.png'
    #Sfigtitle = f'Salinity (Barents Sea)\n{season:02d} - years 2031-2050 (reconstructed)'
    #Sfigfile = f'{figdir}/SprofileBarentsSea_E3SM-Arcticv2.1_historical0151_{season:02d}_years2031-2050_reconstructed.png'
    #Cfigtitle = f'Sound speed (Barents Sea)\n{season:02d} - years 2031-2050 (reconstructed)'
    #Cfigfile = f'{figdir}/CprofileBarentsSea_E3SM-Arcticv2.1_historical0151_{season:02d}_years2031-2050_reconstructed.png'
    Tfigtitle = f'Temperature (Canada Basin)\n{season:02d} - years 2031-2050 (reconstructed)'
    Tfigfile = f'{figdir}/TprofileCanadaBasin_E3SM-Arcticv2.1_historical0151_{season:02d}_years2031-2050_reconstructed.png'
    Sfigtitle = f'Salinity (Canada Basin)\n{season:02d} - years 2031-2050 (reconstructed)'
    Sfigfile = f'{figdir}/SprofileCanadaBasin_E3SM-Arcticv2.1_historical0151_{season:02d}_years2031-2050_reconstructed.png'
    Cfigtitle = f'Sound speed (Canada Basin)\n{season:02d} - years 2031-2050 (reconstructed)'
    Cfigfile = f'{figdir}/CprofileCanadaBasin_E3SM-Arcticv2.1_historical0151_{season:02d}_years2031-2050_reconstructed.png'

    ax_Tprofile.set_xlabel('Temperature ($^\circ$C)', fontsize=fontsize_labels, fontweight='bold')
    ax_Tprofile.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
    ax_Tprofile.set_title(Tfigtitle, fontsize=fontsize_titles, fontweight='bold')
    #ax_Tprofile.set_xlim(-1.85, 1.8)
    ax_Tprofile.set_ylim(-500, 0)
    #
    ax_Sprofile.set_xlabel('Salinity (psu)', fontsize=fontsize_labels, fontweight='bold')
    ax_Sprofile.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
    ax_Sprofile.set_title(Sfigtitle, fontsize=fontsize_titles, fontweight='bold')
    #ax_Sprofile.set_xlim(27.8, 35)
    ax_Sprofile.set_ylim(-500, 0)
    #
    ax_Cprofile.set_xlabel('C (m/s)', fontsize=fontsize_labels, fontweight='bold')
    ax_Cprofile.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
    ax_Cprofile.set_title(Cfigtitle, fontsize=fontsize_titles, fontweight='bold')
    #ax_Cprofile.set_xlim(1430., 1470.)
    ax_Cprofile.set_ylim(-500, 0)

    #infileE3SM0 = f'./TSprofiles_data/E3SM-Arcticv2.1_historical/0151/BarentsSea_profiles_E3SM-Arcticv2.1_historical0151_{season:02d}_years1950-1970.nc'
    #infileE3SM1 = f'./TSprofiles_data/E3SM-Arcticv2.1_historical/0151/BarentsSea_profiles_E3SM-Arcticv2.1_historical0151_{season:02d}_years2000-2014.nc'
    #infileHighresMIP0 = f'./TSprofiles_data/HighresMIP/hist-1950/BarentsSea_profiles_HighresMIP_hist-1950_{season:02d}_years1950-1970.nc'
    #infileHighresMIP1 = f'./TSprofiles_data/HighresMIP/hist-1950/BarentsSea_profiles_HighresMIP_hist-1950_{season:02d}_years2000-2014.nc'
    #infileHighresMIP2 = f'./TSprofiles_data/HighresMIP/highres-future/BarentsSea_profiles_HighresMIP_highres-future_{season:02d}_years2031-2050.nc'
    infileE3SM0 = f'./TSprofiles_data/E3SM-Arcticv2.1_historical/0151/CanadaBasin_profiles_E3SM-Arcticv2.1_historical0151_{season:02d}_years1950-1970.nc'
    infileE3SM1 = f'./TSprofiles_data/E3SM-Arcticv2.1_historical/0151/CanadaBasin_profiles_E3SM-Arcticv2.1_historical0151_{season:02d}_years2000-2014.nc'
    infileHighresMIP0 = f'./TSprofiles_data/HighresMIP/hist-1950/CanadaBasin_profiles_HighresMIP_hist-1950_{season:02d}_years1950-1970.nc'
    infileHighresMIP1 = f'./TSprofiles_data/HighresMIP/hist-1950/CanadaBasin_profiles_HighresMIP_hist-1950_{season:02d}_years2000-2014.nc'
    infileHighresMIP2 = f'./TSprofiles_data/HighresMIP/highres-future/CanadaBasin_profiles_HighresMIP_highres-future_{season:02d}_years2031-2050.nc'
    dsE3SM0 = xr.open_dataset(infileE3SM0)
    dsE3SM1 = xr.open_dataset(infileE3SM1)
    dsHighresMIP0 = xr.open_dataset(infileHighresMIP0)
    dsHighresMIP1 = xr.open_dataset(infileHighresMIP1)
    dsHighresMIP2 = xr.open_dataset(infileHighresMIP2)

    depthE3SM = dsE3SM0['depth'].values
    depthHighresMIP = dsHighresMIP0['depth'].values
    pres = gsw.conversions.p_from_z(-depthE3SM, latMean)
    presHighresMIP = gsw.conversions.p_from_z(-depthHighresMIP, latMean)

    Thist0 = dsHighresMIP0['Tprofile'].values
    Shist0 = dsHighresMIP0['Sprofile'].values
    SA = dsHighresMIP0['SAprofile'].values
    CT = dsHighresMIP0['CTprofile'].values
    soundspeedhist0 = gsw.sound_speed(SA, CT, presHighresMIP)

    Thist1 = dsHighresMIP1['Tprofile'].values
    Shist1 = dsHighresMIP1['Sprofile'].values
    SA = dsHighresMIP1['SAprofile'].values
    CT = dsHighresMIP1['CTprofile'].values
    soundspeedhist1 = gsw.sound_speed(SA, CT, presHighresMIP)

    Tssp  = dsHighresMIP2['Tprofile'].values
    Sssp  = dsHighresMIP2['Sprofile'].values
    SA = dsHighresMIP2['SAprofile'].values
    CT = dsHighresMIP2['CTprofile'].values
    soundspeedssp = gsw.sound_speed(SA, CT, presHighresMIP)

    TE3SMhist0 = dsE3SM0['Tprofile'].values
    SE3SMhist0 = dsE3SM0['Sprofile'].values
    SA = dsE3SM0['SAprofile'].values
    CT = dsE3SM0['CTprofile'].values
    soundspeedE3SMhist0 = gsw.sound_speed(SA, CT, pres)

    TE3SMhist1 = dsE3SM1['Tprofile'].values
    SE3SMhist1 = dsE3SM1['Sprofile'].values
    SA = dsE3SM1['SAprofile'].values
    CT = dsE3SM1['CTprofile'].values
    soundspeedE3SMhist1 = gsw.sound_speed(SA, CT, pres)

    Tanomaly = Tssp-Thist1
    Sanomaly = Sssp-Shist1
    TanomalyE3SM = np.interp(depthE3SM, depthHighresMIP, Tanomaly)
    SanomalyE3SM = np.interp(depthE3SM, depthHighresMIP, Sanomaly)
    TE3SMssp = TE3SMhist1 + TanomalyE3SM
    SE3SMssp = SE3SMhist1 + SanomalyE3SM
    SA = gsw.conversions.SA_from_SP(SE3SMssp, pres, lonMean, latMean)
    CT = gsw.conversions.CT_from_pt(SA, TE3SMssp)
    soundspeedE3SMssp = gsw.sound_speed(SA, CT, pres)
    #outfile = f'{outdir}/BarentsSea_profiles_E3SM-Arcticv2.1_historical0151_{season:02d}_years2031-2050_reconstructed.nc'
    outfile = f'{outdir}/CanadaBasin_profiles_E3SM-Arcticv2.1_historical0151_{season:02d}_years2031-2050_reconstructed.nc'
    dsOut = xr.Dataset()
    dsOut['Tprofile'] = TE3SMssp
    dsOut['Tprofile'].attrs['units'] = 'degC'
    dsOut['Tprofile'].attrs['long_name'] = 'Potential temperature'
    dsOut['Sprofile'] = SE3SMssp
    dsOut['Sprofile'].attrs['units'] = 'psu'
    dsOut['Sprofile'].attrs['long_name'] = 'Salinity'
    dsOut['CTprofile'] = CT
    dsOut['CTprofile'].attrs['units'] = 'degC'
    dsOut['CTprofile'].attrs['long_name'] = 'Conservative temperature'
    dsOut['SAprofile'] = SA
    dsOut['SAprofile'].attrs['units'] = 'psu'
    dsOut['SAprofile'].attrs['long_name'] = 'Absolute salinity'
    dsOut['Cprofile'] = soundspeedE3SMssp
    dsOut['Cprofile'].attrs['units'] = 'm/s'
    dsOut['Cprofile'].attrs['long_name'] = 'Sound speed (computed with python gsw package)'
    dsOut['depth'] = depthE3SM
    dsOut['depth'].attrs['units'] = 'm'
    dsOut['depth'].attrs['long_name'] = 'depth levels'
    dsOut.to_netcdf(outfile)

    ax_Tprofile.plot(Thist0[::-1], -depthHighresMIP[::-1], '-', color='yellow', linewidth=3, label='HighresMIP 1950-1970')
    ax_Tprofile.plot(Thist1[::-1], -depthHighresMIP[::-1], '-', color='gold', linewidth=3, label='HighresMIP 2000-2014')
    ax_Tprofile.plot(Tssp[::-1], -depthHighresMIP[::-1], '-', color='darkgoldenrod', linewidth=3, label='HighresMIP 2031-2050')
    ax_Tprofile.plot(TE3SMhist0[::-1], -depthE3SM[::-1], '-', color='lightblue', linewidth=3, label='E3SM-Arctic-0151 1950-1970')
    ax_Tprofile.plot(TE3SMhist1[::-1], -depthE3SM[::-1], '-', color='dodgerblue', linewidth=3, label='E3SM-Arctic-0151 2000-2014')
    ax_Tprofile.plot(TE3SMssp[::-1], -depthE3SM[::-1], '-', color='steelblue', linewidth=3, label='E3SM-Arctic-0151 2031-2050 (reconstructed)')

    ax_Sprofile.plot(Shist0[::-1], -depthHighresMIP[::-1], '-', color='yellow', linewidth=3, label='HighresMIP 1950-1970')
    ax_Sprofile.plot(Shist1[::-1], -depthHighresMIP[::-1], '-', color='gold', linewidth=3, label='HighresMIP 2000-2014')
    ax_Sprofile.plot(Sssp[::-1], -depthHighresMIP[::-1], '-', color='darkgoldenrod', linewidth=3, label='HighresMIP 2031-2050')
    ax_Sprofile.plot(SE3SMhist0[::-1], -depthE3SM[::-1], '-', color='lightblue', linewidth=3, label='E3SM-Arctic-0151 1950-1970')
    ax_Sprofile.plot(SE3SMhist1[::-1], -depthE3SM[::-1], '-', color='dodgerblue', linewidth=3, label='E3SM-Arctic-0151 2000-2014')
    ax_Sprofile.plot(SE3SMssp[::-1], -depthE3SM[::-1], '-', color='steelblue', linewidth=3, label='E3SM-Arctic-0151 2031-2050 (reconstructed)')

    ax_Cprofile.plot(soundspeedhist0[::-1], -depthHighresMIP[::-1], '-', color='yellow', linewidth=3, label='HighresMIP 1950-1970')
    ax_Cprofile.plot(soundspeedhist1[::-1], -depthHighresMIP[::-1], '-', color='gold', linewidth=3, label='HighresMIP 2000-2014')
    ax_Cprofile.plot(soundspeedssp[::-1], -depthHighresMIP[::-1], '-', color='darkgoldenrod', linewidth=3, label='HighresMIP 2031-2050')
    ax_Cprofile.plot(soundspeedE3SMhist0[::-1], -depthE3SM[::-1], '-', color='lightblue', linewidth=3, label='E3SM-Arctic-0151 1950-1970')
    ax_Cprofile.plot(soundspeedE3SMhist1[::-1], -depthE3SM[::-1], '-', color='dodgerblue', linewidth=3, label='E3SM-Arctic-0151 2000-2014')
    ax_Cprofile.plot(soundspeedE3SMssp[::-1], -depthE3SM[::-1], '-', color='steelblue', linewidth=3, label='E3SM-Arctic-0151 2031-2050 (reconstructed)')

    ax_Tprofile.legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1, 0.5))
    ax_Tprofile.grid(visible=True, which='both')
    fig_Tprofile.savefig(Tfigfile, bbox_inches='tight')
    plt.close(fig_Tprofile)

    ax_Sprofile.legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1, 0.5))
    ax_Sprofile.grid(visible=True, which='both')
    fig_Sprofile.savefig(Sfigfile, bbox_inches='tight')
    plt.close(fig_Sprofile)

    ax_Cprofile.legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1, 0.5))
    ax_Cprofile.grid(visible=True, which='both')
    fig_Cprofile.savefig(Cfigfile, bbox_inches='tight')
    plt.close(fig_Cprofile)
