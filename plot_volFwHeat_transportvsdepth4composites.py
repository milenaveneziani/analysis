#!/usr/bin/env python

# ensure plots are rendered on ICC
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean

from common_functions import add_inset
from geometric_features import FeatureCollection, read_feature_collection
from mpas_analysis.shared.io.utility import decode_strings
from make_plots import _make_discrete_colormap


# Settings for erdc.hpc.mil
#ensemble = ''
#casename = 'E3SMv2.1B60to10rA02'
#maskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20240910.nc'
#featurefile = '/p/home/milena/mpas-region_masks/atlanticZonal_sections20240910.geojson'
#transportfile0 = './transports_data/{casename}/atlanticZonalSectionsTransportsvsdepth_{casename}'
#maskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_arcticSections20220916.nc'
#featurefile = '/p/home/milena/mpas-region_masks/arcticSections20210323.geojson'
#transportfile0 = './transports_data/{casename}/arcticSectionsTransportsvsdepth_{casename}'
#years_maxMLDhighFile = './composites_maxMLDbased_data/{casename}/Years1-386/years_maxMLDhigh.dat'
#years_maxMLDlowFile = './composites_maxMLDbased_data/{casename}/Years1-386/years_maxMLDlow.dat'

# Settings for nersc
ensemble = '0301'
casename = 'E3SM-Arcticv2.1_historical'
#maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20240910.nc'
#featurefile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/atlanticZonal_sections20240910.geojson'
#transportfile0 = f'./transports_data/{casename}{ensemble}/atlanticZonalSectionsTransportsvsdepth_{casename}{ensemble}'
maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arcticSections20220916.nc'
featurefile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/arcticSections20210323.geojson'
transportfile0 = f'./transports_data/{casename}{ensemble}/arcticSectionsTransportsvsdepth_{casename}{ensemble}'
years_maxMLDhighFile = f'./composites_maxMLDbased_data/{casename}/years_maxMLDhigh_{ensemble}.dat'
years_maxMLDlowFile = f'./composites_maxMLDbased_data/{casename}/years_maxMLDlow_{ensemble}.dat'

# Choose years
year1 = 1950
year2 = 2014
#year1 = 1
#year2 = 386
years = range(year1, year2+1)
nyears = len(years)

#transectsToPlot = ['Atlantic zonal 27.2N', 'Atlantic zonal 45N', 'Atlantic zonal OSNAP East', 'Atlantic zonal 65N']
#transectsToPlot = ['Fram Strait', 'Barents Sea Opening', 'Davis Strait', 'Denmark Strait', 'Iceland-Faroe-Scotland']
transectsToPlot = ['Iceland-Faroe-Scotland']

figdir = f'./transports/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found')

figsize = (12, 8)
figdpi = 300
##################################################

years_maxMLDhigh = np.loadtxt(years_maxMLDhighFile)
years_maxMLDlow = np.loadtxt(years_maxMLDlowFile)
# remove year 1, if in there
years_maxMLDhigh = years_maxMLDhigh[years_maxMLDhigh>year1]
years_maxMLDlow = years_maxMLDlow[years_maxMLDlow>year1]
# for monthly data:
#years = np.array([year*np.ones(12) for year in range(year1, year2+1)]).flatten()
# for yearly data:
indYears_preMLDhigh = np.array([np.where(years==year-1) for year in years_maxMLDhigh]).flatten()
indYears_preMLDlow = np.array([np.where(years==year-1) for year in years_maxMLDlow]).flatten()

infiles = []
for year in years:
    infiles.append(f'{transportfile0}_year{year:04d}.nc')
dsIn = xr.open_mfdataset(infiles, decode_times=False)
transects = decode_strings(dsIn.transectNames.isel(Time=0))
z = dsIn['refBottomDepth'].isel(Time=0)
t = dsIn['Time']
t_annual = t.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
[x, y] = np.meshgrid(t_annual.values, z.values)
#[x, y] = np.meshgrid(t.values, z.values)
x = x.T
y = y.T

nTransects = len(transectsToPlot)
for i in range(nTransects):
    transectToPlot = transectsToPlot[i]
    transectName = transectToPlot.replace(' ', '')
    print(transectToPlot)

    fc = FeatureCollection()
    if transectToPlot=='Atlantic zonal 65N':
        for feature in fcAll.features:
            if feature['properties']['name']=='Davis Strait South' or \
               feature['properties']['name']=='Denmark Strait' or \
               feature['properties']['name']=='Iceland Norway 65N':
                fc.add_feature(feature)
        transectIndex = transects.index('Davis Strait South')
        dsTransect1 = dsIn.isel(nTransects=transectIndex)
        transectIndex = transects.index('Denmark Strait')
        dsTransect2 = dsIn.isel(nTransects=transectIndex)
        transectIndex = transects.index('Iceland Norway 65N')
        dsTransect3 = dsIn.isel(nTransects=transectIndex)
        volTransport = dsTransect1.volTransport + dsTransect2.volTransport + dsTransect3.volTransport
        heatTransport = dsTransect1.heatTransport + dsTransect2.heatTransport + dsTransect3.heatTransport
        heatTransportTfp = dsTransect1.heatTransportTfp + dsTransect2.heatTransportTfp + dsTransect3.heatTransportTfp
        FWTransportSref = dsTransect1.FWTransportSref + dsTransect2.FWTransportSref + dsTransect3.FWTransportSref
        tempTransect = dsTransect1.tempTransect + dsTransect2.tempTransect + dsTransect3.tempTransect
        saltTransect = dsTransect1.saltTransect + dsTransect2.saltTransect + dsTransect3.saltTransect
        spice0Transect = dsTransect1.spice0Transect + dsTransect2.spice0Transect + dsTransect3.spice0Transect
        depthTransect = xr.concat([dsTransect1.depthTransect.isel(Time=0),
                                   dsTransect2.depthTransect.isel(Time=0),
                                   dsTransect3.depthTransect.isel(Time=0)], dim='nEdges')
    else:
        for feature in fcAll.features:
            if feature['properties']['name']==transectToPlot:
                fc.add_feature(feature)
        transectIndex = transects.index(transectToPlot)
        dsTransect = dsIn.isel(nTransects=transectIndex)
        volTransport = dsTransect.volTransport
        heatTransport = dsTransect.heatTransport
        heatTransportTfp = dsTransect.heatTransportTfp
        FWTransportSref = dsTransect.FWTransportSref
        tempTransect = dsTransect.tempTransect
        saltTransect = dsTransect.saltTransect
        spice0Transect = dsTransect.spice0Transect
        depthTransect = dsTransect.depthTransect.isel(Time=0)

    volTransport_annual = volTransport.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
    heatTransport_annual = heatTransport.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
    heatTransportTfp_annual = heatTransportTfp.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
    FWTransportSref_annual = FWTransportSref.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
    tempTransect_annual = tempTransect.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
    saltTransect_annual = saltTransect.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
    spiceTransect_annual = spice0Transect.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
    zmax = depthTransect.max()

    # Plot Volume Transport
    figfile = f'{figdir}/transportsvsdepth4composites_{transectName}_{casename}{ensemble}.png'
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(321)
    fld = volTransport_annual.values
    colormap = cmocean.cm.balance
    cf = ax1.contourf(x, y, fld, cmap=colormap, extend='both')
    for k in indYears_preMLDhigh:
        ax1.axvline(x=t_annual.isel(Time=k), linewidth=1, color='white', alpha=0.6)
    for k in indYears_preMLDlow:
        ax1.axvline(x=t_annual.isel(Time=k), linewidth=1, color='black', alpha=0.4)
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label('Volume transport (Sv)', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, zmax)
    ax1.invert_yaxis()
    ax1.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')

    # Plot Heat Transport wrt Tref=0
    ax2 = plt.subplot(322)
    fld = heatTransport_annual.values
    cf = ax2.contourf(x, y, fld, cmap=colormap, extend='both')
    for k in indYears_preMLDhigh:
        ax2.axvline(x=t_annual.isel(Time=k), linewidth=1, color='white', alpha=0.6)
    for k in indYears_preMLDlow:
        ax2.axvline(x=t_annual.isel(Time=k), linewidth=1, color='black', alpha=0.4)
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label('Heat transport (0$^\circ$C; TW)', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, zmax)
    ax2.invert_yaxis()
    ax2.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')

    # Plot Heat Transport wrt Tref=TfreezingPoint
    ax3 = plt.subplot(323)
    fld = heatTransportTfp_annual.values
    cf = ax3.contourf(x, y, fld, cmap=colormap, extend='both')
    for k in indYears_preMLDhigh:
        ax3.axvline(x=t_annual.isel(Time=k), linewidth=1, color='white', alpha=0.6)
    for k in indYears_preMLDlow:
        ax3.axvline(x=t_annual.isel(Time=k), linewidth=1, color='black', alpha=0.4)
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label('Heat transport (Tfp; TW)', fontsize=10, fontweight='bold')
    ax3.set_ylim(0, zmax)
    ax3.invert_yaxis()
    ax3.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')

    # Plot transect mean temperature
    ax4 = plt.subplot(324)
    fld = tempTransect_annual.values
    colormap = cmocean.cm.thermal
    cf = ax4.contourf(x, y, fld, cmap=colormap, extend='both')
    for k in indYears_preMLDhigh:
        ax4.axvline(x=t_annual.isel(Time=k), linewidth=1, color='white', alpha=0.6)
    for k in indYears_preMLDlow:
        ax4.axvline(x=t_annual.isel(Time=k), linewidth=1, color='black', alpha=0.4)
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label('Temperature ($^\circ$C)', fontsize=10, fontweight='bold')
    ax4.set_ylim(0, zmax)
    ax4.invert_yaxis()
    ax4.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')

    # Plot FW Transport wrt Sref
    ax5 = plt.subplot(325)
    fld = FWTransportSref_annual.values
    colormap = cmocean.cm.balance
    cf = ax5.contourf(x, y, fld, cmap=colormap, extend='both')
    for k in indYears_preMLDhigh:
        ax5.axvline(x=t_annual.isel(Time=k), linewidth=1, color='white', alpha=0.6)
    for k in indYears_preMLDlow:
        ax5.axvline(x=t_annual.isel(Time=k), linewidth=1, color='black', alpha=0.4)
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    #cbar.set_label(f'FW transport wrt {saltRef:4.1f} (mSv)', fontsize=10, fontweight='bold')
    cbar.set_label(f'FW transport (Sref; mSv)', fontsize=10, fontweight='bold')
    ax5.set_ylim(0, zmax)
    ax5.invert_yaxis()
    ax5.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Time (Years)', fontsize=10, fontweight='bold')

    # Plot transect mean salinity
    ax6 = plt.subplot(326)
    fld = saltTransect_annual.values
    colormap = cmocean.cm.haline
    cf = ax6.contourf(x, y, fld, cmap=colormap, extend='both')
    for k in indYears_preMLDhigh:
        ax6.axvline(x=t_annual.isel(Time=k), linewidth=1, color='white', alpha=0.6)
    for k in indYears_preMLDlow:
        ax6.axvline(x=t_annual.isel(Time=k), linewidth=1, color='black', alpha=0.4)
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label(f'Salinity (psu)', fontsize=10, fontweight='bold')
    ax6.set_ylim(0, zmax)
    ax6.invert_yaxis()
    ax6.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Time (Years)', fontsize=10, fontweight='bold')

    fig.suptitle(f'Transect = {transectName}\nrunname = {casename}{ensemble}\n years preceding HC (white) and LC (black)', fontsize=12, fontweight='bold', y=1)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=1.5, ybuffer=-0.7)
    #fig.tight_layout(pad=0.5)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

    # Plot spiciness0 in different figure
    figfile = f'{figdir}/spicevsdepth4composites_{transectName}_{casename}{ensemble}.png'
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot()
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(14)
        tick.set_weight('bold')
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(14)
        tick.set_weight('bold')
    fld = spiceTransect_annual.values
    colormap = plt.get_cmap('cividis')
    colorIndices = [0, 14, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]
    clevels = [0, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.1, 1.21, 1.32]
    #clevels = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]
    #clevels = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 1.65, 1.8]
    [cmap, cnorm] = _make_discrete_colormap(colormap, colorIndices, clevels)
    cf = ax.contourf(x, y, fld, cmap=cmap, norm=cnorm, levels=clevels, extend='both')
    for k in indYears_preMLDhigh:
        ax.axvline(x=t_annual.isel(Time=k), linewidth=1.5, color='white', alpha=0.6)
    for k in indYears_preMLDlow:
        ax.axvline(x=t_annual.isel(Time=k), linewidth=1.5, color='black', alpha=0.4)
    cbar = plt.colorbar(cf, location='bottom', pad=0.11, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=16, labelcolor='black')
    cbar.set_label('Spiciness0 (Kg/m$^3$)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, zmax)
    ax.invert_yaxis()
    ax.set_xlabel('Time (years)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=18, fontweight='bold')
    #fig.suptitle(f'Transect = {transectName}\nrunname = {casename}{ensemble}\n years preceding HC (white) and LC (black)', \
    #             fontsize=20, fontweight='bold', y=1)
    #add_inset(fig, fc, width=1.5, height=1.5, xbuffer=1.5, ybuffer=-0.7)
    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
