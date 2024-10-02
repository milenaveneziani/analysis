#
# This script takes previously computed time series of volume, heat, and FW
# transports across a number of transects, and replots them based on high
# and low convection composites calculated for the Greenland Sea (or any
# other region). The composites are computed considering winter (JFMA)
# maximum MLD in the region of interest: therefore, the parts of the 
# transport time series that correspond to the years *preceding* the 
# high-convection winters are colored in red, whereas the parts that 
# correspond to the years preceding the low-convection winters are colored
# in blue.
#


from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset

from common_functions import add_inset
from geometric_features import FeatureCollection, read_feature_collection
from mpas_analysis.shared.io.utility import decode_strings


# Settings for erdc.hpc.mil
maskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20240910.nc'
featurefile = '/p/home/milena/mpas-region_masks/atlanticZonal_sections20240910.geojson'
casename = 'E3SMv2.1B60to10rA02'
transportfile = './transports_data/E3SMv2.1B60to10rA02/atlanticZonalSectionsTransports_E3SMv2.1B60to10rA02_years0001-0386.nc'
years_maxMLDhighFile = './composites_maxMLDbased_data/E3SMv2.1B60to10rA02/Years1-386/years_maxMLDhigh.dat'
years_maxMLDlowFile = './composites_maxMLDbased_data/E3SMv2.1B60to10rA02/Years1-386/years_maxMLDlow.dat'

year1 = 1
year2 = 386
nyears = year2-year1+1

transectsToPlot = ['Atlantic zonal 27.2N', 'Atlantic zonal 45N', 'Atlantic zonal OSNAP East', 'Atlantic zonal 65N']

figdir = f'./transports/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found')

figsize = (16, 8)
figdpi = 300
##################################################

years_maxMLDhigh = np.loadtxt(years_maxMLDhighFile)
years_maxMLDlow = np.loadtxt(years_maxMLDlowFile)
# remove year 1, if in there
years_maxMLDhigh = years_maxMLDhigh[years_maxMLDhigh>1]
years_maxMLDlow = years_maxMLDlow[years_maxMLDlow>1]
# for monthly data:
#allyears = np.array([year*np.ones(12) for year in range(year1, year2+1)]).flatten()
# for yearly data:
allyears = np.arange(year1, year2+1)
indYears_preMLDhigh = np.array([np.where(allyears==year-1) for year in years_maxMLDhigh]).flatten()
indYears_preMLDlow = np.array([np.where(allyears==year-1) for year in years_maxMLDlow]).flatten()

ds = xr.open_dataset(transportfile)
transects = decode_strings(ds.transectNames)
t = ds.Time
tannual = t.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})

#days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

nTransects = len(transectsToPlot)
for n in range(nTransects):
    transectToPlot = transectsToPlot[n]
    transectName = transectToPlot.replace(' ', '')

    fc = FeatureCollection()

    if transectToPlot=='Atlantic zonal 65N':
        for feature in fcAll.features:
            if feature['properties']['name']=='Davis Strait South' or \
               feature['properties']['name']=='Denmark Strait' or \
               feature['properties']['name']=='Iceland Norway 65N':
                fc.add_feature(feature)
        transectIndex = transects.index('Davis Strait South')
        dsTransect1 = ds.isel(nTransects=transectIndex)
        transectIndex = transects.index('Denmark Strait')
        dsTransect2 = ds.isel(nTransects=transectIndex)
        transectIndex = transects.index('Iceland Norway 65N')
        dsTransect3 = ds.isel(nTransects=transectIndex)
        volTransport = dsTransect1.volTransport + dsTransect2.volTransport + dsTransect3.volTransport
        heatTransport = dsTransect1.heatTransport + dsTransect2.heatTransport + dsTransect3.heatTransport
        heatTransportTfp = dsTransect1.heatTransportTfp + dsTransect2.heatTransportTfp + dsTransect3.heatTransportTfp
        FWTransport = dsTransect1.FWTransport + dsTransect2.FWTransport + dsTransect3.FWTransport
    else:
        for feature in fcAll.features:
            if feature['properties']['name']==transectToPlot:
                fc.add_feature(feature)
        transectIndex = transects.index(transectToPlot)
        dsTransect = ds.isel(nTransects=transectIndex)
        volTransport = dsTransect.volTransport
        heatTransport = dsTransect.heatTransport
        heatTransportTfp = dsTransect.heatTransportTfp
        FWTransport = dsTransect.FWTransport

    volTransport_annual = volTransport.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
    heatTransport_annual = heatTransport.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
    heatTransportTfp_annual = heatTransportTfp.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
    FWTransport_annual = FWTransport.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})

    fig = plt.figure(figsize=figsize)

    # Volume transport
    ax1 = plt.subplot(221)
    #ax1.plot(t, volTransport, 'k', linewidth=2)
    ax1.plot(tannual, volTransport_annual, 'silver', linewidth=1, zorder=0)
    ax1.scatter(tannual.isel(Time=indYears_preMLDhigh), volTransport_annual.isel(Time=indYears_preMLDhigh), s=5, c='r', marker='o', zorder=1)
    ax1.scatter(tannual.isel(Time=indYears_preMLDlow), volTransport_annual.isel(Time=indYears_preMLDlow), s=5, c='b', marker='o', zorder=1)
    #ax1.plot(tannual, np.zeros_like(tannual), 'k', linewidth=1)
    ax1.grid(color='k', linestyle=':', linewidth = 0.5)
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel('Volume transport (Sv)', fontsize=12, fontweight='bold')

    # Heat transport wrt 0 Celsius
    ax2 = plt.subplot(222)
    #ax2.plot(t, heatTransport, 'k', linewidth=2)
    ax2.plot(tannual, heatTransport_annual, 'silver', linewidth=1, zorder=0)
    ax2.scatter(tannual.isel(Time=indYears_preMLDhigh), heatTransport_annual.isel(Time=indYears_preMLDhigh), s=5, c='r', marker='o', zorder=1)
    ax2.scatter(tannual.isel(Time=indYears_preMLDlow), heatTransport_annual.isel(Time=indYears_preMLDlow), s=5, c='b', marker='o', zorder=1)
    #ax2.plot(tannual, np.zeros_like(tannual), 'k', linewidth=1)
    ax2.grid(color='k', linestyle=':', linewidth = 0.5)
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_ylabel('Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')

    # Heat transport wrt T freezing point
    ax3 = plt.subplot(223)
    #ax3.plot(t, heatTransportTfp, 'k', linewidth=2)
    ax3.plot(tannual, heatTransportTfp_annual, 'silver', linewidth=1, zorder=0)
    ax3.scatter(tannual.isel(Time=indYears_preMLDhigh), heatTransportTfp_annual.isel(Time=indYears_preMLDhigh), s=5, c='r', marker='o', zorder=1)
    ax3.scatter(tannual.isel(Time=indYears_preMLDlow), heatTransportTfp_annual.isel(Time=indYears_preMLDlow), s=5, c='b', marker='o', zorder=1)
    #ax3.plot(tannual, np.zeros_like(tannual), 'k', linewidth=1)
    ax3.grid(color='k', linestyle=':', linewidth = 0.5)
    ax3.autoscale(enable=True, axis='x', tight=True)
    ax3.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Heat transport wrt freezing point (TW)', fontsize=12, fontweight='bold')

    # FW transport wrt Sref
    ax4 = plt.subplot(224)
    #ax4.plot(t, FWTransport, 'k', linewidth=2)
    ax4.plot(tannual, FWTransport_annual, 'silver', linewidth=1, zorder=0)
    ax4.scatter(tannual.isel(Time=indYears_preMLDhigh), FWTransport_annual.isel(Time=indYears_preMLDhigh), s=5, c='r', marker='o', zorder=1)
    ax4.scatter(tannual.isel(Time=indYears_preMLDlow), FWTransport_annual.isel(Time=indYears_preMLDlow), s=5, c='b', marker='o', zorder=1)
    #ax4.plot(tannual, np.zeros_like(tannual), 'k', linewidth=1)
    ax4.grid(color='k', linestyle=':', linewidth = 0.5)
    ax4.autoscale(enable=True, axis='x', tight=True)
    ax4.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('FW transport (mSv)', fontsize=12, fontweight='bold')

    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Transect = {transectToPlot}\nrunname = {casename}', fontsize=14, fontweight='bold', y=1.045)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=-0.5, ybuffer=-1.65)
    figfile = f'{figdir}/transports4composites_{transectName}_{casename}.png'
    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
