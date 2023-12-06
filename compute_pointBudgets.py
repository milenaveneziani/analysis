#!/usr/bin/env python
"""
Name: compute_pointBudgets.py
Author: Milena Veneziani


"""

# ensure plots are rendered on ICC
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import datetime
import netCDF4

from common_functions import plot_xtick_format, days_to_datetime

# Choose years
year1 = 1
year2 = 65
years = range(year1, year2+1)

referenceDate = '0001-01-01'

#movingAverageMonths = 1
movingAverageMonths = 12

# Settings for lcrc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#regionmaskfile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/EC30to60E2r2_arctic_atlantic_budget_regions20230313.nc'
#regionfeaturefile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/arctic_atlantic_budget_regions.geojson'
#casenameFull = 'v2_1.LR.historical_0101'
#casename = 'v2_1.LR.historical_0101'
#modeldir = f'/lcrc/group/e3sm/ac.golaz/E3SMv2_1/{casenameFull}/archive/ocn/hist'

# Settings for nersc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
projectdir = '/global/cfs/projectdirs/e3sm'
meshfile = f'{projectdir}/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
regionmaskfile = './test.nc'
regionfeaturefile = './test.geojson'
casename = 'GM600_Redi600'
casenameFull = 'GMPAS-JRA1p4_EC30to60E2r2_GM600_Redi600_perlmutter'
modeldir = f'{projectdir}/maltrud/archive/onHPSS/{casenameFull}/ocn/hist'
#meshfile = f'{projectdir}/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
#regionmaskfile = f'{projectdir}/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regions20230313.nc'
#regionfeaturefile = f'{projectdir}/milena/mpas-region_masks/arctic_atlantic_budget_regions.geojson'
#casenameFull = '20221201.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.lat-dep-bd-submeso.cori-knl'
#casename = 'fullyRRM_lat-dep-bd-submeso'
#modeldir = f'/global/cscratch1/sd/milena/e3sm_scratch/cori-knl/{casenameFull}/run'

# Coordinates of point where to compute budgets
lonPoint = -10
latPoint = 67

m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
rho0 = 1027.0 # kg/m^3
earthRadius = 6367.44 # km

figdir = f'./volBudget/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

nTime = 12*len(years)

# Read in relevant global mesh information
dsMesh = xr.open_dataset(meshfile)
nLevels = dsMesh.dims['nVertLevels']
nCells = dsMesh.dims['nCells']
# Identify index of selected ocean cell, by computing the minimum
# of the spherical distance between all points and lonPoint,latPoint
lonCell = dsMesh.lonCell
latCell = dsMesh.latCell
lonPoint = np.pi/180*lonPoint
latPoint = np.pi/180*latPoint
dlon = lonCell-lonPoint
dlat = latCell-latPoint
a = np.sin(dlat / 2) ** 2 + np.cos(latCell) * np.cos(latPoint) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
spherDist = earthRadius * c
indices = xr.DataArray(data=np.arange(nCells).astype(int), dims='nCells')
iCell = indices.where(spherDist==np.min(spherDist), drop=True).values.astype(int)[0]
print(lonCell.values[iCell]*180/np.pi, latCell.values[iCell]*180/np.pi)
areaCell = dsMesh.areaCell.isel(nCells=iCell)
#
# edgeID of all edges bordering the chosen cell. If 0, edge is on land, so remove it.
edgesOnCell = dsMesh.edgesOnCell.isel(nCells=iCell).values
edgesOnCell = edgesOnCell[np.where(edgesOnCell>0)]
# for each ocean edge bordering the chosen cell, select IDs of straddling cells
cellsOnEdge = dsMesh.cellsOnEdge.isel(nEdges=edgesOnCell-1).values
coe0 = cellsOnEdge[:, 0] - 1
coe1 = cellsOnEdge[:, 1] - 1
dvEdge = dsMesh.dvEdge.isel(nEdges=edgesOnCell-1)

# Compute volume budget

# Initialize volume budget terms
volNetLateralFlux = np.zeros(nTime)
evapFlux = np.zeros(nTime)
rainFlux = np.zeros(nTime)
snowFlux = np.zeros(nTime)
riverRunoffFlux = np.zeros(nTime)
iceRunoffFlux = np.zeros(nTime)
seaIceFreshWaterFlux = np.zeros(nTime)
layerThick = np.zeros(nTime)
frazilThick = np.zeros(nTime)
t = np.zeros(nTime)

ktime = 0
for year in years:
    print(f'Year = {year:04d} out of {len(years)} years total')
    for month in range(1, 13):
        print(f'  Month= {month:02d}')
        modelfile = f'{modeldir}/{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

        ds = xr.open_dataset(modelfile, decode_times=False)

        t[ktime] = ds.Time.isel(Time=0).values

        # Compute net lateral fluxes:
        if 'timeMonthly_avg_normalTransportVelocity' in ds.keys():
            vel = ds.timeMonthly_avg_normalTransportVelocity.isel(Time=0, nEdges=edgesOnCell-1)
        elif 'timeMonthly_avg_normalVelocity' in ds.keys():
            vel = ds.timeMonthly_avg_normalVelocity.isel(Time=0, nEdges=edgesOnCell-1)
            if 'timeMonthly_avg_normalGMBolusVelocity' in ds.keys():
                vel = vel + ds.timeMonthly_avg_normalGMBolusVelocity.isel(Time=0, nEdges=edgesOnCell-1)
            if 'timeMonthly_avg_normalMLEvelocity' in ds.keys():
                vel = vel + ds.timeMonthly_avg_normalMLEvelocity.isel(Time=0, nEdges=edgesOnCell-1)
        else:
            raise KeyError('no appropriate normalVelocity variable found')
        dzOnCells0 = ds.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe0)
        dzOnCells1 = ds.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe1)
        #  Then, interpolate dz's onto edges, also considering the topomask
        dzOnEdges = 0.5 * (dzOnCells0 + dzOnCells1)
        dzOnEdges = dzOnEdges.rename({'nCells': 'nEdges'})
        dArea = dvEdge * dzOnEdges
        lateralFlux = (vel * dArea).sum(dim='nVertLevels', skipna=True).sum(dim='nEdges')
        volNetLateralFlux[ktime] = lateralFlux.values

        # Compute net surface fluxes:
        if 'timeMonthly_avg_evaporationFlux' in ds.keys():
            flux = ds.timeMonthly_avg_evaporationFlux.isel(Time=0, nCells=iCell)
            evapFlux[ktime] = (flux * areaCell).values
        else:
            raise KeyError('no evaporation flux variable found')
        if 'timeMonthly_avg_rainFlux' in ds.keys():
            flux = ds.timeMonthly_avg_rainFlux.isel(Time=0, nCells=iCell)
            rainFlux[ktime] = (flux * areaCell).values
        else:
            raise KeyError('no rain flux variable found')
        if 'timeMonthly_avg_snowFlux' in ds.keys():
            flux = ds.timeMonthly_avg_snowFlux.isel(Time=0, nCells=iCell)
            snowFlux[ktime] = (flux * areaCell).values
        else:
            raise KeyError('no snow flux variable found')
        if 'timeMonthly_avg_riverRunoffFlux' in ds.keys():
            flux = ds.timeMonthly_avg_riverRunoffFlux.isel(Time=0, nCells=iCell)
            riverRunoffFlux[ktime] = (flux * areaCell).values
        else:
            raise KeyError('no river runoff flux variable found')
        if 'timeMonthly_avg_iceRunoffFlux' in ds.keys():
            flux = ds.timeMonthly_avg_iceRunoffFlux.isel(Time=0, nCells=iCell)
            iceRunoffFlux[ktime] = (flux * areaCell).values
        else:
            raise KeyError('no ice runoff flux variable found')
        if 'timeMonthly_avg_seaIceFreshWaterFlux' in ds.keys():
            flux = ds.timeMonthly_avg_seaIceFreshWaterFlux.isel(Time=0, nCells=iCell)
            seaIceFreshWaterFlux[ktime] = (flux * areaCell).values
        else:
            raise KeyError('no sea ice freshwater flux variable found')
        if 'timeMonthly_avg_icebergFlux' in ds.keys():
            flux = ds.timeMonthly_avg_icebergFlux.isel(Time=0, nCells=iCell)
            icebergFlux[ktime] = (flux * areaCell).values
        if 'timeMonthly_avg_landIceFlux' in ds.keys():
            flux = ds.timeMonthly_avg_landIceFlux.isel(Time=0, nCells=iCell)
            landIceFlux[ktime] = (flux * areaCell).values

        # Compute layer thickness tendencies:
        if 'timeMonthly_avg_tendLayerThickness' in ds.keys():
            layerThickTend = ds.timeMonthly_avg_tendLayerThickness.isel(Time=0, nCells=iCell)
            layerThick[ktime] = (layerThickTend.sum(dim='nVertLevels', skipna=True) * areaCell).values
        else:
            raise KeyError('no layer thickness tendency variable found')
        if 'timeMonthly_avg_frazilLayerThicknessTendency' in ds.keys():
            frazilThickTend = ds.timeMonthly_avg_frazilLayerThicknessTendency.isel(Time=0, nCells=iCell)
            frazilThick[ktime] = (frazilThickTend.sum(dim='nVertLevels', skipna=True) * areaCell).values
        else:
            raise KeyError('no frazil layer thickness tendency variable found')

        ktime = ktime + 1

#print('\nnetlateral + evap + rain + snow + riverrunoff + icerunoff + seaiceflux, layerThickTend + frazilThickTend')
#print(m3ps_to_Sv*volNetLateralFlux+1/rho0*m3ps_to_Sv*evapFlux+1/rho0*m3ps_to_Sv*rainFlux+1/rho0*m3ps_to_Sv*snowFlux+1/rho0*m3ps_to_Sv*riverRunoffFlux+1/rho0*m3ps_to_Sv*iceRunoffFlux+1/rho0*m3ps_to_Sv*seaIceFreshWaterFlux, m3ps_to_Sv*layerThick+m3ps_to_Sv*frazilThick)
#print('\nnetlateral, evap + rain + snow, riverrunoff + icerunoff, seaiceflux, layerThickTend + frazilThickTend')
#print(m3ps_to_Sv*volNetLateralFlux, 1/rho0*m3ps_to_Sv*evapFlux+1/rho0*m3ps_to_Sv*rainFlux+1/rho0*m3ps_to_Sv*snowFlux, 1/rho0*m3ps_to_Sv*riverRunoffFlux+1/rho0*m3ps_to_Sv*iceRunoffFlux, 1/rho0*m3ps_to_Sv*seaIceFreshWaterFlux, m3ps_to_Sv*layerThick+m3ps_to_Sv*frazilThick)
#print('\nnetlateral, allSurfFluxes, layerThickTend + frazilThickTend')
#print(m3ps_to_Sv*volNetLateralFlux, 1/rho0*m3ps_to_Sv*evapFlux+1/rho0*m3ps_to_Sv*rainFlux+1/rho0*m3ps_to_Sv*snowFlux+1/rho0*m3ps_to_Sv*riverRunoffFlux+1/rho0*m3ps_to_Sv*iceRunoffFlux+1/rho0*m3ps_to_Sv*seaIceFreshWaterFlux, m3ps_to_Sv*layerThick+m3ps_to_Sv*frazilThick)
volNetLateralFlux = m3ps_to_Sv * volNetLateralFlux
evapFlux = 1/rho0 * m3ps_to_Sv * evapFlux
rainFlux = 1/rho0 * m3ps_to_Sv * rainFlux
snowFlux = 1/rho0 * m3ps_to_Sv * snowFlux
riverRunoffFlux = 1/rho0 * m3ps_to_Sv * riverRunoffFlux
iceRunoffFlux = 1/rho0 * m3ps_to_Sv * iceRunoffFlux
seaIceFreshWaterFlux = 1/rho0 * m3ps_to_Sv * seaIceFreshWaterFlux
thickTend = m3ps_to_Sv * (layerThick + frazilThick)
res = thickTend - (volNetLateralFlux + evapFlux + rainFlux + snowFlux + riverRunoffFlux + iceRunoffFlux + seaIceFreshWaterFlux)

figdpi = 300
figsize = (16, 16)
volNetLateralFlux_runavg = pd.Series.rolling(pd.DataFrame(volNetLateralFlux), movingAverageMonths, center=True).mean()
evapFlux_runavg = pd.Series.rolling(pd.DataFrame(evapFlux), movingAverageMonths, center=True).mean()
rainFlux_runavg = pd.Series.rolling(pd.DataFrame(rainFlux), movingAverageMonths, center=True).mean()
snowFlux_runavg = pd.Series.rolling(pd.DataFrame(snowFlux), movingAverageMonths, center=True).mean()
riverRunoffFlux_runavg = pd.Series.rolling(pd.DataFrame(riverRunoffFlux), movingAverageMonths, center=True).mean()
iceRunoffFlux_runavg = pd.Series.rolling(pd.DataFrame(iceRunoffFlux), movingAverageMonths, center=True).mean()
seaIceFreshWaterFlux_runavg = pd.Series.rolling(pd.DataFrame(seaIceFreshWaterFlux), movingAverageMonths, center=True).mean()
thickTend_runavg = pd.Series.rolling(pd.DataFrame(thickTend), movingAverageMonths, center=True).mean()
res_runavg = pd.Series.rolling(pd.DataFrame(res), movingAverageMonths, center=True).mean()
figfile = f'{figdir}/volBudget_pointIcell{iCell:d}_{casename}_years{year1:04d}-{year2:04d}.png'
fig, ax = plt.subplots(5, 2, figsize=figsize)
ax[0, 0].plot(t, volNetLateralFlux, 'k', alpha=0.5, linewidth=1.5)
ax[0, 1].plot(t, evapFlux, 'k', alpha=0.5, linewidth=1.5)
ax[1, 0].plot(t, rainFlux, 'k', alpha=0.5, linewidth=1.5)
ax[1, 1].plot(t, snowFlux, 'k', alpha=0.5, linewidth=1.5)
ax[2, 0].plot(t, riverRunoffFlux, 'k', alpha=0.5, linewidth=1.5)
ax[2, 1].plot(t, iceRunoffFlux, 'k', alpha=0.5, linewidth=1.5)
ax[3, 0].plot(t, seaIceFreshWaterFlux, 'k', alpha=0.5, linewidth=1.5)
ax[3, 1].plot(t, thickTend, 'k', alpha=0.5, linewidth=1.5)
ax[4, 0].plot(t, res, 'k', alpha=0.5, linewidth=1.5)
if movingAverageMonths==1:
    ax[4, 1].plot(t, volNetLateralFlux, 'r', linewidth=2, label='netLateral')
    ax[4, 1].plot(t, evapFlux+rainFlux+snowFlux, 'c', linewidth=2, label='E-P')
    ax[4, 1].plot(t, riverRunoffFlux+iceRunoffFlux, 'g', linewidth=2, label='runoff')
    ax[4, 1].plot(t, seaIceFreshWaterFlux, 'b', linewidth=2, label='seaiceFW')
    ax[4, 1].plot(t, -thickTend, 'm', linewidth=2, label='thickTend')
    ax[4, 1].plot(t, res, 'k', linewidth=2, label='res')
else:
    ax[0, 0].plot(t, volNetLateralFlux_runavg, 'k', linewidth=3)
    ax[0, 1].plot(t, evapFlux_runavg, 'k', linewidth=3)
    ax[1, 0].plot(t, rainFlux_runavg, 'k', linewidth=3)
    ax[1, 1].plot(t, snowFlux_runavg, 'k', linewidth=3)
    ax[2, 0].plot(t, riverRunoffFlux_runavg, 'k', linewidth=3)
    ax[2, 1].plot(t, iceRunoffFlux_runavg, 'k', linewidth=3)
    ax[3, 0].plot(t, seaIceFreshWaterFlux_runavg, 'k', linewidth=3)
    ax[3, 1].plot(t, thickTend_runavg, 'k', linewidth=3)
    ax[4, 0].plot(t, res_runavg, 'k', linewidth=3)
    ax[4, 1].plot(t, volNetLateralFlux_runavg, 'r', linewidth=2, label='netLateral')
    ax[4, 1].plot(t, evapFlux_runavg+rainFlux_runavg+snowFlux_runavg, 'c', linewidth=2, label='E-P')
    ax[4, 1].plot(t, riverRunoffFlux_runavg+iceRunoffFlux_runavg, 'g', linewidth=2, label='runoff')
    ax[4, 1].plot(t, seaIceFreshWaterFlux_runavg, 'b', linewidth=2, label='seaiceFW')
    ax[4, 1].plot(t, -thickTend_runavg, 'm', linewidth=2, label='thickTend')
    ax[4, 1].plot(t, res_runavg, 'k', linewidth=2, label='res')
    ax[4, 1].set_title(f'{movingAverageMonths}-month running averages', fontsize=16, fontweight='bold')
ax[4, 1].legend(loc='lower left')

#ax[0, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.5)
#ax[0, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.5)
#ax[1, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.5)
#ax[1, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.5)
#ax[2, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.5)
#ax[2, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.5)
#ax[3, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.5)
#ax[3, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.5)
#ax[4, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.5)
#ax[4, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.5)

ax[0, 0].autoscale(enable=True, axis='x', tight=True)
ax[0, 1].autoscale(enable=True, axis='x', tight=True)
ax[1, 0].autoscale(enable=True, axis='x', tight=True)
ax[1, 1].autoscale(enable=True, axis='x', tight=True)
ax[2, 0].autoscale(enable=True, axis='x', tight=True)
ax[2, 1].autoscale(enable=True, axis='x', tight=True)
ax[3, 0].autoscale(enable=True, axis='x', tight=True)
ax[3, 1].autoscale(enable=True, axis='x', tight=True)
ax[4, 0].autoscale(enable=True, axis='x', tight=True)
ax[4, 1].autoscale(enable=True, axis='x', tight=True)
plot_xtick_format('gregorian', np.min(t), np.max(t), maxXTicks=20)

ax[0, 0].grid(color='k', linestyle=':', linewidth = 0.5)
ax[0, 1].grid(color='k', linestyle=':', linewidth = 0.5)
ax[1, 0].grid(color='k', linestyle=':', linewidth = 0.5)
ax[1, 1].grid(color='k', linestyle=':', linewidth = 0.5)
ax[2, 0].grid(color='k', linestyle=':', linewidth = 0.5)
ax[2, 1].grid(color='k', linestyle=':', linewidth = 0.5)
ax[3, 0].grid(color='k', linestyle=':', linewidth = 0.5)
ax[3, 1].grid(color='k', linestyle=':', linewidth = 0.5)
ax[4, 0].grid(color='k', linestyle=':', linewidth = 0.5)
ax[4, 1].grid(color='k', linestyle=':', linewidth = 0.5)

ax[0, 0].set_title(f'mean={np.nanmean(volNetLateralFlux):.2e} $\pm$ {np.nanstd(volNetLateralFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[0, 1].set_title(f'mean={np.nanmean(evapFlux):.2e} $\pm$ {np.nanstd(evapFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[1, 0].set_title(f'mean={np.nanmean(rainFlux):.2e} $\pm$ {np.nanstd(rainFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[1, 1].set_title(f'mean={np.nanmean(snowFlux):.2e} $\pm$ {np.nanstd(snowFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[2, 0].set_title(f'mean={np.nanmean(riverRunoffFlux):.2e} $\pm$ {np.nanstd(riverRunoffFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[2, 1].set_title(f'mean={np.nanmean(iceRunoffFlux):.2e} $\pm$ {np.nanstd(iceRunoffFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[3, 0].set_title(f'mean={np.nanmean(seaIceFreshWaterFlux):.2e} $\pm$ {np.nanstd(seaIceFreshWaterFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[3, 1].set_title(f'mean={np.nanmean(thickTend):.2e} $\pm$ {np.nanstd(thickTend):.2e}', \
                   fontsize=16, fontweight='bold')
ax[4, 0].set_title(f'mean={np.nanmean(res):.2e} $\pm$ {np.nanstd(res):.2e}', \
                   fontsize=16, fontweight='bold')

ax[4, 0].set_xlabel('Time (Days)', fontsize=12, fontweight='bold')
ax[4, 1].set_xlabel('Time (Years)', fontsize=12, fontweight='bold')

ax[0, 0].set_ylabel('Net lateral flux (Sv)', fontsize=12, fontweight='bold')
ax[0, 1].set_ylabel('Evap flux (Sv)', fontsize=12, fontweight='bold')
ax[1, 0].set_ylabel('Rain flux (Sv)', fontsize=12, fontweight='bold')
ax[1, 1].set_ylabel('Snow flux (Sv)', fontsize=12, fontweight='bold')
ax[2, 0].set_ylabel('River runoff flux (Sv)', fontsize=12, fontweight='bold')
ax[2, 1].set_ylabel('Ice runoff flux (Sv)', fontsize=12, fontweight='bold')
ax[3, 0].set_ylabel('Sea ice FW flux (Sv)', fontsize=12, fontweight='bold')
ax[3, 1].set_ylabel('Layer+frazil thickness tend (Sv)', fontsize=12, fontweight='bold')
ax[4, 0].set_ylabel('Sum of all terms (Sv)', fontsize=12, fontweight='bold')
ax[4, 1].set_ylabel('Sv', fontsize=12, fontweight='bold')

fig.tight_layout(pad=0.5)
fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
