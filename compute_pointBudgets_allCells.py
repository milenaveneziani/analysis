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

from common_functions import plot_xtick_format


# Note: frazilLayerThicknessTendency is added to layerThicknessTend in
#       shared/mpas_ocn_frazil_forcing.F, therefore no need to take into
#       account for volume budget purposes.

# Settings for lcrc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#casenameFull = 'v2_1.LR.historical_0101'
#casename = 'v2_1.LR.historical_0101'
#modeldir = f'/lcrc/group/e3sm/ac.golaz/E3SMv2_1/{casenameFull}/archive/ocn/hist'

# Settings for nersc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
maindir = '/global/cfs/projectdirs/e3sm'
meshfile = f'{maindir}/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
casename = 'GM600_Redi600'
casenameFull = 'GMPAS-JRA1p4_EC30to60E2r2_GM600_Redi600_perlmutter'
modeldir = f'{maindir}/maltrud/archive/onHPSS/{casenameFull}/ocn/hist'
#meshfile = f'{maindir}/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
#casenameFull = 'E3SM-Arcticv2.1_historical0101'
#casename = 'E3SM-Arcticv2.1_historical0101'
#modeldir = f'/global/cfs/projectdirs/m1199/e3sm-arrm-simulations/{casenameFull}/ocn/hist'

# Choose years
#year1 = 1950
#year2 = 1960
year1 = 1
year2 = 1
#year2 = 65
years = range(year1, year2+1)

referenceDate = '0001-01-01'

movingAverageMonths = 1
#movingAverageMonths = 12

m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
rho0 = 1027.0 # kg/m^3
earthRadius = 6367.44 # km
dt = 30.0*86400.0 # 1 month dt in seconds for volumeTend calculation # *

figdir = f'./volBudget/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

nTimes = 12*len(years)

# Read in relevant global mesh information
dsMesh = xr.open_dataset(meshfile)
nLevels = dsMesh.dims['nVertLevels']
nCells = dsMesh.dims['nCells']

# Compute volume budget

# Initialize volume budget terms
lateralFluxGlobal = np.zeros(nTimes)
evapFluxGlobal = np.zeros(nTimes)
rainFluxGlobal = np.zeros(nTimes)
snowFluxGlobal = np.zeros(nTimes)
riverRunoffFluxGlobal = np.zeros(nTimes)
iceRunoffFluxGlobal = np.zeros(nTimes)
seaIceFreshWaterFluxGlobal = np.zeros(nTimes)
layerThickGlobal = np.zeros(nTimes)
volume = np.nan*np.ones((nTimes, nCells)) # *
volumeTendGlobal = np.zeros(nTimes) # *
frazilThickGlobal = np.zeros(nTimes) # *
t = np.zeros(nTimes)

ktime = 0
for year in years:
    print(f'Year = {year:04d} out of {len(years)} years total')
    for month in range(1, 13):
    #for month in range(1, 2):
        print(f'  Month= {month:02d}')
        modelfile = f'{modeldir}/{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

        ds = xr.open_dataset(modelfile, decode_times=False)

        t[ktime] = ds.Time.isel(Time=0).values

        volNetLateralFlux = np.nan*np.ones(nCells)
        evapFlux = np.nan*np.ones(nCells)
        rainFlux = np.nan*np.ones(nCells)
        snowFlux = np.nan*np.ones(nCells)
        riverRunoffFlux = np.nan*np.ones(nCells)
        iceRunoffFlux = np.nan*np.ones(nCells)
        seaIceFreshWaterFlux = np.nan*np.ones(nCells)
        layerThick = np.nan*np.ones(nCells)
        frazilThick = np.nan*np.ones(nCells) # *

        for iCell in range(nCells):
        #for iCell in range(10):

            # edgeID of all edges bordering the chosen cell. If 0, edge is on land, so remove it
            edgesOnCell = dsMesh.edgesOnCell.isel(nCells=iCell).values
            edgesOnCell = edgesOnCell[np.where(edgesOnCell>0)]
            # for each ocean edge bordering the chosen cell, select IDs of straddling cells
            cellsOnEdge = dsMesh.cellsOnEdge.isel(nEdges=edgesOnCell-1).values
            # remove edges that have one straddling cell on land
            #coe0 = []
            #coe1 = []
            #for i in range(len(edgesOnCell)):
            #    if cellsOnEdge[i, 0]!=0 and cellsOnEdge[i, 1]!=0:
            #        coe0.append(cellsOnEdge[i, 0] - 1)
            #        coe1.append(cellsOnEdge[i, 1] - 1)
            coe0 = cellsOnEdge[:, 0] - 1
            coe1 = cellsOnEdge[:, 1] - 1
            # identify land cellsOnEdge
            coe0[np.where(coe0==-1)] = 0
            coe1[np.where(coe1==-1)] = 0
            # compute edgeSigns
            edgeSigns = np.ones(len(coe0))
            for i in range(len(coe0)):
                if coe0[i]==iCell:
                    edgeSigns[i] = -1
            areaCell = dsMesh.areaCell.isel(nCells=iCell)
            print('iCell = ', iCell)
            #print('areaCell = ', areaCell.values)
            #print('cellsOnEdge0 = ', coe0)
            #print('cellsOnEdge1 = ', coe1)
            #print('edgeSign = ', edgeSigns)
            #
            dvEdge = dsMesh.dvEdge.isel(nEdges=edgesOnCell-1)

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
            #dzOnCells0 = ds.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe0)
            #dzOnCells1 = ds.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe1)
            ##  Then, interpolate dz's onto edges, also considering the topomask
            #dzOnEdges = 0.5 * (dzOnCells0 + dzOnCells1)
            #dzOnEdges = dzOnEdges.rename({'nCells': 'nEdges'})
            dzOnCells0 = ds.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe0).values
            dzOnCells1 = ds.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe1).values
            #  Then, interpolate dz's onto edges, also considering the topomask
            dzOnEdges = np.nan*np.ones(np.shape(dzOnCells0))
            for i in range(len(coe0)):
                if coe0[i]==0:
                    dzOnEdges[i, :] = dzOnCells1[i, :]
                elif coe1[i]==0:
                    dzOnEdges[i, :] = dzOnCells0[i, :]
                else:
                    dzOnEdges[i, :] = 0.5 * (dzOnCells0[i, :] + dzOnCells1[i, :])
            dzOnEdges = xr.DataArray(dzOnEdges, dims=('nEdges', 'nVertLevels'))
            dArea = dvEdge * dzOnEdges
            normalVel = vel * xr.DataArray(edgeSigns, dims='nEdges')
            lateralFlux = (normalVel * dArea).sum(dim='nVertLevels', skipna=True).sum(dim='nEdges')
            volNetLateralFlux[iCell] = lateralFlux.values

            # Compute net surface fluxes:
            if 'timeMonthly_avg_evaporationFlux' in ds.keys():
                flux = ds.timeMonthly_avg_evaporationFlux.isel(Time=0, nCells=iCell)
                evapFlux[iCell] = (flux * areaCell).values
            else:
                raise KeyError('no evaporation flux variable found')
            if 'timeMonthly_avg_rainFlux' in ds.keys():
                flux = ds.timeMonthly_avg_rainFlux.isel(Time=0, nCells=iCell)
                rainFlux[iCell] = (flux * areaCell).values
            else:
                raise KeyError('no rain flux variable found')
            if 'timeMonthly_avg_snowFlux' in ds.keys():
                flux = ds.timeMonthly_avg_snowFlux.isel(Time=0, nCells=iCell)
                snowFlux[iCell] = (flux * areaCell).values
            else:
                raise KeyError('no snow flux variable found')
            if 'timeMonthly_avg_riverRunoffFlux' in ds.keys():
                flux = ds.timeMonthly_avg_riverRunoffFlux.isel(Time=0, nCells=iCell)
                riverRunoffFlux[iCell] = (flux * areaCell).values
            else:
                raise KeyError('no river runoff flux variable found')
            if 'timeMonthly_avg_iceRunoffFlux' in ds.keys():
                flux = ds.timeMonthly_avg_iceRunoffFlux.isel(Time=0, nCells=iCell)
                iceRunoffFlux[iCell] = (flux * areaCell).values
            else:
                raise KeyError('no ice runoff flux variable found')
            if 'timeMonthly_avg_seaIceFreshWaterFlux' in ds.keys():
                flux = ds.timeMonthly_avg_seaIceFreshWaterFlux.isel(Time=0, nCells=iCell)
                seaIceFreshWaterFlux[iCell] = (flux * areaCell).values
            else:
                raise KeyError('no sea ice freshwater flux variable found')
            if 'timeMonthly_avg_icebergFlux' in ds.keys():
                flux = ds.timeMonthly_avg_icebergFlux.isel(Time=0, nCells=iCell)
                icebergFlux[iCell] = (flux * areaCell).values
            if 'timeMonthly_avg_landIceFlux' in ds.keys():
                flux = ds.timeMonthly_avg_landIceFlux.isel(Time=0, nCells=iCell)
                landIceFlux[iCell] = (flux * areaCell).values

            # Compute layer thickness tendencies:
            if 'timeMonthly_avg_tendLayerThickness' in ds.keys():
                layerThickTend = ds.timeMonthly_avg_tendLayerThickness.isel(Time=0, nCells=iCell)
                layerThick[iCell] = (layerThickTend.sum(dim='nVertLevels', skipna=True) * areaCell).values
            else:
                raise KeyError('no layer thickness tendency variable found')
            if 'timeMonthly_avg_ssh' in ds.keys(): # *
                volume[ktime, iCell] = (ds.timeMonthly_avg_ssh.isel(Time=0, nCells=iCell) * areaCell).values # *
            else: # *
                raise KeyError('no layer thickness tendency variable found') # *
            if 'timeMonthly_avg_frazilLayerThicknessTendency' in ds.keys(): # *
                frazilThickTend = ds.timeMonthly_avg_frazilLayerThicknessTendency.isel(Time=0, nCells=iCell) # *
                frazilThick[iCell] = (frazilThickTend.sum(dim='nVertLevels', skipna=True) * areaCell).values # *
            else: # *
                raise KeyError('no frazil layer thickness tendency variable found') # *

        lateralFluxGlobal[ktime] = np.nansum(volNetLateralFlux)
        evapFluxGlobal[ktime] = np.nansum(evapFlux)
        rainFluxGlobal[ktime] = np.nansum(rainFlux)
        snowFluxGlobal[ktime] = np.nansum(snowFlux)
        riverRunoffFluxGlobal[ktime] = np.nansum(riverRunoffFlux)
        iceRunoffFluxGlobal[ktime] = np.nansum(iceRunoffFlux)
        seaIceFreshWaterFluxGlobal[ktime] = np.nansum(seaIceFreshWaterFlux)
        layerThickGlobal[ktime] = np.nansum(layerThick)
        frazilThickGlobal[ktime] = np.nansum(frazilThick)

        ktime = ktime + 1

lateralFluxGlobal = m3ps_to_Sv * lateralFluxGlobal
evapFluxGlobal = 1/rho0 * m3ps_to_Sv * evapFluxGlobal
rainFluxGlobal = 1/rho0 * m3ps_to_Sv * rainFluxGlobal
snowFluxGlobal = 1/rho0 * m3ps_to_Sv * snowFluxGlobal
riverRunoffFluxGlobal = 1/rho0 * m3ps_to_Sv * riverRunoffFluxGlobal
iceRunoffFluxGlobal = 1/rho0 * m3ps_to_Sv * iceRunoffFluxGlobal
seaIceFreshWaterFluxGlobal = 1/rho0 * m3ps_to_Sv * seaIceFreshWaterFluxGlobal
layerThickGlobal = m3ps_to_Sv * layerThickGlobal
volume = np.diff(volume, n=1, axis=0, prepend=np.nan) # *
volumeTendGlobal = m3ps_to_Sv * (1/dt * np.nansum(volume, axis=1) + frazilThickGlobal) # *
print(volumeTendGlobal)
print(layerThickGlobal)

res = layerThickGlobal - (lateralFluxGlobal + evapFluxGlobal + rainFluxGlobal + snowFluxGlobal + riverRunoffFluxGlobal + iceRunoffFluxGlobal + seaIceFreshWaterFluxGlobal)
res2 = volumeTendGlobal - (lateralFluxGlobal + evapFluxGlobal + rainFluxGlobal + snowFluxGlobal + riverRunoffFluxGlobal + iceRunoffFluxGlobal + seaIceFreshWaterFluxGlobal)

print('\nresidual1, residual2 (with ssh calculation)')
print(res)
print(res2)
print('\nnetlateral + evap + rain + snow + riverrunoff + icerunoff + seaiceflux, layerThickTend, volumeTend')
print(lateralFluxGlobal[11]+evapFluxGlobal[11]+rainFluxGlobal[11]+snowFluxGlobal[11]+riverRunoffFluxGlobal[11]+iceRunoffFluxGlobal[11]+seaIceFreshWaterFluxGlobal[11], layerThickGlobal[11], volumeTendGlobal[11])
print('\nnetlateral, allSurfFluxes, layerThickTend, volumeTend')
print(lateralFluxGlobal[11], evapFluxGlobal[11]+rainFluxGlobal[11]+snowFluxGlobal[11]+riverRunoffFluxGlobal[11]+iceRunoffFluxGlobal[11]+seaIceFreshWaterFluxGlobal[11], layerThickGlobal[11], volumeTendGlobal[11])
boh

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
    #ax[4, 1].plot(t, res_runavg, 'k', linewidth=2, label=f'res ({np.mean(res):.2e} $\pm$ {np.std(res):.2e}')
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

ax[0, 0].set_title(f'mean={np.mean(volNetLateralFlux):.2e} $\pm$ {np.std(volNetLateralFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[0, 1].set_title(f'mean={np.mean(evapFlux):.2e} $\pm$ {np.std(evapFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[1, 0].set_title(f'mean={np.mean(rainFlux):.2e} $\pm$ {np.std(rainFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[1, 1].set_title(f'mean={np.mean(snowFlux):.2e} $\pm$ {np.std(snowFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[2, 0].set_title(f'mean={np.mean(riverRunoffFlux):.2e} $\pm$ {np.std(riverRunoffFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[2, 1].set_title(f'mean={np.mean(iceRunoffFlux):.2e} $\pm$ {np.std(iceRunoffFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[3, 0].set_title(f'mean={np.mean(seaIceFreshWaterFlux):.2e} $\pm$ {np.std(seaIceFreshWaterFlux):.2e}', \
                   fontsize=16, fontweight='bold')
ax[3, 1].set_title(f'mean={np.mean(thickTend):.2e} $\pm$ {np.std(thickTend):.2e}', \
                   fontsize=16, fontweight='bold')
ax[4, 0].set_title(f'mean={np.mean(res):.2e} $\pm$ {np.std(res):.2e}', \
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
ax[3, 1].set_ylabel('Layer thickness tend (Sv)', fontsize=12, fontweight='bold')
ax[4, 0].set_ylabel('Residual (Sv)', fontsize=12, fontweight='bold')
ax[4, 1].set_ylabel('Sv', fontsize=12, fontweight='bold')

fig.suptitle(f'iCell={iCell}, {pointTitle})', fontsize=24, fontweight='bold', y=1.1)
fig.tight_layout(pad=0.5)
fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
