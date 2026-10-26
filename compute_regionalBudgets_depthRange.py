#!/usr/bin/env python
"""
Name: compute_regionalBudgets_depthRange.py
Author: Milena Veneziani


"""

# ensure plots are rendered on ICC
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import cartopy
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta # usage: datetime.object + relativedelta(years=shiftyear), when wanting to shif by a certain number of years
import netCDF4
import cftime
mpl.use('Agg')

from mpas_analysis.shared.io.utility import decode_strings
from mpas_analysis.shared.io import write_netcdf_with_fill
from mpas_analysis.ocean.utility import compute_zmid

from common_functions import extract_openBoundaries, add_inset, plot_xtick_format
from geometric_features import FeatureCollection, read_feature_collection


################################
# Settings for lcrc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#featurefile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/arctic_atlantic_budget_regions.geojson'
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#regionmaskfile = '/lcrc/group/e3sm/ac.milena/mpas-region_masks/EC30to60E2r2_arctic_atlantic_budget_regions20230313.nc'
#casenameFull = 'v2_1.LR.historical_0101'
#casename = 'v2_1.LR.historical_0101'
#modeldir = f'/lcrc/group/e3sm/ac.golaz/E3SMv2_1/{casenameFull}/archive/ocn/hist'

# Settings for nersc:
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#featurefile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/arctic_atlantic_budget_regions.geojson'
#featurefile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/arctic_atlantic_budget_regions_new20240408.geojson'
#meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/EC30to60E2r2/mpaso.EC30to60E2r2.rstFromG-anvil.201001.nc'
#regionmaskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/EC30to60E2r2_arctic_atlantic_budget_regions20230313.nc'
#casename = 'GM600_Redi600'
#casenameFull = 'GMPAS-JRA1p4_EC30to60E2r2_GM600_Redi600_perlmutter'
#modeldir = f'/global/cfs/cdirs/e3sm/maltrud/archive/onHPSS/{casenameFull}/ocn/hist'
#
#meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
#regionmaskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regions_new20240408.nc'
#regionmaskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_greaterArctic04082024.nc'
#regionmaskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regions20230313.nc'
#regionmaskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regions.nc'
#casenameFull = 'E3SM-Arcticv2.1_historical0101'
#casename = 'E3SM-Arcticv2.1_historical0101'
#modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{casenameFull}/archive/ocn/hist'

# Settings for erdc.hpc.mil
meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
##regionmaskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_NH.nc'
##featurefile = '/p/home/milena/mpas-region_masks/NH.geojson'
#regionmaskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regions_new20240408.nc'
#featurefile = '/p/home/milena/mpas-region_masks/arctic_atlantic_budget_regions_new20240408.geojson'
regionmaskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regions_20260827.nc'
featurefile = '/p/home/milena/mpas-region_masks/arctic_atlantic_budget_regions_20260827.geojson'
#regionmaskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_arcticRegions.nc'
#featurefile = '/p/home/milena/mpas-region_masks/arcticRegions.geojson'
#regionmaskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_amocPaper_regions.nc'
#featurefile = '/p/home/milena/mpas-region_masks/amocPaper_regions.geojson'
#casenameFull = 'E3SMv2.1G60to10_01'
#casename = 'E3SMv2.1G60to10_01'
casenameFull = 'E3SMv2.1B60to10rA02'
casename = 'E3SMv2.1B60to10rA02'
modeldir = f'/p/global/milena/{casenameFull}/archive/ocn/hist'
#casenameFull = 'E3SMv2.1B60to10rA07'
#casename = 'E3SMv2.1B60to10rA07'
#modeldir = f'/p/global/apcraig/archive/{casenameFull}/ocn/hist'
#casenameFull = 'E3SMv3G60to10_01cd25'
#casename = 'E3SMv3G60to10_01cd25'
#modeldir = f'/p/global/osinski/archive/{casenameFull}/ocn/hist'

#regionNames = ['all']
#regionNames = ['Irminger Sea']
#regionNames = ['Irminger Sea', 'Labrador Sea']
#regionNames = ['Arctic Ocean (no Barents/Kara Seas)', 'North Atlantic subpolar gyre', 'Irminger Sea', 'Labrador Sea', 'Greenland Sea', 'Norwegian Sea']
#regionNames = ['Arctic Ocean (no Barents/Kara Seas)', 'Irminger Sea', 'Labrador Sea', 'Greenland Sea', 'Norwegian Sea']
#regionNames = ['Labrador Sea']
regionNames = ['North Atlantic subpolar gyre', 'North Atlantic greater subpolar gyre', 'Greater Arctic', 'Nordic Seas', 'North Atlantic subtropical gyre']
#
#regionNames = ['North Atlantic Wilbert', 'South Atlantic Wilbert']
#regionNames = ['South Atlantic Wilbert']

# Choose years
year1 = 1950
year2 = 1955
#year2 = 2014
#year1 = 1
#year2 = 50
#year2 = 386
years = range(year1, year2+1)
referenceDate = '0001-01-01'
calendar = 'noleap'
#shiftyear = 1900

zmin = -800.0
zmax = 10.0

makePlots = True

# If plotAnnualAvg = False, time-integrated tendency terms 
# will be plotted instead of annual averages
plotAnnualAvg = True

m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
rho0 = 1026.0 # kg/m^3 (config_density0)
perSec_to_perDay = 86400.0 # 1/s to 1/day
cp = 3.996*1e3 # J/(kg*degK) (as defined in mpas_constants)
restoring_pistonvel = 1.585e-6 # m/s (config_salinity_restoring_constant_piston_velocity)
# Apply depth redistribution of surface fluxes only if attCoef is greater 
# than attCoefMax, set to 1 mm by default (as in MPAS-O)
attCoefMax = 0.001

figdir = f'./budgets/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)
outdir = f'./budgets_data/{casename}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found for this region group')

legend_properties = {'size':10, 'weight':'bold'}

###
### PART 1 -- Read/compute mesh and regional mask quantities
###

# Read in regions information
dsRegionMask = xr.open_dataset(regionmaskfile)
regions = decode_strings(dsRegionMask.regionNames)
if regionNames[0]=='all':
    regionNames = regions
nRegions = np.size(regionNames)

# Read in relevant global mesh information
dsMesh = xr.open_dataset(meshfile)
areaCell = dsMesh.areaCell
# Necessary fields for computing depth mask later
maxLevelCell = dsMesh.maxLevelCell - 1 # now compute_zmid uses 0-based indexing
depth = dsMesh.bottomDepth

for n in range(nRegions):
    regionName = regionNames[n]
    print(f'\n**** Regional budgets for: {regionName} ****')

    if regionName=='Arctic Ocean (no Barents/Kara Seas)':
        rname = 'ArcticOcean_noBarentsnoKara'
    else:
        rname = regionName.replace(' ', '').replace('(', '').replace(')', '')
    regionIndex = regions.index(regionName)

    # Get regional mask quantities
    dsMask = dsRegionMask.isel(nRegions=regionIndex)
    [openBryEdges, openBrySigns, landEdges] = extract_openBoundaries(dsMask, dsMesh)
    cellMask = dsMask.regionCellMasks == 1
    regionArea = areaCell.where(cellMask, drop=True)
    regionAreaTot = regionArea.sum(dim='nCells')
    dvEdge = dsMesh.dvEdge[openBryEdges]
    coe0 = dsMesh.cellsOnEdge.isel(TWO=0, nEdges=openBryEdges) - 1
    coe1 = dsMesh.cellsOnEdge.isel(TWO=1, nEdges=openBryEdges) - 1

    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == regionName:
            fc.add_feature(feature)
            break

    ###
    ### PART 2 -- Compute yearly budget terms if outfile does not exist
    ###
    kyear = 0
    outfiles = []
    for year in years:
        kyear = kyear + 1
        if zmax>0:
            outfile = f'{outdir}/budgetTerms_z0000-{np.abs(np.int32(zmin)):04d}_{rname}_year{year:04d}.nc'
        else:
            outfile = f'{outdir}/budgetTerms_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{rname}_year{year:04d}.nc'
        outfiles.append(outfile)

        if not os.path.exists(outfile):
            print(f'  Compute budget terms over depths {zmax},{zmin} for year = {year:04d} ({kyear} out of {len(years)} years total)')
            dsOut = []
            newTime = np.empty(12, dtype=datetime)
            for month in range(1, 13):
                im = month-1
                print(f'  Month= {month:02d}')
                modelfile = f'{modeldir}/{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

                dsIn = xr.open_dataset(modelfile, decode_times=False)
                start, end = [parse(dsIn[f'xtime_{name}Monthly'].astype(str).values[0].split('_')[0]) for name in ('start', 'end')]
                if start.year < 1000:
                    newTime[im] = dsIn['Time'].values
                else:
                    newTime[im] = start + timedelta(days=int((end - start).days / 2))
                attCoef = dsIn.attrs.get('config_flux_attenuation_coefficient')
                attCoefRunoff = dsIn.attrs.get('config_flux_attenuation_coefficient_runoff')
                dsOutMonthly = xr.Dataset()

                dzOnCells = dsIn.timeMonthly_avg_layerThickness
                zMid = compute_zmid(depth, maxLevelCell, dzOnCells)
                depthMask = np.logical_and(zMid >= zmin, zMid <= zmax)
                depthLayer = dzOnCells.where(depthMask, drop=False).sum(dim='nVertLevels')

                print('Compute frazil layer thickness tendency')
                if not 'timeMonthly_avg_frazilLayerThicknessTendency' in dsIn.keys():
                    raise KeyError('no frazil layer thickness tendency variable found')
                frazilThickTend = dsIn.timeMonthly_avg_frazilLayerThicknessTendency
                frazilThickTend = frazilThickTend.where(depthMask, drop=False).where(cellMask, drop=True)
                frazilThickTend = (frazilThickTend * regionArea).sum(dim='nVertLevels', skipna=True).sum(dim='nCells')

                #####
                ##### Salinity budget terms
                #####
                print('Compute salinity budget terms')
                print('  salinity tendency')
                if not 'timeMonthly_avg_activeTracersTend_salinityTend' in dsIn.keys():
                    raise KeyError('no salinity time tendency variable found')
                # This is actually salinity tendency weighted by layer thickness
                salinityTend = dsIn.timeMonthly_avg_activeTracersTend_salinityTend
                salinityTend = salinityTend.where(depthMask, drop=False).where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                salinityTend = salinityTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['saltTendency'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change (depth weigthed, i.e. d(hS)/dt) due to salinity time tendency over depth range {zmax:d},{zmin:d}', units='m 1e-3 s^-1', )
                    )

                if 'timeMonthly_avg_frazilSalinityTendency' in dsIn.keys():
                    print('  frazil salinity tendency')
                    # This is actually tendency weighted by layer thickness
                    salinityTend = dsIn.timeMonthly_avg_frazilSalinityTendency
                    salinityTend = salinityTend.where(depthMask, drop=False).where(cellMask, drop=True)
                    salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                    salinityTend = salinityTend.sum(dim='nVertLevels', skipna=True)
                else:
                    print('  frazil salinity tendency estimated from the frazilThicknessTendency * 4 psu')
                    salinityTend = frazilThickTend / regionAreaTot * 4
                dsOutMonthly['saltFrazilTendency'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change (depth weigthed) due to frazil ice over depth range {zmax:d},{zmin:d}', units='m 1e-3 s^-1', )
                    )

                # The following activeTracerTendency terms are *not* weighted by layer thickness
                print('  horizontal advection (includes GM and MLE)')
                if not 'timeMonthly_avg_activeTracerHorizontalAdvectionTendency_salinityHorizontalAdvectionTendency' in dsIn.keys():
                    raise KeyError('no salinity horizontal advection tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerHorizontalAdvectionTendency_salinityHorizontalAdvectionTendency
                salinityTend = (salinityTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                salinityTend = salinityTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['saltHAdvTendency'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change (depth weigthed) due to horizontal advection (including GM and MLE contributions) over depth range {zmax:d},{zmin:d}', units='m 1e-3 s^-1', )
                    )

                print('  vertical advection')
                if not 'timeMonthly_avg_activeTracerVerticalAdvectionTendency_salinityVerticalAdvectionTendency' in dsIn.keys():
                    raise KeyError('no salinity vertical advection tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerVerticalAdvectionTendency_salinityVerticalAdvectionTendency
                salinityTend = (salinityTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                salinityTend = salinityTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['saltVAdvTendency'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change (depth weigthed) due to vertical advection over depth range {zmax:d},{zmin:d}', units='m 1e-3 s^-1', )
                    )

                print('  horizontal mixing (includes Redi)')
                if not 'timeMonthly_avg_activeTracerHorMixTendency_salinityHorMixTendency' in dsIn.keys():
                    raise KeyError('no salinity horizontal mixing tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerHorMixTendency_salinityHorMixTendency
                salinityTend = (salinityTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                salinityTend = salinityTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['saltHMixTendency'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change (depth weigthed) due to horizontal mixing (including del2, del4, and Redi) over depth range {zmax:d},{zmin:d}', units='m 1e-3 s^-1', )
                    )

                print('  nonlocal tendency')
                if not 'timeMonthly_avg_activeTracerNonLocalTendency_salinityNonLocalTendency' in dsIn.keys():
                    raise KeyError('no salinity non local tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerNonLocalTendency_salinityNonLocalTendency
                salinityTend = (salinityTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                salinityTend = salinityTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['saltNonLocalTendency'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change (depth weigthed) due to non local mixing over depth range {zmax:d},{zmin:d}', units='m 1e-3 s^-1', )
                    )

                print('  vertical mixing')
                if not 'timeMonthly_avg_activeTracerVertMixTendency_salinityVertMixTendency' in dsIn.keys():
                    raise KeyError('no salinity vertical mixing tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerVertMixTendency_salinityVertMixTendency
                salinityTend = (salinityTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                salinityTend = salinityTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['saltVMixTendency'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change (depth weigthed) due to vertical mixing over depth range {zmax:d},{zmin:d}', units='m 1e-3 s^-1', )
                    )

                print('  surface fluxes (sea ice salinity flux and SSS restoring if present)')
                if not 'timeMonthly_avg_activeTracerSurfaceFluxTendency_salinitySurfaceFluxTendency' in dsIn.keys():
                    raise KeyError('no salinity surface flux tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerSurfaceFluxTendency_salinitySurfaceFluxTendency
                salinityTend = (salinityTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                salinityTend = salinityTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['saltSurfaceFluxTendency'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change (depth weigthed) due to surface fluxes (sea ice salinity flux and SSS restoring if present) over depth range {zmax:d},{zmin:d}', units='m 1e-3 s^-1', )
                        )

                print('  surface fluxes due to sea ice salinity flux only')
                if not 'timeMonthly_avg_seaIceSalinityFlux' in dsIn.keys():
                    raise KeyError('no sea ice salinity flux variable found')
                # seaIceSalinityFlux is Kg of salt (associated to sea ice changes)
                # per (m^2 s), i.e. 10^3 grams of salt per unit area and per second.
                # Hence, If I divide by rho_fw (10^3 Kg of FW per m^3), I get units of
                # psu * m / s (where psu=grams of salt per Kg of FW). After that,
                # we can treat it as a thickness weigthed salinity tendency, similarly to
                # the other salinity tendency terms.
                saltFlux_to_salinityFlux = 1.0
                salinityTend = dsIn.timeMonthly_avg_seaIceSalinityFlux
                salinityTend = saltFlux_to_salinityFlux * salinityTend
                salinityTend = salinityTend.where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['saltSeaIceTendency'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change (depth weigthed) due to sea ice salinity flux', units='m 1e-3 s^-1', )
                    )

                if 'timeMonthly_avg_salinitySurfaceRestoringTendency' in dsIn.keys():
                    print('  surface fluxes due to SSS-restoring salinity tendency only')
                    # This is piston_velocity * Delta(SSS_restoringValue), hence in units of m * 1e-3 (psu)/s.
                    # We therefore treat it as a thickness weigthed salinity tendency, similarly to
                    # the other salinity tendency terms.
                    salinityTend = dsIn.timeMonthly_avg_salinitySurfaceRestoringTendency
                    salinityTend = salinityTend.where(cellMask, drop=True)
                    salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                    dsOutMonthly['saltSurfaceRestoringTendency'] = xr.DataArray(
                        data=salinityTend,
                        dims=('Time', ),
                        attrs=dict(description='Salinity change (depth weigthed) due to SSS restoring', units='m 1e-3 s^-1', )
                        )
                else:
                    print('  WARNING: variable salinitySurfaceRestoringTendency unavailable: skipping SSS-restoring salinity tendency')

                print('  salinity')
                salinity = dsIn.timeMonthly_avg_activeTracers_salinity
                salinity = (salinity * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                salinity = (salinity * regionArea).sum(dim='nCells') / regionAreaTot
                salinity = salinity.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['salinity'] = xr.DataArray(
                    data=salinity,
                    dims=('Time', ),
                    attrs=dict(description='Averaged regional salinity (depth weigthed) over depth range {zmax:d},{zmin:d}', units='m 1e-3', )
                )

                #####
                ##### Temperature budget terms
                #####
                print('Compute temperature budget terms')
                print('  temperature tendency')
                if not 'timeMonthly_avg_activeTracersTend_temperatureTend' in dsIn.keys():
                    raise KeyError('no temperature time tendency variable found')
                # This is actually temperature tendency weighted by layer thickness
                temperatureTend = dsIn.timeMonthly_avg_activeTracersTend_temperatureTend
                temperatureTend = temperatureTend.where(depthMask, drop=False).where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                temperatureTend = temperatureTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['tempTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed, i.e. d(hT)/dt) due to temperature time tendency over depth range {zmax:d},{zmin:d}', units='m C s^-1', )
                    )

                print('  frazil temperature tendency')
                if not 'timeMonthly_avg_frazilTemperatureTendency' in dsIn.keys():
                    raise KeyError('no temperature frazil tendency variable found')
                # This is actually tendency weighted by layer thickness
                temperatureTend = dsIn.timeMonthly_avg_frazilTemperatureTendency
                temperatureTend = temperatureTend.where(depthMask, drop=False).where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                temperatureTend = temperatureTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['tempFrazilTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to frazil ice over depth range {zmax:d},{zmin:d}', units='m C s^-1', )
                    )

                # The following activeTracerTendency terms are *not* weighted by layer thickness
                print('  horizontal advection (includes GM and MLE)')
                if not 'timeMonthly_avg_activeTracerHorizontalAdvectionTendency_temperatureHorizontalAdvectionTendency' in dsIn.keys():
                    raise KeyError('no temperature horizontal advection tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerHorizontalAdvectionTendency_temperatureHorizontalAdvectionTendency
                temperatureTend = (temperatureTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                temperatureTend = temperatureTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['tempHAdvTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to horizontal advection (including GM and MLE contributions) over depth range {zmax:d},{zmin:d}', units='m C s^-1', )
                    )

                print('  vertical advection')
                if not 'timeMonthly_avg_activeTracerVerticalAdvectionTendency_temperatureVerticalAdvectionTendency' in dsIn.keys():
                    raise KeyError('no temperature vertical advection tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerVerticalAdvectionTendency_temperatureVerticalAdvectionTendency
                temperatureTend = (temperatureTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                temperatureTend = temperatureTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['tempVAdvTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to vertical advection over depth range {zmax:d},{zmin:d}', units='m C s^-1', )
                    )

                print('  horizontal mixing (includes Redi)')
                if not 'timeMonthly_avg_activeTracerHorMixTendency_temperatureHorMixTendency' in dsIn.keys():
                    raise KeyError('no temperature horizontal mixing tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerHorMixTendency_temperatureHorMixTendency
                temperatureTend = (temperatureTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                temperatureTend = temperatureTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['tempHMixTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to horizontal mixing (including del2, del4, and Redi) over depth range {zmax:d},{zmin:d}', units='m C s^-1', )
                    )

                print('  nonlocal tendency')
                if not 'timeMonthly_avg_activeTracerNonLocalTendency_temperatureNonLocalTendency' in dsIn.keys():
                    raise KeyError('no temperature non local tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerNonLocalTendency_temperatureNonLocalTendency
                temperatureTend = (temperatureTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                temperatureTend = temperatureTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['tempNonLocalTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to non local mixing over depth range {zmax:d},{zmin:d}', units='m C s^-1', )
                    )

                print('  vertical mixing')
                if not 'timeMonthly_avg_activeTracerVertMixTendency_temperatureVertMixTendency' in dsIn.keys():
                    raise KeyError('no temperature vertical mixing tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerVertMixTendency_temperatureVertMixTendency
                temperatureTend = (temperatureTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                temperatureTend = temperatureTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['tempVMixTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to vertical mixing over depth range {zmax:d},{zmin:d}', units='m C s^-1', )
                    )

                print('  surface fluxes (sensible + latent + longwaveUp + longwaveDown + seaIceHeatFlux + icebergHeatFlux if present + latentHeatFusion)')
                if not 'timeMonthly_avg_activeTracerSurfaceFluxTendency_temperatureSurfaceFluxTendency' in dsIn.keys():
                    raise KeyError('no temperature surface flux tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerSurfaceFluxTendency_temperatureSurfaceFluxTendency
                temperatureTend = (temperatureTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                temperatureTend = temperatureTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['tempSurfaceFluxTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to surface fluxes (sensible + latent + longwaveUp + longwaveDown + seaIceHeatFlux + icebergHeatFlux if present + latentHeatFusion) over depth range {zmax:d},{zmin:d}', units='m C s^-1', )
                    )

                print('  surface fluxes due to sensible heat flux only')
                if not 'timeMonthly_avg_sensibleHeatFlux' in dsIn.keys():
                    raise KeyError('no sensible heat flux variable found')
                heatFlux_to_temperatureFlux = 1.0/(rho0 * cp)
                temperatureTend = dsIn.timeMonthly_avg_sensibleHeatFlux
                temperatureTend = heatFlux_to_temperatureFlux * temperatureTend
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempSensibleHeatTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to sensible heat flux', units='m C s^-1', )
                    )

                print('  surface fluxes due to latent heat flux only')
                if not 'timeMonthly_avg_latentHeatFlux' in dsIn.keys():
                    raise KeyError('no latent heat flux variable found')
                temperatureTend = dsIn.timeMonthly_avg_latentHeatFlux
                temperatureTend = heatFlux_to_temperatureFlux * temperatureTend
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempLatentHeatTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to latent heat flux', units='m C s^-1', )
                    )

                print('  surface fluxes due to net longwave heat flux only')
                if (not 'timeMonthly_avg_longWaveHeatFluxUp'   in dsIn.keys()) or \
                   (not 'timeMonthly_avg_longWaveHeatFluxDown' in dsIn.keys()):
                    raise KeyError('no longwave heat flux variable found')
                temperatureTend = dsIn.timeMonthly_avg_longWaveHeatFluxUp + \
                                  dsIn.timeMonthly_avg_longWaveHeatFluxDown
                temperatureTend = heatFlux_to_temperatureFlux * temperatureTend
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempLongWaveTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to net longwave heat flux', units='m C s^-1', )
                    )

                print('  surface fluxes due to sea ice heat flux only')
                if not 'timeMonthly_avg_seaIceHeatFlux' in dsIn.keys():
                    raise KeyError('no sea ice heat flux variable found')
                temperatureTend = dsIn.timeMonthly_avg_seaIceHeatFlux
                temperatureTend = heatFlux_to_temperatureFlux * temperatureTend
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempSeaIceHeatTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to sea ice heat flux', units='m C s^-1', )
                    )

                print('  shortwave tendency')
                if not 'timeMonthly_avg_temperatureShortWaveTendency' in dsIn.keys():
                    raise KeyError('no temperature shortwave tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_temperatureShortWaveTendency
                temperatureTend = (temperatureTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                temperatureTend = temperatureTend.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['tempShortWaveTendency'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change (depth weigthed) due to shortwave tendency over depth range {zmax:d},{zmin:d}', units='m C s^-1', )
                    )

                print('  temperature')
                temperature = dsIn.timeMonthly_avg_activeTracers_temperature
                temperature = (temperature * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                temperature = (temperature * regionArea).sum(dim='nCells') / regionAreaTot
                temperature = temperature.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['temperature'] = xr.DataArray(
                    data=temperature,
                    dims=('Time', ),
                    attrs=dict(description='Averaged regional temperature (depth weigthed) over depth range {zmax:d},{zmin:d}', units='m C', )
                    )

                dsOutMonthly['depth'] = xr.DataArray(
                    data=depthLayer,
                    dims=('Time', 'nCells', ),
                    attrs=dict(description='Averaged thickness of layer spanning depth range {zmax:d},{zmin:d} for each cell in the region', units='m', )
                    )

                dsOut.append(dsOutMonthly)

            dsOut = xr.concat(dsOut, dim='Time')
            dsOut.to_netcdf(outfile)
        else:
            print(f'  File for year = {year:04d} ({kyear} out of {len(years)} years total) already exists. Moving to the next one...')

    ###
    ### PART 3 -- Plotting
    ###
    if makePlots is True:
        dsBudgets = xr.open_mfdataset(outfiles)

        # Read in previously computed salinity budget quantities
        saltTend = dsBudgets['saltTendency']
        saltHadvTend = dsBudgets['saltHAdvTendency']
        saltVadvTend = dsBudgets['saltVAdvTendency']
        saltHmixTend = dsBudgets['saltHMixTendency']
        saltVmixTend = dsBudgets['saltVMixTendency']
        saltNonLocalTend = dsBudgets['saltNonLocalTendency']
        saltSurfaceFluxTend = dsBudgets['saltSurfaceFluxTendency']
        if 'saltFrazilTendency' in dsBudgets.keys():
            saltFrazilTend = dsBudgets['saltFrazilTendency']
            # vmix is not included in the MPAS salinityTendency term, so do not add it to tot:
            tot = saltHadvTend + saltVadvTend + saltHmixTend + saltNonLocalTend + saltSurfaceFluxTend + saltFrazilTend
        else:
            # vmix is not included in the MPAS salinityTendency term, so do not add it to tot:
            tot = saltHadvTend + saltVadvTend + saltHmixTend + saltNonLocalTend + saltSurfaceFluxTend
        # Separate contributions of seaIceSalinityFlux and SSSrestoring to saltSurfaceFluxTend
        # (note that these are not depth weigthed)
        saltSeaIceTend = dsBudgets['saltSeaIceTendency']
        if 'saltSurfaceRestoringTendency' in dsBudgets.keys():
            saltSurfaceRestoringTend = dsBudgets['saltSurfaceRestoringTendency']
        saltRes = saltTend - tot
        saltTotTend = saltTend + saltVmixTend
        salt = dsBudgets['salinity']

        # Read in previously computed temperature budget quantities
        tempTend = dsBudgets['tempTendency']
        tempFrazilTend = dsBudgets['tempFrazilTendency']
        tempHadvTend = dsBudgets['tempHAdvTendency']
        tempVadvTend = dsBudgets['tempVAdvTendency']
        tempHmixTend = dsBudgets['tempHMixTendency']
        tempVmixTend = dsBudgets['tempVMixTendency']
        tempNonLocalTend = dsBudgets['tempNonLocalTendency']
        tempSurfaceFluxTend = dsBudgets['tempSurfaceFluxTendency']
        tempSensibleHeatTend = dsBudgets['tempSensibleHeatTendency']
        tempLatentHeatTend = dsBudgets['tempLatentHeatTendency']
        tempLongWaveTend = dsBudgets['tempLongWaveTendency']
        tempSeaIceTend = dsBudgets['tempSeaIceHeatTendency']
        tempShortWaveTend = dsBudgets['tempShortWaveTendency']
        # vmix is not included in the MPAS temperatureTendency term, so do not add it to tot:
        tot = tempHadvTend + tempVadvTend + tempHmixTend + tempNonLocalTend + tempSurfaceFluxTend + tempShortWaveTend + tempFrazilTend
        tempRes = tempTend - tot
        tempSFluxRes = tempSurfaceFluxTend - (tempSensibleHeatTend + tempLatentHeatTend + tempLongWaveTend + tempSeaIceTend)
        tempTotTend = tempTend + tempVmixTend
        temp = dsBudgets['temperature']

        # Compute optional annual averages
        if plotAnnualAvg is True:
            t = dsBudgets['Time']
            datetimes = netCDF4.num2date(t, f'days since {referenceDate}', calendar=calendar)
            t_years = np.empty(np.shape(t), dtype=np.int64)
            for i, date in enumerate(datetimes.flat):
                t_years[i] = date.year
            weights = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])/daysInYear
            weights = xr.DataArray(weights, dims=('Time', ))
            t_annual = []
            saltTend_annual = []
            saltTotTend_annual = []
            saltRes_annual = []
            salt_annual = []
            saltHadvTend_annual = []
            saltVadvTend_annual = []
            saltHmixTend_annual = []
            saltVmixTend_annual = []
            saltNonLocalTend_annual = []
            saltSurfaceFluxTend_annual = []
            saltSeaIceTend_annual = []
            if 'saltSurfaceRestoringTendency' in dsBudgets.keys():
                saltSurfaceRestoringTend_annual = []
            tempTend_annual = []
            tempTotTend_annual = []
            tempRes_annual = []
            temp_annual = []
            tempHadvTend_annual = []
            tempVadvTend_annual = []
            tempHmixTend_annual = []
            tempVmixTend_annual = []
            tempNonLocalTend_annual = []
            tempSurfaceFluxTend_annual = []
            tempSeaIceTend_annual = []
            tempFrazilTend_annual = []
            tempSensibleHeatTend_annual = []
            tempLatentHeatTend_annual = []
            tempLongWaveTend_annual = []
            tempShortWaveTend_annual = []
            tempSFluxRes_annual = []
            for year in years:
                indexYears = np.where(t_years==year)[0]
                t_annual.append(t.isel(Time=indexYears).mean().values)
                saltTend_annual.append((weights * saltTend.isel(Time=indexYears)).sum().values)
                saltTotTend_annual.append((weights * saltTotTend.isel(Time=indexYears)).sum().values)
                saltRes_annual.append((weights * saltRes.isel(Time=indexYears)).sum().values)
                salt_annual.append((weights * salt.isel(Time=indexYears)).sum().values)
                saltHadvTend_annual.append((weights * saltHadvTend.isel(Time=indexYears)).sum().values)
                saltVadvTend_annual.append((weights * saltVadvTend.isel(Time=indexYears)).sum().values)
                saltHmixTend_annual.append((weights * saltHmixTend.isel(Time=indexYears)).sum().values)
                saltVmixTend_annual.append((weights * saltVmixTend.isel(Time=indexYears)).sum().values)
                saltNonLocalTend_annual.append((weights * saltNonLocalTend.isel(Time=indexYears)).sum().values)
                saltSurfaceFluxTend_annual.append((weights * saltSurfaceFluxTend.isel(Time=indexYears)).sum().values)
                saltSeaIceTend_annual.append((weights * saltSeaIceTend.isel(Time=indexYears)).sum().values)
                if 'saltSurfaceRestoringTendency' in dsBudgets.keys():
                    saltSurfaceRestoringTend_annual.append((weights * saltSurfaceRestoringTend.isel(Time=indexYears)).sum().values)
                tempTend_annual.append((weights * tempTend.isel(Time=indexYears)).sum().values)
                tempTotTend_annual.append((weights * tempTotTend.isel(Time=indexYears)).sum().values)
                tempRes_annual.append((weights * tempRes.isel(Time=indexYears)).sum().values)
                temp_annual.append((weights * temp.isel(Time=indexYears)).sum().values)
                tempHadvTend_annual.append((weights * tempHadvTend.isel(Time=indexYears)).sum().values)
                tempVadvTend_annual.append((weights * tempVadvTend.isel(Time=indexYears)).sum().values)
                tempHmixTend_annual.append((weights * tempHmixTend.isel(Time=indexYears)).sum().values)
                tempVmixTend_annual.append((weights * tempVmixTend.isel(Time=indexYears)).sum().values)
                tempNonLocalTend_annual.append((weights * tempNonLocalTend.isel(Time=indexYears)).sum().values)
                tempSurfaceFluxTend_annual.append((weights * tempSurfaceFluxTend.isel(Time=indexYears)).sum().values)
                tempSeaIceTend_annual.append((weights * tempSeaIceTend.isel(Time=indexYears)).sum().values)
                tempFrazilTend_annual.append((weights * tempFrazilTend.isel(Time=indexYears)).sum().values)
                tempSensibleHeatTend_annual.append((weights * tempSensibleHeatTend.isel(Time=indexYears)).sum().values)
                tempLatentHeatTend_annual.append((weights * tempLatentHeatTend.isel(Time=indexYears)).sum().values)
                tempLongWaveTend_annual.append((weights * tempLongWaveTend.isel(Time=indexYears)).sum().values)
                tempShortWaveTend_annual.append((weights * tempShortWaveTend.isel(Time=indexYears)).sum().values)
                tempSFluxRes_annual.append((weights * tempSFluxRes.isel(Time=indexYears)).sum().values)
        else:
            t = cftime.date2num(np.hstack(dsBudgets['Time']), f'days since {referenceDate}') # days
            depthRegion = dsBudgets['depth'].mean('nCells')
            # Factor to apply for time-integrated tracer tendency summary plots:
            fac = perSec_to_perDay / depthRegion
            weights = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
            months = np.empty(np.shape(t), dtype=np.int64)
            datetimes = netCDF4.num2date(t, f'days since {referenceDate}', calendar=calendar)
            for i, date in enumerate(datetimes.flat):
                months[i] = date.month
            monthlyMask = np.empty(np.shape(t), dtype=np.float64)
            for im in range(1, 13):
                monthlyMask[months==im] = weights[im-1]
            t = t/365 # from days to years

        # Compute long-term means
        saltTendMean = saltTend.mean().values
        saltTotTendMean = saltTotTend.mean().values
        if 'saltFrazilTend' in locals():
            saltFrazilTendMean = saltFrazilTend.mean().values
        saltHadvTendMean = saltHadvTend.mean().values
        saltVadvTendMean = saltVadvTend.mean().values
        saltAdvTendMean = (saltHadvTend+saltVadvTend).mean().values
        saltHmixTendMean = saltHmixTend.mean().values
        saltVmixTendMean = saltVmixTend.mean().values
        saltNonLocalTendMean = saltNonLocalTend.mean().values
        saltSurfaceFluxTendMean = saltSurfaceFluxTend.mean().values
        saltSeaIceTendMean = saltSeaIceTend.mean().values
        if 'saltSurfaceRestoringTend' in locals():
            saltSurfaceRestoringTendMean = saltSurfaceRestoringTend.mean().values
        saltResMean = saltRes.mean().values
        #
        tempTendMean = tempTend.mean().values
        tempTotTendMean = tempTotTend.mean().values
        tempFrazilTendMean = tempFrazilTend.mean().values
        tempHadvTendMean = tempHadvTend.mean().values
        tempVadvTendMean = tempVadvTend.mean().values
        tempAdvTendMean = (tempHadvTend+tempVadvTend).mean().values
        tempHmixTendMean = tempHmixTend.mean().values
        tempVmixTendMean = tempVmixTend.mean().values
        tempNonLocalTendMean = tempNonLocalTend.mean().values
        tempSurfaceFluxTendMean = tempSurfaceFluxTend.mean().values
        tempShortWaveTendMean = tempShortWaveTend.mean().values
        tempSeaIceTendMean = tempSeaIceTend.mean().values
        tempSensibleHeatTendMean = tempSensibleHeatTend.mean().values
        tempLatentHeatTendMean = tempLatentHeatTend.mean().values
        tempLongWaveTendMean = tempLongWaveTend.mean().values
        tempSFluxResMean = tempSFluxRes.mean().values
        tempResMean = tempRes.mean().values

        figdpi = 300

        # Plot summary of salinity budget terms
        figsize = (14, 8)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(14)
            tick.set_weight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(14)
            tick.set_weight('bold')
        ax.yaxis.get_offset_text().set_fontsize(14)
        ax.yaxis.get_offset_text().set_weight('bold')
        #
        if plotAnnualAvg is True:
            if zmax>0:
                figfile = f'{figdir}/saltBudgetSummaryAnnual_z0000-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
            else:
                figfile = f'{figdir}/saltBudgetSummaryAnnual_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'

            #ax.plot(t_annual, saltHadvTend_annual, 'r', linewidth=2, label=f'hor-adv ({saltHadvTendMean:.2e})')
            #ax.plot(t_annual, saltVadvTend_annual, 'g', linewidth=2, label=f'ver-adv ({saltVadvTendMean:.2e})')
            ax.plot(t_annual, saltHadvTend_annual+saltVadvTend_annual, 'r', linewidth=2, label=f'tot-adv ({saltAdvTendMean:.2e})')
            ax.plot(t_annual, saltVmixTend_annual, 'salmon', linewidth=2, label=f'ver-mix ({saltVmixTendMean:.2e})')
            ax.plot(t_annual, saltNonLocalTend_annual, 'c', linewidth=2, label=f'non-local ({saltNonLocalTendMean:.2e})')
            ax.plot(t_annual, saltHmixTend_annual, 'k', linewidth=2, label=f'hor-mix ({saltHmixTendMean:.2e})')
            ax.plot(t_annual, saltSurfaceFluxTend_annual, 'b', linewidth=2, label=f'sfc-flux ({saltSurfaceFluxTendMean:.2e})')
            if 'saltFrazilTend' in locals():
                ax.plot(t_annual, saltFrazilTend_annual, 'brown', linewidth=2, label=f'saltFrazilTend ({saltFrazilTendMean:.2e})')
            #ax.plot(t_annual, saltTend_annual, 'm', linewidth=2, label=f'saltTend ({saltTendMean:.2e})')
            ax.plot(t_annual, saltTotTend_annual, 'm', linewidth=2, label=f'saltTend+vmix ({saltTotTendMean:.2e})')
            ax.plot(t_annual, saltRes_annual, 'k', alpha=0.5, linewidth=1, label=f'res ({saltResMean:.2e})')
            ax.set_ylabel('psu/s', fontsize=12, fontweight='bold')
        else:
            if zmax>0:
                figfile = f'{figdir}/saltBudgetSummary_z0000-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
            else:
                figfile = f'{figdir}/saltBudgetSummary_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'

            #ax.plot(t, fac * np.cumsum(monthlyMask*saltHadvTend), 'r', linewidth=2, label=f'hor-adv ({saltHadvTendMean:.2e})')
            #ax.plot(t, fac * np.cumsum(monthlyMask*saltVadvTend), 'g', linewidth=2, label=f'ver-adv ({saltVadvTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*(saltHadvTend+saltVadvTend)), 'r', linewidth=2, label=f'tot-adv ({saltAdvTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltVmixTend), 'salmon', linewidth=2, label=f'ver-mix ({saltVmixTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltNonLocalTend), 'c', linewidth=2, label=f'non-local ({saltNonLocalTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltHmixTend), 'k', linewidth=2, label=f'hor-mix ({saltHmixTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltSurfaceFluxTend), 'b', linewidth=2, label=f'sfc-flux ({saltSurfaceFluxTendMean:.2e})')
            if 'saltFrazilTend' in locals():
                ax.plot(t, fac * np.cumsum(monthlyMask*saltFrazilTend), 'brown', linewidth=2, label=f'saltFrazilTend ({saltFrazilTendMean:.2e})')
            #ax.plot(t, fac * np.cumsum(monthlyMask*saltTend), 'm', linewidth=2, label=f'saltTend ({saltTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltTotTend), 'm', linewidth=2, label=f'saltTend+vmix ({saltTotTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltRes), 'k', alpha=0.5, linewidth=1, label=f'res ({saltResMean:.2e})')
            ax.plot(t, (salt-salt[0]) / depthRegion, color='m', linestyle=':', linewidth=1.5, label='d(salt)')
            ax.set_ylabel('psu (delta)', fontsize=12, fontweight='bold')

        ax.plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax.legend(prop=legend_properties)
        ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=0.5)
        if zmax>0:
            fig.suptitle(f'Region = {regionName}, runname = {casename}, depth range=0000-{np.abs(np.int32(zmin)):04d}', fontsize=14, fontweight='bold', y=1.025)
        else:   
            fig.suptitle(f'Region = {regionName}, runname = {casename}, depth range={np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}', fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

        # Plot components of salinity surface flux budget terms
        figsize = (14, 8)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(14)
            tick.set_weight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(14)
            tick.set_weight('bold')
        ax.yaxis.get_offset_text().set_fontsize(14)
        ax.yaxis.get_offset_text().set_weight('bold')
        #
        if plotAnnualAvg is True:
            if zmax>0:
                figfile = f'{figdir}/saltBudgetSurfaceFluxesAnnual_z0000-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
            else:
                figfile = f'{figdir}/saltBudgetSurfaceFluxesAnnual_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'

            ax.plot(t_annual, saltSurfaceFluxTend_annual, 'b', linewidth=2, label=f'sfc-flux ({saltSurfaceFluxTendMean:.2e})')
            ax.plot(t_annual, saltSeaIceTend_annual, 'c', linewidth=2, label=f'sea-ice-saltFlux ({saltSeaIceTendMean:.2e})')
            if 'saltSurfaceRestoringTend' in locals():
                ax.plot(t_annual, saltSurfaceRestoringTend_annual, 'g', linewidth=2, label=f'SSS-restoring ({saltSurfaceRestoringTendMean:.2e})')
            ax.set_ylabel('psu/s', fontsize=12, fontweight='bold')
        else:
            if zmax>0:
                figfile = f'{figdir}/saltBudgetSurfaceFluxes_z0000-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
            else:
                figfile = f'{figdir}/saltBudgetSurfaceFluxes_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'

            ax.plot(t, fac * np.cumsum(monthlyMask*saltSurfaceFluxTend), 'b', linewidth=2, label=f'sfc-flux ({saltSurfaceFluxTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltSeaIceTend), 'c', linewidth=2, label=f'sea-ice-saltFlux ({saltSeaIceTendMean:.2e})')
            if 'saltSurfaceRestoringTend' in locals():
                ax.plot(t, fac * np.cumsum(monthlyMask*saltSurfaceRestoringTend), 'g', linewidth=2, label=f'SSS-restoring ({saltSurfaceRestoringTendMean:.2e})')
            ax.set_ylabel('psu (delta)', fontsize=12, fontweight='bold')

        ax.plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax.legend(prop=legend_properties)
        ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=0.5)
        if zmax>0:
            fig.suptitle(f'Region = {regionName}, runname = {casename}, depth range=0000-{np.abs(np.int32(zmin)):04d}', fontsize=14, fontweight='bold', y=1.025)
        else:   
            fig.suptitle(f'Region = {regionName}, runname = {casename}, depth range={np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}', fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.2, height=1.2, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

        # Plot summary of temperature budget terms
        figsize = (14, 8)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(14)
            tick.set_weight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(14)
            tick.set_weight('bold')
        ax.yaxis.get_offset_text().set_fontsize(14)
        ax.yaxis.get_offset_text().set_weight('bold')
        #
        if plotAnnualAvg is True:
            if zmax>0:
                figfile = f'{figdir}/tempBudgetSummaryAnnual_z0000-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
            else:
                figfile = f'{figdir}/tempBudgetSummaryAnnual_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'

            #ax.plot(t_annual, tempHadvTend_annual, 'r', linewidth=2, label=f'hor-adv ({tempHadvTendMean:.2e})')
            #ax.plot(t_annual, tempVadvTend_annual, 'g', linewidth=2, label=f'ver-adv ({tempVadvTendMean:.2e})')
            ax.plot(t_annual, tempHadvTend_annual+tempVadvTend_annual, 'r', linewidth=2, label=f'tot-adv ({tempAdvTendMean:.2e})')
            ax.plot(t_annual, tempVmixTend_annual, 'salmon', linewidth=2, label=f'ver-mix ({tempVmixTendMean:.2e})')
            ax.plot(t_annual, tempNonLocalTend_annual, 'c', linewidth=2, label=f'non-local ({tempNonLocalTendMean:.2e})')
            ax.plot(t_annual, tempHmixTend_annual, 'k', linewidth=2, label=f'hor-mix ({tempHmixTendMean:.2e})')
            ax.plot(t_annual, tempSurfaceFluxTend_annual+tempShortWaveTend_annual, 'b', linewidth=2, \
                    label=f'sfc-flux+SW ({tempSurfaceFluxTend.mean().values+tempShortWaveTend.mean().values:.2e})')
            ax.plot(t_annual, tempFrazilTend_annual, 'brown', linewidth=2, label=f'tempFrazilTend ({tempFrazilTendMean:.2e})')
            #ax.plot(t_annual, tempTend_annual, 'm', linewidth=2, label=f'tempTend ({tempTendMean:.2e})')
            ax.plot(t_annual, tempTotTend_annual, 'm', linewidth=2, label=f'tempTend+vmix ({tempTotTendMean:.2e})')
            ax.plot(t_annual, tempRes_annual, 'k', alpha=0.5, linewidth=1, label=f'res ({tempResMean:.2e})')
            ax.set_ylabel('C/s', fontsize=12, fontweight='bold')
        else:
            if zmax>0:
                figfile = f'{figdir}/tempBudgetSummary_z0000-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
            else:
                figfile = f'{figdir}/tempBudgetSummary_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'

            #ax.plot(t, fac * np.cumsum(monthlyMask*tempHadvTend), 'r', linewidth=2, label=f'hor-adv ({tempHadvTendMean:.2e})')
            #ax.plot(t, fac * np.cumsum(monthlyMask*tempVadvTend), 'g', linewidth=2, label=f'ver-adv ({tempVadvTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*(tempHadvTend+tempVadvTend)), 'r', linewidth=2, label=f'tot-adv ({tempAdvTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempVmixTend), 'salmon', linewidth=2, label=f'ver-mix ({tempVmixTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempNonLocalTend), 'c', linewidth=2, label=f'non-local ({tempNonLocalTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempHmixTend), 'k', linewidth=2, label=f'hor-mix ({tempHmixTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*(tempSurfaceFluxTend+tempShortWaveTend)), 'b', linewidth=2, \
                    label=f'sfc-flux+SW ({tempSurfaceFluxTend.mean().values+tempShortWaveTend.mean().values:.2e})')
            #ax.plot(t, fac * np.cumsum(monthlyMask*tempSurfaceFluxTend), 'b', linewidth=2, label=f'sfc-flux ({tempSurfaceFluxTendMean:.2e})')
            #ax.plot(t, fac * np.cumsum(monthlyMask*tempShortWaveTend), 'lightblue', linewidth=2, label=f'shortwave ({tempShortWaveTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempFrazilTend), 'brown', linewidth=2, label=f'tempFrazilTend ({tempFrazilTendMean:.2e})')
            #ax.plot(t, fac * np.cumsum(monthlyMask*tempTend), 'm', linewidth=2, label=f'tempTend ({tempTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempTotTend), 'm', linewidth=2, label=f'tempTend+vmix ({tempTotTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempRes), 'k', alpha=0.5, linewidth=1, label=f'res ({tempResMean:.2e})')
            ax.plot(t, (temp-temp[0]) / depthRegion, color='m', linestyle=':', linewidth=1.5, label='d(temp)')
            ax.set_ylabel('C (delta)', fontsize=12, fontweight='bold')

        ax.plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax.legend(prop=legend_properties)
        ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=0.5)
        if zmax>0:
            fig.suptitle(f'Region = {regionName}, runname = {casename}, depth range=0000-{np.abs(np.int32(zmin)):04d}', fontsize=14, fontweight='bold', y=1.025)
        else:   
            fig.suptitle(f'Region = {regionName}, runname = {casename}, depth range={np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}', fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.2, height=1.2, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

        # Plot components of temperature surface flux budget terms
        figsize = (14, 8)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(14)
            tick.set_weight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(14)
            tick.set_weight('bold')
        ax.yaxis.get_offset_text().set_fontsize(14)
        ax.yaxis.get_offset_text().set_weight('bold')
        #
        if plotAnnualAvg is True:
            if zmax>0:
                figfile = f'{figdir}/tempBudgetSurfaceFluxesAnnual_z0000-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
            else:
                figfile = f'{figdir}/tempBudgetSurfaceFluxesAnnual_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'

            ax.plot(t_annual, tempSurfaceFluxTend_annual+tempShortWaveTend_annual, 'b', linewidth=2, \
                    label=f'sfc-flux+SW ({tempSurfaceFluxTend.mean().values+tempShortWaveTend.mean().values:.2e})')
            ax.plot(t_annual, tempSeaIceTend_annual, 'c', linewidth=2, label=f'sea-ice-tempFlux ({tempSeaIceTendMean:.2e})')
            ax.plot(t_annual, tempSensibleHeatTend_annual, 'teal', linewidth=2, label=f'sensible ({tempSensibleHeatTendMean:.2e})')
            ax.plot(t_annual, tempLatentHeatTend_annual, 'dodgerblue', linewidth=2, label=f'latent ({tempLatentHeatTendMean:.2e})')
            ax.plot(t_annual, tempLongWaveTend_annual, 'navy', linewidth=2, label=f'longwave ({tempLongWaveTendMean:.2e})')
            ax.plot(t_annual, tempSFluxRes_annual, 'b', linestyle=':', linewidth=2, label=f'res ({tempSFluxResMean:.2e})')
            ax.plot(t_annual, tempShortWaveTend_annual, 'lightblue', linewidth=2, label=f'shortwave ({tempShortWaveTendMean:.2e})')
            ax.set_ylabel('C/s', fontsize=12, fontweight='bold')
        else:
            if zmax>0:
                figfile = f'{figdir}/tempBudgetSurfaceFluxes_z0000-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
            else:
                figfile = f'{figdir}/tempBudgetSurfaceFluxes_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'

            #ax.plot(t, fac * np.cumsum(monthlyMask*tempSurfaceFluxTend), 'b', linewidth=2, label=f'sfc-flux ({tempSurfaceFluxTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*(tempSurfaceFluxTend+tempShortWaveTend)), 'b', linewidth=2, \
                    label=f'sfc-flux+SW ({tempSurfaceFluxTend.mean().values+tempShortWaveTend.mean().values:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempSeaIceTend), 'c', linewidth=2, label=f'sea-ice-tempFlux ({tempSeaIceTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempSensibleHeatTend), 'teal', linewidth=2, label=f'sensible ({tempSensibleHeatTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempLatentHeatTend), 'dodgerblue', linewidth=2, label=f'latent ({tempLatentHeatTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempLongWaveTend), 'navy', linewidth=2, label=f'longwave ({tempLongWaveTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempSFluxRes), 'b', linestyle=':', linewidth=2, label=f'res ({tempSFluxResMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*tempShortWaveTend), 'lightblue', linewidth=2, label=f'shortwave ({tempShortWaveTendMean:.2e})')
            ax.set_ylabel('C (delta)', fontsize=12, fontweight='bold')

        ax.plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax.legend(prop=legend_properties)
        ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=0.5)
        if zmax>0:
            fig.suptitle(f'Region = {regionName}, runname = {casename}, depth range=0000-{np.abs(np.int32(zmin)):04d}', fontsize=14, fontweight='bold', y=1.025)
        else:   
            fig.suptitle(f'Region = {regionName}, runname = {casename}, depth range={np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}', fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.2, height=1.2, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
