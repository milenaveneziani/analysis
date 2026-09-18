#!/usr/bin/env python
"""
Name: compute_regionalMLBudgets.py
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

from common_functions import add_inset
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
casenameFull = 'E3SMv2.1G60to10_01'
casename = 'E3SMv2.1G60to10_01'
#casenameFull = 'E3SMv2.1B60to10rA02'
#casename = 'E3SMv2.1B60to10rA02'
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
#year1 = 1950
#year2 = 1952
#year2 = 2014
year1 = 1
year2 = 50
#year2 = 386
years = range(year1, year2+1)
referenceDate = '0001-01-01'
calendar = 'noleap'
#shiftyear = 1900

makePlots = True

perSec_to_perDay = 86400.0 # 1/s to 1/day

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
    cellMask = dsMask.regionCellMasks == 1
    regionArea = areaCell.where(cellMask, drop=True)
    regionAreaTot = regionArea.sum(dim='nCells')

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
        outfile = f'{outdir}/budgetTerms_mixedLayer_{rname}_year{year:04d}.nc'
        outfiles.append(outfile)

        if not os.path.exists(outfile):
            print(f'  Compute mixed layer budget terms for year = {year:04d} ({kyear} out of {len(years)} years total)')
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

                dsOutMonthly = xr.Dataset()

                #####
                ##### Salinity budget terms
                #####
                print('Compute salinity mixed layer budget terms')
                print('  salinity tendency')
                if not 'timeMonthly_avg_activeTracersTendML_salinityTendML' in dsIn.keys():
                    raise KeyError('no ML salinity time tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracersTendML_salinityTendML
                salinityTend = salinityTend.where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['saltTendency_mixedLayer'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change in the mixed layer due to salinity time tendency', units='1e-3 s^-1', )
                    )

                print('  horizontal advection')
                if not 'timeMonthly_avg_activeTracerHorAdvectionMLTend_salinityHorAdvectionMLTend' in dsIn.keys():
                    raise KeyError('no ML salinity horizontal advection tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerHorAdvectionMLTend_salinityHorAdvectionMLTend
                salinityTend = salinityTend.where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['saltHAdvTendency_mixedLayer'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change in the mixed layer due to horizontal advection', units='1e-3 s^-1', )
                    )

                print('  vertical advection')
                if not 'timeMonthly_avg_activeTracerVertAdvectionMLTend_salinityVertAdvectionMLTend' in dsIn.keys():
                    raise KeyError('no ML salinity vertical advection tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerVertAdvectionMLTend_salinityVertAdvectionMLTend
                salinityTend = salinityTend.where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['saltVAdvTendency_mixedLayer'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change in the mixed layer due to vertical advection', units='1e-3 s^-1', )
                    )

                print('  horizontal mixing')
                if not 'timeMonthly_avg_activeTracerHorMixMLTend_salinityHorMixMLTend' in dsIn.keys():
                    raise KeyError('no ML salinity horizontal mixing tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerHorMixMLTend_salinityHorMixMLTend
                salinityTend = salinityTend.where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['saltHMixTendency_mixedLayer'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change in the mixed layer due to horizontal mixing', units='1e-3 s^-1', )
                    )

                print('  nonlocal tendency')
                if not 'timeMonthly_avg_activeTracerNonLocalMLTend_salinityNonLocalMLTend' in dsIn.keys():
                    raise KeyError('no ML salinity non local tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerNonLocalMLTend_salinityNonLocalMLTend
                salinityTend = salinityTend.where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['saltNonLocalTendency_mixedLayer'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change in the mixed layer due to non local mixing', units='1e-3 s^-1', )
                    )

                print('  vertical mixing')
                if not 'timeMonthly_avg_activeTracerVertMixMLTend_salinityVertMixMLTend' in dsIn.keys():
                    raise KeyError('no ML salinity vertical mixing tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerVertMixMLTend_salinityVertMixMLTend
                salinityTend = salinityTend.where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['saltVMixTendency_mixedLayer'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change in the mixed layer due to vertical mixing', units='1e-3 s^-1', )
                    )

                print('  forcing')
                if not 'timeMonthly_avg_activeTracerForcingMLTend_salinityForcingMLTend' in dsIn.keys():
                    raise KeyError('no ML salinity forcing tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerForcingMLTend_salinityForcingMLTend
                salinityTend = salinityTend.where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['saltForcingTendency_mixedLayer'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change in the mixed layer due to forcing', units='1e-3 s^-1', )
                        )

                #####
                ##### Temperature budget terms
                #####
                print('Compute temperature mixed layer budget terms')
                print('  temperature tendency')
                if not 'timeMonthly_avg_activeTracersTendML_temperatureTendML' in dsIn.keys():
                    raise KeyError('no ML temperature time tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracersTendML_temperatureTendML
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempTendency_mixedLayer'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change in the mixed layer due to temperature time tendency', units='m C s^-1', )
                    )

                print('  horizontal advection')
                if not 'timeMonthly_avg_activeTracerHorAdvectionMLTend_temperatureHorAdvectionMLTend' in dsIn.keys():
                    raise KeyError('no ML temperature horizontal advection tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerHorAdvectionMLTend_temperatureHorAdvectionMLTend
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempHAdvTendency_mixedLayer'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change in the mixed layer due to horizontal advection', units='C s^-1', )
                    )

                print('  vertical advection')
                if not 'timeMonthly_avg_activeTracerVertAdvectionMLTend_temperatureVertAdvectionMLTend' in dsIn.keys():
                    raise KeyError('no ML temperature vertical advection tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerVertAdvectionMLTend_temperatureVertAdvectionMLTend
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempVAdvTendency_mixedLayer'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change in the mixed layer due to vertical advection', units='C s^-1', )
                    )

                print('  horizontal mixing')
                if not 'timeMonthly_avg_activeTracerHorMixMLTend_temperatureHorMixMLTend' in dsIn.keys():
                    raise KeyError('no ML temperature horizontal mixing tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerHorMixMLTend_temperatureHorMixMLTend
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempHMixTendency_mixedLayer'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change in the mixed layer due to horizontal mixing', units='C s^-1', )
                    )

                print('  nonlocal tendency')
                if not 'timeMonthly_avg_activeTracerNonLocalMLTend_temperatureNonLocalMLTend' in dsIn.keys():
                    raise KeyError('no ML temperature non local tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerNonLocalMLTend_temperatureNonLocalMLTend
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempNonLocalTendency_mixedLayer'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change in the mixed layer due to non local mixing', units='C s^-1', )
                    )

                print('  vertical mixing')
                if not 'timeMonthly_avg_activeTracerVertMixMLTend_temperatureVertMixMLTend' in dsIn.keys():
                    raise KeyError('no ML temperature vertical mixing tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerVertMixMLTend_temperatureVertMixMLTend
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempVMixTendency_mixedLayer'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change in the mixed layer due to vertical mixing', units='C s^-1', )
                    )

                print('  forcing')
                if not 'timeMonthly_avg_activeTracerForcingMLTend_temperatureForcingMLTend' in dsIn.keys():
                    raise KeyError('no ML temperature forcing tendency variable found')
                temperatureTend = dsIn.timeMonthly_avg_activeTracerForcingMLTend_temperatureForcingMLTend
                temperatureTend = temperatureTend.where(cellMask, drop=True)
                temperatureTend = (temperatureTend * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['tempForcingTendency_mixedLayer'] = xr.DataArray(
                    data=temperatureTend,
                    dims=('Time', ),
                    attrs=dict(description='Temperature change in the mixed layer due to forcing', units='C s^-1', )
                    )

                print('  mixed layer depth')
                mld = dsIn.timeMonthly_avg_dThreshMLD
                mld = mld.where(cellMask, drop=True)
                mld = (mld * regionArea).sum(dim='nCells') / regionAreaTot
                dsOutMonthly['mld'] = xr.DataArray(
                    data=mld,
                    dims=('Time', ),
                    attrs=dict(description='Averaged regional mixed layer depth', units='m', )
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
        t = cftime.date2num(np.hstack(dsBudgets['Time']), f'days since {referenceDate}') # days

        mldRegion = dsBudgets['mld']
        # Factor to apply for time-integrated tracer tendency summary plots:
        fac = perSec_to_perDay

        weights = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        months = np.empty(np.shape(t), dtype=np.int64)
        datetimes = netCDF4.num2date(t, f'days since {referenceDate}', calendar=calendar)
        for i, date in enumerate(datetimes.flat):
            months[i] = date.month
        monthlyMask = np.empty(np.shape(t), dtype=np.float64)
        for im in range(1, 13):
            monthlyMask[months==im] = weights[im-1]

        t = t/365 # from days to years

        # Read in previously computed salinity budget quantities
        saltTend = dsBudgets['saltTendency_mixedLayer']
        saltHadvTend = dsBudgets['saltHAdvTendency_mixedLayer']
        saltVadvTend = dsBudgets['saltVAdvTendency_mixedLayer']
        saltHmixTend = dsBudgets['saltHMixTendency_mixedLayer']
        saltVmixTend = dsBudgets['saltVMixTendency_mixedLayer']
        saltNonLocalTend = dsBudgets['saltNonLocalTendency_mixedLayer']
        saltForcingTend = dsBudgets['saltForcingTendency_mixedLayer']
        # vmix is not included in the MPAS salinityTendency term, so do not add it to tot:
        tot = saltHadvTend + saltVadvTend + saltHmixTend + saltNonLocalTend + saltForcingTend
        saltRes = saltTend - tot
        saltTotTend = saltTend + saltVmixTend

        # Read in previously computed temperature budget quantities
        tempTend = dsBudgets['tempTendency_mixedLayer']
        tempHadvTend = dsBudgets['tempHAdvTendency_mixedLayer']
        tempVadvTend = dsBudgets['tempVAdvTendency_mixedLayer']
        tempHmixTend = dsBudgets['tempHMixTendency_mixedLayer']
        tempVmixTend = dsBudgets['tempVMixTendency_mixedLayer']
        tempNonLocalTend = dsBudgets['tempNonLocalTendency_mixedLayer']
        tempForcingTend = dsBudgets['tempForcingTendency_mixedLayer']
        # vmix is not included in the MPAS temperatureTendency term, so do not add it to tot:
        tot = tempHadvTend + tempVadvTend + tempHmixTend + tempNonLocalTend + tempForcingTend
        tempRes = tempTend - tot
        tempTotTend = tempTend + tempVmixTend

        # Compute long-term means
        saltTendMean = saltTend.mean().values
        saltTotTendMean = saltTotTend.mean().values
        saltHadvTendMean = saltHadvTend.mean().values
        saltVadvTendMean = saltVadvTend.mean().values
        saltAdvTendMean = (saltHadvTend+saltVadvTend).mean().values
        saltHmixTendMean = saltHmixTend.mean().values
        saltVmixTendMean = saltVmixTend.mean().values
        saltNonLocalTendMean = saltNonLocalTend.mean().values
        saltForcingTendMean = saltForcingTend.mean().values
        saltResMean = saltRes.mean().values
        #
        tempTendMean = tempTend.mean().values
        tempTotTendMean = tempTotTend.mean().values
        tempHadvTendMean = tempHadvTend.mean().values
        tempVadvTendMean = tempVadvTend.mean().values
        tempAdvTendMean = (tempHadvTend+tempVadvTend).mean().values
        tempHmixTendMean = tempHmixTend.mean().values
        tempVmixTendMean = tempVmixTend.mean().values
        tempNonLocalTendMean = tempNonLocalTend.mean().values
        tempForcingTendMean = tempForcingTend.mean().values
        tempResMean = tempRes.mean().values

        # Plot summary of salinity budget terms
        figdpi = 300
        figsize = (14, 8)
        figfile = f'{figdir}/saltBudgetML_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
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
        #print(saltHadvTend.values[0:24])
        #print(monthlyMask[0:24])
        #print(fac*monthlyMask[0:24]*saltHadvTend.values[0:24])
        #print(fac*np.cumsum(monthlyMask[0:24]*saltHadvTend.values[0:24]))
        #boh
        #ax.plot(t, fac * np.cumsum(monthlyMask*saltHadvTend), 'r', linewidth=2, label=f'hor-adv ({saltHadvTendMean:.2e})')
        #ax.plot(t, fac * np.cumsum(monthlyMask*saltVadvTend), 'g', linewidth=2, label=f'ver-adv ({saltVadvTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*(saltHadvTend+saltVadvTend)), 'r', linewidth=2, label=f'tot-adv ({saltAdvTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*saltVmixTend), 'salmon', linewidth=2, label=f'ver-mix ({saltVmixTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*saltNonLocalTend), 'c', linewidth=2, label=f'non-local ({saltNonLocalTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*saltHmixTend), 'k', linewidth=2, label=f'hor-mix ({saltHmixTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*saltForcingTend), 'b', linewidth=2, label=f'forcing ({saltForcingTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*saltTotTend), 'm', linewidth=2, label=f'tend ({saltTotTendMean:.2e})')
        #ax.plot(t, fac * np.cumsum(monthlyMask*saltTend), 'm', linewidth=2, label=f'saltTend ({saltTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*saltRes), 'k', alpha=0.5, linewidth=1, label=f'res ({saltResMean:.2e})')
        #
        ax.plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax.legend(prop=legend_properties)
        ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('psu (delta)', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=0.5)
        fig.suptitle(f'Region = {regionName}, runname = {casename}', \
                     fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

        # Plot summary of temperature budget terms
        figsize = (14, 8)
        figfile = f'{figdir}/tempBudgetML_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
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
        #ax.plot(t, fac * np.cumsum(monthlyMask*tempHadvTend), 'r', linewidth=2, label=f'hor-adv ({tempHadvTendMean:.2e})')
        #ax.plot(t, fac * np.cumsum(monthlyMask*tempVadvTend), 'g', linewidth=2, label=f'ver-adv ({tempVadvTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*(tempHadvTend+tempVadvTend)), 'r', linewidth=2, label=f'tot-adv ({tempAdvTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempVmixTend), 'salmon', linewidth=2, label=f'ver-mix ({tempVmixTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempNonLocalTend), 'c', linewidth=2, label=f'non-local ({tempNonLocalTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempHmixTend), 'k', linewidth=2, label=f'hor-mix ({tempHmixTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempForcingTend), 'b', linewidth=2, label=f'forcing ({tempForcingTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempTotTend), 'm', linewidth=2, label=f'tend ({tempTotTendMean:.2e})')
        #ax.plot(t, fac * np.cumsum(monthlyMask*tempTend), 'm', linewidth=2, label=f'tempTend ({tempTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempRes), 'k', alpha=0.5, linewidth=1, label=f'res ({tempResMean:.2e})')
        #
        ax.set_ylabel('C (delta)', fontsize=12, fontweight='bold')
        ax.plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax.legend(prop=legend_properties)
        ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=0.5)
        fig.suptitle(f'Region = {regionName}, runname = {casename}', \
                     fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.2, height=1.2, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
