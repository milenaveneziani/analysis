#!/usr/bin/env python
"""
Name: compute_regionalBudgets.py
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

from common_functions import extract_openBoundaries, add_inset, plot_xtick_format
from geometric_features import FeatureCollection, read_feature_collection


################################
def _vertical_transmission_function(z, attenuationCoefficient):
    # vertical distribution function for external surface fluxes
    transmissionCoefficient = np.exp( np.maximum(z/attenuationCoefficient, -100.0) )
    return transmissionCoefficient

def _apply_attCoef_to_externalFlux(layerThickness, attenuationCoefficient, externalFlux):

    z = layerThickness.cumsum(dim='nVertLevels')
    # Create a zero dataset with the same metadata for concatenation
    zero_z = z.isel(nVertLevels=[0]).astype(z.dtype) * 0
    # Concatenate the zero dataset at the beginning of z
    z = xr.concat([zero_z, z], dim='nVertLevels')
    nLevels = z.sizes['nVertLevels']
    transmissionCoeff = _vertical_transmission_function(-z, attenuationCoefficient)
    fractionAbsorbed = transmissionCoeff.isel(nVertLevels=range(0, nLevels-1)) - \
                       transmissionCoeff.isel(nVertLevels=range(1, nLevels))

    externalFlux = externalFlux * fractionAbsorbed

    # From MPAS-ocean code:
    #zTop = 0.0
    #transmissionCoeffTop = vertical_transmission_function(zTop, attenuationCoefficient)
    #do k = minLevelCell(iCell), maxLevelCell(iCell)
    #   zBot = zTop - layerThickness(k,iCell)
    #   transmissionCoeffBot = ocn_forcing_transmission(zBot, sfcFlxAttCoeff(iCell))
    #   fractionAbsorbed(k, iCell) = transmissionCoeffTop - transmissionCoeffBot
    #   zTop = zBot
    #   transmissionCoeffTop = transmissionCoeffBot
    #end do
    #remainingFlux = 1.0
    #do k = minLevelCell(iCell), maxLevelCell(iCell)
    #  remainingFlux = remainingFlux - transmissionCoefficient(k, iCell)
    #  surfaceThicknessFlux(iCell) * transmissionCoefficient(k, iCell) # thicknessTend
    #end do
    #if(maxLevelCell(iCell) > 0 .and. remainingFlux > 0.0_RKIND) then
    #  tend(maxLevelCell(iCell), iCell) = tend(maxLevelCell(iCell), iCell) + remainingFlux * surfaceThicknessFlux(iCell)
    #end if

    return externalFlux
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
#regionmaskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_arcticRegions.nc'
#featurefile = '/p/home/milena/mpas-region_masks/arcticRegions.geojson'
regionmaskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_amocPaper_regions.nc'
featurefile = '/p/home/milena/mpas-region_masks/amocPaper_regions.geojson'
#casenameFull = 'E3SMv2.1G60to10_01'
#casename = 'E3SMv2.1G60to10_01'
casenameFull = 'E3SMv2.1B60to10rA02'
casename = 'E3SMv2.1B60to10rA02'
modeldir = f'/p/global/milena/{casenameFull}/archive/ocn/hist'
#casenameFull = 'E3SMv2.1B60to10rA07'
#casename = 'E3SMv2.1B60to10rA07'
#modeldir = f'/p/global/apcraig/archive/{casenameFull}/ocn/hist'

#regionNames = ['all']
#regionNames = ['Irminger Sea']
#regionNames = ['Irminger Sea', 'Labrador Sea']
#regionNames = ['Arctic Ocean (no Barents/Kara Seas)', 'North Atlantic subpolar gyre', 'Irminger Sea', 'Labrador Sea', 'Greenland Sea', 'Norwegian Sea']
#regionNames = ['North Atlantic subtropical gyre']
#
regionNames = ['North Atlantic Wilbert', 'South Atlantic Wilbert']

# Choose years
#year1 = 1950
#year2 = 1952
#year2 = 2014
year1 = 1
#year2 = 10
year2 = 386
years = range(year1, year2+1)
referenceDate = '0001-01-01'
calendar = 'noleap'
#shiftyear = 1900

makePlots = True

movingAverageMonths = 1
#movingAverageMonths = 12
#movingAverageMonths = 5*12

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
# Make depth mask
maxLevelCell = dsMesh.maxLevelCell
nVertLevels = dsMesh.sizes['nVertLevels']
vertIndex = xr.DataArray.from_dict({'dims': ('nVertLevels',), 'data': np.arange(nVertLevels)})
depthMask = (vertIndex < maxLevelCell).transpose('nCells', 'nVertLevels')

## The following was originally created to properly mask points on land and 
## topography when computing the lateral fluxes. But I believe this is no
## longer necessary because 1) we only consider the open boundary edges
## (which are open ocean edges by definition) and 2) the layerThickness is
## already masked (nan) below maxLevelCell.
##
## Create land/bathymetry mask separately for the two neighboring cells of each edge
##  Landmask
##cellsOnEdge = dsMesh.cellsOnEdge # cellID of the 2 cells straddling each edge. If 0, cell is on land.
##coe0 = cellsOnEdge.isel(TWO=0).values - 1
##coe1 = cellsOnEdge.isel(TWO=1).values - 1
##landmask0 = coe0==0
##landmask1 = coe1==0
##  Topomask
##nLevels = dsMesh.dims['nVertLevels']
##maxLevelCell = dsMesh.maxLevelCell.values
##kmaxOnCells0 = maxLevelCell[coe0]
##kmaxOnCells1 = maxLevelCell[coe1]
##karray = np.array(range(nLevels))
##topomask0 = np.ones((len(landmask0), nLevels)) * karray[np.newaxis, :]
##topomask1 = np.ones((len(landmask1), nLevels)) * karray[np.newaxis, :]
##for k in range(len(kmaxOnCells0)):
##    topomask0[k, kmaxOnCells0[k]:] = 0
##for k in range(len(kmaxOnCells1)):
##    topomask1[k, kmaxOnCells1[k]:] = 0
## Save to mesh dataset
##dsMesh['cellsOnEdge0'] = (('nEdges'), coe0)
##dsMesh['cellsOnEdge1'] = (('nEdges'), coe1)
##dsMesh['landmask0'] = (('nEdges'), landmask0)
##dsMesh['landmask1'] = (('nEdges'), landmask1)
##dsMesh['topomask0'] = (('nEdges', 'nVertLevels'), topomask0)
##dsMesh['topomask1'] = (('nEdges', 'nVertLevels'), topomask1)

# Commenting this out after regions have been verified
#print('\nPlotting region masks...')
#for n in range(nRegions):
#    regionName = regionNames[n]
#    if regionName=='Arctic Ocean (no Barents/Kara Seas)':
#        rname = 'ArcticOcean_noBarentsnoKara'
#    else:
#        rname = regionName.replace(' ', '').replace('(', '').replace(')', '')
#    regionIndex = regions.index(regionName)
#    print(f'  Region: {regionName}  (rname={rname})')
#
#    #### Plot regional masks
#    lonCell = 180./np.pi * dsMesh.lonCell
#    latCell = 180./np.pi * dsMesh.latCell
#    lonEdge = 180./np.pi * dsMesh.lonEdge
#    latEdge = 180./np.pi * dsMesh.latEdge
#    dsMask = dsRegionMask.isel(nRegions=regionIndex)
#    [openBryEdges, openBrySigns, landEdges] = extract_openBoundaries(dsMask, dsMesh)
#    landEdges = np.where(landEdges)[0]
#    lonRegion = lonCell.where(dsMask.regionCellMasks==1, drop=True)
#    latRegion = latCell.where(dsMask.regionCellMasks==1, drop=True)
#    data_crs = ccrs.PlateCarree()
#    plt.figure(figsize=[20, 20], dpi=300)
#    if regionName=='Greater Arctic' or regionName=='Nordic Seas':
#        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
#        ax.set_extent([-180, 180, 50, 90], crs=data_crs)
#        gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6, draw_labels=True)
#        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 40))
#        gl.ylocator = mticker.FixedLocator(np.arange(50, 90, 5))
#    elif regionName=='North Atlantic subpolar gyre' or \
#         regionName=='North Atlantic subtropical (north of 27.2N)' or \
#         regionName=='North Atlantic subtropical gyre':
#        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
#        ax.set_extent([-100, 40, 5, 80], crs=data_crs)
#        gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6, draw_labels=True)
#        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
#        gl.ylocator = mticker.FixedLocator(np.arange(5, 85, 5))
#    elif regionName=='Atlantic tropical' or regionName=='South Atlantic subtropical gyre':
#        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
#        ax.set_extent([-65, 25, -40, 10], crs=data_crs)
#        gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6, draw_labels=True)
#        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
#        gl.ylocator = mticker.FixedLocator(np.arange(-40, 15, 5))
#    else:
#        ax = plt.axes(projection=ccrs.Robinson(central_longitude=0))
#        ax.set_extent([-180, 180, -90, 90], crs=data_crs)
#        gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6, draw_labels=True)
#        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 40))
#        gl.ylocator = mticker.FixedLocator(np.arange(-80, 90, 10))
#    gl.n_steps = 100
#    gl.right_labels = False
#    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
#    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
#    gl.rotate_labels = False
#    ax.scatter(lonEdge.values[landEdges], latEdge.values[landEdges], s=0.3, c='k', marker='*', transform=data_crs)
#    ax.scatter(lonCell, latCell, s=0.1, c='g', marker='*', transform=data_crs)
#    ax.scatter(lonRegion, latRegion, s=0.1, c='b', marker='*', transform=data_crs)
#    ax.scatter(lonEdge.isel(nEdges=openBryEdges), latEdge.isel(nEdges=openBryEdges), s=0.02, c='r', marker='*', transform=data_crs)
#    ax.set_title(f'{regionName} region mask (blue) and openbry edges (red dots)', y=1.04, fontsize=16)
#    plt.savefig(f'{figdir}/{rname}_regionMaskOpenbry.png', bbox_inches='tight')
#    plt.close()

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
        outfile = f'{outdir}/budgetTerms_{rname}_year{year:04d}.nc'
        outfiles.append(outfile)

        if not os.path.exists(outfile):
            print(f'  Compute budget terms for year = {year:04d} ({kyear} out of {len(years)} years total)')
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
                depth = dzOnCells.where(depthMask, drop=False).sum(dim='nVertLevels')

                #####
                ##### Volume budget terms
                #####
                print('Compute volume budget terms')
                # Compute net lateral fluxes
                print('  net lateral fluxes')
                if 'timeMonthly_avg_normalTransportVelocity' in dsIn.keys():
                    vel = dsIn.timeMonthly_avg_normalTransportVelocity.isel(nEdges=openBryEdges)
                elif 'timeMonthly_avg_normalVelocity' in dsIn.keys():
                    vel = dsIn.timeMonthly_avg_normalVelocity.isel(nEdges=openBryEdges)
                    if 'timeMonthly_avg_normalGMBolusVelocity' in dsIn.keys():
                        vel = vel + dsIn.timeMonthly_avg_normalGMBolusVelocity.isel(nEdges=openBryEdges)
                    if 'timeMonthly_avg_normalMLEvelocity' in dsIn.keys():
                        vel = vel + dsIn.timeMonthly_avg_normalMLEvelocity.isel(nEdges=openBryEdges)
                else:
                    raise KeyError('no normalVelocity variable found')
                #  First, get dz's from two neighboring cells for each edge
                dzOnCells0 = dsIn.timeMonthly_avg_layerThickness.isel(nCells=coe0)
                dzOnCells1 = dsIn.timeMonthly_avg_layerThickness.isel(nCells=coe1)
                #  Then, interpolate dz's onto edges
                dzOnEdges = 0.5 * (dzOnCells0 + dzOnCells1)
                dArea = dvEdge * dzOnEdges
                normalVel = vel * xr.DataArray(openBrySigns, dims='nEdges')
                lateralFlux = (normalVel * dArea).sum(dim='nVertLevels', skipna=True).sum(dim='nEdges')
                dsOutMonthly['volNetLateralFlux'] = xr.DataArray(
                    data=m3ps_to_Sv * lateralFlux,
                    dims=('Time', ),
                    attrs=dict(description='Net lateral volume transport across all open boundaries', units='Sv', )
                    )

                # Compute net vertical fluxes
                # Note: vertAleTransportTop is not routinely saved as monthly output,
                # but this should be OK for regional budgets because the net vertical
                # advection of mass over the whole water column is very small compared
                # to the net lateral advection.
                if 'timeMonthly_avg_vertAleTransportTop' in dsIn.keys():
                    print('  net vertical fluxes (using vertAleTransportTop)')
                    vel = dsIn.timeMonthly_avg_vertAleTransportTop
                    dvel = vel.isel(nVertLevelsP1=range(1, nVertLevels+1)) - vel.isel(nVertLevelsP1=range(0, nVertLevels))
                    dvel = dvel.rename({'nVertLevelsP1': 'nVertLevels'})
                    dvel = dvel.where(depthMask, drop=False).where(cellMask, drop=True)
                    verticalFlux = (dvel * regionArea).sum(dim='nVertLevels', skipna=True).sum(dim='nCells')
                    dsOutMonthly['volNetVerticalFlux'] = xr.DataArray(
                        data=m3ps_to_Sv * verticalFlux,
                        dims=('Time', ),
                        attrs=dict(description='Net vertical volume transport', units='Sv', )
                        )
                else:
                    print('  WARNING: variable vertAleTransportTop unavailable: skipping net vertical fluxes')

                # Compute net surface fluxes
                print('  surface fluxes (evaporation)')
                if not 'timeMonthly_avg_evaporationFlux' in dsIn.keys():
                    raise KeyError('no evaporation flux variable found')
                flux = dsIn.timeMonthly_avg_evaporationFlux
                if attCoef > attCoefMax:
                    flux = flux.expand_dims({'nVertLevels':nVertLevels}, axis=1).where(depthMask, drop=False)
                    flux = _apply_attCoef_to_externalFlux(dzOnCells, attCoef, flux)
                    flux = flux.sum('nVertLevels')
                flux = flux.where(cellMask, drop=True)
                evapFlux = (flux * regionArea).sum(dim='nCells')
                dsOutMonthly['evapFlux'] = xr.DataArray(
                    data=1/rho0 * m3ps_to_Sv * evapFlux,
                    dims=('Time', ),
                    attrs=dict(description='Volume change due to region integrated evaporation', units='Sv', )
                    )

                print('  surface fluxes (rain)')
                if not 'timeMonthly_avg_rainFlux' in dsIn.keys():
                    raise KeyError('no rain flux variable found')
                flux = dsIn.timeMonthly_avg_rainFlux
                if attCoef > attCoefMax:
                    flux = flux.expand_dims({'nVertLevels':nVertLevels}, axis=1).where(depthMask, drop=False)
                    flux = _apply_attCoef_to_externalFlux(dzOnCells, attCoef, flux)
                    flux = flux.sum('nVertLevels')
                flux = flux.where(cellMask, drop=True)
                rainFlux = (flux * regionArea).sum(dim='nCells')
                dsOutMonthly['rainFlux'] = xr.DataArray(
                    data=1/rho0 * m3ps_to_Sv * rainFlux,
                    dims=('Time', ),
                    attrs=dict(description='Volume change due to region integrated rain precipitation', units='Sv', )
                    )

                print('  surface fluxes (snow)')
                if not 'timeMonthly_avg_snowFlux' in dsIn.keys():
                    raise KeyError('no snow flux variable found')
                flux = dsIn.timeMonthly_avg_snowFlux
                if attCoef > attCoefMax:
                    flux = flux.expand_dims({'nVertLevels':nVertLevels}, axis=1).where(depthMask, drop=False)
                    flux = _apply_attCoef_to_externalFlux(dzOnCells, attCoef, flux)
                    flux = flux.sum('nVertLevels')
                flux = flux.where(cellMask, drop=True)
                snowFlux = (flux * regionArea).sum(dim='nCells')
                dsOutMonthly['snowFlux'] = xr.DataArray(
                    data=1/rho0 * m3ps_to_Sv * snowFlux,
                    dims=('Time', ),
                    attrs=dict(description='Volume change due to region integrated snow precipitation', units='Sv', )
                    )

                print('  surface fluxes (river runoff)')
                if not 'timeMonthly_avg_riverRunoffFlux' in dsIn.keys():
                    raise KeyError('no river runoff flux variable found')
                flux = dsIn.timeMonthly_avg_riverRunoffFlux
                if attCoefRunoff > attCoefMax:
                    flux = flux.expand_dims({'nVertLevels':nVertLevels}, axis=1).where(depthMask, drop=False)
                    flux = _apply_attCoef_to_externalFlux(dzOnCells, attCoefRunoff, flux)
                    flux = flux.sum('nVertLevels')
                flux = flux.where(cellMask, drop=True)
                riverRunoffFlux = (flux * regionArea).sum(dim='nCells')
                dsOutMonthly['riverRunoffFlux'] = xr.DataArray(
                    data=1/rho0 * m3ps_to_Sv * riverRunoffFlux,
                    dims=('Time', ),
                    attrs=dict(description='Volume change due to region integrated liquid runoff', units='Sv', )
                    )

                print('  surface fluxes (ice runoff)')
                if not 'timeMonthly_avg_iceRunoffFlux' in dsIn.keys():
                    raise KeyError('no ice runoff flux variable found')
                flux = dsIn.timeMonthly_avg_iceRunoffFlux
                flux = flux.where(cellMask, drop=True)
                iceRunoffFlux = (flux * regionArea).sum(dim='nCells')
                dsOutMonthly['iceRunoffFlux'] = xr.DataArray(
                    data=1/rho0 * m3ps_to_Sv * iceRunoffFlux,
                    dims=('Time', ),
                    attrs=dict(description='Volume change due to region integrated solid runoff', units='Sv', )
                    )

                print('  surface fluxes (sea ice freshwater)')
                if not 'timeMonthly_avg_seaIceFreshWaterFlux' in dsIn.keys():
                    raise KeyError('no sea ice freshwater flux variable found')
                flux = dsIn.timeMonthly_avg_seaIceFreshWaterFlux
                flux = flux.where(cellMask, drop=True)
                seaIceFreshWaterFlux = (flux * regionArea).sum(dim='nCells')
                dsOutMonthly['seaIceFreshWaterFlux'] = xr.DataArray(
                    data=1/rho0 * m3ps_to_Sv * seaIceFreshWaterFlux,
                    dims=('Time', ),
                    attrs=dict(description='Volume change due to region integrated sea-ice freshwater flux', units='Sv', )
                    )

                if 'timeMonthly_avg_icebergFlux' in dsIn.keys():
                    print('  surface fluxes (iceberg freshwater)')
                    flux = dsIn.timeMonthly_avg_icebergFlux
                    flux = flux.where(cellMask, drop=True)
                    icebergFlux = (flux * regionArea).sum(dim='nCells')
                    dsOutMonthly['icebergFlux'] = xr.DataArray(
                        data=1/rho0 * m3ps_to_Sv * icebergFlux,
                        dims=('Time', ),
                        attrs=dict(description='Volume change due to region integrated iceberg flux', units='Sv', )
                        )

                if 'timeMonthly_avg_landIceFlux' in dsIn.keys():
                    print('  surface fluxes (land ice freshwater)')
                    flux = dsIn.timeMonthly_avg_landIceFlux
                    flux = flux.where(cellMask, drop=True)
                    landIceFlux = (flux * regionArea).sum(dim='nCells')
                    dsOutMonthly['landIceFlux'] = xr.DataArray(
                        data=1/rho0 * m3ps_to_Sv * landIceFlux,
                        dims=('Time', ),
                        attrs=dict(description='Volume change due to region integrated land ice flux', units='Sv', )
                        )

                # Compute layer thickness tendencies
                print('  layer thickness tendency')
                if not 'timeMonthly_avg_tendLayerThickness' in dsIn.keys():
                    raise KeyError('no layer thickness tendency variable found')
                layerThickTend = dsIn.timeMonthly_avg_tendLayerThickness
                layerThickTend = layerThickTend.where(depthMask, drop=False).where(cellMask, drop=True)
                layerThickTend = (layerThickTend * regionArea).sum(dim='nVertLevels', skipna=True).sum(dim='nCells')
                dsOutMonthly['thicknessTendency'] = xr.DataArray(
                    data=m3ps_to_Sv * layerThickTend,
                    dims=('Time', ),
                    attrs=dict(description='Volume change due to total water column tendency (SSH changes)', units='Sv', )
                    )

                print('  frazil layer thickness tendency')
                if not 'timeMonthly_avg_frazilLayerThicknessTendency' in dsIn.keys():
                    raise KeyError('no frazil layer thickness tendency variable found')
                frazilThickTend = dsIn.timeMonthly_avg_frazilLayerThicknessTendency
                frazilThickTend = frazilThickTend.where(depthMask, drop=False).where(cellMask, drop=True)
                frazilThickTend = (frazilThickTend * regionArea).sum(dim='nVertLevels', skipna=True).sum(dim='nCells')
                dsOutMonthly['frazilTendency'] = xr.DataArray(
                    data=m3ps_to_Sv * frazilThickTend,
                    dims=('Time', ),
                    attrs=dict(description='Volume change due to frazil ice formation (included in thickTend)', units='Sv', )
                    )

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
                    attrs=dict(description='Salinity change (depth weigthed, i.e. d(hS)/dt) due to salinity time tendency', units='m 1e-3 s^-1', )
                    )

                # Note: unfortunately, this is not routinely added to monthly output.
                # So, considering it an option at the moment.
                if 'timeMonthly_avg_frazilSalinityTendency' in dsIn.keys():
                    print('  frazil salinity tendency')
                    # This is actually tendency weighted by layer thickness
                    salinityTend = dsIn.timeMonthly_avg_frazilSalinityTendency
                    salinityTend = salinityTend.where(depthMask, drop=False).where(cellMask, drop=True)
                    salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                    salinityTend = salinityTend.sum(dim='nVertLevels', skipna=True)
                    dsOutMonthly['saltFrazilTendency'] = xr.DataArray(
                        data=salinityTend,
                        dims=('Time', ),
                        attrs=dict(description='Salinity change (depth weigthed) due to frazil ice', units='m 1e-3 s^-1', )
                        )
                else:
                    print('  WARNING: variable frazilSalinityTendency unavailable: skipping frazil salinity tendency')

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
                    attrs=dict(description='Salinity change (depth weigthed) due to horizontal advection (including GM and MLE contributions)', units='m 1e-3 s^-1', )
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
                    attrs=dict(description='Salinity change (depth weigthed) due to vertical advection', units='m 1e-3 s^-1', )
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
                    attrs=dict(description='Salinity change (depth weigthed) due to horizontal mixing (including del2, del4, and Redi)', units='m 1e-3 s^-1', )
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
                    attrs=dict(description='Salinity change (depth weigthed) due to non local mixing', units='m 1e-3 s^-1', )
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
                    attrs=dict(description='Salinity change (depth weigthed) due to vertical mixing', units='m 1e-3 s^-1', )
                    )

                print('  surface fluxes (sea ice salinity flux and SSS retoring if present)')
                if not 'timeMonthly_avg_activeTracerSurfaceFluxTendency_salinitySurfaceFluxTendency' in dsIn.keys():
                    raise KeyError('no salinity surface flux tendency variable found')
                salinityTend = dsIn.timeMonthly_avg_activeTracerSurfaceFluxTendency_salinitySurfaceFluxTendency
                salinityTend = (salinityTend * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                salinityTend = (salinityTend * regionArea).sum(dim='nCells') / regionAreaTot
                salinityTend = salinityTend.sum(dim='nVertLevels', skipna=True)

                dsOutMonthly['saltSurfaceFluxTendency'] = xr.DataArray(
                    data=salinityTend,
                    dims=('Time', ),
                    attrs=dict(description='Salinity change (depth weigthed) due to surface fluxes (sea ice salinity flux and SSS restoring if present)', units='m 1e-3 s^-1', )
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
                    attrs=dict(description='Averaged regional salinity (depth weigthed)', units='m 1e-3', )
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
                    attrs=dict(description='Temperature change (depth weigthed, i.e. d(hT)/dt) due to temperature time tendency', units='m C s^-1', )
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
                    attrs=dict(description='Temperature change (depth weigthed) due to frazil ice', units='m C s^-1', )
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
                    attrs=dict(description='Temperature change (depth weigthed) due to horizontal advection (including GM and MLE contributions)', units='m C s^-1', )
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
                    attrs=dict(description='Temperature change (depth weigthed) due to vertical advection', units='m C s^-1', )
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
                    attrs=dict(description='Temperature change (depth weigthed) due to horizontal mixing (including del2, del4, and Redi)', units='m C s^-1', )
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
                    attrs=dict(description='Temperature change (depth weigthed) due to non local mixing', units='m C s^-1', )
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
                    attrs=dict(description='Temperature change (depth weigthed) due to vertical mixing', units='m C s^-1', )
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
                    attrs=dict(description='Temperature change (depth weigthed) due to surface fluxes (sensible + latent + longwaveUp + longwaveDown + seaIceHeatFlux + icebergHeatFlux if present + latentHeatFusion)', units='m C s^-1', )
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
                    attrs=dict(description='Temperature change (depth weigthed) due to shortwave tendency', units='m C s^-1', )
                    )

                print('  temperature')
                temperature = dsIn.timeMonthly_avg_activeTracers_temperature
                temperature = (temperature * dzOnCells).where(depthMask, drop=False).where(cellMask, drop=True)
                temperature = (temperature * regionArea).sum(dim='nCells') / regionAreaTot
                temperature = temperature.sum(dim='nVertLevels', skipna=True)
                dsOutMonthly['temperature'] = xr.DataArray(
                    data=temperature,
                    dims=('Time', ),
                    attrs=dict(description='Averaged regional temperature (depth weigthed)', units='m C', )
                    )

                dsOutMonthly['depth'] = xr.DataArray(
                    data=depth,
                    dims=('Time', 'nCells', ),
                    attrs=dict(description='Averaged depth for each cell in the region', units='m', )
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
        #t = np.hstack(dsBudgets['Time']) # days
        t = cftime.date2num(np.hstack(dsBudgets['Time']), f'days since {referenceDate}') # days
        #hours = 24*(t-np.int64(t))
        #minutes = 60*(hours-np.int64(hours))
        #seconds = 60*(minutes-np.int64(minutes))
        print(t)
        #dt = timedelta(days=int(t[0]), hours=int(hours[0]), minutes=int(minutes[0]), seconds=int(seconds[0]))
        #print(datetime(year=1, month=1, day=1) + dt - timedelta(days=1))

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
        print(monthlyMask)

        t = t/365 # from days to years

        # Read in previously computed volume budget quantities
        volNetLateralFlux = dsBudgets['volNetLateralFlux']
        evapFlux = dsBudgets['evapFlux']
        rainFlux = dsBudgets['rainFlux']
        snowFlux = dsBudgets['snowFlux']
        riverRunoffFlux = dsBudgets['riverRunoffFlux']
        iceRunoffFlux = dsBudgets['iceRunoffFlux']
        seaIceFreshWaterFlux = dsBudgets['seaIceFreshWaterFlux']
        thickTend = dsBudgets['thicknessTendency']
        frazilTend = dsBudgets['frazilTendency']
        if 'volNetVerticalFlux' in dsBudgets.keys():
            volNetVerticalFlux = dsBudgets['volNetVerticalFlux']
            tot = volNetLateralFlux + volNetVerticalFlux + evapFlux + rainFlux + snowFlux + riverRunoffFlux + iceRunoffFlux + seaIceFreshWaterFlux + frazilTend
        else:
            tot = volNetLateralFlux + evapFlux + rainFlux + snowFlux + riverRunoffFlux + iceRunoffFlux + seaIceFreshWaterFlux + frazilTend
        volRes = thickTend - tot

        # Read in previously computed salinity budget quantities
        saltTend = dsBudgets['saltTendency']
        saltHadvTend = dsBudgets['saltHAdvTendency']
        saltVadvTend = dsBudgets['saltVAdvTendency']
        saltHmixTend = dsBudgets['saltHMixTendency']
        saltVmixTend = dsBudgets['saltVMixTendency']
        saltNonLocalTend = dsBudgets['saltNonLocalTendency']
        saltSurfaceFluxTend = dsBudgets['saltSurfaceFluxTendency']
        if 'saltFrazilTend' in dsBudgets.keys():
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
        temp = dsBudgets['temperature']

        # Compute running averages
        if movingAverageMonths!=1:
            window = int(movingAverageMonths)
            volNetLateralFlux_runavg = pd.Series(volNetLateralFlux).rolling(window, center=True).mean()
            if 'volNetVerticalFlux' in locals():
                volNetVerticalFlux_runavg = pd.Series(volNetVerticalFlux).rolling(window, center=True).mean()
            evapFlux_runavg = pd.Series(evapFlux).rolling(window, center=True).mean()
            rainFlux_runavg = pd.Series(rainFlux).rolling(window, center=True).mean()
            snowFlux_runavg = pd.Series(snowFlux).rolling(window, center=True).mean()
            riverRunoffFlux_runavg = pd.Series(riverRunoffFlux).rolling(window, center=True).mean()
            iceRunoffFlux_runavg = pd.Series(iceRunoffFlux).rolling(window, center=True).mean()
            seaIceFreshWaterFlux_runavg = pd.Series(seaIceFreshWaterFlux).rolling(window, center=True).mean()
            frazilTend_runavg = pd.Series(frazilTend).rolling(window, center=True).mean()
            thickTend_runavg = pd.Series(thickTend).rolling(window, center=True).mean()
            volRes_runavg = pd.Series(volRes).rolling(window, center=True).mean()
            #
            saltTend_runavg = pd.Series(saltTend).rolling(window, center=True).mean()
            if 'saltFrazilTend' in locals():
                saltFrazilTend_runavg = pd.Series(saltFrazilTend).rolling(window, center=True).mean()
            saltHadvTend_runavg = pd.Series(saltHadvTend).rolling(window, center=True).mean()
            saltVadvTend_runavg = pd.Series(saltVadvTend).rolling(window, center=True).mean()
            saltHmixTend_runavg = pd.Series(saltHmixTend).rolling(window, center=True).mean()
            saltVmixTend_runavg = pd.Series(saltVmixTend).rolling(window, center=True).mean()
            saltNonLocalTend_runavg = pd.Series(saltNonLocalTend).rolling(window, center=True).mean()
            saltSurfaceFluxTend_runavg = pd.Series(saltSurfaceFluxTend).rolling(window, center=True).mean()
            saltSeaIceTend_runavg = pd.Series(saltSeaIceTend).rolling(window, center=True).mean()
            if 'saltSurfaceRestoringTend' in locals():
                saltSurfaceRestoringTend_runavg = pd.Series(saltSurfaceRestoringTend).rolling(window, center=True).mean()
            saltRes_runavg = pd.Series(saltRes).rolling(window, center=True).mean()
            salt_runavg = pd.Series(salt).rolling(window, center=True).mean()
            #
            tempTend_runavg = pd.Series(tempTend).rolling(window, center=True).mean()
            tempFrazilTend_runavg = pd.Series(tempFrazilTend).rolling(window, center=True).mean()
            tempHadvTend_runavg = pd.Series(tempHadvTend).rolling(window, center=True).mean()
            tempVadvTend_runavg = pd.Series(tempVadvTend).rolling(window, center=True).mean()
            tempHmixTend_runavg = pd.Series(tempHmixTend).rolling(window, center=True).mean()
            tempVmixTend_runavg = pd.Series(tempVmixTend).rolling(window, center=True).mean()
            tempNonLocalTend_runavg = pd.Series(tempNonLocalTend).rolling(window, center=True).mean()
            tempSurfaceFluxTend_runavg = pd.Series(tempSurfaceFluxTend).rolling(window, center=True).mean()
            tempSeaIceTend_runavg = pd.Series(tempSeaIceTend).rolling(window, center=True).mean()
            tempSensibleHeatTend_runavg = pd.Series(tempSensibleHeatTend).rolling(window, center=True).mean()
            tempLatentHeatTend_runavg = pd.Series(tempLatentHeatTend).rolling(window, center=True).mean()
            tempLongwaveTend_runavg = pd.Series(tempLongwaveTend).rolling(window, center=True).mean()
            tempShortwaveTend_runavg = pd.Series(tempShortwaveTend).rolling(window, center=True).mean()
            tempSFluxRes_runavg = pd.Series(tempSFluxRes).rolling(window, center=True).mean()
            tempRes_runavg = pd.Series(tempRes).rolling(window, center=True).mean()
            temp_runavg = pd.Series(temp).rolling(window, center=True).mean()

        # Compute long-term means
        volNetLateralFluxMean = volNetLateralFlux.mean().values
        if 'volNetVerticalFlux' in locals():
            volNetVerticalFluxMean = volNetVerticalFlux.mean().values
        evapFluxMean = evapFlux.mean().values
        rainFluxMean = rainFlux.mean().values
        snowFluxMean = snowFlux.mean().values
        empMean = (evapFlux + rainFlux + snowFlux).mean().values
        riverRunoffFluxMean = riverRunoffFlux.mean().values
        iceRunoffFluxMean = iceRunoffFlux.mean().values
        runoffMean = (riverRunoffFlux + iceRunoffFlux).mean().values
        seaIceFreshWaterFluxMean = seaIceFreshWaterFlux.mean().values
        frazilTendMean = frazilTend.mean().values
        thickTendMean = thickTend.mean().values
        volResMean = volRes.mean().values
        #
        saltTendMean = saltTend.mean().values
        if 'saltFrazilTend' in locals():
            saltFrazilTendMean = saltFrazilTend.mean().values
        saltHadvTendMean = saltHadvTend.mean().values
        saltVadvTendMean = saltVadvTend.mean().values
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
        tempFrazilTendMean = tempFrazilTend.mean().values
        tempHadvTendMean = tempHadvTend.mean().values
        tempVadvTendMean = tempVadvTend.mean().values
        tempHmixTendMean = tempHmixTend.mean().values
        tempVmixTendMean = tempVmixTend.mean().values
        tempNonLocalTendMean = tempNonLocalTend.mean().values
        tempSurfaceFluxTendMean = tempSurfaceFluxTend.mean().values
        tempSeaIceTendMean = tempSeaIceTend.mean().values
        tempSensibleHeatTendMean = tempSensibleHeatTend.mean().values
        tempLatentHeatTendMean = tempLatentHeatTend.mean().values
        tempLongWaveTendMean = tempLongWaveTend.mean().values
        tempShortWaveTendMean = tempShortWaveTend.mean().values
        tempSFluxResMean = tempSFluxRes.mean().values
        tempResMean = tempRes.mean().values

        # Plot single terms of volume budget
        figdpi = 300
        figsize = (16, 16)
        figfile = f'{figdir}/volBudget_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
        fig, ax = plt.subplots(6, 2, figsize=figsize)
        ax[0, 0].plot(t, volNetLateralFlux, 'k', alpha=0.5, linewidth=1.5)
        if 'volNetVerticalFlux' in locals():
            ax[0, 1].plot(t, volNetVerticalFlux, 'k', alpha=0.5, linewidth=1.5)
            ax[0, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
            ax[0, 1].autoscale(enable=True, axis='x', tight=True)
            ax[0, 1].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
            ax[0, 1].set_title(f'mean={volNetVerticalFluxMean:.2e}', fontsize=16, fontweight='bold')
            ax[0, 1].set_ylabel('Vertical flux (Sv)', fontsize=12, fontweight='bold')
        ax[1, 0].plot(t, seaIceFreshWaterFlux, 'k', alpha=0.5, linewidth=1.5)
        ax[1, 1].plot(t, evapFlux, 'k', alpha=0.5, linewidth=1.5)
        ax[2, 0].plot(t, rainFlux, 'k', alpha=0.5, linewidth=1.5)
        ax[2, 1].plot(t, snowFlux, 'k', alpha=0.5, linewidth=1.5)
        ax[3, 0].plot(t, riverRunoffFlux, 'k', alpha=0.5, linewidth=1.5)
        ax[3, 1].plot(t, iceRunoffFlux, 'k', alpha=0.5, linewidth=1.5)
        ax[4, 0].plot(t, frazilTend, 'k', alpha=0.5, linewidth=1.5)
        ax[4, 1].plot(t, thickTend, 'k', alpha=0.5, linewidth=1.5)
        ax[5, 0].plot(t, volRes, 'k', alpha=0.5, linewidth=1.5)
        if movingAverageMonths!=1:
            ax[0, 0].plot(t, volNetLateralFlux_runavg, 'k', linewidth=3)
            if 'volNetVerticalFlux' in locals():
                ax[0, 1].plot(t, volNetVerticalFlux_runavg, 'k', linewidth=3)
                ax[0, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
                ax[0, 1].autoscale(enable=True, axis='x', tight=True)
                ax[0, 1].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
                ax[0, 1].set_title(f'mean={volNetVerticalFluxMean:.2e}', fontsize=16, fontweight='bold')
                ax[0, 1].set_ylabel('Vertical flux (Sv)', fontsize=12, fontweight='bold')
            ax[1, 0].plot(t, seaIceFreshWaterFlux_runavg, 'k', linewidth=3)
            ax[1, 1].plot(t, evapFlux_runavg, 'k', linewidth=3)
            ax[2, 0].plot(t, rainFlux_runavg, 'k', linewidth=3)
            ax[2, 1].plot(t, snowFlux_runavg, 'k', linewidth=3)
            ax[3, 0].plot(t, riverRunoffFlux_runavg, 'k', linewidth=3)
            ax[3, 1].plot(t, iceRunoffFlux_runavg, 'k', linewidth=3)
            ax[4, 0].plot(t, frazilTend_runavg, 'k', linewidth=3)
            ax[4, 1].plot(t, thickTend_runavg, 'k', linewidth=3)
            ax[5, 0].plot(t, volRes_runavg, 'k', linewidth=3)
     
        ax[0, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax[1, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax[1, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax[2, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax[2, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax[3, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax[3, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax[4, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax[4, 1].plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax[5, 0].plot(t, np.zeros_like(t), 'k', linewidth=0.8)

        ax[0, 0].autoscale(enable=True, axis='x', tight=True)
        ax[1, 0].autoscale(enable=True, axis='x', tight=True)
        ax[1, 1].autoscale(enable=True, axis='x', tight=True)
        ax[2, 0].autoscale(enable=True, axis='x', tight=True)
        ax[2, 1].autoscale(enable=True, axis='x', tight=True)
        ax[3, 0].autoscale(enable=True, axis='x', tight=True)
        ax[3, 1].autoscale(enable=True, axis='x', tight=True)
        ax[4, 0].autoscale(enable=True, axis='x', tight=True)
        ax[4, 1].autoscale(enable=True, axis='x', tight=True)
        ax[5, 0].autoscale(enable=True, axis='x', tight=True)

        ax[0, 0].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax[1, 0].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax[1, 1].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax[2, 0].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax[2, 1].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax[3, 0].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax[3, 1].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax[4, 0].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax[4, 1].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax[5, 0].grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)

        ax[0, 0].set_title(f'mean={volNetLateralFluxMean:.2e}', fontsize=16, fontweight='bold')
        ax[1, 0].set_title(f'mean={seaIceFreshWaterFluxMean:.2e}', fontsize=16, fontweight='bold')
        ax[1, 1].set_title(f'mean={evapFluxMean:.2e}', fontsize=16, fontweight='bold')
        ax[2, 0].set_title(f'mean={rainFluxMean:.2e}', fontsize=16, fontweight='bold')
        ax[2, 1].set_title(f'mean={snowFluxMean:.2e}', fontsize=16, fontweight='bold')
        ax[3, 0].set_title(f'mean={riverRunoffFluxMean:.2e}', fontsize=16, fontweight='bold')
        ax[3, 1].set_title(f'mean={iceRunoffFluxMean:.2e}', fontsize=16, fontweight='bold')
        ax[4, 0].set_title(f'mean={frazilTendMean:.2e}', fontsize=14, fontweight='bold')
        ax[4, 1].set_title(f'mean={thickTendMean:.2e}', fontsize=16, fontweight='bold')
        ax[5, 0].set_title(f'mean={volResMean:.2e}', fontsize=16, fontweight='bold')

        ax[4, 1].set_xlabel('Time (Days)', fontsize=12, fontweight='bold')
        ax[5, 0].set_xlabel('Time (Days)', fontsize=12, fontweight='bold')

        ax[0, 0].set_ylabel('Lateral flux (Sv)', fontsize=12, fontweight='bold')
        ax[1, 0].set_ylabel('Sea ice FW flux (Sv)', fontsize=12, fontweight='bold')
        ax[1, 1].set_ylabel('Evap flux (Sv)', fontsize=12, fontweight='bold')
        ax[2, 0].set_ylabel('Rain flux (Sv)', fontsize=12, fontweight='bold')
        ax[2, 1].set_ylabel('Snow flux (Sv)', fontsize=12, fontweight='bold')
        ax[3, 0].set_ylabel('River runoff flux (Sv)', fontsize=12, fontweight='bold')
        ax[3, 1].set_ylabel('Ice runoff flux (Sv)', fontsize=12, fontweight='bold')
        ax[4, 0].set_ylabel('Frazil thicTend (Sv)', fontsize=12, fontweight='bold')
        ax[4, 1].set_ylabel('Layer thickTend (Sv)', fontsize=12, fontweight='bold')
        ax[5, 0].set_ylabel('Res = LHS - sumRHSTerms', fontsize=12, fontweight='bold')

        fig.tight_layout(pad=0.5)
        fig.suptitle(f'Region = {regionName}, runname = {casename}', \
                     fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

        # Plot summary of volume budget terms
        figsize = (14, 8)
        figfile = f'{figdir}/volBudgetSummary_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
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
        if movingAverageMonths==1:
            emp = evapFlux + rainFlux + snowFlux
            runoff = riverRunoffFlux + iceRunoffFlux
            ax.plot(t, volNetLateralFlux, 'r', linewidth=2, label=f'netLateral ({volNetLateralFluxMean:.2e} Sv)')
            if 'volNetVerticalFlux' in locals():
                ax.plot(t, volNetVerticalFlux, 'g', linewidth=2, label=f'netVertical ({volNetVerticalFluxMean:.2e}) Sv')
            ax.plot(t, emp, 'c', linewidth=2, label=f'E-P ({empMean:.2e} Sv)')
            ax.plot(t, runoff, 'salmon', linewidth=2, label=f'runoff ({runoffMean:.2e} Sv)')
            ax.plot(t, seaIceFreshWaterFlux, 'b', linewidth=2, label=f'seaiceFW ({seaIceFreshWaterFluxMean:.2e} Sv)')
            ax.plot(t, thickTend, 'm', linewidth=2, label=f'thickTend ({thickTendMean:.2e} Sv)')
            ax.plot(t, volRes, 'k', alpha=0.5, linewidth=1, label=f'res ({volResMean:.2e} Sv)')
        else:
            emp = evapFlux_runavg + rainFlux_runavg + snowFlux_runavg
            runoff = riverRunoffFlux_runavg + iceRunoffFlux_runavg
            ax.plot(t, volNetLateralFlux_runavg, 'r', linewidth=2, label=f'netLateral ({volNetLateralFluxMean:.2e} Sv)')
            if 'volNetVerticalFlux' in locals():
                ax.plot(t, volNetVerticalFlux_runavg, 'g', linewidth=2, label=f'netVertical ({volNetVerticalFluxMean:.2e}) Sv')
            ax.plot(t, emp, 'c', linewidth=2, label=f'E-P ({empMean:.2e} Sv)')
            ax.plot(t, runoff, 'salmon', linewidth=2, label=f'runoff ({runoffMean:.2e} Sv)')
            ax.plot(t, seaIceFreshWaterFlux_runavg, 'b', linewidth=2, label=f'seaiceFW ({seaIceFreshWaterFluxMean:.2e} Sv)')
            ax.plot(t, thickTend_runavg, 'm', linewidth=2, label=f'thickTend ({thickTendMean:.2e} Sv)')
            ax.plot(t, volRes_runavg, 'k', alpha=0.5, linewidth=1, label=f'res ({volResMean:.2e} Sv)')
            ax.set_title(f'{int(movingAverageMonths/12)}-year running averages', fontsize=16, fontweight='bold')
        ax.plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax.legend(prop=legend_properties)
        #plot_xtick_format('gregorian', np.min(t), np.max(t), maxXTicks=20)
        ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sv', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=0.5)
        fig.suptitle(f'Region = {regionName}, runname = {casename}', \
                     fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

        # Plot summary of salinity budget terms
        figsize = (14, 8)
        figfile = f'{figdir}/saltBudgetSummary_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
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
        if movingAverageMonths==1:
            ax.plot(t, fac * np.cumsum(monthlyMask*saltHadvTend), 'r', linewidth=2, label=f'hor-adv ({saltHadvTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltVadvTend), 'g', linewidth=2, label=f'ver-adv ({saltVadvTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltVmixTend), 'salmon', linewidth=2, label=f'ver-mix ({saltVmixTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltNonLocalTend), 'c', linewidth=2, label=f'non-local ({saltNonLocalTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltHmixTend), 'k', linewidth=2, label=f'hor-mix ({saltHmixTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltSurfaceFluxTend), 'b', linewidth=2, label=f'sfc-flux ({saltSurfaceFluxTendMean:.2e})')
            if 'saltFrazilTend' in locals():
                ax.plot(t, fac * np.cumsum(monthlyMask*saltFrazilTend), 'brown', linewidth=2, label=f'saltFrazilTend ({saltFrazilTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltTend), 'm', linewidth=2, label=f'saltTend ({saltTendMean:.2e})')
            ax.plot(t, fac * np.cumsum(monthlyMask*saltRes), 'k', alpha=0.5, linewidth=1, label=f'res ({saltResMean:.2e})')
            ax.plot(t, (salt-salt[0]) / depthRegion, color='m', linestyle=':', linewidth=1.5, label='d(salt)')
        else:
            ax.plot(t, fac*saltHadvTend_runavg, 'r', linewidth=2, label=f'hor-adv ({saltHadvTendMean:.2e})')
            ax.plot(t, fac*saltVadvTend_runavg, 'g', linewidth=2, label=f'ver-adv ({saltVadvTendMean:.2e})')
            ax.plot(t, fac*saltVmixTend_runavg, 'salmon', linewidth=2, label=f'ver-mix ({saltVmixTendMean:.2e})')
            ax.plot(t, fac*saltNonLocalTend_runavg, 'c', linewidth=2, label=f'non-local ({saltNonLocalTendMean:.2e})')
            ax.plot(t, fac*saltHmixTend_runavg, 'k', linewidth=2, label=f'hor-mix ({saltHmixTendMean:.2e})')
            ax.plot(t, fac*saltSurfaceFluxTend_runavg, 'b', linewidth=2, label=f'sfc-flux ({saltSurfaceFluxTendMean:.2e})')
            if 'saltFrazilTend' in locals():
                ax.plot(t, fac*saltFrazilTend_runavg, 'brown', linewidth=2, label=f'saltFrazilTend ({saltFrazilTendMean:.2e})')
            ax.plot(t, fac*saltTend_runavg, 'm', linewidth=2, label=f'saltTend ({saltTendMean:.2e})')
            ax.plot(t, fac*saltRes_runavg, 'k', alpha=0.5, linewidth=1, label=f'res ({saltResMean:.2e})')
            ax.plot(t, (salt_runavg-salt_runavg[0]) / depthRegion, color='m', linestyle=':', linewidth=1.5, label='d(salt)')
            ax.set_title(f'{int(movingAverageMonths/12)}-year running averages', fontsize=16, fontweight='bold')
            ax.set_ylabel('psu (delta)', fontsize=12, fontweight='bold')
        ax.plot(t, np.zeros_like(t), 'k', linewidth=0.8)
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.grid(color='k', linestyle=':', linewidth=0.5, alpha=0.75)
        ax.legend(prop=legend_properties)
        #plot_xtick_format('gregorian', np.min(t), np.max(t), maxXTicks=20)
        ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
        fig.tight_layout(pad=0.5)
        fig.suptitle(f'Region = {regionName}, runname = {casename}', \
                     fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.5, height=1.5, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

        # Plot components of salinity surface flux budget terms
        figsize = (14, 8)
        figfile = f'{figdir}/saltBudgetSurfaceFluxes_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
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
        fig.suptitle(f'Region = {regionName}, runname = {casename}', \
                     fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.2, height=1.2, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

        # Plot summary of temperature budget terms
        figsize = (14, 8)
        figfile = f'{figdir}/tempBudgetSummary_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
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
        ax.plot(t, fac * np.cumsum(monthlyMask*tempHadvTend), 'r', linewidth=2, label=f'hor-adv ({tempHadvTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempVadvTend), 'g', linewidth=2, label=f'ver-adv ({tempVadvTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempVmixTend), 'salmon', linewidth=2, label=f'ver-mix ({tempVmixTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempNonLocalTend), 'c', linewidth=2, label=f'non-local ({tempNonLocalTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempHmixTend), 'k', linewidth=2, label=f'hor-mix ({tempHmixTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempSurfaceFluxTend), 'b', linewidth=2, label=f'sfc-flux ({tempSurfaceFluxTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempShortWaveTend), 'lightblue', linewidth=2, label=f'shortwave ({tempShortWaveTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempFrazilTend), 'brown', linewidth=2, label=f'tempFrazilTend ({tempFrazilTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempTend), 'm', linewidth=2, label=f'tempTend ({tempTendMean:.2e})')
        ax.plot(t, fac * np.cumsum(monthlyMask*tempRes), 'k', alpha=0.5, linewidth=1, label=f'res ({tempResMean:.2e})')
        ax.plot(t, (temp-temp[0]) / depthRegion, color='m', linestyle=':', linewidth=1.5, label='d(temp)')
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

        # Plot components of temperature surface flux budget terms
        figsize = (14, 8)
        figfile = f'{figdir}/tempBudgetSurfaceFluxes_{rname}_{casename}_years{year1:04d}-{year2:04d}.png'
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
        ax.plot(t, fac * np.cumsum(monthlyMask*tempSurfaceFluxTend), 'b', linewidth=2, label=f'sfc-flux ({tempSurfaceFluxTendMean:.2e})')
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
        fig.suptitle(f'Region = {regionName}, runname = {casename}', \
                     fontsize=14, fontweight='bold', y=1.025)
        add_inset(fig, fc, width=1.2, height=1.2, xbuffer=0.5, ybuffer=-1.2)
        fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
