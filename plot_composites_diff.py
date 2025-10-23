from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import subprocess
from subprocess import call
import xarray as xr
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import cmocean

from make_plots import make_scatter_plot, make_streamline_plot, make_contourf_plot, make_mosaic_descriptor, make_mosaic_plot


startYear = [1950]
endYear = [2014]
#startYear = [21, 245]
#endYear = [140, 386]
#startYear = [1]
#endYear = [140]
#startYear = [141]
#endYear = [386]

# Settings for nersc
meshFile = f'/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runName = 'E3SM-Arcticv2.1_historical'
#runName = 'E3SM-Arcticv2.1_historical0151'
##runName = 'E3SMv2.1B60to10rA02'
# Relevant for streamlines case
remap = 'cmip6_720x1440_aave.20240401'
remapFile = f'/global/cfs/cdirs/m1199/diagnostics/maps/map_ARRM10to60E2r1_to_{remap}.nc'

# Settings for erdc.hpc.mil
#meshFile = f'/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runName = 'E3SMv2.1B60to10rA02'
# Relevant for streamlines case
#remap = 'cmip6_720x1440_aave.20240401'
#remapFile = f'/p/home/milena/diagnostics/maps/map_ARRM10to60E2r1_to_{remap}.nc'


# variable name with respect to which composites were computed
varRef = 'maxMLD'
#varRef = 'iceArea'

indir0 = f'./composites_{varRef}based_data/{runName}'
figdir0 = f'./composites_{varRef}based/{runName}'
# For 1950-control
#indir = f'Years{startYear[0]}-{endYear[0]}'
#figdir = f'Years{startYear[0]}-{endYear[0]}'
#for iy in range(1, np.size(startYear)):
#    indir = f'{indir}_{startYear[iy]}-{endYear[iy]}'
#    figdir = f'{figdir}_{startYear[iy]}-{endYear[iy]}'
#indir = f'{indir0}/{indir}'
#figdir = f'{figdir0}/{figdir}'
##indir = f'{indir0}/Years1-386_combiningYears1-140andYears141-386'
##figdir = f'{figdir0}/Years1-386_combiningYears1-140andYears141-386'
# For historical ensemble
indir = indir0
figdir = figdir0
if not os.path.isdir(figdir):
    os.makedirs(figdir)

regionGroup = 'Arctic Regions'
groupName = regionGroup[0].lower() + regionGroup[1:].replace(' ', '')
regions = ['Greenland Sea']
#regions = ['Norwegian Sea']

climoMonths = 'JFMA' # should be consistent with composites calculation

# Critical value of t-test (from tables, depending on the max level of the p-value
# (confidence level) and on the number of independent data)
tcritical = 2.011 # for alpha=0.05 and nind=49-1
#tcritical = 2.407 # for alpha=0.02 and nind=49-1
#tcritical = 2.682 # for alpha=0.01 and nind=49-1
#conf_level = 1.96 # 95% confidence level

colorIndices = [0, 14, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]
lon0 = -50.0
lon1 = 50.0
dlon = 10.0
lat0 = 60.0
lat1 = 80.0
dlat = 4.0

# Choose either variables in timeSeriesStatsMonthly
# or variables in timeSeriesStatsMonthlyMax (2d only) or
# ice variables (2d only)
#
#   Ocean variables
modelComp = 'ocn'
modelName = 'mpaso'
mpasFile = 'timeSeriesStatsMonthly'
variables = [
#             {'name': 'barotropicStreamfunction',
#              'title': 'Barotropic streamfunction',
#              'units': 'Sv',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'barotropicStreamfunction',
#              'clevels': [-3.6, -3.0, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.balance},
##             #{'name': 'velSpeed',
##             # 'title': 'Velocity magnitude',
##             # 'units': 'cm/s',
##             # 'factor': 1e2,
##             # 'isvar3d': True,
##             # 'mpas': None,
##             # 'clevels': [0.5, 1, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40],
##             # 'cIndices': colorIndices,
##             # 'colormap': cmocean.cm.balance},
#             {'name': 'windStressSpeed',
#              'title': 'Wind stress magnitude',
#              'units': 'N/m$^2$',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'windStressSpeed',
#              'clevels': [-0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.balance},
##             {'name': 'spiciness',
##              'title': 'Spiciness0',
##              'units': '',
##              'factor': 1,
##              'isvar3d': True,
##              'mpas': 'spiciness0',
##              'clevels': [-0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75],
##              'cIndices': colorIndices,
##              'colormap': cmocean.cm.balance},
#             {'name': 'activeTracers_temperature',
#              'title': 'Potential Temperature',
#              'units': '$^\circ$C',
#              'factor': 1,
#              'isvar3d': True,
#              'mpas': 'timeMonthly_avg_activeTracers_temperature',
#              'clevels': [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2., 2.5, 3.],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.balance},
#             {'name': 'activeTracers_salinity',
#              'title': 'Salinity',
#              'units': 'psu',
#              'factor': 1,
#              'isvar3d': True,
#              'mpas': 'timeMonthly_avg_activeTracers_salinity',
#              'clevels': [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.balance},
#             {'name': 'surfaceBuoyancyForcing',
#              'title': 'Surface buoyancy flux',
#              'units': '10$^{-8}$ m$^2$ s$^{-3}$',
#              'factor': 1e8,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_surfaceBuoyancyForcing',
#              'clevels': [-2.4, -2, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2, 2.4],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.balance},
##             {'name': 'sensibleHeatFlux',
##              'title': 'Sensible Heat Flux',
##              'units': 'W/m$^2$',
##              'factor': 1,
##              'isvar3d': False,
##              'mpas': 'timeMonthly_avg_sensibleHeatFlux',
##              'clevels': [-120, -100, -80, -60, -40, -20, 0.0, 20, 40, 60, 80, 100, 120], # winter months
##              #'clevels': [-60, -50, -40, -30, -20, -10, 0.0, 10, 20, 30, 40, 50, 60], # summer months
##              'cIndices': colorIndices,
##              'colormap': cmocean.cm.balance},
##             {'name': 'latentHeatFlux',
##              'title': 'Latent Heat Flux',
##              'units': 'W/m$^2$',
##              'factor': 1,
##              'isvar3d': False,
##              'mpas': 'timeMonthly_avg_latentHeatFlux',
##              'clevels': [-120, -100, -80, -60, -40, -20, 0.0, 20, 40, 60, 80, 100, 120], # winter months
##              #'clevels': [-60, -50, -40, -30, -20, -10, 0.0, 10, 20, 30, 40, 50, 60], # summer months
##              'cIndices': colorIndices,
##              'colormap': cmocean.cm.balance},
#             {'name': 'totalHeatFlux',
#              'title': 'Total Heat Flux',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'totalHeatFlux',
#              'clevels': [-180, -150, -120, -90, -60, -30, 0.0, 30, 60, 90, 120, 150, 180], # winter months
#              #'clevels': [-60, -50, -40, -30, -20, -10, 0.0, 10, 20, 30, 40, 50, 60], # summer months
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.balance},
             {'name': 'seaIceFreshWaterFlux',
              'title': 'Sea ice FW flux',
              'units': '10$^{-6}$ kg m$^-2$ s$^-1$',
              'factor': 1e6,
              'isvar3d': False,
              'mpas': 'timeMonthly_avg_seaIceFreshWaterFlux',
              'clevels': [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150],
              'cIndices': colorIndices,
              'colormap': cmocean.cm.balance}
             ]
#
#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'title': 'Maximum MLD',
#              'units': 'm',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthlyMax_max_dThreshMLD',
#              #'clevels': [-120, -100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, 120], # summer months
#              'clevels': [-1200, -1000, -800, -600, -400, -200, 0, 200, 400, 600, 800, 1000, 1200], # winter months
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.balance},
#            ]
#   Sea ice variables
#modelComp = 'ice'
#modelName = 'mpassi'
#mpasFile = 'timeSeriesStatsMonthly'
#variables = [
#             {'name': 'iceArea',
#              'title': 'Sea Ice Concentration',
#              'units': '%',
#              'factor': 100,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_iceAreaCell',
#              'clevels': [-90, -75, -60, -45, -30, -15, 0.0, 15, 30, 45, 60, 75, 90],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.balance},
#             {'name': 'iceVolume',
#              'title': 'Sea Ice Thickness',
#              'units': 'm',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_iceVolumeCell',
#              'clevels': [-1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.balance},
#             {'name': 'iceSpeed',
#              'title': 'Sea Ice speed',
#              'units': 'm/s',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'iceSpeed',
#              'clevels': [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.balance},
#            ]
clevelsrel = [-60, -50, -40, -30, -20, -10, 0.0, 10, 20, 30, 40, 50, 60]

plotDepthAvg = False
# zmins/zmaxs [m] (relevant for 3d variables and if plotDepthAvg = True)
#zmins = [-50., -600., -8000.]
#zmaxs = [0., -100., 0.]
zmins = [-50.]
zmaxs = [0.]
#zmins = [-50., -600.]
#zmaxs = [0., -100.]
#zmins = [-600.]
#zmaxs = [0.]
# z levels [m] (relevant for 3d variables and if plotDepthAvg = False)
dlevels = [50.]
#dlevels = [50., 100.]
#dlevels = [0., 50., 100.]
#dlevels = [300.]

# Info about MPAS mesh
dsMesh = xr.open_dataset(meshFile)
lonCell = dsMesh.lonCell.values
latCell = dsMesh.latCell.values
lonCell = 180/np.pi*lonCell
latCell = 180/np.pi*latCell
lonVertex = dsMesh.lonVertex.values
latVertex = dsMesh.latVertex.values
lonVertex = 180/np.pi*lonVertex
latVertex = 180/np.pi*latVertex
# Find model levels for each depth level (relevant if plotDepthAvg = False)
z = dsMesh.refBottomDepth
zlevels = np.zeros(np.shape(dlevels), dtype=np.int64)
for id in range(len(dlevels)):
    dz = np.abs(z.values-dlevels[id])
    zlevels[id] = np.argmin(dz)
# Make depth mask (also relevant if plotDepthAvg = False, because depth averaged fields are already masked)
maxLevelCell = dsMesh.maxLevelCell
nVertLevels = dsMesh.sizes['nVertLevels']
vertIndex = xr.DataArray.from_dict({'dims': ('nVertLevels',), 'data': np.arange(nVertLevels)})
depthMask = (vertIndex < maxLevelCell).transpose('nCells', 'nVertLevels')

# For mosaic plots
# restart files are missing this attribute that is needed for mosaic,
# so for now adding this manually:
dsMesh.attrs['is_periodic'] = 'NO'
mosaic_descriptor = make_mosaic_descriptor(dsMesh, 'NorthPolarStereo')

# Plot monthly climatologies associated with previously computed composites
for regionName in regions:
    print(f'\nPlot composites based on {varRef} for region: {regionName}')
    regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')

    for im in range(1, 13):
    #for im in range(1, 5): # winter months
    #for im in range(5, 13): # summer months
    #for im in range(1, 2):
        print(f'\n  Month: {im}')
        for var in variables:
            varname = var['name']
            print(f'    var: {varname}')
            varmpasname = var['mpas']
            isvar3d = var['isvar3d']
            varfactor = var['factor']
            varunits = var['units']
            vartitle = var['title']
            colormap = var['colormap']
            colorIndices = var['cIndices']
            clevels = var['clevels']

            if varname=='iceSpeed' or varname=='barotropicStreamfunction':
                x = lonVertex
                y = latVertex
            else:
                x = lonCell
                y = latCell

            if varname=='velSpeed':
                # Regrid velocityZonal and velocityMeridional (if necessary)
                if plotDepthAvg:
                    # Compute the depth average first and then regrid
                    for iz in range(len(zmins)):
                        zmin = zmins[iz]
                        zmax = zmaxs[iz]

                        # Check if regridded uLow file exists
                        fileHead = f'velocityZonalDepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}'
                        infileLow_u  = f'{indir}/{fileHead}_{remap}.nc'
                        if not os.path.isfile(infileLow_u):
                            print(f'\nRegridded file {infileLow_u} does not exist. Creating it with ncremap...')
                            infileLow_uNative = f'{indir}/{fileHead}.nc'
                            if not os.path.isfile(infileLow_uNative):
                                raise IOError(f'Native file {infileLow_uNative} does not exist. Need to create it with compute_composites')
                            else:
                                args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                                args.append(infileLow_uNative)
                                args.append(infileLow_u)
                                subprocess.check_call(args)
                        # Check if regridded vLow file exists
                        fileHead = f'velocityMeridionalDepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}'
                        infileLow_v  = f'{indir}/{fileHead}_{remap}.nc'
                        if not os.path.isfile(infileLow_v):
                            print(f'\nRegridded file {infileLow_v} does not exist. Creating it with ncremap...')
                            infileLow_vNative = f'{indir}/{fileHead}.nc'
                            if not os.path.isfile(infileLow_vNative):
                                raise IOError(f'Native file {infileLow_vNative} does not exist. Need to create it with compute_composites')
                            else:
                                args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                                args.append(infileLow_vNative)
                                args.append(infileLow_v)
                                subprocess.check_call(args)
                        # Check if regridded uHigh file exists
                        fileHead = f'velocityZonalDepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}'
                        infileHigh_u  = f'{indir}/{fileHead}_{remap}.nc'
                        if not os.path.isfile(infileHigh_u):
                            print(f'\nRegridded file {infileHigh_u} does not exist. Creating it with ncremap...')
                            infileHigh_uNative = f'{indir}/{fileHead}.nc'
                            if not os.path.isfile(infileHigh_uNative):
                                raise IOError(f'Native file {infileHigh_uNative} does not exist. Need to create it with compute_composites')
                            else:
                                args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                                args.append(infileHigh_uNative)
                                args.append(infileHigh_u)
                                subprocess.check_call(args)
                        # Check if regridded vHigh file exists
                        fileHead = f'velocityMeridionalDepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}'
                        infileHigh_v  = f'{indir}/{fileHead}_{remap}.nc'
                        if not os.path.isfile(infileHigh_v):
                            print(f'\nRegridded file {infileHigh_v} does not exist. Creating it with ncremap...')
                            infileHigh_vNative = f'{indir}/{fileHead}.nc'
                            if not os.path.isfile(infileHigh_vNative):
                                raise IOError(f'Native file {infileHigh_vNative} does not exist. Need to create it with compute_composites')
                            else:
                                args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                                args.append(infileHigh_vNative)
                                args.append(infileHigh_v)
                                subprocess.check_call(args)

                else: # plotDepthAvg = False
                    # Regrid the 3d fields

                    # Check if regridded uLow file exists
                    fileHead = f'velocityZonal_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}'
                    infileLow_u  = f'{indir}/{fileHead}_{remap}.nc'
                    if not os.path.isfile(infileLow_u):
                        print(f'\nRegridded file {infileLow_u} does not exist. Creating it with ncremap...')
                        infileLow_uNative = f'{indir}/{fileHead}.nc'
                        if not os.path.isfile(infileLow_uNative):
                            raise IOError(f'Native file {infileLow_uNative} does not exist. Need to create it with compute_composites')
                        else:
                            args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                            args.append(infileLow_uNative)
                            args.append(infileLow_u)
                            subprocess.check_call(args)
                    # Check if regridded vLow file exists
                    fileHead = f'velocityMeridional_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}'
                    infileLow_v  = f'{indir}/{fileHead}_{remap}.nc'
                    if not os.path.isfile(infileLow_v):
                        print(f'\nRegridded file {infileLow_v} does not exist. Creating it with ncremap...')
                        infileLow_vNative = f'{indir}/{fileHead}.nc'
                        if not os.path.isfile(infileLow_vNative):
                            raise IOError(f'Native file {infileLow_vNative} does not exist. Need to create it with compute_composites')
                        else:
                            args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                            args.append(infileLow_vNative)
                            args.append(infileLow_v)
                            subprocess.check_call(args)
                    # Check if regridded uHigh file exists
                    fileHead = f'velocityZonal_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}'
                    infileHigh_u  = f'{indir}/{fileHead}_{remap}.nc'
                    if not os.path.isfile(infileHigh_u):
                        print(f'\nRegridded file {infileHigh_u} does not exist. Creating it with ncremap...')
                        infileHigh_uNative = f'{indir}/{fileHead}.nc'
                        if not os.path.isfile(infileHigh_uNative):
                            raise IOError(f'Native file {infileHigh_uNative} does not exist. Need to create it with compute_composites')
                        else:
                            args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                            args.append(infileHigh_uNative)
                            args.append(infileHigh_u)
                            subprocess.check_call(args)
                    # Check if regridded vHigh file exists
                    fileHead = f'velocityMeridional_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}'
                    infileHigh_v  = f'{indir}/{fileHead}_{remap}.nc'
                    if not os.path.isfile(infileHigh_v):
                        print(f'\nRegridded file {infileHigh_v} does not exist. Creating it with ncremap...')
                        infileHigh_vNative = f'{indir}/{fileHead}.nc'
                        if not os.path.isfile(infileHigh_vNative):
                            raise IOError(f'Native file {infileHigh_vNative} does not exist. Need to create it with compute_composites')
                        else:
                            args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                            args.append(infileHigh_vNative)
                            args.append(infileHigh_v)
                            subprocess.check_call(args)
            # The following is no longer necessary because the complex fields are computed in compute_regionalComposites:
            #elif varname=='windStressSpeed':
            #    infileLow_u  = f'{indir}/windStressZonal_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileLow_u):
            #        raise IOError(f'File {infileLow_u} does not exist. Need to create it with compute_composites')
            #    infileLow_v  = f'{indir}/windStressMeridional_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileLow_v):
            #        raise IOError(f'File {infileLow_v} does not exist. Need to create it with compute_composites')
            #    infileHigh_u  = f'{indir}/windStressZonal_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileHigh_u):
            #        raise IOError(f'File {infileHigh_u} does not exist. Need to create it with compute_composites')
            #    infileHigh_v  = f'{indir}/windStressMeridional_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileHigh_v):
            #        raise IOError(f'File {infileHigh_v} does not exist. Need to create it with compute_composites')
            #elif varname=='iceSpeed':
            #    infileLow_u  = f'{indir}/uVelocityGeo_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileLow_u):
            #        raise IOError(f'File {infileLow_u} does not exist. Need to create it with compute_composites')
            #    infileLow_v  = f'{indir}/vVelocityGeo_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileLow_v):
            #        raise IOError(f'File {infileLow_v} does not exist. Need to create it with compute_composites')
            #    infileHigh_u  = f'{indir}/uVelocityGeo_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileHigh_u):
            #        raise IOError(f'File {infileHigh_u} does not exist. Need to create it with compute_composites')
            #    infileHigh_v  = f'{indir}/vVelocityGeo_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileHigh_v):
            #        raise IOError(f'File {infileHigh_v} does not exist. Need to create it with compute_composites')
            #elif varname=='totalHeatFlux':
            #    infileLow_sensible = f'{indir}/sensibleHeatFlux_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileLow_sensible):
            #        raise IOError(f'File {infileLow_sensible} does not exist. Need to create it with compute_composites')
            #    infileLow_latent = f'{indir}/latentHeatFlux_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileLow_latent):
            #        raise IOError(f'File {infileLow_latent} does not exist. Need to create it with compute_composites')
            #    infileLow_LRdown = f'{indir}/longWaveHeatFluxDown_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileLow_LRdown):
            #        raise IOError(f'File {infileLow_LRdown} does not exist. Need to create it with compute_composites')
            #    infileLow_LRup = f'{indir}/longWaveHeatFluxUp_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileLow_LRup):
            #        raise IOError(f'File {infileLow_LRup} does not exist. Need to create it with compute_composites')
            #    infileLow_SR = f'{indir}/shortWaveHeatFlux_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileLow_SR):
            #        raise IOError(f'File {infileLow_SR} does not exist. Need to create it with compute_composites')
            #    infileHigh_sensible = f'{indir}/sensibleHeatFlux_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileHigh_sensible):
            #        raise IOError(f'File {infileHigh_sensible} does not exist. Need to create it with compute_composites')
            #    infileHigh_latent = f'{indir}/latentHeatFlux_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileHigh_latent):
            #        raise IOError(f'File {infileHigh_latent} does not exist. Need to create it with compute_composites')
            #    infileHigh_LRdown = f'{indir}/longWaveHeatFluxDown_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileHigh_LRdown):
            #        raise IOError(f'File {infileHigh_LRdown} does not exist. Need to create it with compute_composites')
            #    infileHigh_LRup = f'{indir}/longWaveHeatFluxUp_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileHigh_LRup):
            #        raise IOError(f'File {infileHigh_LRup} does not exist. Need to create it with compute_composites')
            #    infileHigh_SR = f'{indir}/shortWaveHeatFlux_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
            #    if not os.path.isfile(infileHigh_SR):
            #        raise IOError(f'File {infileHigh_SR} does not exist. Need to create it with compute_composites')
            else: # other variables
                infileLow  = f'{indir}/{varname}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileLow):
                    raise IOError(f'Native file {infileLow} does not exist. Need to create it with compute_composites')
                infileLowStd  = f'{indir}/{varname}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}std.nc'
                if not os.path.isfile(infileLowStd):
                    raise IOError(f'Native file {infileLowStd} does not exist. Need to create it with compute_composites')
                infileHigh = f'{indir}/{varname}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileHigh):
                    raise IOError(f'Native file {infileHigh} does not exist. Need to create it with compute_composites')
                infileHighStd = f'{indir}/{varname}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}std.nc'
                if not os.path.isfile(infileHighStd):
                    raise IOError(f'Native file {infileHighStd} does not exist. Need to create it with compute_composites')

            if isvar3d:
                if plotDepthAvg:
                    for iz in range(len(zmins)):
                        zmin = zmins[iz]
                        zmax = zmaxs[iz]

                        figtitle   = f'HC-LC difference, {vartitle} (avg over z=[{np.abs(np.int32(zmax))}-{np.abs(np.int32(zmin))}] m), month={im}'
                        figfile  = f'{figdir}/{varname}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{varRef}diff_{climoMonths}_{regionNameShort}_M{im:02d}.png'
                        #figfilerel  = f'{figdir}/{varname}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{varRef}diffrel_{climoMonths}_{regionNameShort}_M{im:02d}.png'

                        if varname=='velSpeed':
                            uLow  = xr.open_dataset(infileLow_u).isel(Time=0).timeMonthly_avg_velocityZonal.values
                            vLow  = xr.open_dataset(infileLow_v).isel(Time=0).timeMonthly_avg_velocityMeridional.values
                            uHigh  = xr.open_dataset(infileHigh_u).isel(Time=0).timeMonthly_avg_velocityZonal.values
                            vHigh  = xr.open_dataset(infileHigh_v).isel(Time=0).timeMonthly_avg_velocityMeridional.values
                            uLow = varfactor * uLow
                            vLow = varfactor * vLow
                            uHigh = varfactor * uHigh
                            vHigh = varfactor * vHigh
                            speedLow  = 0.5 * np.sqrt(uLow*uLow + vLow*vLow)
                            speedHigh = 0.5 * np.sqrt(uHigh*uHigh + vHigh*vHigh)
                            diff = speedHigh-speedLow
                            #diffrel = diff/np.abs(0.5*(speedHigh+speedLow))
                            lon = xr.open_dataset(infileLow_u).lon.values
                            lat = xr.open_dataset(infileLow_u).lat.values

                            make_contourf_plot(lon, lat, diff, colormap, clevels, colorIndices, varunits,
                                               figtitle, figfile, contourFld=None, contourValues=None, projectionName='NorthPolarStereo',
                                               lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                            #make_contourf_plot(lon, lat, diffrel, colormap, clevelsrel, colorIndices, '%',
                            #                   figtitle, figfilerel, contourFld=None, contourValues=None, projectionName='NorthPolarStereo',
                            #                   lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                        else:
                            infileLow = f'{indir}/{varname}DepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                            if not os.path.isfile(infileLow):
                                raise IOError(f'Native file {infileLow} does not exist. Need to create it with compute_composites')
                            infileHigh = f'{indir}/{varname}DepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                            if not os.path.isfile(infileHigh):
                                raise IOError(f'Native file {infileHigh} does not exist. Need to create it with compute_composites')
                            dsFieldLow  = xr.open_dataset(infileLow).isel(Time=0)[varmpasname]
                            dsFieldHigh = xr.open_dataset(infileHigh).isel(Time=0)[varmpasname]

                            fldLow  = varfactor * dsFieldLow.values
                            fldHigh = varfactor * dsFieldHigh.values
                            diff = fldHigh-fldLow
                            #diffrel = diff/np.abs(0.5*(fldHigh+fldLow))

                            dotSize = 1.2 # this should go up as resolution decreases
                            make_scatter_plot(x, y, dotSize, figtitle, figfile, projectionName='NorthPolarStereo',
                                              lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                              fld=diff, ttestMask=None, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                            #make_scatter_plot(x, y, dotSize, figtitle, figfilerel, projectionName='NorthPolarStereo',
                            #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                            #                  fld=diffrel, ttestMask=None, cmap=colormap, clevels=clevelsrel, cindices=colorIndices, cbarLabel='%')
                else:
                    for iz in range(len(dlevels)):
                        figtitle   = f'HC-LC difference, {vartitle} (z={z[zlevels[iz]]:5.1f} m), month={im}'
                        figfile  = f'{figdir}/{varname}_depth{int(dlevels[iz]):04d}_{varRef}diff_{climoMonths}_{regionNameShort}_M{im:02d}.png'
                        #figfilerel  = f'{figdir}/{varname}_depth{int(dlevels[iz]):04d}_{varRef}diffrel_{climoMonths}_{regionNameShort}_M{im:02d}.png'

                        if varname=='velSpeed':
                            uLow = xr.open_dataset(infileLow_u).isel(Time=0, nVertLevels=zlevels[iz]).timeMonthly_avg_velocityZonal
                            vLow = xr.open_dataset(infileLow_v).isel(Time=0, nVertLevels=zlevels[iz]).timeMonthly_avg_velocityMeridional
                            uHigh = xr.open_dataset(infileHigh_u).isel(Time=0, nVertLevels=zlevels[iz]).timeMonthly_avg_velocityZonal
                            vHigh = xr.open_dataset(infileHigh_v).isel(Time=0, nVertLevels=zlevels[iz]).timeMonthly_avg_velocityMeridional
                             
                            uLow = varfactor * uLow.values
                            vLow = varfactor * vLow.values
                            uHigh = varfactor * uHigh.values
                            vHigh = varfactor * vHigh.values
                             
                            speedLow  = 0.5 * np.sqrt(uLow*uLow + vLow*vLow)
                            speedHigh = 0.5 * np.sqrt(uHigh*uHigh + vHigh*vHigh)
                            diff = speedHigh-speedLow
                            #diffrel = diff/np.abs(0.5*(speedHigh+speedLow))
                            lon = xr.open_dataset(infileLow_u).lon.values
                            lat = xr.open_dataset(infileLow_u).lat.values

                            make_contourf_plot(lon, lat, diff, colormap, clevels, colorIndices, varunits,
                                               figtitle, figfile, contourFld=None, contourValues=None, projectionName='NorthPolarStereo', 
                                               lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                            #make_contourf_plot(lon, lat, diffrel, colormap, clevelsrel, colorIndices, '%',
                            #                   figtitle, figfilerel, contourFld=None, contourValues=None, projectionName='NorthPolarStereo', 
                            #                   lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                        else:
                            dsFieldLow  = xr.open_dataset(infileLow).isel(Time=0, nVertLevels=zlevels[iz])[varmpasname]
                            dsFieldHigh = xr.open_dataset(infileHigh).isel(Time=0, nVertLevels=zlevels[iz])[varmpasname]
                            dsFieldLowStd  = xr.open_dataset(infileLowStd).isel(Time=0, nVertLevels=zlevels[iz])
                            dsFieldHighStd = xr.open_dataset(infileHighStd).isel(Time=0, nVertLevels=zlevels[iz])

                            topoMask = depthMask.isel(nVertLevels=zlevels[iz])
                            dsFieldLow  = dsFieldLow.where(topoMask, drop=False)
                            dsFieldHigh = dsFieldHigh.where(topoMask, drop=False)
                            dsFieldLowStd  = dsFieldLowStd.where(topoMask, drop=False)
                            dsFieldHighStd = dsFieldHighStd.where(topoMask, drop=False)

                            fldLow  = varfactor * dsFieldLow.values
                            fldHigh = varfactor * dsFieldHigh.values
                            diff = fldHigh-fldLow
                            #diffrel = diff/np.abs(0.5*(fldHigh+fldLow))
                            fldLowStd  = varfactor * dsFieldLowStd[varmpasname].values
                            fldHighStd = varfactor * dsFieldHighStd[varmpasname].values
                            fldLowStd[np.where(fldLowStd<1e-15)] = np.nan
                            fldHighStd[np.where(fldHighStd<1e-15)] = np.nan
                            ndataLow = dsFieldLowStd.nind_data.values
                            ndataHigh = dsFieldHighStd.nind_data.values

                            # two-sample t-test
                            combinedStd = np.sqrt( ((ndataLow-1)*fldLowStd**2 + (ndataHigh-1)*fldHighStd**2) / (ndataLow+ndataHigh-2) )
                            tvalue = (fldHigh - fldLow) / (combinedStd * np.sqrt(1/ndataLow+1/ndataHigh))
                            mask_ttest = np.logical_and(tvalue<tcritical, tvalue>-tcritical)
                            print('min(tvalue), max(tvalue)')
                            print(np.nanmin(tvalue), np.nanmax(tvalue))
                            print(np.shape(np.where(mask_ttest)))

                            #dotSize = 1.2 # this should go up as resolution decreases
                            #make_scatter_plot(x, y, dotSize, figtitle, figfile, projectionName='NorthPolarStereo',
                            #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                            #                  fld=diff, ttestMask=mask_ttest, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                            ##make_scatter_plot(x, y, dotSize, figtitle, figfilerel, projectionName='NorthPolarStereo',
                            ##                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                            ##                  fld=diffrel, ttestMask=mask_ttest, cmap=colormap, clevels=clevelsrel, cindices=colorIndices, cbarLabel='%')
                            make_mosaic_plot(x, y, varfactor*(dsFieldHigh-dsFieldLow), mosaic_descriptor, figtitle, figfile, ttestMask=mask_ttest, showEdges=None,
                                             cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                                             projectionName='NorthPolarStereo', lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
            else:
                figtitle = f'HC-LC difference, {vartitle}, month={im}'
                figfile  = f'{figdir}/{varname}_{varRef}diff_{climoMonths}_{regionNameShort}_M{im:02d}.png'
                #figfilerel  = f'{figdir}/{varname}_{varRef}diffrel_{climoMonths}_{regionNameShort}_M{im:02d}.png'

                if varname=='barotropicStreamfunction':
                    dsFieldLow  = xr.open_dataset(infileLow)[varmpasname]
                    dsFieldHigh = xr.open_dataset(infileHigh)[varmpasname]
                    dsFieldLowStd  = xr.open_dataset(infileLowStd)
                    dsFieldHighStd = xr.open_dataset(infileHighStd)
                # Piece of old code:
                #if varname=='windStressSpeed':
                #    dsFieldLow_u  = xr.open_dataset(infileLow_u).isel(Time=0)['timeMonthly_avg_windStressZonal']
                #    dsFieldLow_v  = xr.open_dataset(infileLow_v).isel(Time=0)['timeMonthly_avg_windStressMeridional']
                #    dsFieldHigh_u = xr.open_dataset(infileHigh_u).isel(Time=0)['timeMonthly_avg_windStressZonal']
                #    dsFieldHigh_v = xr.open_dataset(infileHigh_v).isel(Time=0)['timeMonthly_avg_windStressMeridional']
                #    fldLow  = varfactor * 0.5 * np.sqrt(dsFieldLow_u.values**2  + dsFieldLow_v.values**2)
                #    fldHigh = varfactor * 0.5 * np.sqrt(dsFieldHigh_u.values**2 + dsFieldHigh_v.values**2)
                #elif varname=='iceSpeed':
                #    dsFieldLow_u  = xr.open_dataset(infileLow_u).isel(Time=0)['timeMonthly_avg_uVelocityGeo']
                #    dsFieldLow_v  = xr.open_dataset(infileLow_v).isel(Time=0)['timeMonthly_avg_vVelocityGeo']
                #    dsFieldHigh_u = xr.open_dataset(infileHigh_u).isel(Time=0)['timeMonthly_avg_uVelocityGeo']
                #    dsFieldHigh_v = xr.open_dataset(infileHigh_v).isel(Time=0)['timeMonthly_avg_vVelocityGeo']
                #    fldLow  = varfactor * 0.5 * np.sqrt(dsFieldLow_u.values**2  + dsFieldLow_v.values**2)
                #    fldHigh = varfactor * 0.5 * np.sqrt(dsFieldHigh_u.values**2 + dsFieldHigh_v.values**2)
                #elif varname=='barotropicStreamfunction':
                #   dsFieldLow  = xr.open_dataset(infileLow)[varmpasname]
                #   dsFieldHigh = xr.open_dataset(infileHigh)[varmpasname]
                #   fldLow  = varfactor * dsFieldLow.values
                #   fldHigh = varfactor * dsFieldHigh.values
                #elif varname=='totalHeatFlux':
                #   sensibleLow = xr.open_dataset(infileLow_sensible)['timeMonthly_avg_sensibleHeatFlux']
                #   latentLow = xr.open_dataset(infileLow_latent)['timeMonthly_avg_latentHeatFlux']
                #   LRdownLow = xr.open_dataset(infileLow_LRdown)['timeMonthly_avg_longWaveHeatFluxDown']
                #   LRupLow = xr.open_dataset(infileLow_LRup)['timeMonthly_avg_longWaveHeatFluxUp']
                #   SRLow = xr.open_dataset(infileLow_SR)['timeMonthly_avg_shortWaveHeatFlux']
                #   sensibleHigh = xr.open_dataset(infileHigh_sensible)['timeMonthly_avg_sensibleHeatFlux']
                #   latentHigh = xr.open_dataset(infileHigh_latent)['timeMonthly_avg_latentHeatFlux']
                #   LRdownHigh = xr.open_dataset(infileHigh_LRdown)['timeMonthly_avg_longWaveHeatFluxDown']
                #   LRupHigh = xr.open_dataset(infileHigh_LRup)['timeMonthly_avg_longWaveHeatFluxUp']
                #   SRHigh = xr.open_dataset(infileHigh_SR)['timeMonthly_avg_shortWaveHeatFlux']
                #   dsFieldLow = sensibleLow + latentLow + LRdownLow + LRupLow + SRLow
                #   dsFieldHigh = sensibleHigh + latentHigh + LRdownHigh + LRupHigh + SRHigh
                #   fldLow  = varfactor * dsFieldLow.values
                #   fldHigh = varfactor * dsFieldHigh.values
                else:
                   dsFieldLow  = xr.open_dataset(infileLow).isel(Time=0)[varmpasname]
                   dsFieldHigh = xr.open_dataset(infileHigh).isel(Time=0)[varmpasname]
                   dsFieldLowStd  = xr.open_dataset(infileLowStd).isel(Time=0)
                   dsFieldHighStd = xr.open_dataset(infileHighStd).isel(Time=0)

                fldLow  = varfactor * dsFieldLow.values
                fldHigh = varfactor * dsFieldHigh.values
                diff = fldHigh-fldLow
                #diffrel = diff/np.abs(0.5*(fldHigh+fldLow))
                fldLowStd  = varfactor * dsFieldLowStd[varmpasname].values
                fldHighStd = varfactor * dsFieldHighStd[varmpasname].values
                fldLowStd[np.where(fldLowStd<1e-15)] = np.nan
                fldHighStd[np.where(fldHighStd<1e-15)] = np.nan
                ndataLow = dsFieldLowStd.nind_data.values
                ndataHigh = dsFieldHighStd.nind_data.values

                # two-sample t-test
                combinedStd = np.sqrt( ((ndataLow-1)*fldLowStd**2 + (ndataHigh-1)*fldHighStd**2) / (ndataLow+ndataHigh-2) )
                tvalue = (fldHigh - fldLow) / (combinedStd * np.sqrt(1/ndataLow+1/ndataHigh))
                mask_ttest = np.logical_and(tvalue<tcritical, tvalue>-tcritical)
                print('min(tvalue), max(tvalue)')
                print(np.nanmin(tvalue), np.nanmax(tvalue))
                print(np.shape(np.where(mask_ttest)))

                # Mask areas with effective no ice
                if varname=='iceArea' or varname=='iceVolume' or varname=='iceDivergence' or varname=='iceSpeed' or varname=='seaIceFreshWaterFlux':
                    fldLow[np.where(np.abs(fldLow)<1e-15)]    = np.nan
                    fldHigh[np.where(np.abs(fldHigh)<1e-15)]  = np.nan
                    diff = fldHigh-fldLow
                    dsFieldLow = dsFieldLow.where(np.abs(dsFieldLow)>1e-15)
                    dsFieldHigh = dsFieldHigh.where(np.abs(dsFieldHigh)>1e-15)

                #dotSize = 1.2 # this should go up as resolution decreases
                #make_scatter_plot(x, y, dotSize, figtitle, figfile, projectionName='NorthPolarStereo',
                #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                #                  fld=diff, ttestMask=mask_ttest, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                ##make_scatter_plot(x, y, dotSize, figtitle, figfilerel, projectionName='NorthPolarStereo',
                ##                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                ##                  fld=diffrel, ttestMask=mask_ttest, cmap=colormap, clevels=clevelsrel, cindices=colorIndices, cbarLabel='%')
                make_mosaic_plot(x, y, varfactor*(dsFieldHigh-dsFieldLow), mosaic_descriptor, figtitle, figfile, ttestMask=mask_ttest, showEdges=None,
                                 cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                                 projectionName='NorthPolarStereo', lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
