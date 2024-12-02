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

from make_plots import make_scatter_plot, make_streamline_plot, make_contourf_plot


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

indir0 = f'./composites_maxMLDbased_data/{runName}'
figdir0 = f'./composites_maxMLDbased/{runName}'
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
#regions = ['Greenland Sea', 'Norwegian Sea']

climoMonths = 'JFMA' # should be consistent with composites calculation

colorIndices = [0, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 250, 255]
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
#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'title': 'Maximum MLD',
#              'units': 'm',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthlyMax_max_dThreshMLD'}
#            ]
#
mpasFile = 'timeSeriesStatsMonthly'
variables = [
#             {'name': 'streamlines',
#              'title': 'Velocity',
#              'units': 'cm/s',
#              'factor': 1e2,
#              'isvar3d': True,
#              'mpas': None,
#              'clevels': [0.5, 1, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40],
#              'colormap': cmocean.cm.speed_r},
#             {'name': 'velSpeed',
#              'title': 'Velocity magnitude',
#              'units': 'cm/s',
#              'factor': 1e2,
#              'isvar3d': True,
#              'mpas': None,
#              'clevels': [0.5, 1, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40],
#              'colormap': cmocean.cm.speed_r},
#             {'name': 'windStressSpeed',
#              'title': 'Wind stress magnitude',
#              'units': 'N/m$^2$',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': None,
#              'clevels': [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.14, 0.15],
#              'colormap': cmocean.cm.speed_r},
#             {'name': 'dThreshMLD',
#              'title': 'Mean MLD',
#              'units': 'm',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_dThreshMLD',
#              'clevels': [10, 20, 50, 80, 100, 120, 150, 180, 250, 300, 400, 500, 800],
#              'colormap': plt.get_cmap('viridis')},
#             {'name': 'sensibleHeatFlux',
#              'title': 'Sensible Heat Flux',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_sensibleHeatFlux',
#              'clevels': [-250, -200, -150, -120, -100, -80, -60, -40, -20, -10, 0, 10, 20],
#              'colormap': cmocean.cm.solar_r},
             {'name': 'spiciness',
              'title': 'Spiciness0',
              'units': '',
              'factor': 1,
              'isvar3d': True,
              'mpas': 'spiciness0',
              'clevels': [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
              'colormap': cmocean.cm.balance}
#             {'name': 'activeTracers_temperature',
#              'title': 'Potential Temperature',
#              'units': 'degC',
#              'factor': 1,
#              'isvar3d': True,
#              'mpas': 'timeMonthly_avg_activeTracers_temperature',
#              'clevels': [-1.0, -0.5, 0.0, 0.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8., 10., 12.],
#              'colormap': cmocean.cm.thermal},
#             {'name': 'activeTracers_salinity',
#              'title': 'Salinity',
#              'units': 'psu',
#              'factor': 1,
#              'isvar3d': True,
#              'mpas': 'timeMonthly_avg_activeTracers_salinity',
#              'clevels': [31.0, 33.0, 34.2,  34.4,  34.6, 34.7,  34.8,  34.87, 34.9, 34.95, 35.0, 35.2, 35.4],
#              'colormap': cmocean.cm.haline}
             ]
             #{'name': 'surfaceBuoyancyForcing',
             # 'title': 'Surface buoyancy flux',
             # 'units': 'm$^2$ s$^{-3}$',
             # 'factor': 1,
             # 'isvar3d': False,
             # 'mpas': 'timeMonthly_avg_surfaceBuoyancyForcing'}
             #{'name': 'latentHeatFlux',
             # 'title': 'Latent Heat Flux',
             # 'units': 'W/m$^2$',
             # 'factor': 1,
             # 'isvar3d': False,
             # 'mpas': 'timeMonthly_avg_latentHeatFlux'}
#   Sea ice variables
#modelComp = 'ice'
#modelName = 'mpassi'
#mpasFile = 'timeSeriesStatsMonthly'
#variables = [
#             {'name': 'iceArea',
#              'title': 'Sea Ice Concentration',
#              'units': '%',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_iceAreaCell',
#              'clevels': [0.15, 0.3, 0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.0],
#              'colormap': cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
#                                               (0.169, 0.435, 0.223), (0.455, 0.478, 0.196), (0.757, 0.474, 0.435),\
#                                               (0.827, 0.561, 0.772), (0.761, 0.757, 0.949), (0.808, 0.921, 0.937)])},
#             {'name': 'iceVolume',
#              'title': 'Sea Ice Thickness',
#              'units': 'm',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_iceVolumeCell',
#              'clevels': [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
#              'colormap': plt.get_cmap('YlGnBu_r')},
#             {'name': 'iceDivergence',
#              'title': 'Sea Ice divergence',
#              'units': '%/day',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_divergence',
#              'clevels': [-10.0, -8.0, -6.0, -4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0, 8.0, 10.0],
#              'colormap': cmocean.cm.balance},
#             {'name': 'iceSpeed',
#              'title': 'Sea Ice speed',
#              'units': 'm/s',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': None,
#              'clevels': [0.04, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
#              'colormap': cmocean.cm.solar}
#            ]
#   Atmosphere variables
#modelComp = 'atm'
#modelName = 'eam'

plotDepthAvg = False
# zmins/zmaxs [m] (relevant for 3d variables and if plotDepthAvg = True)
#zmins = [-50., -600., -8000.]
#zmaxs = [0., -100., 0.]
#zmins = [-50.]
#zmaxs = [0.]
#zmins = [-50., -600.]
#zmaxs = [0., -100.]
zmins = [-600.]
zmaxs = [0.]
# z levels [m] (relevant for 3d variables and if plotDepthAvg = False)
#dlevels = [0., 500.]
#dlevels = [50., 100.]
#dlevels = [0., 100.]
#dlevels = [0., 50., 100.]
dlevels = [300.]

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

# Plot monthly climatologies associated with previously computed composites
for regionName in regions:
    print(f'\nPlot low/high convection composites based on maxMLD for region: {regionName}')
    regionNameShort = regionName[0].lower() + regionName[1:].replace(' ', '').replace('(', '_').replace(')', '').replace('/', '_')

    for im in range(1, 13):
        print(f'  Month: {im}')
        for var in variables:
            varname = var['name']
            print(f'    var: {varname}')
            varmpasname = var['mpas']
            isvar3d = var['isvar3d']
            varfactor = var['factor']
            varunits = var['units']
            vartitle = var['title']
            colormap = var['colormap']
            clevels = var['clevels']

            if varname=='iceSpeed':
                x = lonVertex
                y = latVertex
            else:
                x = lonCell
                y = latCell

            if varname=='streamlines' or varname=='velSpeed':
                # Regrid velocityZonal and velocityMeridional (if necessary)
                if plotDepthAvg:
                    # Compute the depth average first and then regrid
                    for iz in range(len(zmins)):
                        zmin = zmins[iz]
                        zmax = zmaxs[iz]

                        # Check if regridded uLow file exists
                        fileHead = f'velocityZonalDepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}'
                        infileLow_u  = f'{indir}/{fileHead}_{remap}.nc'
                        if not os.path.isfile(infileLow_u):
                            print(f'\nRegridded file {infileLow_u} does not exist. Creating it with ncremap...')
                            infileLow_uNative = f'{indir}/{fileHead}.nc'
                            if not os.path.isfile(infileLow_uNative):
                                raise IOError(f'Native file {infileLow_uNative} does not exist. Need to create it with compute_composites_maxMLDbased')
                            else:
                                args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                                args.append(infileLow_uNative)
                                args.append(infileLow_u)
                                subprocess.check_call(args)
                        # Check if regridded vLow file exists
                        fileHead = f'velocityMeridionalDepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}'
                        infileLow_v  = f'{indir}/{fileHead}_{remap}.nc'
                        if not os.path.isfile(infileLow_v):
                            print(f'\nRegridded file {infileLow_v} does not exist. Creating it with ncremap...')
                            infileLow_vNative = f'{indir}/{fileHead}.nc'
                            if not os.path.isfile(infileLow_vNative):
                                raise IOError(f'Native file {infileLow_vNative} does not exist. Need to create it with compute_composites_maxMLDbased')
                            else:
                                args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                                args.append(infileLow_vNative)
                                args.append(infileLow_v)
                                subprocess.check_call(args)
                        # Check if regridded uHigh file exists
                        fileHead = f'velocityZonalDepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}'
                        infileHigh_u  = f'{indir}/{fileHead}_{remap}.nc'
                        if not os.path.isfile(infileHigh_u):
                            print(f'\nRegridded file {infileHigh_u} does not exist. Creating it with ncremap...')
                            infileHigh_uNative = f'{indir}/{fileHead}.nc'
                            if not os.path.isfile(infileHigh_uNative):
                                raise IOError(f'Native file {infileHigh_uNative} does not exist. Need to create it with compute_composites_maxMLDbased')
                            else:
                                args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                                args.append(infileHigh_uNative)
                                args.append(infileHigh_u)
                                subprocess.check_call(args)
                        # Check if regridded vHigh file exists
                        fileHead = f'velocityMeridionalDepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}'
                        infileHigh_v  = f'{indir}/{fileHead}_{remap}.nc'
                        if not os.path.isfile(infileHigh_v):
                            print(f'\nRegridded file {infileHigh_v} does not exist. Creating it with ncremap...')
                            infileHigh_vNative = f'{indir}/{fileHead}.nc'
                            if not os.path.isfile(infileHigh_vNative):
                                raise IOError(f'Native file {infileHigh_vNative} does not exist. Need to create it with compute_composites_maxMLDbased')
                            else:
                                args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                                args.append(infileHigh_vNative)
                                args.append(infileHigh_v)
                                subprocess.check_call(args)

                else: # plotDepthAvg = False
                    # Regrid the 3d fields

                    # Check if regridded uLow file exists
                    fileHead = f'velocityZonal_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}'
                    infileLow_u  = f'{indir}/{fileHead}_{remap}.nc'
                    if not os.path.isfile(infileLow_u):
                        print(f'\nRegridded file {infileLow_u} does not exist. Creating it with ncremap...')
                        infileLow_uNative = f'{indir}/{fileHead}.nc'
                        if not os.path.isfile(infileLow_uNative):
                            raise IOError(f'Native file {infileLow_uNative} does not exist. Need to create it with compute_composites_maxMLDbased')
                        else:
                            args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                            args.append(infileLow_uNative)
                            args.append(infileLow_u)
                            subprocess.check_call(args)
                    # Check if regridded vLow file exists
                    fileHead = f'velocityMeridional_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}'
                    infileLow_v  = f'{indir}/{fileHead}_{remap}.nc'
                    if not os.path.isfile(infileLow_v):
                        print(f'\nRegridded file {infileLow_v} does not exist. Creating it with ncremap...')
                        infileLow_vNative = f'{indir}/{fileHead}.nc'
                        if not os.path.isfile(infileLow_vNative):
                            raise IOError(f'Native file {infileLow_vNative} does not exist. Need to create it with compute_composites_maxMLDbased')
                        else:
                            args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                            args.append(infileLow_vNative)
                            args.append(infileLow_v)
                            subprocess.check_call(args)
                    # Check if regridded uHigh file exists
                    fileHead = f'velocityZonal_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}'
                    infileHigh_u  = f'{indir}/{fileHead}_{remap}.nc'
                    if not os.path.isfile(infileHigh_u):
                        print(f'\nRegridded file {infileHigh_u} does not exist. Creating it with ncremap...')
                        infileHigh_uNative = f'{indir}/{fileHead}.nc'
                        if not os.path.isfile(infileHigh_uNative):
                            raise IOError(f'Native file {infileHigh_uNative} does not exist. Need to create it with compute_composites_maxMLDbased')
                        else:
                            args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                            args.append(infileHigh_uNative)
                            args.append(infileHigh_u)
                            subprocess.check_call(args)
                    # Check if regridded vHigh file exists
                    fileHead = f'velocityMeridional_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}'
                    infileHigh_v  = f'{indir}/{fileHead}_{remap}.nc'
                    if not os.path.isfile(infileHigh_v):
                        print(f'\nRegridded file {infileHigh_v} does not exist. Creating it with ncremap...')
                        infileHigh_vNative = f'{indir}/{fileHead}.nc'
                        if not os.path.isfile(infileHigh_vNative):
                            raise IOError(f'Native file {infileHigh_vNative} does not exist. Need to create it with compute_composites_maxMLDbased')
                        else:
                            args = ['ncremap', '-a', 'trbilin', '-m', remapFile, '-P', 'mpaso']
                            args.append(infileHigh_vNative)
                            args.append(infileHigh_v)
                            subprocess.check_call(args)
            elif varname=='windStressSpeed':
                infileLow_u  = f'{indir}/windStressZonal_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileLow_u):
                    raise IOError(f'File {infileLow_u} does not exist. Need to create it with compute_composites_maxMLDbased')
                infileLow_v  = f'{indir}/windStressMeridional_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileLow_v):
                    raise IOError(f'File {infileLow_v} does not exist. Need to create it with compute_composites_maxMLDbased')
                infileHigh_u  = f'{indir}/windStressZonal_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileHigh_u):
                    raise IOError(f'File {infileHigh_u} does not exist. Need to create it with compute_composites_maxMLDbased')
                infileHigh_v  = f'{indir}/windStressMeridional_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileHigh_v):
                    raise IOError(f'File {infileHigh_v} does not exist. Need to create it with compute_composites_maxMLDbased')
            elif varname=='iceSpeed':
                infileLow_u  = f'{indir}/uVelocityGeo_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileLow_u):
                    raise IOError(f'File {infileLow_u} does not exist. Need to create it with compute_composites_maxMLDbased')
                infileLow_v  = f'{indir}/vVelocityGeo_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileLow_v):
                    raise IOError(f'File {infileLow_v} does not exist. Need to create it with compute_composites_maxMLDbased')
                infileHigh_u  = f'{indir}/uVelocityGeo_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileHigh_u):
                    raise IOError(f'File {infileHigh_u} does not exist. Need to create it with compute_composites_maxMLDbased')
                infileHigh_v  = f'{indir}/vVelocityGeo_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileHigh_v):
                    raise IOError(f'File {infileHigh_v} does not exist. Need to create it with compute_composites_maxMLDbased')
            else: # other variables
                infileLow  = f'{indir}/{varname}_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileLow):
                    raise IOError(f'Native file {infileLow} does not exist. Need to create it with compute_composites_maxMLDbased')
                infileHigh = f'{indir}/{varname}_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                if not os.path.isfile(infileHigh):
                    raise IOError(f'Native file {infileHigh} does not exist. Need to create it with compute_composites_maxMLDbased')

            if isvar3d:
                if plotDepthAvg:
                    for iz in range(len(zmins)):
                        zmin = zmins[iz]
                        zmax = zmaxs[iz]

                        figtitleLow   = f'Composite for low maxMLD ({regionName})\nmonth={im}, {vartitle}, avg over z=[{np.abs(np.int32(zmax))}-{np.abs(np.int32(zmin))}] m'
                        figtitleHigh  = f'Composite for high maxMLD ({regionName})\nmonth={im}, {vartitle}, avg over z=[{np.abs(np.int32(zmax))}-{np.abs(np.int32(zmin))}] m'
                        figfileLow  = f'{figdir}/{varname}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}.png'
                        figfileHigh = f'{figdir}/{varname}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}.png'

                        if varname=='streamlines' or varname=='velSpeed':
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
                            lon = xr.open_dataset(infileLow_u).lon.values
                            lat = xr.open_dataset(infileLow_u).lat.values

                            if varname=='streamlines':
                                streamlineDensity = 4
                                make_streamline_plot(lon, lat, uLow, vLow, speedLow, streamlineDensity, 
                                                     colormap, clevels, colorIndices, varunits,
                                                     'NorthPolarStereo', figtitleLow, figfileLow,
                                                     lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                                make_streamline_plot(lon, lat, uHigh, vHigh, speedHigh, streamlineDensity, 
                                                     colormap, clevels, colorIndices, varunits,
                                                     'NorthPolarStereo', figtitleHigh, figfileHigh,
                                                     lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                            else:
                                make_contourf_plot(lon, lat, speedLow, colormap, clevels, colorIndices, varunits,
                                                   figtitleLow, figfileLow, contourFld=None, contourValues=None, projectionName='NorthPolarStereo',
                                                   lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                                make_contourf_plot(lon, lat, speedHigh, colormap, clevels, colorIndices, varunits,
                                                   figtitleHigh, figfileHigh, contourFld=None, contourValues=None, projectionName='NorthPolarStereo',
                                                   lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                        else:
                            infileLow = f'{indir}/{varname}DepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                            if not os.path.isfile(infileLow):
                                raise IOError(f'Native file {infileLow} does not exist. Need to create it with compute_composites_maxMLDbased')
                            infileHigh = f'{indir}/{varname}DepthAvg_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                            if not os.path.isfile(infileHigh):
                                raise IOError(f'Native file {infileHigh} does not exist. Need to create it with compute_composites_maxMLDbased')
                            dsFieldLow  = xr.open_dataset(infileLow).isel(Time=0)[varmpasname]
                            dsFieldHigh = xr.open_dataset(infileHigh).isel(Time=0)[varmpasname]

                            fldLow  = varfactor * dsFieldLow.values
                            fldHigh = varfactor * dsFieldHigh.values

                            dotSize = 1.2 # this should go up as resolution decreases
                            make_scatter_plot(x, y, dotSize, figtitleLow, figfileLow, projectionName='NorthPolarStereo',
                                              lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                              fld=fldLow, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                            make_scatter_plot(x, y, dotSize, figtitleHigh, figfileHigh, projectionName='NorthPolarStereo',
                                              lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                              fld=fldHigh, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                else:
                    for iz in range(len(dlevels)):
                        figtitleLow   = f'Composite for low maxMLD ({regionName})\nmonth={im}, {vartitle}, z={z[zlevels[iz]]:5.1f} m'
                        figtitleHigh  = f'Composite for high maxMLD ({regionName})\nmonth={im}, {vartitle}, z={z[zlevels[iz]]:5.1f} m'
                        figfileLow  = f'{figdir}/{varname}_depth{int(dlevels[iz]):04d}_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}.png'
                        figfileHigh = f'{figdir}/{varname}_depth{int(dlevels[iz]):04d}_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}.png'

                        if varname=='streamlines' or varname=='velSpeed':
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
                            lon = xr.open_dataset(infileLow_u).lon.values
                            lat = xr.open_dataset(infileLow_u).lat.values

                            if varname=='streamlines':
                                streamlineDensity = 4
                                make_streamline_plot(lon, lat, uLow, vLow, speedLow, streamlineDensity, 
                                                     colormap, clevels, colorIndices, varunits,
                                                     'NorthPolarStereo', figtitleLow, figfileLow,
                                                     lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                                make_streamline_plot(lon, lat, uHigh, vHigh, speedHigh, streamlineDensity, 
                                                     colormap, clevels, colorIndices, varunits,
                                                     'NorthPolarStereo', figtitleHigh, figfileHigh,
                                                     lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                            else:
                                make_contourf_plot(lon, lat, speedLow, colormap, clevels, colorIndices, varunits,
                                                   figtitleLow, figfileLow, contourFld=None, contourValues=None, projectionName='NorthPolarStereo', 
                                                   lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                                make_contourf_plot(lon, lat, speedHigh, colormap, clevels, colorIndices, varunits,
                                                   figtitleHigh, figfileHigh, contourFld=None, contourValues=None, projectionName='NorthPolarStereo', 
                                                   lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                        else:
                            dsFieldLow  = xr.open_dataset(infileLow).isel(Time=0, nVertLevels=zlevels[iz])[varmpasname]
                            dsFieldHigh = xr.open_dataset(infileHigh).isel(Time=0, nVertLevels=zlevels[iz])[varmpasname]

                            topoMask = depthMask.isel(nVertLevels=zlevels[iz])
                            dsFieldLow  = dsFieldLow.where(topoMask, drop=False)
                            dsFieldHigh = dsFieldHigh.where(topoMask, drop=False)

                            fldLow  = varfactor * dsFieldLow.values
                            fldHigh = varfactor * dsFieldHigh.values

                            dotSize = 1.2 # this should go up as resolution decreases
                            make_scatter_plot(x, y, dotSize, figtitleLow, figfileLow, projectionName='NorthPolarStereo',
                                              lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                              fld=fldLow, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                            make_scatter_plot(x, y, dotSize, figtitleHigh, figfileHigh, projectionName='NorthPolarStereo',
                                              lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                              fld=fldHigh, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
            else:
                figtitleLow   = f'Composite for low maxMLD ({regionName})\nmonth={im}, {vartitle}'
                figtitleHigh  = f'Composite for high maxMLD ({regionName})\nmonth={im}, {vartitle}'
                figfileLow  = f'{figdir}/{varname}_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}.png'
                figfileHigh = f'{figdir}/{varname}_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}.png'

                if varname=='windStressSpeed':
                    dsFieldLow_u  = xr.open_dataset(infileLow_u).isel(Time=0)['timeMonthly_avg_windStressZonal']
                    dsFieldLow_v  = xr.open_dataset(infileLow_v).isel(Time=0)['timeMonthly_avg_windStressMeridional']
                    dsFieldHigh_u = xr.open_dataset(infileHigh_u).isel(Time=0)['timeMonthly_avg_windStressZonal']
                    dsFieldHigh_v = xr.open_dataset(infileHigh_v).isel(Time=0)['timeMonthly_avg_windStressMeridional']
                    fldLow  = varfactor * 0.5 * np.sqrt(dsFieldLow_u.values**2  + dsFieldLow_v.values**2)
                    fldHigh = varfactor * 0.5 * np.sqrt(dsFieldHigh_u.values**2 + dsFieldHigh_v.values**2)
                elif varname=='iceSpeed':
                    dsFieldLow_u  = xr.open_dataset(infileLow_u).isel(Time=0)['timeMonthly_avg_uVelocityGeo']
                    dsFieldLow_v  = xr.open_dataset(infileLow_v).isel(Time=0)['timeMonthly_avg_vVelocityGeo']
                    dsFieldHigh_u = xr.open_dataset(infileHigh_u).isel(Time=0)['timeMonthly_avg_uVelocityGeo']
                    dsFieldHigh_v = xr.open_dataset(infileHigh_v).isel(Time=0)['timeMonthly_avg_vVelocityGeo']
                    fldLow  = varfactor * 0.5 * np.sqrt(dsFieldLow_u.values**2  + dsFieldLow_v.values**2)
                    fldHigh = varfactor * 0.5 * np.sqrt(dsFieldHigh_u.values**2 + dsFieldHigh_v.values**2)
                else:
                   dsFieldLow  = xr.open_dataset(infileLow).isel(Time=0)[varmpasname]
                   dsFieldHigh = xr.open_dataset(infileHigh).isel(Time=0)[varmpasname]
                   fldLow  = varfactor * dsFieldLow.values
                   fldHigh = varfactor * dsFieldHigh.values

                if varname=='iceVolume' or varname=='iceDivergence':
                    # Mask fields where ice concentration < 15%
                    infile  = f'{indir}/iceArea_maxMLDlow_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                    if not os.path.isfile(infile):
                        raise IOError(f'File {infile} does not exist. Need to create it with compute_composites_maxMLDbased')
                    iceArea  = xr.open_dataset(infile).isel(Time=0)['timeMonthly_avg_iceAreaCell']
                    fldLow[np.where(iceArea<0.15)]   = np.nan
                    infile = f'{indir}/iceArea_maxMLDhigh_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                    if not os.path.isfile(infile):
                        raise IOError(f'File {infile} does not exist. Need to create it with compute_composites_maxMLDbased')
                    iceArea  = xr.open_dataset(infile).isel(Time=0)['timeMonthly_avg_iceAreaCell']
                    fldHigh[np.where(iceArea<0.15)]   = np.nan

                #print(np.nanmin(fldLow), np.nanmax(fldLow))

                dotSize = 1.2 # this should go up as resolution decreases
                if varname=='iceArea':
                    fldLow[np.where(fldLow<0.15)]   = np.nan
                    fldHigh[np.where(fldHigh<0.15)] = np.nan
                    make_scatter_plot(x, y, dotSize, figtitleLow, figfileLow, projectionName='NorthPolarStereo',
                                      lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                      fld=fldLow, cmap=colormap, clevels=clevels, cindices=None, cbarLabel=varunits)
                    make_scatter_plot(x, y, dotSize, figtitleHigh, figfileHigh, projectionName='NorthPolarStereo',
                                      lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                      fld=fldHigh, cmap=colormap, clevels=clevels, cindices=None, cbarLabel=varunits)
                else:
                    make_scatter_plot(x, y, dotSize, figtitleLow, figfileLow, projectionName='NorthPolarStereo',
                                      lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                      fld=fldLow, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                    make_scatter_plot(x, y, dotSize, figtitleHigh, figfileHigh, projectionName='NorthPolarStereo',
                                      lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                      fld=fldHigh, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
