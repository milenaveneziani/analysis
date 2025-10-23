from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import subprocess
from subprocess import call
import xarray as xr
import numpy as np
import netCDF4
from scipy import stats
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
meandir = '/global/cfs/cdirs/m1199/e3sm-arrm-simulations/E3SM-Arcticv2.1_historical/postprocessing'

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
#modelComp = 'ocn'
#modelName = 'mpaso'
#mpasFile = 'timeSeriesStatsMonthly'
#variables = [
#             {'name': 'barotropicStreamfunction',
#              'title': 'Barotropic streamfunction',
#              'units': 'Sv',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'barotropicStreamfunction',
#              #'clevels': [-18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18],
#              'clevels': [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.curl},
#              #'colormap': cmocean.cm.tarn}
##             {'name': 'streamlines',
##              'title': 'Velocity',
##              'units': 'cm/s',
##              'factor': 1e2,
##              'isvar3d': True,
##              'mpas': None,
##              'clevels': [0.5, 1, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40],
##              'cIndices': colorIndices,
##              'colormap': cmocean.cm.speed_r},
##             {'name': 'velSpeed',
##              'title': 'Velocity magnitude',
##              'units': 'cm/s',
##              'factor': 1e2,
##              'isvar3d': True,
##              'mpas': None,
##              'clevels': [0.5, 1, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40],
##              'cIndices': colorIndices,
##              'colormap': cmocean.cm.speed_r},
#             {'name': 'windStressSpeed',
#              'title': 'Wind stress magnitude',
#              'units': 'N/m$^2$',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'windStressSpeed',
##              'mpas': None,
#              'clevels': [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.14, 0.15],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.speed_r},
##             {'name': 'dThreshMLD',
##              'title': 'Mean MLD',
##              'units': 'm',
##              'factor': 1,
##              'isvar3d': False,
##              'mpas': 'timeMonthly_avg_dThreshMLD',
##              'clevels': [10, 20, 50, 80, 100, 120, 150, 180, 250, 300, 400, 500, 800],
##              'cIndices': colorIndices,
##              'colormap': plt.get_cmap('viridis')},
##             {'name': 'spiciness',
##              'title': 'Spiciness0',
##              'units': '',
##              'factor': 1,
##              'isvar3d': True,
##              'mpas': 'spiciness0',
##              'clevels': [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
##              'cIndices': colorIndices,
##              'colormap': cmocean.cm.curl},
##              #'colormap': cmocean.cm.balance},
#             {'name': 'activeTracers_temperature',
#              'title': 'Potential Temperature',
#             'units': '$^\circ$C',
#              'factor': 1,
#              'isvar3d': True,
#              'mpas': 'timeMonthly_avg_activeTracers_temperature',
#              'clevels': [-1.0, -0.5, 0.0, 0.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8., 10., 12.],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.thermal},
#             {'name': 'activeTracers_salinity',
#              'title': 'Salinity',
#              'units': 'psu',
#              'factor': 1,
#              'isvar3d': True,
#              'mpas': 'timeMonthly_avg_activeTracers_salinity',
#              'clevels': [31.0, 33.0, 34.2,  34.4,  34.6, 34.7,  34.8,  34.87, 34.9, 34.95, 35.0, 35.2, 35.4],
#              'cIndices': colorIndices,
#              'colormap': cmocean.cm.haline},
#             {'name': 'surfaceBuoyancyForcing',
#              'title': 'Surface buoyancy flux',
#              'units': '10$^{-8}$ m$^2$ s$^{-3}$',
#              'factor': 1e8,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_surfaceBuoyancyForcing',
#              'clevels': [-4.8, -4, -3.2, -2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4, 3.2, 4, 4.8],
#             # 'clevels': [-2.4, -2, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2, 2.4],
#              'cIndices': colorIndices,
#             # 'cIndices': [0, 14, 28, 57, 80, 113, 125, 155, 170, 198, 227, 242, 255],
#              'colormap': plt.get_cmap('BrBG_r')},
#             # 'cIndices': None,
#             # 'colormap': cols.ListedColormap([(37/255,  52/255,  148/255), (44/255,  127/255, 184/255),     \
#             #                                  (65/255,  182/255, 196/255), (127/255, 205/255, 187/255),     \
#             #                                  (199/255, 233/255, 180/255), (1, 1, 204/255), (1, 1, 217/255),\
#             #                                  (254/255, 240/255, 217/255), (253/255, 212/255, 158/255),     \
#             #                                  (253/255, 187/255, 132/255), (252/255, 141/255, 89/255),      \
#             #                                  (227/255, 74/255,  51/255),  (179/255, 0,   0)])},
##             {'name': 'sensibleHeatFlux',
##              'title': 'Sensible Heat Flux',
##              'units': 'W/m$^2$',
##              'factor': 1,
##              'isvar3d': False,
##              'mpas': 'timeMonthly_avg_sensibleHeatFlux',
##              'clevels': [-250, -200, -180, -160, -140,  -120, -100, -80, -60, -40, -20, -10, 0],
##              'cIndices': [0, 28, 40, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255],
##              'colormap': cmocean.cm.solar_r},
##             {'name': 'latentHeatFlux',
##              'title': 'Latent Heat Flux',
##              'units': 'W/m$^2$',
##              'factor': 1,
##              'isvar3d': False,
##              'mpas': 'timeMonthly_avg_latentHeatFlux',
##              'clevels': [-250, -200, -180, -160, -140,  -120, -100, -80, -60, -40, -20, -10, 0],
##              'cIndices': [0, 28, 40, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255],
##              'colormap': cmocean.cm.solar_r},
#             {'name': 'totalHeatFlux',
#              'title': 'Total Heat (sensible+latent+netLR+netSR) Flux',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'totalHeatFlux',
##              'mpas': None,
#              'clevels': [-250, -200, -180, -160, -140,  -120, -100, -80, -60, -40, -20, -10, 0],
#              'cIndices': [0, 28, 40, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255],
#              'colormap': cmocean.cm.solar_r},
#             {'name': 'seaIceFreshWaterFlux',
#              'title': 'Sea ice FW flux',
#              'units': '10$^{-6}$ kg m$^-2$ s$^-1$',
#              'factor': 1e6,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_seaIceFreshWaterFlux',
#              'clevels': [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150],
#              'cIndices': colorIndices,
#              'colormap': plt.get_cmap('PuOr_r')}
#              #'colormap': plt.get_cmap('BrBG_r')}
#             ]
#
#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'title': 'Maximum MLD',
#              'units': 'm',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthlyMax_max_dThreshMLD',
#              'clevels': [20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 250, 300], # summer months
#              #'clevels': [50, 80, 100, 120, 150, 180, 250, 300, 400, 500, 800, 1000, 1200], # winter months
#              'cIndices': colorIndices,
#              'colormap': plt.get_cmap('viridis')},
#            ]
#   Sea ice variables
modelComp = 'ice'
modelName = 'mpassi'
mpasFile = 'timeSeriesStatsMonthly'
variables = [
             {'name': 'iceArea',
              'title': 'Sea Ice Concentration',
              'units': '%',
              'factor': 100,
              'isvar3d': False,
              'mpas': 'timeMonthly_avg_iceAreaCell',
              #'clevels': [15, 30, 50, 80, 90, 95, 97, 98, 99, 100],
              'clevels': [10, 15, 30, 50, 80, 90, 95, 97, 98, 99, 100],
              'cIndices': colorIndices,
              #'colormap': cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
              #                                 (0.169, 0.435, 0.223), (0.455, 0.478, 0.196), (0.757, 0.474, 0.435),\
              #                                 (0.827, 0.561, 0.772), (0.761, 0.757, 0.949), (0.808, 0.921, 0.937)])},
              'colormap': cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
                                               (0,     0.4,   0.4),   (0.169, 0.435, 0.223), (0.455, 0.478, 0.196),\
                                               (0.757, 0.474, 0.435), (0.827, 0.561, 0.772), (0.761, 0.757, 0.949),\
                                               (0.808, 0.921, 0.937)])},
             {'name': 'iceVolume',
              'title': 'Sea Ice Thickness',
              'units': 'm',
              'factor': 1,
              'isvar3d': False,
              'mpas': 'timeMonthly_avg_iceVolumeCell',
              'clevels': [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
              'cIndices': colorIndices,
              'colormap': plt.get_cmap('YlGnBu_r')},
#             {'name': 'iceDivergence',
#              'title': 'Sea Ice divergence',
#              'units': '%/day',
#              'factor': 1,
#              'isvar3d': False,
#              'mpas': 'timeMonthly_avg_divergence',
#              'cIndices': colorIndices,
#              'clevels': [-10.0, -8.0, -6.0, -4.0, -2.0, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0, 8.0, 10.0],
#              'colormap': cmocean.cm.balance},
             {'name': 'iceSpeed',
              'title': 'Sea Ice speed',
              'units': 'm/s',
              'factor': 1,
              'isvar3d': False,
              'mpas': 'iceSpeed',
#              'mpas': None,
              'clevels': [0.04, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6],
              'cIndices': colorIndices,
              'colormap': cmocean.cm.solar}
            ]
#   Atmosphere variables
#modelComp = 'atm'
#modelName = 'eam'

plotDepthAvg = False
# zmins/zmaxs [m] (relevant for 3d variables and if plotDepthAvg = True)
#zmins = [-100.]
#zmaxs = [10.]
#zmins = [-600.]
#zmaxs = [-100.]
#zmins = [-8000.]
#zmaxs = [-600.]
zmins = [-100., -600., -8000.]
zmaxs = [10., -100., -600.]
# z levels [m] (relevant for 3d variables and if plotDepthAvg = False)
dlevels = [50.]
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
            print(f'\n    var: {varname}')
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

            # Read in long-term ensemble average for the specific month and variable
            if varname=='windStressSpeed':
                meanfileU  = f'{meandir}/windStressZonal_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfileU):
                    raise IOError(f'Native file {meanfileU} does not exist. Need to create it with interannual_ensemblemeans.bash')
                meanfileV  = f'{meandir}/windStressMeridional_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfileV):
                    raise IOError(f'Native file {meanfileV} does not exist. Need to create it with interannual_ensemblemeans.bash')
                dsmeanU  = xr.open_dataset(meanfileU).isel(Time=0)['timeMonthly_avg_windStressZonal']
                dsmeanV  = xr.open_dataset(meanfileV).isel(Time=0)['timeMonthly_avg_windStressMeridional']
                dsmean  = 0.5 * np.sqrt(dsmeanU**2  + dsmeanV**2)
            elif varname=='iceSpeed':
                meanfileU  = f'{meandir}/uVelocityGeo_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfileU):
                    raise IOError(f'Native file {meanfileU} does not exist. Need to create it with interannual_ensemblemeans.bash')
                meanfileV  = f'{meandir}/vVelocityGeo_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfileV):
                    raise IOError(f'Native file {meanfileV} does not exist. Need to create it with interannual_ensemblemeans.bash')
                dsmeanU  = xr.open_dataset(meanfileU).isel(Time=0)['timeMonthly_avg_uVelocityGeo']
                dsmeanV  = xr.open_dataset(meanfileV).isel(Time=0)['timeMonthly_avg_vVelocityGeo']
                dsmean  = 0.5 * np.sqrt(dsmeanU**2  + dsmeanV**2)
            elif varname=='totalHeatFlux':
                meanfileSens = f'{meandir}/sensibleHeatFlux_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfileSens):
                    raise IOError(f'Native file {meanfileSens} does not exist. Need to create it with interannual_ensemblemeans.bash')
                meanfileLate = f'{meandir}/latentHeatFlux_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfileLate):
                    raise IOError(f'Native file {meanfileLate} does not exist. Need to create it with interannual_ensemblemeans.bash')
                meanfileLRdown = f'{meandir}/longWaveHeatFluxDown_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfileLRdown):
                    raise IOError(f'Native file {meanfileLRdown} does not exist. Need to create it with interannual_ensemblemeans.bash')
                meanfileLRup = f'{meandir}/longWaveHeatFluxUp_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfileLRup):
                    raise IOError(f'Native file {meanfileLRup} does not exist. Need to create it with interannual_ensemblemeans.bash')
                meanfileSR = f'{meandir}/shortWaveHeatFlux_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfileSR):
                    raise IOError(f'Native file {meanfileSR} does not exist. Need to create it with interannual_ensemblemeans.bash')
                dsmeanSensible= xr.open_dataset(meanfileSens).isel(Time=0)['timeMonthly_avg_sensibleHeatFlux']
                dsmeanLatent  = xr.open_dataset(meanfileLate).isel(Time=0)['timeMonthly_avg_latentHeatFlux']
                dsmeanLRdown  = xr.open_dataset(meanfileLRdown).isel(Time=0)['timeMonthly_avg_longWaveHeatFluxDown']
                dsmeanLRup    = xr.open_dataset(meanfileLRup).isel(Time=0)['timeMonthly_avg_longWaveHeatFluxUp']
                dsmeanSR      = xr.open_dataset(meanfileSR).isel(Time=0)['timeMonthly_avg_shortWaveHeatFlux']
                dsmean = dsmeanSensible + dsmeanLatent + dsmeanLRdown + dsmeanLRup + dsmeanSR
            elif varname=='streamlines' or varname=='velSpeed':
                dsmean = None
            elif varname=='barotropicStreamfunction':
                meanfile = f'{meandir}/{varname}_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfile):
                    raise IOError(f'Native file {meanfile} does not exist. Need to create it with interannualBSF_ensemblemeans.bash')
                dsmean = xr.open_dataset(meanfile)[varmpasname]
            elif varname=='maxMLD':
                meanfile = f'{meandir}/dThreshMLDMax_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfile):
                    raise IOError(f'Native file {meanfile} does not exist. Need to create it with interannual_ensemblemeans.bash')
                dsmean = xr.open_dataset(meanfile).isel(Time=0)[varmpasname]
            elif varname=='iceArea':
                meanfile = f'{meandir}/iceAreaCell_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfile):
                    raise IOError(f'Native file {meanfile} does not exist. Need to create it with interannual_ensemblemeans.bash')
                dsmean = xr.open_dataset(meanfile).isel(Time=0)[varmpasname]
            elif varname=='iceVolume':
                meanfile = f'{meandir}/iceVolumeCell_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfile):
                    raise IOError(f'Native file {meanfile} does not exist. Need to create it with interannual_ensemblemeans.bash')
                dsmean = xr.open_dataset(meanfile).isel(Time=0)[varmpasname]
            else:
                meanfile = f'{meandir}/{varname}_ensembleMean_{im:02d}_{startYear[0]}_{endYear[0]}.nc'
                if not os.path.isfile(meanfile):
                    raise IOError(f'Native file {meanfile} does not exist. Need to create it with interannual_ensemblemeans.bash')
                dsmean = xr.open_dataset(meanfile).isel(Time=0)[varmpasname]

            if varname=='streamlines' or varname=='velSpeed':
                # Regrid velocityZonal and velocityMeridional (if necessary)
                if plotDepthAvg:
                    # Compute the depth average first and then regrid
                    for iz in range(len(zmins)):
                        zmin = zmins[iz]
                        zmax = zmaxs[iz]

                        # Check if regridded uLow file exists
                        fileHead = f'velocityZonalDepthAvg_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}'
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
                        fileHead = f'velocityMeridionalDepthAvg_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}'
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
                        fileHead = f'velocityZonalDepthAvg_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}'
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
                        fileHead = f'velocityMeridionalDepthAvg_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}'
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
                    dsmean = None
                    for iz in range(len(zmins)):
                        zmin = zmins[iz]
                        zmax = zmaxs[iz]

                        figtitleLow   = f'LC composite, {vartitle} (avg over z=({np.int32(zmin)}, {np.int32(zmax)}) m), month={im}'
                        figtitleHigh  = f'HC Composite, {vartitle} (avg over z=({np.int32(zmin)}, {np.int32(zmax)}) m), month={im}'
                        figfileLow  = f'{figdir}/{varname}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.png'
                        figfileHigh = f'{figdir}/{varname}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.png'

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
                            infileLow = f'{indir}/{varname}DepthAvg_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                            if not os.path.isfile(infileLow):
                                raise IOError(f'Native file {infileLow} does not exist. Need to create it with compute_composites')
                            infileHigh = f'{indir}/{varname}DepthAvg_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                            if not os.path.isfile(infileHigh):
                                raise IOError(f'Native file {infileHigh} does not exist. Need to create it with compute_composites')
                            dsFieldLow  = xr.open_dataset(infileLow).isel(Time=0)[varmpasname]
                            dsFieldHigh = xr.open_dataset(infileHigh).isel(Time=0)[varmpasname]

                            fldLow  = varfactor * dsFieldLow.values
                            fldHigh = varfactor * dsFieldHigh.values

                            dotSize = 1.2 # this should go up as resolution decreases
                            make_scatter_plot(x, y, dotSize, figtitleLow, figfileLow, projectionName='NorthPolarStereo',
                                              lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                              fld=fldLow, ttestMask=None, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                            make_scatter_plot(x, y, dotSize, figtitleHigh, figfileHigh, projectionName='NorthPolarStereo',
                                              lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                              fld=fldHigh, ttestMask=None, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                else:
                    for iz in range(len(dlevels)):
                        figtitleLow   = f'LC composite, {vartitle} (z={z[zlevels[iz]]:5.1f} m), month={im}'
                        figtitleHigh  = f'HC composite, {vartitle} (z={z[zlevels[iz]]:5.1f} m), month={im}'
                        figfileLow  = f'{figdir}/{varname}_depth{int(dlevels[iz]):04d}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.png'
                        figfileHigh = f'{figdir}/{varname}_depth{int(dlevels[iz]):04d}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.png'

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
                            dsFieldLowStd  = xr.open_dataset(infileLowStd).isel(Time=0, nVertLevels=zlevels[iz])
                            dsFieldHighStd = xr.open_dataset(infileHighStd).isel(Time=0, nVertLevels=zlevels[iz])

                            topoMask = depthMask.isel(nVertLevels=zlevels[iz])
                            dsFieldLow  = dsFieldLow.where(topoMask, drop=False)
                            dsFieldHigh = dsFieldHigh.where(topoMask, drop=False)
                            dsFieldLowStd  = dsFieldLowStd.where(topoMask, drop=False)
                            dsFieldHighStd = dsFieldHighStd.where(topoMask, drop=False)

                            fldLow  = varfactor * dsFieldLow.values
                            fldHigh = varfactor * dsFieldHigh.values
                            fldLowStd  = varfactor * dsFieldLowStd[varmpasname].values
                            fldHighStd = varfactor * dsFieldHighStd[varmpasname].values
                            fldLowStd[np.where(fldLowStd<1e-15)] = np.nan
                            fldHighStd[np.where(fldHighStd<1e-15)] = np.nan
                            ndataLow = dsFieldLowStd.nind_data.values
                            ndataHigh = dsFieldHighStd.nind_data.values

                            # one-sample t-test
                            if dsmean is not None:
                                fldMean = dsmean.isel(nVertLevels=zlevels[iz])
                                fldMean = fldMean.where(topoMask, drop=False)
                                fldMean = varfactor * fldMean.values
                                tvalue = np.sqrt(ndataLow) * (fldLow - fldMean) / fldLowStd
                                mask_ttestLow = np.logical_and(tvalue<tcritical, tvalue>-tcritical)
                                print('min(mean), max(mean)')
                                print(np.nanmin(fldMean), np.nanmax(fldMean))
                                print('min(fldLow), max(fldLow)')
                                print(np.nanmin(fldLow), np.nanmax(fldLow))
                                print('min(tvalue), max(tvalue)')
                                print(np.nanmin(tvalue), np.nanmax(tvalue))
                                print(np.shape(np.where(mask_ttestLow)))
                                tvalue = np.sqrt(ndataHigh) * (fldHigh - fldMean) / fldHighStd
                                mask_ttestHigh = np.logical_and(tvalue<tcritical, tvalue>-tcritical)
                                print('min(fldHigh), max(fldHigh)')
                                print(np.nanmin(fldHigh), np.nanmax(fldHigh))
                                print('min(tvalue), max(tvalue)')
                                print(np.nanmin(tvalue), np.nanmax(tvalue))
                                print(np.shape(np.where(mask_ttestHigh)))
                            else:
                                mask_ttestLow = None
                                mask_ttestHigh = None

                            #dotSize = 1.2 # this should go up as resolution decreases
                            #make_scatter_plot(x, y, dotSize, figtitleLow, figfileLow, projectionName='NorthPolarStereo',
                            #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                            #                  fld=fldLow, ttestMask=mask_ttestLow, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                            #make_scatter_plot(x, y, dotSize, figtitleHigh, figfileHigh, projectionName='NorthPolarStereo',
                            #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                            #                  fld=fldHigh, ttestMask=mask_ttestHigh, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                            make_mosaic_plot(x, y, varfactor*dsFieldLow, mosaic_descriptor, figtitleLow, figfileLow, ttestMask=mask_ttestLow, showEdges=None,
                                             cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                                             projectionName='NorthPolarStereo', lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                            make_mosaic_plot(x, y, varfactor*dsFieldHigh, mosaic_descriptor, figtitleHigh, figfileHigh, ttestMask=mask_ttestHigh, showEdges=None,
                                             cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                                             projectionName='NorthPolarStereo', lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
            else:
                figtitleLow   = f'LC composite, {vartitle}, month={im}'
                figtitleHigh  = f'HC composite, {vartitle}, month={im}'
                figfileLow  = f'{figdir}/{varname}_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.png'
                figfileHigh = f'{figdir}/{varname}_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.png'

                if varname=='barotropicStreamfunction':
                    dsFieldLow  = xr.open_dataset(infileLow)[varmpasname]
                    dsFieldHigh = xr.open_dataset(infileHigh)[varmpasname]
                    dsFieldLowStd  = xr.open_dataset(infileLowStd)
                    dsFieldHighStd = xr.open_dataset(infileHighStd)
                # Piece of old code:
                #elif varname=='windStressSpeed':
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
                fldLowStd  = varfactor * dsFieldLowStd[varmpasname].values
                fldHighStd = varfactor * dsFieldHighStd[varmpasname].values
                fldLowStd[np.where(fldLowStd<1e-15)] = np.nan
                fldHighStd[np.where(fldHighStd<1e-15)] = np.nan
                ndataLow = dsFieldLowStd.nind_data.values
                ndataHigh = dsFieldHighStd.nind_data.values

                # one-sample t-test
                #tcritical = 1
                #tvalue = fldLow / fldLowStd
                #mask_ttestLow = tvalue<tcritical
                #tvalue = fldHigh / fldHighStd
                #mask_ttestHigh = tvalue<tcritical
                if dsmean is not None:
                    fldMean = varfactor * dsmean.values
                    tvalue = np.sqrt(ndataLow) * (fldLow - fldMean) / fldLowStd
                    mask_ttestLow = np.logical_and(tvalue<tcritical, tvalue>-tcritical)
                    print('min(mean), max(mean)')
                    print(np.nanmin(fldMean), np.nanmax(fldMean))
                    print('min(fldLow), max(fldLow)')
                    print(np.nanmin(fldLow), np.nanmax(fldLow))
                    print('min(tvalue), max(tvalue)')
                    print(np.nanmin(tvalue), np.nanmax(tvalue))
                    print(np.shape(np.where(mask_ttestLow)))
                    tvalue = np.sqrt(ndataHigh) * (fldHigh - fldMean) / fldHighStd
                    mask_ttestHigh = np.logical_and(tvalue<tcritical, tvalue>-tcritical)
                    print('min(fldHigh), max(fldHigh)')
                    print(np.nanmin(fldHigh), np.nanmax(fldHigh))
                    print('min(tvalue), max(tvalue)')
                    print(np.nanmin(tvalue), np.nanmax(tvalue))
                    print(np.shape(np.where(mask_ttestHigh)))
                else:
                    mask_ttestLow = None
                    mask_ttestHigh = None

                if varname=='iceVolume' or varname=='iceDivergence' or varname=='seaIceFreshWaterFlux':
                    # Mask fields where ice concentration < 15% (or =0%)
                    infile  = f'{indir}/iceArea_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                    if not os.path.isfile(infile):
                        raise IOError(f'File {infile} does not exist. Need to create it with compute_composites')
                    iceArea  = xr.open_dataset(infile).isel(Time=0)['timeMonthly_avg_iceAreaCell']
                    #fldLow[np.where(iceArea<0.15)]   = np.nan
                    fldLow[np.where(iceArea<1e-15)]   = np.nan
                    dsFieldLow = dsFieldLow.where(iceArea>1e-15)

                    infile = f'{indir}/iceArea_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                    if not os.path.isfile(infile):
                        raise IOError(f'File {infile} does not exist. Need to create it with compute_composites')
                    iceArea  = xr.open_dataset(infile).isel(Time=0)['timeMonthly_avg_iceAreaCell']
                    #fldHigh[np.where(iceArea<0.15)]   = np.nan
                    fldHigh[np.where(iceArea<1e-15)]   = np.nan
                    dsFieldHigh = dsFieldHigh.where(iceArea>1e-15)

                if varname=='iceSpeed': # iceSpeed is on vertices, so iceArea doesn't work for masking
                    fldLow[np.where(fldLow<1e-15)]    = np.nan
                    fldHigh[np.where(fldHigh<1e-15)]  = np.nan
                    dsFieldLow = dsFieldLow.where(dsFieldLow>1e-15)
                    dsFieldHigh = dsFieldHigh.where(dsFieldHigh>1e-15)

                #print(np.nanmin(fldLow), np.nanmax(fldLow))
                #print(np.nanmin(fldHigh), np.nanmax(fldHigh))

                dotSize = 1.2 # this should go up as resolution decreases
                if varname=='iceArea':
                    #fldLow[np.where(fldLow<0.15*varfactor)]   = np.nan
                    #fldHigh[np.where(fldHigh<0.15*varfactor)] = np.nan
                    fldLow[np.where(fldLow<1e-15)]   = np.nan
                    fldHigh[np.where(fldHigh<1e-15)] = np.nan
                    dsFieldLow = dsFieldLow.where(dsFieldLow>1e-15)
                    dsFieldHigh = dsFieldHigh.where(dsFieldHigh>1e-15)
                    # Plot iceArea with iceThickness contours (with scatter only):
                    #infile  = f'{indir}/iceVolume_{varRef}low_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                    #if not os.path.isfile(infile):
                    #    raise IOError(f'File {infile} does not exist. Need to create it with compute_composites')
                    #iceVol = xr.open_dataset(infile).isel(Time=0)['timeMonthly_avg_iceVolumeCell']
                    #make_scatter_plot(x, y, dotSize, figtitleLow, figfileLow, projectionName='NorthPolarStereo',
                    #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                    #                  fld=fldLow, ttestMask=mask_ttestLow, cmap=colormap, clevels=clevels, cindices=None, cbarLabel=varunits,
                    #                  contourfld=iceVol, contourLevels=[1, 2], contourColors=['w', 'k'])
                    ##make_scatter_plot(x, y, dotSize, figtitleLow, figfileLow, projectionName='NorthPolarStereo',
                    ##                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                    ##                  fld=fldLow, ttestMask=mask_ttestLow, cmap=colormap, clevels=clevels, cindices=None, cbarLabel=varunits)
                    make_mosaic_plot(x, y, varfactor*dsFieldLow, mosaic_descriptor, figtitleLow, figfileLow, ttestMask=mask_ttestLow, showEdges=None,
                                     cmap=colormap, clevels=clevels, cindices=None, cbarLabel=varunits,
                                     projectionName='NorthPolarStereo', lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                    #infile  = f'{indir}/iceVolume_{varRef}high_{climoMonths}_{regionNameShort}_M{im:02d}.nc'
                    #if not os.path.isfile(infile):
                    #    raise IOError(f'File {infile} does not exist. Need to create it with compute_composites')
                    #iceVol = xr.open_dataset(infile).isel(Time=0)['timeMonthly_avg_iceVolumeCell']
                    #make_scatter_plot(x, y, dotSize, figtitleHigh, figfileHigh, projectionName='NorthPolarStereo',
                    #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                    #                  fld=fldHigh, ttestMask=mask_ttestHigh, cmap=colormap, clevels=clevels, cindices=None, cbarLabel=varunits,
                    #                  contourfld=iceVol, contourLevels=[1, 2], contourColors=['w', 'k'])
                    ##make_scatter_plot(x, y, dotSize, figtitleHigh, figfileHigh, projectionName='NorthPolarStereo',
                    ##                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                    ##                  fld=fldHigh, ttestMask=mask_ttestHigh, cmap=colormap, clevels=clevels, cindices=None, cbarLabel=varunits)
                    make_mosaic_plot(x, y, varfactor*dsFieldHigh, mosaic_descriptor, figtitleHigh, figfileHigh, ttestMask=mask_ttestHigh, showEdges=None,
                                     cmap=colormap, clevels=clevels, cindices=None, cbarLabel=varunits,
                                     projectionName='NorthPolarStereo', lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                else:
                    #make_scatter_plot(x, y, dotSize, figtitleLow, figfileLow, projectionName='NorthPolarStereo',
                    #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                    #                  fld=fldLow, ttestMask=mask_ttestLow, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                    #make_scatter_plot(x, y, dotSize, figtitleHigh, figfileHigh, projectionName='NorthPolarStereo',
                    #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                    #                  fld=fldHigh, ttestMask=mask_ttestHigh, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
                    make_mosaic_plot(x, y, varfactor*dsFieldLow, mosaic_descriptor, figtitleLow, figfileLow, ttestMask=mask_ttestLow, showEdges=None,
                                     cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                                     projectionName='NorthPolarStereo', lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                    make_mosaic_plot(x, y, varfactor*dsFieldHigh, mosaic_descriptor, figtitleHigh, figfileHigh, ttestMask=mask_ttestHigh, showEdges=None,
                                     cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                                     projectionName='NorthPolarStereo', lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
