from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import cmocean


from make_plots import make_scatter_plot, make_mosaic_descriptor, make_mosaic_plot


# Settings for lcrc
#meshName = 'oEC60to30v3'
#meshfile = f'/lcrc/group/acme/public_html/inputdata/ocn/mpas-o/{meshName}/oEC60to30v3_60layer.170905.nc'
##runname = '20200116.Redioff.GMPAS-IAF.oEC60to30v3.anvil'
##runname = '20200122.RedionLowTapering.GMPAS-IAF.oEC60to30v3.anvil'
#runname = '20200122.RedionHighTapering.GMPAS-IAF.oEC60to30v3.anvil'
#modeldir = F'/lcrc/group/acme/milena/acme_scratch/anvil/{runname}/run'
#
#meshName = 'ECwISC30to60E1r2'
#meshfile = f'/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/{meshName}/ocean.ECwISC30to60E1r2.200408.nc'
#runname = '20210614.A_WCYCL1850-DIB-ISMF_CMIP6.ne30_ECwISC30to60E1r2.anvil.DIBbugFixMGM'
#modeldir = f'/lcrc/group/acme/ac.sprice/acme_scratch/anvil/{runname}/archive/ocn/hist'
#
#meshName = 'SOwISC12to60E2r4'
#meshfile = f'/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/{meshName}/ocean.SOwISC12to60E2r4.210107.nc'
#runname = '20211026.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.GMtaperMinKappa500plusRedi'
#runname = '20220223.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.CryoBranchBaseline'
#runname = '20220401.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.CryoBranchGMHorResFun'
#modeldir = f'/lcrc/group/e3sm/ac.sprice/scratch/chrys/{runname}/run'
#runname = '20220418.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.CryoBranchHorTaperGMRediconstant'
#runname = '20220418.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.CryoBranchHorTaperGMVisbeckRediequalGM'
#runname = '20220419.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.CryoBranchRediEqGM2'
#modeldir = f'/lcrc/group/e3sm/ac.milena/scratch/chrys/{runname}/run'
#
#meshName = 'EC30to60E2r2'
#meshfile = f'/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/{meshName}/ocean.EC30to60E2r2.210210.nc'
#runname = 'v2Visbeck_RediequalGM_lowMaxKappa.LR.picontrol'
#modeldir = f'/lcrc/group/e3sm/ac.milena/E3SMv2/{runname}/run'
#
#meshName = 'EC30to60E2r2'
#meshfile = f'/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/{meshName}/ocean.EC30to60E2r2.210210.nc'
#runname = '20241030.EC30to60_test.anvil'
#modeldir = f'/lcrc/group/e3sm/ac.vanroekel/scratch/anvil/{runname}/run'
#isShortTermArchive = False

# Settings for nersc
#meshName = 'ARRM10to60E2r1'
#meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runname = 'E3SM-Arcticv2.1_historical0151'
##runname = 'E3SMv2.1B60to10rA02' # 1950-control
#modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runname}/archive'
#isShortTermArchive = True

# Settings for erdc.hpc.mil
meshName = 'ARRM10to60E2r1'
meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runname = 'E3SMv2.1B60to10rA02'
modeldir = f'/p/cwfs/milena/{runname}/archive'
#meshName = 'ARRMwISC3to18E3r1'
#meshfile = f'/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/{meshName}/mpaso.ARRMwISC3to18E3r1.rstFromG-pm.241022.nc'
#runname = 'E3SMv3G18to3'
#modeldir = f'/p/global/milena/{runname}/archive'
isShortTermArchive = True # if True, {modelname}/hist will be appended to modeldir

year = 20
months = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
months = [9, 10]
months = [6]

# z levels [m] (relevant for 3d variables)
#dlevels = [100.0, 250.0]
#dlevels = [50.0, 100.0, 250.0, 500.0, 3000.0]
#dlevels = [50., 100.0, 300.0, 800.0]
#dlevels = [0., 100., 600., 2000.]
dlevels = [0., 50.]

colorIndices = [0, 14, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]

showedges = None
projectionNH = 'NorthPolarStereo'
lon0NH = 10.0
lon1NH = 50.0
dlonNH = 5.0
lat0NH = 69.0
lat1NH = 82.0
dlatNH = 2.0
projectionSH = 'SouthPolarStereo'
lon0SH = -180.0
lon1SH = 180.0
dlonSH = 20.0
lat0SH = -55.0
lat1SH = -90.0
dlatSH = 10.0
projectionGlobal = 'Robinson'
lon0 = -180.0
lon1 = 180.0
dlon = 40.0
lat0 = -90.0
lat1 = 90.0
dlat = 20.0

############# MPAS-Ocean fields
#modelname = 'ocn'
#modelnameOut = 'ocean'
#modelcomp = 'mpaso'

#mpasFile = 'highFrequencyOutput'
#mpasFileDayformat = '01_00.00.00'
#timeIndex = 334
#variables = [
#             {'name': 'temperatureAtSurface',
#              'mpasvarname': 'temperatureAtSurface',
#              'title': 'SST',
#              'units': '$^\circ$C',
#              'factor': 1,
#              'colormap': plt.get_cmap('RdBu_r'),
#              'clevels':   [-1.8, -1.6, -0.5, 0.0, 0.5, 2.0, 4.0, 8.0, 12., 16., 22.],
#              'clevelsNH': [-1.8, -1.6, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0, 8.0, 12.],
#              'clevelsSH': [-1.8, -1.6, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0, 8.0, 12.],
#              'is3d': False},
#             #{'name': 'temperatureAt250m',
#             # 'mpasvarname': 'temperatureAt250m',
#             # 'title': 'Temperature at 250m',
#             # 'units': '$^\circ$C',
#             # 'factor': 1,
#             # 'colormap': plt.get_cmap('RdBu_r'),
#             # 'clevels':   [-1.8, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 8.0, 12., 16., 22.],
#             # 'clevelsNH': [-1.8, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0, 8.0, 12., 16.],
#             # 'clevelsSH': [-1.8, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 6.0, 8.0, 12., 16.],
#             # 'is3d': False},
#             #{'name': 'temperatureAtBottom',
#             # 'mpasvarname': 'temperatureAtBottom',
#             # 'title': 'Temperature at bottom',
#             # 'units': '$^\circ$C',
#             # 'factor': 1,
#             # 'colormap': plt.get_cmap('RdBu_r'),
#             # 'clevels':   [-1.8, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
#             # 'clevelsNH': [-1.8, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
#             # 'clevelsSH': [-1.8, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
#             # 'is3d': False},
#             {'name': 'salinityAtSurface',
#              'mpasvarname': 'salinityAtSurface',
#              'title': 'SSS',
#              'units': 'psu',
#              'factor': 1,
#              'colormap': cmocean.cm.haline,
#              'clevels':   [27., 28., 29., 29.5, 30., 30.5, 31., 32., 33., 34., 35.],
#              'clevelsNH': [27., 28., 29., 29.5, 30., 30.5, 31., 32., 33., 34., 35.],
#              'clevelsSH': [31., 31.5, 32., 32.3, 32.6, 32.9, 33.2, 33.5, 34., 34.5, 35.],
#             # 'is3d': False},
#             #{'name': 'salinityAt250m',
#             # 'mpasvarname': 'salinityAt250m',
#             # 'title': 'Salinity at 250m',
#             # 'units': 'psu',
#             # 'factor': 1,
#             # 'colormap': cmocean.cm.haline,
#             # 'clevels':   [30., 30.5, 31., 32., 32.5, 33., 33.5, 34., 34.5, 35., 36.],
#             # 'clevelsNH':   [30., 30.5, 31., 32., 32.5, 33., 33.5, 34., 34.5, 35., 36.],
#             # 'clevelsSH': [31., 31.5, 32., 32.3, 32.6, 32.9, 33.2, 33.5, 34., 34.5, 35.],
#             # 'is3d': False},
#             #{'name': 'salinityAtBottom',
#             # 'mpasvarname': 'salinityAtBottom',
#             # 'title': 'Salinity at bottom',
#             # 'units': 'psu',
#             # 'factor': 1,
#             # 'colormap': cmocean.cm.haline,
#             # 'clevels': [31., 31.5, 32., 32.3, 32.6, 32.9, 33.2, 33.5, 34., 34.5, 35.],
#             # 'clevelsNH': [31., 31.5, 32., 32.3, 32.6, 32.9, 33.2, 33.5, 34., 34.5, 35.],
#             # 'clevelsSH': [31., 31.5, 32., 32.3, 32.6, 32.9, 33.2, 33.5, 34., 34.5, 35.],
#              'is3d': False}
#            ]

#mpasFile = 'timeSeriesStatsMonthlyMax'
#mpasFileDayformat = '01'
#timeIndex = 0
#variables = [
#             {'name': 'maxMLD',
#              'mpasvarname': 'timeMonthlyMax_max_dThreshMLD',
#              'title': 'Maximum MLD',
#              'units': 'm',
#              'factor': 1,
#              'colormap': plt.get_cmap('viridis'),
#              'clevels': [10., 50., 80., 100., 150., 200., 300., 400., 800., 1200., 2000.],
#              'clevelsNH': [50., 100., 150., 200., 300., 500., 800., 1200., 1500., 2000., 3000.],
#              'clevelsSH': [10., 50., 80., 100., 150., 200., 300., 400., 800., 1200., 2000.],
#              'is3d': False}
#            ]

#mpasFile = 'timeSeriesStatsMonthly'
#mpasFileDayformat = '01'
#timeIndex = 0
#variables = [
#             #{'name': 'GMkappa',
#             # 'mpasvarname': 'timeMonthly_avg_gmKappaScaling',
#             # 'title': 'GM Kappa (w/ horTaper & kappaScaling)',
#             # 'units': 'm$^2$/s',
#             # 'factor': 1,
#             # 'colormap': plt.get_cmap('terrain'),
#             # 'clevels':   [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'clevelsNH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'clevelsSH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'is3d': True},
#             #{'name': 'gmBolusKappa',
#             # 'mpasvarname': 'timeMonthly_avg_gmBolusKappa',
#             # 'title': 'gmBolusKappa',
#             # 'units': 'm$^2$/s',
#             # 'factor': 1,
#             # 'colormap': plt.get_cmap('terrain'),
#             # 'clevels':   [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'clevelsNH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'clevelsSH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'is3d': False},
#             #{'name': 'gmKappaScaling',
#             # 'mpasvarname': 'timeMonthly_avg_gmKappaScaling',
#             # 'title': 'gmKappaScaling',
#             # 'units': '',
#             # 'factor': 1,
#             # 'colormap': plt.get_cmap('terrain'),
#             # 'clevels':   [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#             # 'clevelsNH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#             # 'clevelsSH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#             # 'is3d': True},
#             #{'name': 'gmHorizontalTaper',
#             # 'mpasvarname': 'timeMonthly_avg_gmHorizontalTaper',
#             # 'title': 'gmHorizontalTaper',
#             # 'units': '',
#             # 'factor': 1,
#             # 'colormap': plt.get_cmap('terrain'),
#             # 'clevels':   [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#             # 'clevelsNH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#             # 'clevelsSH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#             # 'is3d': False},
#             #{'name': 'Redikappa',
#             # 'mpasvarname': 'timeMonthly_avg_RediKappa',
#             # 'title': 'Redi Kappa (w/ horTaper)',
#             # 'units': 'm$^2$/s',
#             # 'factor': 1,
#             # 'colormap': plt.get_cmap('terrain'),
#             # 'clevels':   [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'clevelsNH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'clevelsSH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'is3d': False},
#             #{'name': 'RedikappaNoTaper',
#             # 'mpasvarname': 'timeMonthly_avg_RediKappa',
#             # 'title': 'RediKappa',
#             # 'units': 'm$^2$/s',
#             # 'factor': 1,
#             # 'colormap': plt.get_cmap('terrain'),
#             # 'clevels':   [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'clevelsNH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'clevelsSH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
#             # 'is3d': False},
#             #{'name': 'RediHorizontalTaper',
#             # 'mpasvarname': 'timeMonthly_avg_RediHorizontalTaper',
#             # 'title': 'RediHorizontalTaper',
#             # 'units': '',
#             # 'factor': 1,
#             # 'colormap': plt.get_cmap('terrain'),
#             # 'clevels':   [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#             # 'clevelsNH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#             # 'clevelsSH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#             # 'is3d': False},
#             {'name': 'temperature',
#              'mpasvarname': 'timeMonthly_avg_activeTracers_temperature',
#              'title': 'Potential Temperature',
#              'units': '$^\circ$C',
#              'factor': 1,
#              'colormap': cmocean.cm.thermal,
#              'clevels':   [-1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 22.0],
#              'clevelsNH': [-1.0, -0.5, 0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,  6.0,  7.0,  8.0],
#              'clevelsSH': [-1.0, -0.5, 0.0, 0.5, 2.0, 2.5, 3.0, 3.5,  4.0,  6.0,  8.0, 10., 12.],
#              'is3d': True},
#             {'name': 'salinity',
#              'mpasvarname': 'timeMonthly_avg_activeTracers_salinity',
#              'title': 'Salinity',
#              'units': 'psu',
#              'factor': 1,
#              'colormap': cmocean.cm.haline,
#              'clevels': [31.0, 33.0, 34.2,  34.4,  34.6, 34.7,  34.8,  34.87, 34.9, 34.95, 35.0, 35.2, 35.4],
#              'clevelsNH': [31.0, 33.0, 34.2,  34.4,  34.6, 34.7,  34.8,  34.87, 34.9, 34.95, 35.0, 35.2, 35.4],
#              'clevelsSH': [31.0, 33.0, 34.2,  34.4,  34.6, 34.7,  34.8,  34.87, 34.9, 34.95, 35.0, 35.2, 35.4],
#              'is3d': True},
#             #{'name': 'GMnormalVel',
#             # 'mpasvarname': 'timeMonthly_avg_normalGMBolusVelocity',
#             # 'title': 'GM Normal Velocity',
#             # 'units': 'cm/s',
#             # 'factor': 1e2,
#             # 'colormap': plt.get_cmap('RdBu_r'),
#             # 'clevels':   [-2, -1, -0.5, -0.25, -0.05, 0., 0.05, 0.25, 0.5, 1, 2],
#             # 'clevelsNH': [-2, -1, -0.5, -0.25, -0.05, 0., 0.05, 0.25, 0.5, 1, 2],
#             # 'clevelsSH': [-2, -1, -0.5, -0.25, -0.05, 0., 0.05, 0.25, 0.5, 1, 2],
#             # 'is3d': True}
#            ]

############# MPAS-Seaice fields
modelname = 'ice'
modelnameOut = 'seaice'
modelcomp = 'mpassi'
mpasFile = 'timeSeriesStatsMonthly'
mpasFileDayformat = '01'
timeIndex = 0
variables = [
             {'name': 'iceArea',
              'mpasvarname': 'timeMonthly_avg_iceAreaCell',
              'title': 'Sea Ice Concentration',
              'units': '%',
              'factor': 1e2,
              'colormap': cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
                                               (0,     0.4,   0.4),   (0.169, 0.435, 0.223), (0.455, 0.478, 0.196),\
                                               (0.757, 0.474, 0.435), (0.827, 0.561, 0.772), (0.761, 0.757, 0.949),\
                                               (0.808, 0.921, 0.937)]),
              'clevels':   [10, 15, 30, 50, 80, 90, 95, 97, 98, 99, 100],
              'clevelsNH': [10, 15, 30, 50, 80, 90, 95, 97, 98, 99, 100],
              'clevelsSH': [10, 15, 30, 50, 80, 90, 95, 97, 98, 99, 100],
              'is3d': False}
            ]

if isShortTermArchive:
    modeldir = f'{modeldir}/{modelname}/hist'

figdir = f'./{modelnameOut}_native/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

# Info about MPAS mesh
dsMesh = xr.open_dataset(meshfile)
lonCell = dsMesh.lonCell.values
latCell = dsMesh.latCell.values
lonCell = 180/np.pi*lonCell
latCell = 180/np.pi*latCell
lonEdge = dsMesh.lonEdge.values
latEdge = dsMesh.latEdge.values
lonEdge = 180/np.pi*lonEdge
latEdge = 180/np.pi*latEdge
lonVertex = dsMesh.lonVertex.values
latVertex = dsMesh.latVertex.values
lonVertex = 180/np.pi*lonVertex
latVertex = 180/np.pi*latVertex
depth = dsMesh.bottomDepth
# Find model levels for each depth level (relevant if plotDepthAvg = False)
z = dsMesh.refBottomDepth
zlevels = np.zeros(np.shape(dlevels), dtype=np.int64)
for id in range(len(dlevels)):
    dz = np.abs(z.values-dlevels[id])
    zlevels[id] = np.argmin(dz)
# Make depth mask
maxLevelCell = dsMesh.maxLevelCell
nVertLevels = dsMesh.sizes['nVertLevels']
vertIndex = xr.DataArray.from_dict({'dims': ('nVertLevels',), 'data': np.arange(nVertLevels)})
depthMask = (vertIndex < maxLevelCell).transpose('nCells', 'nVertLevels')
# restart files are missing this attribute that is needed for mosaic,
# so for now adding this manually:
dsMesh.attrs['is_periodic'] = 'NO'
mosaic_descriptorGlobal = make_mosaic_descriptor(dsMesh, projectionGlobal)
mosaic_descriptorNH = make_mosaic_descriptor(dsMesh, projectionNH)
mosaic_descriptorSH = make_mosaic_descriptor(dsMesh, projectionSH)

#figtitle = f'Bottom depth {meshName}'
#figfileGlobal = f'{figdir}/DepthGlobal_{meshName}.png'
#figfileNH = f'{figdir}/DepthNH_{meshName}.png'
#figfileSH = f'{figdir}/DepthSH_{meshName}.png'
#clevels = [0., 10., 50., 100., 250., 500.0, 750., 1000., 2000., 3000., 5000.]
#colormap = cmocean.cm.deep_r
#dotSize = 0.25
#make_scatter_plot(lonCell, latCell, dotSize, figtitle, figfileGlobal, projectionName='Robinson',
#                  lon0=-180, lon1=180, dlon=40, lat0=-90, lat1=90, dlat=20,
#                  fld=depth, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel='m')
#dotSize = 5.0 # this should go up as resolution decreases # for ARRM
#dotSize = 25.0 # this should go up as resolution decreases # for LR
#make_scatter_plot(lonCell, latCell, dotSize, figtitle, figfileNH, projectionName='NorthPolarStereo',
#                  lon0=lon0NH, lon1=lon1NH, dlon=20.0, lat0=lat0NH, lat1=lat1NH, dlat=10.0,
#                  fld=depth, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel='m')
#dotSize = 25.0 # this should go up as resolution decreases
#make_scatter_plot(lonCell, latCell, dotSize, figtitle, figfileSH, projectionName='SouthPolarStereo',
#                  lon0=lon0SH, lon1=lon1SH, dlon=20.0, lat0=lat0SH, lat1=lat1SH, dlat=10.0,
#                  fld=depth, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel='m')

for month in months:
    modelfile = f'{modeldir}/{runname}.{modelcomp}.hist.am.{mpasFile}.{year:04d}-{month:02d}-{mpasFileDayformat}.nc'
    #modelfile = f'{modeldir}/{modelcomp}.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-{mpasFileDayformat}.nc' # old (v1) filename format
    ds = xr.open_dataset(modelfile).isel(Time=timeIndex)
    for var in variables:
        varname = var['name']
        mpasvarname = var['mpasvarname']
        factor = var['factor']
        clevels = var['clevels']
        clevelsNH = var['clevelsNH']
        clevelsSH = var['clevelsSH']
        colormap = var['colormap']
        vartitle = var['title']
        varunits = var['units']

        if varname=='GMkappa' or varname=='Redikappa' or \
           varname=='gmBolusKappa' or varname=='gmKappaScaling' or \
           varname=='gmHorizontalTaper' or varname=='RedikappaNoTaper' or \
           varname=='RediHorizontalTaper' or varname=='GMnormalVel':
            lon = lonEdge
            lat = latEdge
        else:
            lon = lonCell
            lat = latCell

        if var['is3d']:
            for iz in range(len(dlevels)):
                figtitle = f'{vartitle}, z={z[zlevels[iz]]:5.1f} m, year={year}, month={month}'
                figfileGlobal = f'{figdir}/{varname}Global_depth{int(dlevels[iz]):04d}_{runname}_{year:04d}-{month:02d}timeIndex{timeIndex:02d}.png'
                figfileNH = f'{figdir}/{varname}NH_depth{int(dlevels[iz]):04d}_{runname}_{year:04d}-{month:02d}timeIndex{timeIndex:02d}.png'
                figfileSH = f'{figdir}/{varname}SH_depth{int(dlevels[iz]):04d}_{runname}_{year:04d}-{month:02d}timeIndex{timeIndex:02d}.png'

                fld = ds[mpasvarname].isel(nVertLevels=zlevels[iz])
                topoMask = depthMask.isel(nVertLevels=zlevels[iz])
                fld = fld.where(topoMask, drop=False)
                fld = factor*fld
                if varname=='GMkappa':
                    kappa = ds['timeMonthly_avg_gmBolusKappa']
                    kappa_horScaling = ds['timeMonthly_avg_gmHorizontalTaper']
                    fld = kappa * fld * kappa_horScaling
                print('varname=', varname, 'month=', month, 'zlev=', int(dlevels[iz]), 'fldmin=', np.min(fld), 'fldmax=', np.max(fld))

                #make_mosaic_plot(lon, lat, fld, mosaic_descriptorGlobal, figtitle, figfileGlobal, showEdges=None,
                #                 cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                #                 projectionName=projectionGlobal, lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
                #dotSize = 0.25
                #make_scatter_plot(lon, lat, dotSize, figtitle, figfileGlobal, projectionName=projectionGlobal,
                #  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                #  fld=fld, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)

                make_mosaic_plot(lon, lat, fld, mosaic_descriptorNH, figtitle, figfileNH, showEdges=showedges,
                                 cmap=colormap, clevels=clevelsNH, cindices=colorIndices, cbarLabel=varunits,
                                 projectionName=projectionNH, lon0=lon0NH, lon1=lon1NH, dlon=dlonNH, lat0=lat0NH, lat1=lat1NH, dlat=dlatNH)
                ##dotSize = 5.0 # this should go up as resolution decreases # for ARRM
                #dotSize = 25.0 # this should go up as resolution decreases # for LR
                #make_scatter_plot(lon, lat, dotSize, figtitle, figfileNH, projectionName=projectionNH,
                #  lon0=lon0NH, lon1=lon1NH, dlon=dlonNH, lat0=lat0NH, lat1=lat1NH, dlat=dlatNH,
                #  fld=fld, cmap=colormap, clevels=clevelsNH, cindices=colorIndices, cbarLabel=varunits)

                #make_mosaic_plot(lon, lat, fld, mosaic_descriptorSH, figtitle, figfileSH, showEdges=showedges,
                #                 cmap=colormap, clevels=clevelsSH, cindices=colorIndices, cbarLabel=varunits,
                #                 projectionName=projectionSH, lon0=lon0SH, lon1=lon1SH, dlon=dlonSH, lat0=lat0SH, lat1=lat1SH, dlat=dlatSH)
                #dotSize = 25.0 # this should go up as resolution decreases
                #make_scatter_plot(lon, lat, dotSize, figtitle, figfileSH, projectionName=projectionSH,
                #  lon0=lon0SH, lon1=lon1SH, dlon=dlonSH, lat0=lat0SH, lat1=lat1SH, dlat=dlatSH,
                #  fld=fld, cmap=colormap, clevels=clevelsSH, cindices=colorIndices, cbarLabel=varunits)

        else:
            figtitle = f'{vartitle}, year={year}, month={month}'
            figfileGlobal = f'{figdir}/{varname}Global_{runname}_{year:04d}-{month:02d}timeIndex{timeIndex:02d}.png'
            figfileNH = f'{figdir}/{varname}NH_{runname}_{year:04d}-{month:02d}timeIndex{timeIndex:02d}.png'
            figfileSH = f'{figdir}/{varname}SH_{runname}_{year:04d}-{month:02d}timeIndex{timeIndex:02d}.png'

            fld = ds[mpasvarname]
            fld = factor*fld
            if varname=='Redikappa':
                kappa_horScaling = ds['timeMonthly_avg_RediHorizontalTaper']
                fld = fld * kappa_horScaling
            print('varname=', varname, 'month=', month, 'fldmin=', np.min(fld), 'fldmax=', np.max(fld))

            if varname=='iceArea':
                cindices = None
            else:
                cindices = colorIndices

            #make_mosaic_plot(lon, lat, fld, mosaic_descriptorGlobal, figtitle, figfileGlobal, showEdges=None,
            #                 cmap=colormap, clevels=clevels, cindices=cindices, cbarLabel=varunits,
            #                 projectionName=projectionGlobal, lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
            #dotSize = 0.25
            #make_scatter_plot(lon, lat, dotSize, figtitle, figfileGlobal, projectionName=projectionGlobal,
            #  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
            #  fld=fld, cmap=colormap, clevels=clevels, cindices=cindices, cbarLabel=varunits)

            make_mosaic_plot(lon, lat, fld, mosaic_descriptorNH, figtitle, figfileNH, showEdges=showedges,
                             cmap=colormap, clevels=clevelsNH, cindices=cindices, cbarLabel=varunits,
                             projectionName=projectionNH, lon0=lon0NH, lon1=lon1NH, dlon=dlonNH, lat0=lat0NH, lat1=lat1NH, dlat=dlatNH)
            ##dotSize = 5.0 # this should go up as resolution decreases # for ARRM
            #dotSize = 25.0 # this should go up as resolution decreases # for LR
            #make_scatter_plot(lon, lat, dotSize, figtitle, figfileNH, projectionName=projectionNH,
            #  lon0=lon0NH, lon1=lon1NH, dlon=dlonNH, lat0=lat0NH, lat1=lat1NH, dlat=dlatNH,
            #  fld=fld, cmap=colormap, clevels=clevelsNH, cindices=cindices, cbarLabel=varunits)

            #make_mosaic_plot(lon, lat, fld, mosaic_descriptorSH, figtitle, figfileSH, showEdges=showedges,
            #                 cmap=colormap, clevels=clevelsSH, cindices=cindices, cbarLabel=varunits,
            #                 projectionName=projectionSH, lon0=lon0SH, lon1=lon1SH, dlon=dlonSH, lat0=lat0SH, lat1=lat1SH, dlat=dlatSH)
            #dotSize = 25.0 # this should go up as resolution decreases
            #make_scatter_plot(lon, lat, dotSize, figtitle, figfileSH, projectionName=projectionSH,
            #  lon0=lon0SH, lon1=lon1SH, dlon=dlonSH, lat0=lat0SH, lat1=lat1SH, dlat=dlatSH,
            #  fld=fld, cmap=colormap, clevels=clevelsSH, cindices=cindices, cbarLabel=varunits)
