from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cmocean

from make_plots import make_scatter_plot


# Settings for lcrc
#meshName = 'oEC60to30v3'
#meshfile = '/lcrc/group/acme/public_html/inputdata/ocn/mpas-o/{}/oEC60to30v3_60layer.170905.nc'.format(meshName)
##runname = '20200116.Redioff.GMPAS-IAF.oEC60to30v3.anvil'
##runname = '20200122.RedionLowTapering.GMPAS-IAF.oEC60to30v3.anvil'
#runname = '20200122.RedionHighTapering.GMPAS-IAF.oEC60to30v3.anvil'
#modeldir = '/lcrc/group/acme/milena/acme_scratch/anvil/{}/run'.format(runname)
#
#meshName = 'ECwISC30to60E1r2'
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/{}/ocean.ECwISC30to60E1r2.200408.nc'.format(meshName)
#runname = '20210614.A_WCYCL1850-DIB-ISMF_CMIP6.ne30_ECwISC30to60E1r2.anvil.DIBbugFixMGM'
#modeldir = '/lcrc/group/acme/ac.sprice/acme_scratch/anvil/{}/archive/ocn/hist'.format(runname)
#
#meshName = 'SOwISC12to60E2r4'
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/{}/ocean.SOwISC12to60E2r4.210107.nc'.format(meshName)
#runname = '20211026.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.GMtaperMinKappa500plusRedi'
#runname = '20220223.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.CryoBranchBaseline'
#runname = '20220401.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.CryoBranchGMHorResFun'
#modeldir = '/lcrc/group/e3sm/ac.sprice/scratch/chrys/{}/run'.format(runname)
#runname = '20220418.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.CryoBranchHorTaperGMRediconstant'
#runname = '20220418.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.CryoBranchHorTaperGMVisbeckRediequalGM'
#runname = '20220419.CRYO1850.ne30pg2_SOwISC12to60E2r4.chrysalis.CryoBranchRediEqGM2'
#modeldir = '/lcrc/group/e3sm/ac.milena/scratch/chrys/{}/run'.format(runname)
#
#meshName = 'EC30to60E2r2'
#meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/{}/ocean.EC30to60E2r2.210210.nc'.format(meshName)
#runname = 'v2Visbeck_RediequalGM_lowMaxKappa.LR.picontrol'
#modeldir = '/lcrc/group/e3sm/ac.milena/E3SMv2/{}/run'.format(runname)

# Settings for nersc
meshName = 'ARRM10to60E2r1'
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runname = 'E3SM-Arcticv2.1_historical0151'
#runname = 'E3SMv2.1B60to10rA02' # 1950-control
modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runname}'
# Note: the following two variables cannot be both True
isShortTermArchive = True

if isShortTermArchive:
    modeldir = f'{modeldir}/archive/ocn/hist'

figdir = './ocean_native/{}'.format(runname)
if not os.path.isdir(figdir):
    os.makedirs(figdir)

#year = 5
year = 1972
#months = [2, 8]
months = [8]

# z levels [m] (relevant for 3d variables)
#dlevels = [100.0, 250.0]
#dlevels = [50.0, 100.0, 250.0, 500.0, 3000.0]
#dlevels = [50., 100.0, 300.0, 800.0]
dlevels = [0.]

colorIndices = [0, 10, 28, 57, 85, 113, 142, 170, 198, 227, 242, 255]
lon0NH = -180.0
lon1NH = 180.0
lat0NH = 50.0
lat1NH = 90.0
lon0SH = -180.0
lon1SH = 180.0
lat0SH = -55.0
lat1SH = -90.0

pi2deg = 180/np.pi

variables = [
             #{'name': 'GMkappa',
             # 'mpasvarname': 'timeMonthly_avg_gmKappaScaling',
             # 'component': 'mpaso',
             # 'title': 'GM Kappa (w/ horTaper & kappaScaling)',
             # 'units': 'm$^2$/s',
             # 'factor': 1,
             # 'colormap': plt.get_cmap('terrain'),
             # 'clevels':   [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'clevelsNH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'clevelsSH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'is3d': True},
             #{'name': 'gmBolusKappa',
             # 'mpasvarname': 'timeMonthly_avg_gmBolusKappa',
             # 'component': 'mpaso',
             # 'title': 'gmBolusKappa',
             # 'units': 'm$^2$/s',
             # 'factor': 1,
             # 'colormap': plt.get_cmap('terrain'),
             # 'clevels':   [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'clevelsNH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'clevelsSH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'is3d': False},
             #{'name': 'gmKappaScaling',
             # 'mpasvarname': 'timeMonthly_avg_gmKappaScaling',
             # 'component': 'mpaso',
             # 'title': 'gmKappaScaling',
             # 'units': '',
             # 'factor': 1,
             # 'colormap': plt.get_cmap('terrain'),
             # 'clevels':   [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             # 'clevelsNH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             # 'clevelsSH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             # 'is3d': True},
             #{'name': 'gmHorizontalTaper',
             # 'mpasvarname': 'timeMonthly_avg_gmHorizontalTaper',
             # 'component': 'mpaso',
             # 'title': 'gmHorizontalTaper',
             # 'units': '',
             # 'factor': 1,
             # 'colormap': plt.get_cmap('terrain'),
             # 'clevels':   [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             # 'clevelsNH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             # 'clevelsSH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             # 'is3d': False},
             #{'name': 'Redikappa',
             # 'mpasvarname': 'timeMonthly_avg_RediKappa',
             # 'component': 'mpaso',
             # 'title': 'Redi Kappa (w/ horTaper)',
             # 'units': 'm$^2$/s',
             # 'factor': 1,
             # 'colormap': plt.get_cmap('terrain'),
             # 'clevels':   [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'clevelsNH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'clevelsSH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'is3d': False},
             #{'name': 'RedikappaNoTaper',
             # 'mpasvarname': 'timeMonthly_avg_RediKappa',
             # 'component': 'mpaso',
             # 'title': 'RediKappa',
             # 'units': 'm$^2$/s',
             # 'factor': 1,
             # 'colormap': plt.get_cmap('terrain'),
             # 'clevels':   [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'clevelsNH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'clevelsSH': [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800],
             # 'is3d': False},
             #{'name': 'RediHorizontalTaper',
             # 'mpasvarname': 'timeMonthly_avg_RediHorizontalTaper',
             # 'component': 'mpaso',
             # 'title': 'RediHorizontalTaper',
             # 'units': '',
             # 'factor': 1,
             # 'colormap': plt.get_cmap('terrain'),
             # 'clevels':   [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             # 'clevelsNH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             # 'clevelsSH': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
             # 'is3d': False},
             {'name': 'temperature',
              'mpasvarname': 'timeMonthly_avg_activeTracers_temperature',
              'component': 'mpaso',
              'title': 'Potential Temperature',
              'units': '$^\circ$C',
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-1.8, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 8.0, 12., 16., 22.],
              'clevelsNH': [-5, -1.8, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 8.0, 12., 16.],
              'clevelsSH': [-5, -1.8, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 8.0, 12., 16.],
              'is3d': True},
             {'name': 'salinity',
              'mpasvarname': 'timeMonthly_avg_activeTracers_salinity',
              'component': 'mpaso',
              'title': 'Salinity',
              'units': 'psu',
              'factor': 1,
              'colormap': cmocean.cm.haline,
              'clevels': [27., 28., 29., 29.5, 30., 30.5, 31., 32., 33., 34., 35.],
              'clevelsNH': [27., 28., 29., 29.5, 30., 30.5, 31., 32., 33., 34., 35.],
              'clevelsSH': [31., 31.5, 32., 32.3, 32.6, 32.9, 33.2, 33.5, 34., 34.5, 35.],
              'is3d': True}
             #{'name': 'GMnormalVel',
             # 'mpasvarname': 'timeMonthly_avg_normalGMBolusVelocity',
             # 'component': 'mpaso',
             # 'title': 'GM Normal Velocity',
             # 'units': 'cm/s',
             # 'factor': 1e2,
             # 'colormap': plt.get_cmap('RdBu_r'),
             # 'clevels':   [-2, -1, -0.5, -0.25, -0.05, 0., 0.05, 0.25, 0.5, 1, 2],
             # 'clevelsNH': [-2, -1, -0.5, -0.25, -0.05, 0., 0.05, 0.25, 0.5, 1, 2],
             # 'clevelsSH': [-2, -1, -0.5, -0.25, -0.05, 0., 0.05, 0.25, 0.5, 1, 2],
             # 'is3d': True}
            ]

# Info about MPAS mesh
f = netcdf_dataset(meshfile, mode='r')
lonCell = f.variables['lonCell'][:]
latCell = f.variables['latCell'][:]
lonEdge = f.variables['lonEdge'][:]
latEdge = f.variables['latEdge'][:]
z = f.variables['refBottomDepth'][:]
depth = f.variables['bottomDepth'][:]
f.close()
lonCell = pi2deg*lonCell
latCell = pi2deg*latCell
lonEdge = pi2deg*lonEdge
latEdge = pi2deg*latEdge
# Find model levels for each depth level
zlevels = np.zeros(np.shape(dlevels), dtype=np.int64)
for id in range(len(dlevels)):
    dz = np.abs(z-dlevels[id])
    zlevels[id] = np.argmin(dz)
#print('Model levels = ', z[zlevels])
#print(np.min(depth), np.max(depth))
#print('z levels = ', z)

#figtitle = 'Bottom depth {}'.format(meshName)
#figfileGlobal = '{}/DepthGlobal_{}.png'.format(figdir, meshName)
#figfileNH = '{}/DepthNH_{}.png'.format(figdir, meshName)
#figfileSH = '{}/DepthSH_{}.png'.format(figdir, meshName)
#clevels = [0., 10., 50., 100., 250., 500.0, 750., 1000., 2000., 3000., 5000.]
#colormap = cmocean.cm.deep_r
#dotSize = 0.25
#make_scatter_plot(lonCell, latCell, depth, dotSize, colormap, clevels, colorIndices, 'm', 'Robinson', figtitle, figfileGlobal)
#dotSize = 5.0 # this should go up as resolution decreases
#make_scatter_plot(lonCell, latCell, depth, dotSize, colormap, clevels, colorIndices, 'm', 'NorthPolarStereo', figtitle, figfileNH, lon0=lon0NH, lon1=lon1NH, dlon=20.0, lat0=lat0NH, lat1=lat1NH, dlat=10.0)
#dotSize = 25.0 # this should go up as resolution decreases
#make_scatter_plot(lonCell, latCell, depth, dotSize, colormap, clevels, colorIndices, 'm', 'SouthPolarStereo', figtitle, figfileSH, lon0=lon0SH, lon1=lon1SH, dlon=20.0, lat0=lat0SH, lat1=lat1SH, dlat=10.0)

for month in months:
    modelfile = '{}/{}.mpaso.hist.am.timeSeriesStatsMonthly.{:04d}-{:02d}-01.nc'.format(
                 modeldir, runname, year, month)
    #modelfile = '{}/mpaso.hist.am.timeSeriesStatsMonthly.{:04d}-{:02d}-01.nc'.format(
    #             modeldir, year, month)  # old (v1) filename format
    f = netcdf_dataset(modelfile, mode='r')
    for var in variables:
        varname = var['name']
        mpasvarname = var['mpasvarname']
        factor = var['factor']
        clevels = var['clevels']
        clevelsNH = var['clevelsNH']
        clevelsSH = var['clevelsSH']
        colormap = var['colormap']

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
                figtitle = '{}, z={:5.1f} m, year={}, month={}'.format(
                            var['title'], z[zlevels[iz]], year, month)
                figfileGlobal = '{}/{}Global_depth{:04d}_{}_{:04d}-{:02d}.png'.format(
                                 figdir, varname, int(dlevels[iz]), runname, year, month)
                figfileNH = '{}/{}NH_depth{:04d}_{}_{:04d}-{:02d}.png'.format(
                                 figdir, varname, int(dlevels[iz]), runname, year, month)
                figfileSH = '{}/{}SH_depth{:04d}_{}_{:04d}-{:02d}.png'.format(
                                 figdir, varname, int(dlevels[iz]), runname, year, month)

                # What I have below could instead be:
                # dsIn  = xr.open_dataset(modelfile).isel(Time=0, nVertLevels=zlevels[iz]) # this would go outside the var loop, right after modelfile is defined
                # fld = factor * dsIn[mpasvarname].values # this would go here in place of the next 5 lines
                fld = f.variables[mpasvarname][0, :, zlevels[iz]]
                fld = ma.masked_greater(fld, 1e15)
                fld = ma.masked_less(fld, -1e15)
                fld = factor*fld
                fld = np.squeeze(fld)
                if varname=='GMkappa':
                    kappa = np.squeeze(f.variables['timeMonthly_avg_gmBolusKappa'])
                    kappa_horScaling = np.squeeze(f.variables['timeMonthly_avg_gmHorizontalTaper'])
                    fld = kappa * fld * kappa_horScaling
                print('varname=', varname, 'month=', month, 'zlev=', int(dlevels[iz]), 'fldmin=', np.min(fld), 'fldmax=', np.max(fld))
                #fld3d = f.variables[varname][0, :, :]
                #fld3d = ma.masked_greater(fld3d, 1e15)
                #fld3d = ma.masked_less(fld3d, -1e15)
                #fld3d = factor*fld3d
                #print(np.min(fld3d), np.max(fld3d))

                dotSize = 0.25
                make_scatter_plot(lon, lat, fld, dotSize, colormap, clevels, colorIndices, var['units'], 'Robinson', figtitle, figfileGlobal)

                dotSize = 5.0 # this should go up as resolution decreases
                make_scatter_plot(lon, lat, fld, dotSize, colormap, clevelsNH, colorIndices, var['units'], 'NorthPolarStereo', figtitle, figfileNH, lon0=lon0NH, lon1=lon1NH, dlon=20.0, lat0=lat0NH, lat1=lat1NH, dlat=10.0)

                dotSize = 25.0 # this should go up as resolution decreases
                make_scatter_plot(lon, lat, fld, dotSize, colormap, clevelsSH, colorIndices, var['units'], 'SouthPolarStereo', figtitle, figfileSH, lon0=lon0SH, lon1=lon1SH, dlon=20.0, lat0=lat0SH, lat1=lat1SH, dlat=10.0)

        else:
            figtitle = '{}, year={}, month={}'.format(
                        var['title'], year, month)
            figfileGlobal = '{}/{}Global_{}_{:04d}-{:02d}.png'.format(
                             figdir, varname, runname, year, month)
            figfileNH = '{}/{}NH_{}_{:04d}-{:02d}.png'.format(
                             figdir, varname, runname, year, month)
            figfileSH = '{}/{}SH_{}_{:04d}-{:02d}.png'.format(
                             figdir, varname, runname, year, month)

            fld = f.variables[mpasvarname]
            fld = ma.masked_greater(fld, 1e15)
            fld = ma.masked_less(fld, -1e15)
            fld = factor*fld
            fld = np.squeeze(fld)
            if varname=='Redikappa':
                kappa_horScaling = np.squeeze(f.variables['timeMonthly_avg_RediHorizontalTaper'])
                fld = fld * kappa_horScaling
            print('varname=', varname, 'month=', month, 'fldmin=', np.min(fld), 'fldmax=', np.max(fld))

            dotSize = 0.25
            make_scatter_plot(lon, lat, fld, dotSize, colormap, clevels, colorIndices, var['units'], 'Robinson', figtitle, figfileGlobal)

            dotSize = 5.0 # this should go up as resolution decreases
            make_scatter_plot(lon, lat, fld, dotSize, colormap, clevelsNH, colorIndices, var['units'], 'NorthPolarStereo', figtitle, figfileNH, lon0=lon0NH, lon1=lon1NH, dlon=20.0, lat0=lat0NH, lat1=lat1NH, dlat=10.0)

            dotSize = 25.0 # this should go up as resolution decreases
            make_scatter_plot(lon, lat, fld, dotSize, colormap, clevelsSH, colorIndices, var['units'], 'SouthPolarStereo', figtitle, figfileSH, lon0=lon0SH, lon1=lon1SH, dlon=20.0, lat0=lat0SH, lat1=lat1SH, dlat=10.0)

    f.close()
