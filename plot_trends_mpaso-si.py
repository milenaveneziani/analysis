from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import cmocean

import time


from make_plots import make_scatter_plot, make_mosaic_descriptor, make_mosaic_plot


############################
# Machine related settings (paths, etc)
# Settings for erdc.hpc.mil
#meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runname = 'E3SMv2.1B60to10rA02'
#indir0 = f'/p/global/milena/{runname}/archive'
#isShortTermArchive = True # if True, {modelname}/hist will be appended to indir0
#isSingleVarFiles = False # if True, {modelname}/singleVarFiles will be appended to indir0

# Settings for nersc
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
runname = 'E3SMv2.1B60to10rA02'
indir0 = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runname}/archive'
isShortTermArchive = True # these two options should be switched for ocean variables
isSingleVarFiles = False

############################
# Specific script settings (plus plotting stuff)
startYearPhase = [1, 201, 276, 301, 351]
endYearPhase = [200, 275, 300, 350, 386]
#startYearPhase = [1]
#endYearPhase = [50]
nPhases = len(startYearPhase)
if nPhases!=len(endYearPhase):
    raise ValueError('Variables startYearPhase and endYearPhase need to be of the same length')

projection = 'NorthPolarStereo'
lon0 = 0.0
lon1 = 360.0
dlon = 30.0
lat0 = 45.0
lat1 = 90.0
dlat = 5.0

colorIndices = [0, 14, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]

############################
# Variable settings
#modelname = 'ocn'
#modelnameOut = 'ocean'
#mpascomp = 'mpaso'
#mpasFile = 'timeSeriesStatsMonthly'
#variables = [
#             {'name': 'totalHeatFlux',
#              'mpasvarname': None,
#              'title': 'Total heat flux (sens+lat+netLW+netSW)',
#              'units': 'W/m$^2$',
#              'factor': 1,
#              'isvar3d': False,
#              'colormap_means': cmocean.cm.balance,
#              'clevels_means': [-200, -150, -120, -90, -60, -20, 0, 20, 60, 90, 120, 150, 200],
#              'colormap_trends': cmocean.cm.balance,
#              'clevels_trends': [-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]},
#            ]

#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'mpasvarname': 'timeMonthlyMax_max_dThreshMLD',
#              'title': 'Maximum MLD',
#              'units': 'm',
#              'factor': 1,
#              'isvar3d': False,
#              'colormap_means': plt.get_cmap('viridis'),
#              'clevels_means': [20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 250, 300],
#              'colormap_trends': cmocean.cm.balance,
#              'clevels_trends': [-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]},
#            ]

modelname = 'ice'
modelnameOut = 'seaice'
mpascomp = 'mpassi'
mpasFile = 'timeSeriesStatsMonthly'
variables = [
#             {'name': 'iceArea',
#              'mpasvarname': 'timeMonthly_avg_iceAreaCell',
#              'title': 'Sea Ice Concentration',
#              'units': '%',
#              'factor': 1e2,
#              'isvar3d': False,
#              'colormap_means': cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
#                                                     (0,     0.4,   0.4),   (0.169, 0.435, 0.223), (0.455, 0.478, 0.196),\
#                                                     (0.757, 0.474, 0.435), (0.827, 0.561, 0.772), (0.761, 0.757, 0.949),\
#                                                     (0.808, 0.921, 0.937)]),
#              'clevels_means': [10, 15, 20, 40, 60, 70, 80, 90, 93, 95, 96],
#              'colormap_trends': cmocean.cm.balance,
#              'clevels_trends': [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]},
             {'name': 'iceVolume',
             'mpasvarname': 'timeMonthly_avg_iceVolumeCell',
              'title': 'Sea Ice Thickness',
              'units': 'm',
              'factor': 1,
              'isvar3d': False,
              'colormap_means': plt.get_cmap('YlGnBu_r'),
              'clevels_means': [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
              'colormap_trends': cmocean.cm.balance,
              'clevels_trends': [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]},
           ]

############################
if isSingleVarFiles:
    indir = f'{indir0}/{modelname}/singleVarFiles'
else:
    if isShortTermArchive:
        indir = f'{indir0}/{modelname}/hist'
    else:
        indir = indir0

# For annual means:
postprocdir = f'{indir0}/{modelname}/postproc'
if not os.path.isdir(postprocdir):
    os.makedirs(postprocdir)

figdir = f'./{modelnameOut}_native/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

# Info about MPAS mesh
dsMesh = xr.open_dataset(meshfile)
lonCell = 180/np.pi*dsMesh.lonCell
latCell = 180/np.pi*dsMesh.latCell
lonVertex = 180/np.pi*dsMesh.lonVertex
latVertex = 180/np.pi*dsMesh.latVertex
#maskLatCell = latCell >= minLat
#maskLatVert = latVertex >= minLat
# restart files are missing this attribute that is needed for mosaic,
# so for now adding this manually:
dsMesh.attrs['is_periodic'] = 'NO'
mosaic_descriptor = make_mosaic_descriptor(dsMesh, projection)

# First compute annual means, if not already there
print('Compute annual means for each variable, if they do not exist...')
daysInMonth = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
timeFactor = xr.DataArray(
                          data=daysInMonth/365,
                          dims=('Time', )
                          )
years = range(startYearPhase[0], endYearPhase[-1]+1)
for var in variables:
    varname = var['name']
    print(f'  Variable = {varname}')
    mpasvarname = var['mpasvarname']
    factor = var['factor']
    vartitle = var['title']
    varunits = var['units']
    isvar3d = var['isvar3d']
    for year in years:
        print(f'    Year = {year}')
        outfile = f'{postprocdir}/{varname}.{runname}_year{year:04d}.nc'
        if not os.path.isfile(outfile):
            dsOut = []
            #newTime = np.empty(12, dtype=datetime)
            for month in range(1, 13):
                dsOutMonthly = xr.Dataset()
                if isSingleVarFiles:
                    if varname=='totalHeatFlux':
                        infile = f'{indir}/sensibleHeatFlux/sensibleHeatFlux.{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                        if not os.path.isfile(infile):
                            raise SystemExit(f'*** File {infile} not found. Exiting...')
                        dsIn = xr.open_dataset(infile, decode_times=False)
                        sensible = dsIn['timeMonthly_avg_sensibleHeatFlux']
                        infile = f'{indir}/latentHeatFlux/latentHeatFlux.{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                        if not os.path.isfile(infile):
                            raise SystemExit(f'*** File {infile} not found. Exiting...')
                        dsIn = xr.open_dataset(infile, decode_times=False)
                        latent = dsIn['timeMonthly_avg_latentHeatFlux']
                        infile = f'{indir}/longWaveHeatFluxDown/longWaveHeatFluxDown.{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                        if not os.path.isfile(infile):
                            raise SystemExit(f'*** File {infile} not found. Exiting...')
                        dsIn = xr.open_dataset(infile, decode_times=False)
                        longwave_down = dsIn['timeMonthly_avg_longWaveHeatFluxDown']
                        infile = f'{indir}/longWaveHeatFluxUp/longWaveHeatFluxUp.{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                        if not os.path.isfile(infile):
                            raise SystemExit(f'*** File {infile} not found. Exiting...')
                        dsIn = xr.open_dataset(infile, decode_times=False)
                        longwave_up = dsIn['timeMonthly_avg_longWaveHeatFluxUp']
                        infile = f'{indir}/shortWaveHeatFlux/shortWaveHeatFlux.{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                        if not os.path.isfile(infile):
                            raise SystemExit(f'*** File {infile} not found. Exiting...')
                        dsIn = xr.open_dataset(infile, decode_times=False)
                        shotwave_net = dsIn['timeMonthly_avg_shortWaveHeatFlux']
                        fld = factor * (sensible + latent + longwave_down + longwave_up + shotwave_net)
                    elif varname=='maxMLD':
                        infile = f'{indir}/maxMLD/dThreshMLD.{runname}.{mpascomp}.hist.am.timeSeriesStatsMonthlyMax.{year:04d}-{month:02d}-01.nc'
                        if not os.path.isfile(infile):
                            raise SystemExit(f'*** File {infile} not found. Exiting...')
                        dsIn = xr.open_dataset(infile, decode_times=False)
                        fld = factor * dsIn[mpasvarname]
                    else:
                        infile = f'{indir}/{varname}/{varname}.{runname}.{mpascomp}.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                        if not os.path.isfile(infile):
                            raise SystemExit(f'*** File {infile} not found. Exiting...')
                        dsIn = xr.open_dataset(infile, decode_times=False)
                        fld = factor * dsIn[mpasvarname]
                else:
                    infile = f'{indir}/{runname}.{mpascomp}.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
                    if not os.path.isfile(infile):
                        raise SystemExit(f'*** File {infile} not found. Exiting...')
                    dsIn = xr.open_dataset(infile, decode_times=False)
                    #start, end = [parse(dsIn[f'xtime_{name}Monthly'].astype(str).values[0].split('_')[0]) for name in ('start', 'end')]
                    #if start.year < 1000:
                    #    newTime[im] = dsIn['Time'].values
                    #else:
                    #    newTime[im] = start + timedelta(days=int((end - start).days / 2))

                    if varname=='totalHeatFlux':
                        sensible = dsIn['timeMonthly_avg_sensibleHeatFlux']
                        latent = dsIn['timeMonthly_avg_latentHeatFlux']
                        longwave_down = dsIn['timeMonthly_avg_longWaveHeatFluxDown']
                        longwave_up = dsIn['timeMonthly_avg_longWaveHeatFluxUp']
                        shotwave_net = dsIn['timeMonthly_avg_shortWaveHeatFlux']
                        fld = factor * (sensible + latent + longwave_down + longwave_up + shotwave_net)
                    else:
                        fld = factor * dsIn[mpasvarname]

                # Note: this will have to be changed for vertex-centered variables:
                if isvar3d:
                    dsOutMonthly[varname] = xr.DataArray(
                            data=fld,
                            dims=('Time', 'nCells', 'nVertLevels', ),
                            attrs=dict(description=vartitle, units=varunits, )
                            )
                else:
                    dsOutMonthly[varname] = xr.DataArray(
                            data=fld,
                            dims=('Time', 'nCells', ),
                            attrs=dict(description=vartitle, units=varunits, )
                            )

                dsOut.append(dsOutMonthly)

            dsOut = xr.concat(dsOut, dim='Time')
            dsOut = (timeFactor * dsOut).sum('Time')
            dsOut = dsOut.expand_dims(dim='Time')
            dsOut.to_netcdf(outfile)
        else:
            print(f'    File {outfile} already exists. Skipping it...')

# Now compute means and trends for identified phase periods, and plot them
print('\nCompute means and trends for each variable and each identified period...')
for var in variables:
    varname = var['name']
    print(f'  Variable = {varname}')
    mpasvarname = var['mpasvarname']
    factor = var['factor']
    vartitle = var['title']
    varunits = var['units']
    colormap_means = var['colormap_means']
    clevels_means = var['clevels_means']
    colormap_trends = var['colormap_trends']
    clevels_trends = var['clevels_trends']

    for iphase in range(0, nPhases):
        t0 = time.time()
        print(f'    Phase period = {startYearPhase[iphase]:04d}-{endYearPhase[iphase]:04d}')
        infiles = []
        for year in range(startYearPhase[iphase], endYearPhase[iphase]+1):
            infiles.append(f'{postprocdir}/{varname}.{runname}_year{year:04d}.nc')

        dsIn = xr.open_mfdataset(infiles, combine='nested', concat_dim='Time')
        fld = dsIn[varname]
        mean = fld.mean(dim='Time')
        t1 = time.time()
        print('mean calculated, time taken (seconds) = ', t1-t0)
        t0 = t1
        fld = np.squeeze(fld.to_numpy())
        ncells = np.shape(fld)[1]
        trend = np.empty(ncells, )
        #intercept = np.empty(ncells, )
        x = range(1, endYearPhase[iphase]-startYearPhase[iphase]+2)
        for ncell in range(0, ncells):
            poly_coef = np.polyfit(x, fld[:, ncell], 1)
            trend[ncell] = poly_coef[0]
            #intercept[ncell] = poly_coef[1]
        print(np.nanmin(trend), np.nanmax(trend))
        trend = xr.DataArray(data=trend, dims=('nCells', ))
        t1 = time.time()
        print('trend calculated, time taken (seconds) = ', t1-t0)
        t0 = t1

        print('    plot mean...')
        figtitle = f'{vartitle}, mean, years {startYearPhase[iphase]:04d}-{endYearPhase[iphase]:04d}\n{runname}'
        figfile = f'{figdir}/{varname}_mean_years{startYearPhase[iphase]:04d}-{endYearPhase[iphase]:04d}.png'
        if varname=='iceArea':
            cindices = None
        else:
            cindices = colorIndices
        make_mosaic_plot(lonCell, latCell, mean, mosaic_descriptor, figtitle, figfile, showEdges=None,
                         cmap=colormap_means, clevels=clevels_means, cindices=cindices, cbarLabel=varunits,
                         projectionName=projection, lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
        t1 = time.time()
        print('    time taken (seconds) = ', t1-t0)
        t0 = t1


        print('    plot trend...')
        figtitle = f'{vartitle}, trend, years {startYearPhase[iphase]:04d}-{endYearPhase[iphase]:04d}\n{runname}'
        figfile = f'{figdir}/{varname}_trend_years{startYearPhase[iphase]:04d}-{endYearPhase[iphase]:04d}.png'
        make_mosaic_plot(lonCell, latCell, trend, mosaic_descriptor, figtitle, figfile, showEdges=None,
                         cmap=colormap_trends, clevels=clevels_trends, cindices=colorIndices, cbarLabel=varunits,
                         projectionName=projection, lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
        t1 = time.time()
        print('    time taken (seconds) = ', t1-t0)
