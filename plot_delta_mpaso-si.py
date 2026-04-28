#
# This was intended to compute differences of fields wrt their initial
# condition, but of course the only fields available in the ic file are
# the ones that are needed to start the ocean (or seaice), so the prognostic
# variables. Which makes this script basically useless.
#
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import cmocean


from make_plots import make_scatter_plot, make_mosaic_descriptor, make_mosaic_plot


# Settings for erdc.hpc.mil
meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
runname = 'E3SMv2.1B60to10rA02'
icfile_ocean = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
icfile_seaice = '/p/app/unsupported/RASM/acme/inputdata/ice/mpas-seaice/ARRM10to60E2r1/mpassi.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runname = 'E3SMv2.1G60to10_01'
#icfile_ocean = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
#icfile_seaice = '/p/app/unsupported/RASM/acme/inputdata/ice/mpas-seaice/ARRM10to60E2r1/mpassi.ARRM10to60E2r1.220730.nc'
indir = f'/p/global/milena/{runname}/archive'
#runname = 'E3SMv2.1B60to10rA07'
#icfile_ocean = can't find it but I should be able to point to restart files from the E3SMv2.1G60to10_01 run
#                mpaso.E3SMv2.1G60to10_01-Jan1997.230225.nc
#icfile_seaice = mpassi.E3SMv2.1G60to10_01-Jan1997.230225.nc 
#indir = f'/p/global/apcraig/archive/{runname}'
isShortTermArchive = True # if True, {modelname}/hist will be appended to indir

# Initial condition will be plotted first, and then the
# differences between yearsToPlot and the ic will be plotted
yearsToPlot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50]
yearsToPlot = [1]

colorIndices = [0, 14, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]

projection = 'Miller'
lon0 = -90.0
lon1 = 20.0
dlon = 20.0
lat0 = -60.0
lat1 = 90.0
dlat = 30.0

modelname = 'ocn'
modelnameOut = 'ocean'
mpascomp = 'mpaso'
icfile = icfile_ocean
mpasFile = 'timeSeriesStatsMonthly'
mpasvarnameHeader = 'timeMonthly_avg_'
# none of these fields are available in the ic file:
variables = [
             #{'name': 'ssh',
             # 'mpasvarname': 'ssh',
             # 'title': 'SSH',
             # 'units': 'm',
             # 'factor': 1,
             # 'colormap_ic': plt.get_cmap('BrBG_r'),
             # 'clevels_ic': [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
             # 'colormap': cmocean.cm.balance,
             # 'clevels': [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
             {'name': 'sshAdjusted',
              'mpasvarname': 'pressureAdjustedSSH',
              'title': 'SSH (adjusted by sea surface pressure)',
              'units': 'm',
              'factor': 1,
              'colormap_ic': plt.get_cmap('BrBG_r'),
              'clevels_ic': [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
              'colormap': cmocean.cm.balance,
              'clevels': [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
             {'name': 'windStress',
              'mpasvarname': None,
              'title': 'wind stress magnitude',
              'units': 'N/m$^2$',
              'factor': 1,
              'colormap_ic': cmocean.cm.speed_r,
              'clevels_ic': [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.14, 0.15],
              'colormap': cmocean.cm.balance,
              'clevels': [-0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]},
             #{'name': 'evaporationFlux',
             # 'mpasvarname': 'evaporationFlux',
             # 'title': 'E',
             # 'units': '10$^{-6}$ kg m$^{-2}$ s$^{-1}$',
             # 'factor': 1e6,
             # 'colormap_ic': plt.get_cmap('PuOr_r'),
             # 'clevels_ic': [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150],
             # 'colormap': cmocean.cm.balance,
             # 'clevels': [-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50]},
             #{'name': 'precip',
             # 'mpasvarname': None,
             # 'title': 'P (rain+snow)',
             # 'units': '10$^{-6}$ kg m$^{-2}$ s$^{-1}$',
             # 'factor': 1e6,
             # 'colormap_ic': plt.get_cmap('PuOr_r'),
             # 'clevels_ic': [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150],
             # 'colormap': cmocean.cm.balance,
             # 'clevels': [-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50]},
             {'name': 'EminusP',
              'mpasvarname': None,
              'title': 'E-P',
              'units': '10$^{-6}$ kg m$^{-2}$ s$^{-1}$',
              'factor': 1e6,
              'colormap_ic': plt.get_cmap('PuOr_r'),
              'clevels_ic': [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150],
              'colormap': cmocean.cm.balance,
              'clevels': [-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50]},
             {'name': 'seaIceFreshWaterFlux',
              'mpasvarname': 'seaIceFreshWaterFlux',
              'title': 'sea ice FW flux',
              'units': '10$^{-6}$ kg m$^{-2}$ s$^{-1}$',
              'factor': 1e6,
              'colormap_ic': plt.get_cmap('PuOr_r'),
              'clevels_ic': [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150],
              'colormap': cmocean.cm.balance,
              'clevels': [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150]},
             #{'name': 'surfaceBuoyancyForcing',
             # 'mpasvarname': 'surfaceBuoyancyForcing',
             # 'title': 'sfc buoyancy flux',
             # 'units': '10$^{-8}$ m$^2$ s$^{-3}$',
             # 'factor': 1e8,
             # 'colormap_ic': plt.get_cmap('BrBG_r'),
             # 'clevels_ic': [-4.8, -4, -3.2, -2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4, 3.2, 4, 4.8],
             # 'colormap': cmocean.cm.balance,
             # 'clevels': [-2.4, -2, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2, 2.4]},
             {'name': 'MLD',
              'mpasvarname': 'dThreshMLD',
              'title': 'Mean MLD',
              'units': 'm',
              'factor': 1,
              'colormap_ic': plt.get_cmap('viridis'),
              'clevels_ic': [20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 250, 300],
              'colormap': cmocean.cm.balance,
              'clevels': [-100.0, -80.0, -60.0, -40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]},
            ]

#modelname = 'ice'
#modelnameOut = 'seaice'
#mpascomp = 'mpassi'
#icfile = icfile_seaice
#mpasFile = 'timeSeriesStatsMonthly'
#variables = [
#             {'name': 'iceArea',
#              'mpasvarname': 'iceAreaCell',
#              'title': 'Sea Ice Concentration',
#              'units': '%',
#              'factor': 1e2,
#              'colormap_ic': cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
#                                                  (0,     0.4,   0.4),   (0.169, 0.435, 0.223), (0.455, 0.478, 0.196),\
#                                                  (0.757, 0.474, 0.435), (0.827, 0.561, 0.772), (0.761, 0.757, 0.949),\
#                                                  (0.808, 0.921, 0.937)]),
#              'clevels_ic': [10, 15, 30, 50, 80, 90, 95, 97, 98, 99, 100],
#              'colormap': cmocean.cm.balance,
#              'clevels': [-50.0, -40.0, -30.0, -20.0, -15.0, -5.0, 0.0, 5.0, 15.0, 20.0, 30.0, 40.0, 50.0]},
#             {'name': 'iceVolume',
#              'mpasvarname': 'iceVolumeCell',
#              'title': 'Sea Ice Thickness',
#              'units': 'm',
#              'factor': 1,
#              'colormap_ic': plt.get_cmap('YlGnBu_r'),
#              'clevels_ic': [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
#              'colormap': cmocean.cm.balance,
#              'clevels': [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]},
#           ]

if isShortTermArchive:
    indir = f'{indir}/{modelname}/hist'

figdir = f'./{modelnameOut}_native/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

# Info about MPAS mesh
print('\nRead in mesh info')
dsMesh = xr.open_dataset(meshfile)
lonCell = dsMesh.lonCell.values
latCell = dsMesh.latCell.values
lonCell = 180/np.pi*lonCell
latCell = 180/np.pi*latCell
# restart files are missing this attribute that is needed for mosaic,
# so for now adding this manually:
dsMesh.attrs['is_periodic'] = 'NO'
mosaic_descriptor = make_mosaic_descriptor(dsMesh, projection)

####################################################
# First plot the initial condition
#
print(f'\nPlot initial condition from file:\n {icfile}')
dsIC = xr.open_dataset(icfile)

for var in variables:
    varname = var['name']
    mpasvarname = var['mpasvarname']
    factor = var['factor']
    clevels = var['clevels_ic']
    colormap = var['colormap_ic']
    vartitle = var['title']
    varunits = var['units']

    print(f'  Variable: {varname}...')
    if varname=='EminusP':
        evap = dsIC['evaporationFlux']
        rain = dsIC['rainFlux']
        snow = dsIC['snowFlux']
        fld = factor * (evap + rain + snow)
    elif varname=='precip':
        rain = dsIC['rainFlux']
        snow = dsIC['snowFlux']
        fld = factor * (rain + snow)
    elif varname=='windStress':
        ustress = dsIC['windStressZonal']
        vstress = dsIC['windStressMeridional']
        fld = factor * 0.5 * np.sqrt(ustress*ustress + vstress*vstress)
    else:
        fld = factor * dsIC[mpasvarname]
    print(np.nanmin(fld), np.nanmax(fld))

    dsIC[varname] = fld

    # Plot
    figtitle = f'{vartitle}, initial condition\n{runname}'
    figfile = f'{figdir}/{varname}_ic.png'
    if varname=='iceArea':
        make_mosaic_plot(lonCell, latCell, fld, mosaic_descriptor, figtitle, figfile, showEdges=None,
                         cmap=colormap, clevels=clevels, cindices=None, cbarLabel=varunits,
                         projectionName=projection, lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
    else:
        make_mosaic_plot(lonCell, latCell, fld, mosaic_descriptor, figtitle, figfile, showEdges=None,
                         cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                         projectionName=projection, lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
        ##dotSize = 0.25
        #dotSize = 1.0
        #make_scatter_plot(lonCell, latCell, dotSize, figtitle, figfile, projectionName=projection,
        #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
        #                  fld=fld, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)

########################################################
# Then compute annual averages for each subsequent year
# and plot the difference with the initial condition
#
for year in yearsToPlot:
    print(f'\nCompute annual means for year {year} and plot diff with year 1')
    infiles = f'{indir}/{runname}.{mpascomp}.hist.am.{mpasFile}.{year:04d}-*.nc'
    ds = xr.open_mfdataset(infiles, combine='nested', concat_dim='Time', decode_times=False)

    for var in variables:
        varname = var['name']
        mpasvarname = var['mpasvarname']
        factor = var['factor']
        clevels = var['clevels']
        colormap = var['colormap']
        vartitle = var['title']
        varunits = var['units']

        print(f'  Variable: {varname}...')
        # Read in fld and compute annual average
        if varname=='EminusP':
            evap = ds[f'{mpasvarnameHeader}evaporationFlux']
            rain = ds[f'{mpasvarnameHeader}rainFlux']
            snow = ds[f'{mpasvarnameHeader}snowFlux']
            fld = factor * (evap + rain + snow)
        elif varname=='precip':
            rain = ds[f'{mpasvarnameHeader}rainFlux']
            snow = ds[f'{mpasvarnameHeader}snowFlux']
            fld = factor * (rain + snow)
        elif varname=='windStress':
            ustress = ds[f'{mpasvarnameHeader}windStressZonal']
            vstress = ds[f'{mpasvarnameHeader}windStressMeridional']
            fld = factor * 0.5 * np.sqrt(ustress*ustress + vstress*vstress)
        else:
            fld = factor * ds[f'{mpasvarnameHeader}{mpasvarname}']
        fld = fld.mean(dim='Time')

        diff = fld - dsIC[varname]

        # Plot
        figtitle = f'{vartitle}, year {year} minus ic\n{runname}'
        figfile = f'{figdir}/{varname}_annual_diffYears{year:04d}-ic.png'
        make_mosaic_plot(lonCell, latCell, diff, mosaic_descriptor, figtitle, figfile, showEdges=None,
                         cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                         projectionName=projection, lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
        ##dotSize = 0.25
        #dotSize = 1.0
        #make_scatter_plot(lonCell, latCell, dotSize, figtitle, figfile, projectionName=projection,
        #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
        #                  fld=diff, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
