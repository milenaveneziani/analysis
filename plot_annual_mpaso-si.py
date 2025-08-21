from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import numpy as np
import xarray as xr
import cmocean
import matplotlib.pyplot as plt
import matplotlib.colors as cols

from make_plots import make_scatter_plot, make_mosaic_descriptor, make_mosaic_plot


# Settings for erdc.hpc.mil
meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runname = 'E3SMv2.1B60to10rA02'
runname = 'E3SMv2.1G60to10_01'
indir = f'/p/cwfs/milena/{runname}/archive/ocn/hist'
#indir = f'/p/cwfs/milena/{runname}/archive/ice/hist'
#runname = 'E3SMv2.1B60to10rA07'
#indir = f'/p/cwfs/apcraig/archive/{runname}/ocn/hist'
#indir = f'/p/cwfs/apcraig/archive/{runname}/ice/hist'

# Annual mean for year 1 will be plotted first, and then the
# differences between yearsToPlot and year 1 will be plotted
yearsToPlot = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50]
#yearsToPlot = [2, 50]

colorIndices = [0, 14, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]

projection = 'Miller'
lon0 = -90.0
lon1 = 20.0
dlon = 20.0
lat0 = -60.0
lat1 = 90.0
dlat = 30.0

modelname = 'ocean'
mpascomp = 'mpaso'
mpasFile = 'timeSeriesStatsMonthly'
variables = [
             {'name': 'EminusP',
              'mpasvarname': None,
              'title': 'E-P',
              'units': '10$^{-6}$ kg m$^{-2}$ s$^{-1}$',
              'factor': 1e6,
              'colormap_year1': plt.get_cmap('PuOr_r'),
              'clevels_year1': [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150],
              'colormap': cmocean.cm.balance,
              'clevels': [-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50]},
             {'name': 'seaIceFreshWaterFlux',
              'mpasvarname': 'timeMonthly_avg_seaIceFreshWaterFlux',
              'title': 'sea ice FW flux',
              'units': '10$^{-6}$ kg m$^{-2}$ s$^{-1}$',
              'factor': 1e6,
              'colormap_year1': plt.get_cmap('PuOr_r'),
              'clevels_year1': [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150],
              'colormap': cmocean.cm.balance,
              'clevels': [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150]},
             {'name': 'MLD',
              'mpasvarname': 'timeMonthly_avg_dThreshMLD',
              'title': 'Mean MLD',
              'units': 'm',
              'factor': 1,
              'colormap_year1': plt.get_cmap('viridis'),
              'clevels_year1': [20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 250, 300],
              'colormap': cmocean.cm.balance,
              'clevels': [-100.0, -80.0, -60.0, -40.0, -20.0, -10.0, 0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0]},
            ]
#mpasFile = 'timeSeriesStatsMonthlyMax'
#variables = [
#             {'name': 'maxMLD',
#              'mpasvarname': 'timeMonthlyMax_max_dThreshMLD',
#              'title': 'Maximum MLD',
#              'units': 'm',
#              'factor': 1,
#              'colormap_year1': plt.get_cmap('viridis'),
#              'clevels_year1': [20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 250, 300],
#              'colormap': cmocean.cm.balance,
#              'clevels': [-300.0, -250.0, -200.0, -150.0, -100.0, -50.0, 0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0]},
#            ]

#modelname = 'seaice'
#mpascomp = 'mpassi'
#mpasFile = 'timeSeriesStatsMonthly'
#variables = [
#             {'name': 'iceArea',
#              'mpasvarname': 'timeMonthly_avg_iceAreaCell',
#              'title': 'Sea Ice Concentration',
#              'units': '%',
#              'factor': 1e2,
#              'colormap_year1': cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
#                                                     (0,     0.4,   0.4),   (0.169, 0.435, 0.223), (0.455, 0.478, 0.196),\
#                                                     (0.757, 0.474, 0.435), (0.827, 0.561, 0.772), (0.761, 0.757, 0.949),\
#                                                     (0.808, 0.921, 0.937)]),
#              'clevels_year1': [10, 15, 30, 50, 80, 90, 95, 97, 98, 99, 100],
#              'colormap': cmocean.cm.balance,
#              'clevels': [-50.0, -40.0, -30.0, -20.0, -15.0, -5.0, 0.0, 5.0, 15.0, 20.0, 30.0, 40.0, 50.0]},
#           ]

figdir = f'./{modelname}_native/{runname}'
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
# First compute and plot annual average for year 1
#
print('\nCompute and plot annual means for year 1')
dsYear1 = xr.Dataset()
infiles = f'{indir}/{runname}.{mpascomp}.hist.am.{mpasFile}.0001-*.nc'
ds = xr.open_mfdataset(infiles, combine='nested', concat_dim='Time', decode_times=False)

for var in variables:
    varname = var['name']
    mpasvarname = var['mpasvarname']
    factor = var['factor']
    clevels = var['clevels_year1']
    colormap = var['colormap_year1']
    vartitle = var['title']
    varunits = var['units']

    print(f'  Variable: {varname}...')
    # Read in fld and compute annual average
    if varname=='EminusP':
        evap = ds['timeMonthly_avg_evaporationFlux']
        rain = ds['timeMonthly_avg_rainFlux']
        snow = ds['timeMonthly_avg_snowFlux']
        fld = factor * (evap + rain + snow)
    else:
        fld = factor * ds[mpasvarname]
    fld = fld.mean(dim='Time')

    # Save year 1 annual average
    dsYear1[varname] = fld

    # Plot
    figtitle = f'{vartitle}, year 1\n{runname}'
    figfile = f'{figdir}/{varname}_annual_year0001.png'
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
# and plot the difference with year 1
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
            evap = ds['timeMonthly_avg_evaporationFlux']
            rain = ds['timeMonthly_avg_rainFlux']
            snow = ds['timeMonthly_avg_snowFlux']
            fld = factor * (evap + rain + snow)
        else:
            fld = factor * ds[mpasvarname]
        fld = fld.mean(dim='Time')

        diff = fld - dsYear1[varname]

        # Plot
        figtitle = f'{vartitle}, year {year} minus year 1\n{runname}'
        figfile = f'{figdir}/{varname}_annual_diffYears{year:04d}-0001.png'
        make_mosaic_plot(lonCell, latCell, diff, mosaic_descriptor, figtitle, figfile, showEdges=None,
                         cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                         projectionName=projection, lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
        ##dotSize = 0.25
        #dotSize = 1.0
        #make_scatter_plot(lonCell, latCell, dotSize, figtitle, figfile, projectionName=projection,
        #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
        #                  fld=diff, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
