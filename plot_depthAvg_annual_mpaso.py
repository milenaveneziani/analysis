from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean


from mpas_analysis.ocean.utility import compute_zmid
from make_plots import make_scatter_plot, make_mosaic_descriptor, make_mosaic_plot


# Settings for erdc.hpc.mil
meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runname = 'E3SMv2.1B60to10rA02'
#runname = 'E3SMv2.1G60to10_01'
indir = f'/p/cwfs/milena/{runname}/archive/ocn/hist'
#runname = 'E3SMv2.1B60to10rA07'
#indir = f'/p/cwfs/apcraig/archive/{runname}/ocn/hist'

figdir = f'./ocean_native/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

# Annual mean for year 1 will be plotted first, and then the
# differences between yearsToPlot and year 1 will be plotted
yearsToPlot = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50]
#yearsToPlot = [2]

# zmins/zmaxs [m]
zmins = [-50., -300., -1000., -8000.]
zmaxs = [10., -50., -300., -1000.]
#zmins = [-50., -300.]
#zmaxs = [10., -50.]

colorIndices = [0, 14, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]

projection = 'Miller'
lon0 = -90.0
lon1 = 20.0
dlon = 20.0
lat0 = -60.0
lat1 = 90.0
dlat = 30.0

mpasFile = 'timeSeriesStatsMonthly'
variables = [
             {'name': 'velSpeed',
              'mpasvarname': None,
              'title': 'Velocity magnitude',
              'units': 'cm/s',
              'factor': 1e2,
              'colormap_year1': cmocean.cm.speed_r,
              'clevels_year1': [0.5, 1, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40],
              'colormap': cmocean.cm.balance,
              'clevels': [-10.0, -8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]},
             {'name': 'velocityZonal',
              'mpasvarname': 'timeMonthly_avg_velocityZonal',
              'title': 'Zonal velocity',
              'units': 'cm/s',
              'factor': 1e2,
              'colormap_year1': cmocean.cm.speed_r,
              'clevels_year1': [0.5, 1, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40],
              'colormap': cmocean.cm.balance,
              'clevels': [-10.0, -8.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]},
             #{'name': 'velocityMeridional',
             # 'mpasvarname': 'timeMonthly_avg_velocityMeridional',
             # 'title': 'Meridional velocity',
             # 'units': 'cm/s',
             # 'factor': 1e2,
             # 'colormap_year1': cmocean.cm.speed_r,
             # 'clevels_year1': [0.05, 1, 2, 5, 8, 10, 12, 15, 20, 25, 30, 35, 40],
             # 'colormap': cmocean.cm.balance,
             # 'clevels': [-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]}
             {'name': 'salinity',
              'mpasvarname': 'timeMonthly_avg_activeTracers_salinity',
              'title': 'Salinity',
              'units': 'psu',
              'factor': 1,
              'colormap_year1': cmocean.cm.haline,
              'clevels_year1': [33.0, 34.2,  34.4,  34.6, 34.7,  34.8,  34.9, 35.0, 35.2, 35.4, 35.6, 36.0, 36.5],
              'colormap': cmocean.cm.balance,
              'clevels': [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]},
             {'name': 'temperature',
              'mpasvarname': 'timeMonthly_avg_activeTracers_temperature',
              'title': 'Temperature',
              'units': '$^\circ$C',
              'factor': 1,
              'colormap_year1': cmocean.cm.thermal,
              'clevels_year1': [0., 2., 3., 4., 6., 8., 10., 12., 14., 16., 18., 22., 26.],
              'colormap': cmocean.cm.balance,
              'clevels': [-5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]}
            ]

# Info about MPAS mesh
print('\nRead in mesh info')
dsMesh = xr.open_dataset(meshfile)
lonCell = dsMesh.lonCell.values
latCell = dsMesh.latCell.values
lonCell = 180/np.pi*lonCell
latCell = 180/np.pi*latCell
maxLevelCell = dsMesh.maxLevelCell - 1 # now compute_zmid uses 0-based indexing
depth = dsMesh.bottomDepth
# restart files are missing this attribute that is needed for mosaic,
# so for now adding this manually:
dsMesh.attrs['is_periodic'] = 'NO'
mosaic_descriptor = make_mosaic_descriptor(dsMesh, projection)

####################################################
# First compute and plot annual average for year 1
#
print('\nCompute and plot annual means for year 1')
infiles = f'{indir}/{runname}.mpaso.hist.am.timeSeriesStatsMonthly.0001-*.nc'
ds = xr.open_mfdataset(infiles, combine='nested', concat_dim='Time', decode_times=False)
layerThickness = ds.timeMonthly_avg_layerThickness
zMid = compute_zmid(depth, maxLevelCell, layerThickness)

fldYear1 = []
for iz in range(len(zmins)):
    zmin = zmins[iz]
    zmax = zmaxs[iz]
    print(f'  depth range: {zmin}, {zmax}')
    depthMask = np.logical_and(zMid >= zmin, zMid <= zmax)
    dz = layerThickness.where(depthMask, drop=False)
    layerDepth = dz.sum(dim='nVertLevels')

    dsYear1 = xr.Dataset()
    for var in variables:
        varname = var['name']
        mpasvarname = var['mpasvarname']
        factor = var['factor']
        clevels = var['clevels_year1']
        colormap = var['colormap_year1']
        vartitle = var['title']
        varunits = var['units']

        print(f'    variable: {varname}...')
        # Read in fld and compute depth average
        if varname=='velSpeed':
            u  = ds['timeMonthly_avg_velocityZonal']
            v  = ds['timeMonthly_avg_velocityMeridional']
            u = factor * u
            v = factor * v
            fld  = 0.5 * np.sqrt(u*u + v*v)
        else:
            fld = factor * ds[mpasvarname]
        fld = fld.where(depthMask, drop=False)
        fld = (fld * dz).sum(dim='nVertLevels')/layerDepth

        # Compute annual average
        fld = fld.mean(dim='Time')

        # Save year 1 annual average
        dsYear1[varname] = fld.expand_dims(dim='nDepthRanges', axis=0)

        # Plot
        figtitle = f'{vartitle} (avg over z=({np.int32(zmin)}, {np.int32(zmax)}) m), year 1\n{runname}'
        figfile = f'{figdir}/{varname}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_annual_year0001.png'
        make_mosaic_plot(lonCell, latCell, fld, mosaic_descriptor, figtitle, figfile, showEdges=None,
                         cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                         projectionName=projection, lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
        ##dotSize = 0.25
        #dotSize = 1.0
        #make_scatter_plot(lonCell, latCell, dotSize, figtitle, figfile, projectionName=projection,
        #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
        #                  fld=fld, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
    fldYear1.append(dsYear1)
fldYear1 = xr.concat(fldYear1, dim='nDepthRanges')

########################################################
# Then compute annual averages for each subsequent year
# and plot the difference with year 1
#
for year in yearsToPlot:
    print(f'\nCompute annual means for year {year} and plot diff with year 1')
    infiles = f'{indir}/{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-*.nc'
    ds = xr.open_mfdataset(infiles, combine='nested', concat_dim='Time', decode_times=False)
    layerThickness = ds.timeMonthly_avg_layerThickness
    zMid = compute_zmid(depth, maxLevelCell, layerThickness)

    for iz in range(len(zmins)):
        zmin = zmins[iz]
        zmax = zmaxs[iz]
        print(f'  depth range: {zmin}, {zmax}')
        depthMask = np.logical_and(zMid >= zmin, zMid <= zmax)
        dz = layerThickness.where(depthMask, drop=False)
        layerDepth = dz.sum(dim='nVertLevels')

        for var in variables:
            varname = var['name']
            mpasvarname = var['mpasvarname']
            factor = var['factor']
            clevels = var['clevels']
            colormap = var['colormap']
            vartitle = var['title']
            varunits = var['units']
            print(f'    variable: {varname}...')

            # Read in fld and compute depth average
            if varname=='velSpeed':
                u  = ds['timeMonthly_avg_velocityZonal']
                v  = ds['timeMonthly_avg_velocityMeridional']
                u = factor * u
                v = factor * v
                fld  = 0.5 * np.sqrt(u*u + v*v)
            else:
                fld = factor * ds[mpasvarname]
            fld = fld.where(depthMask, drop=False)
            fld = (fld * dz).sum(dim='nVertLevels')/layerDepth
            # Compute annual average
            fld = fld.mean(dim='Time')

            diff = fld - fldYear1[varname].isel(nDepthRanges=iz)

            # Plot
            figtitle = f'{vartitle} (avg over z=({np.int32(zmin)}, {np.int32(zmax)}) m), year {year} minus year 1\n{runname}'
            figfile = f'{figdir}/{varname}_z{np.int32(zmin):05d}_{np.int32(zmax):05d}_annual_diffYears{year:04d}-0001.png'
            make_mosaic_plot(lonCell, latCell, diff, mosaic_descriptor, figtitle, figfile, showEdges=None,
                             cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits,
                             projectionName=projection, lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
            ##dotSize = 0.25
            #dotSize = 1.0
            #make_scatter_plot(lonCell, latCell, dotSize, figtitle, figfile, projectionName=projection,
            #                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
            #                  fld=diff, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)
