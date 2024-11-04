from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import glob
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
from matplotlib.pyplot import cm
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cmocean

from make_plots import make_scatter_plot


# Settings for nersc
#meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oARRM60to10/ocean.ARRM60to10.180715.nc'
#casename = '20210914.WCYCL1950.ne30pg2_oARRM60to10.hybrid.cori-knl'
#runname = '20210914.WCYCL1950.ne30pg2_oARRM60to10.hybrid'
#modeldir = f'/global/cscratch1/sd/dcomeau/e3sm_scratch/cori-knl/{casename}/run'
#
#meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#casename = '20220810.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.baseline_bdvslat.cori-knl'
#runname = 'arcticx4v1pg2_ARRM10to60E2r1.baseline_bdvslat'
#casename = '20220810.WCYCL1950.ne30pg2_ARRM10to60E2r1.baseline_bdvslat.cori-knl'
#runname = 'ne30pg2_ARRM10to60E2r1.baseline_bdvslat'
#modeldir = f'/global/cscratch1/sd/milena/e3sm_scratch/cori-knl/{casename}/run'

# Settings for lcrc
meshfile = '/lcrc/group/e3sm/public_html/inputdata/ocn/mpas-o/EC30to60E2r2/ocean.EC30to60E2r2.210210.nc'
casename = '20241030.EC30to60_test.anvil'
runname = '20241030.EC30to60_test.anvil'
modeldir = f'/lcrc/group/e3sm/ac.vanroekel/scratch/anvil/{casename}/run'

figdir = f'./seaice_native/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

varname = 'iceAreaCell' # ice concentration in fraction units (0-1)
#varname = 'iceVolumeCell' # ice thickness in m

mpasFile = 'highFrequencyOutput'
mpasFileDayformat = '01'
mpasvarname = varname
#timeIndex = 48 # 1 day
timeIndex = 336 # 7 days

#mpasFile = 'timeSeriesStatsMonthly'
#mpasFileDayformat = '01'
#mpasvarname = f'timeMonthly_avg_{varname}'
#timeIndex = 0

years = [1]
months = [1]

pi2deg = 180/np.pi

if varname=='iceAreaCell':
    figtext = f'Sea-ice concentration ({runname})'
    units = 'ice fraction'
    clevels_obs = [0.15, 0.80]
    #clevels_obs = [0.15, 0.80, 0.95]
    # Colormap for model field
    #colormap = cmocean.cm.ice
    #colorIndices = [0, 40, 80, 120, 160, 180, 200, 240, 255]
    #colormap = cols.ListedColormap(colormap(colorIndices))
    colorIndices = None
    colormap = cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
                                    (0.169, 0.435, 0.223), (0.455, 0.478, 0.196), (0.757, 0.474, 0.435),\
                                    (0.827, 0.561, 0.772), (0.761, 0.757, 0.949), (0.808, 0.921, 0.937)])
    clevels = [0.15, 0.3, 0.45, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1]
elif varname=='iceVolumeCell':
    figtext = f'Sea-ice thickness ({runname})'
    units = 'meters'
    #clevels_obs = [2, 3.5]
    clevels_obs = [2]
    # Colormap for model field
    #colormap = cmocean.cm.deep_r
    #colormap = cmocean.cm.thermal
    colormap = plt.get_cmap('YlGnBu_r')
    colorIndices = [0, 40, 80, 120, 160, 180, 200, 240, 255]
    colormap = cols.ListedColormap(colormap(colorIndices))
    #colorIndices = None
    #colormap = cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
    #                                (0.169, 0.435, 0.223), (0.455, 0.478, 0.196), (0.757, 0.474, 0.435),\
    #                                (0.827, 0.561, 0.772), (0.761, 0.757, 0.949), (0.808, 0.921, 0.937)])
    #clevels = [0., 0.5, 1.0, 1.5, 2.0, 2.5, 3., 3.5, 4., 5.]
    clevels = [0., 0.2, 1.0, 1.5, 2.0, 2.5, 3., 3.5, 4., 5.]

else:
    raise SystemExit(f'varname {varname} not supported')
cnorm = mpl.colors.BoundaryNorm(clevels, colormap.N)

# Info about MPAS mesh
f = netcdf_dataset(meshfile, mode='r')
lon = f.variables['lonCell'][:]
lat = f.variables['latCell'][:]
f.close()
lon = pi2deg*lon
lat = pi2deg*lat

figsize = [20, 20]
figdpi = 150

for year in years:
    for month in months:
        modelfile = f'{modeldir}/{casename}.mpassi.hist.am.{mpasFile}.{year:04d}-{month:02d}-{mpasFileDayformat}.nc'
        f = netcdf_dataset(modelfile, mode='r')
        fld = f.variables[mpasvarname][timeIndex, :]
        f.close()
        fld = np.squeeze(fld)
        #if varname!='iceAreaCell':
        #    iceFrac = f.variables['timeMonthly_avg_iceAreaCell'][timeIndex, :]
        #    fld = ma.masked_less(iceFrac, 0.15)
        #else:
        #    fld = ma.masked_less(fld, 0.15)
        ##fld = ma.masked_less(fld, 0.1)

        figtitle = f'{figtext}, Year={year:04d}, Month={month:02d}'

        figfile = f'{figdir}/{varname}NH_{runname}_{year:04d}-{month:02d}timeIndex{timeIndex:02d}.png'
        #dotSize = 5.0 # this should go up as resolution decreases # for ARRM
        #dotSize = 0.2
        dotSize = 25.0 # this should go up as resolution decreases # for LR
        make_scatter_plot(lon, lat, dotSize, figtitle, figfile, projectionName='NorthPolarStereo',
          lon0=-180, lon1=180, dlon=20.0, lat0=50, lat1=90, dlat=10.0,
          fld=fld, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=units)

        figfile = f'{figdir}/{varname}SH_{runname}_{year:04d}-{month:02d}timeIndex{timeIndex:02d}.png'
        dotSize = 25.0 # this should go up as resolution decreases # for LR
        make_scatter_plot(lon, lat, dotSize, figtitle, figfile, projectionName='SouthPolarStereo',
          lon0=-180, lon1=180, dlon=20.0, lat0=-55, lat1=-90, dlat=10.0,
          fld=fld, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=units)
