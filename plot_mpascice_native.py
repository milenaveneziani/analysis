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

def _add_land_lakes_coastline(ax):
    land_50m = cfeature.NaturalEarthFeature(
            'physical', 'land', '50m', edgecolor='face',
            facecolor='lightgray', linewidth=0.5)
    lakes_50m = cfeature.NaturalEarthFeature(
            'physical', 'lakes', '50m', edgecolor='k',
            facecolor='aliceblue',
            linewidth=0.5)
    coast_50m = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '50m', edgecolor='k',
            facecolor='None', linewidth=0.5)
    ax.add_feature(land_50m, zorder=2)
    ax.add_feature(lakes_50m, zorder=3)
    ax.add_feature(coast_50m, zorder=4)

#meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oARRM60to10/ocean.ARRM60to10.180715.nc'
#casename = '20210914.WCYCL1950.ne30pg2_oARRM60to10.hybrid.cori-knl'
#runname = '20210914.WCYCL1950.ne30pg2_oARRM60to10.hybrid'
#modeldir = '/global/cscratch1/sd/dcomeau/e3sm_scratch/cori-knl/{}/run'.format(casename)

meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#casename = '20220810.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.baseline_bdvslat.cori-knl'
#runname = 'arcticx4v1pg2_ARRM10to60E2r1.baseline_bdvslat'
#casename = '20220810.WCYCL1950.ne30pg2_ARRM10to60E2r1.baseline_bdvslat.cori-knl'
#runname = 'ne30pg2_ARRM10to60E2r1.baseline_bdvslat'
#modeldir = '/global/cscratch1/sd/milena/e3sm_scratch/cori-knl/{}/run'.format(casename)
#
#casename = '20220808.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.baseline.cori-knl'
#runname = 'arcticx4v1pg2_ARRM10to60E2r1.baseline'
casename = '20220803.WCYCL1950.ne30pg2_ARRM10to60E2r1.baseline.cori-knl'
runname = 'ne30pg2_ARRM10to60E2r1.baseline'
modeldir = '/global/cscratch1/sd/dcomeau/e3sm_scratch/cori-knl/{}/run'.format(casename)

figdir = './seaice_native/{}'.format(runname)
if not os.path.isdir(figdir):
    os.makedirs(figdir)

varname = 'iceAreaCell' # ice concentration in fraction units (0-1)
varname = 'iceVolumeCell' # ice thickness in m

years = [25]
months = [2, 9]

pi2deg = 180/np.pi

if varname=='iceAreaCell':
    figtext = 'Sea-ice concentration ({})'.format(runname)
    units = 'ice fraction'
    clevels_obs = [0.15, 0.80]
    #clevels_obs = [0.15, 0.80, 0.95]
    # Colormap for model field
    #colormap = cmocean.cm.ice
    #colorIndices = [0, 40, 80, 120, 160, 180, 200, 240, 255]
    #colormap = cols.ListedColormap(colormap(colorIndices))
    colormap = cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
                                    (0.169, 0.435, 0.223), (0.455, 0.478, 0.196), (0.757, 0.474, 0.435),\
                                    (0.827, 0.561, 0.772), (0.761, 0.757, 0.949), (0.808, 0.921, 0.937)])
    clevels = [0.15, 0.3, 0.45, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1]
elif varname=='iceVolumeCell':
    figtext = 'Sea-ice thickness ({})'.format(runname)
    units = 'meters'
    #clevels_obs = [2, 3.5]
    clevels_obs = [2]
    # Colormap for model field
    #colormap = cmocean.cm.deep_r
    #colormap = cmocean.cm.thermal
    colormap = plt.get_cmap('YlGnBu_r')
    colorIndices = [0, 40, 80, 120, 160, 180, 200, 240, 255]
    colormap = cols.ListedColormap(colormap(colorIndices))
    #colormap = cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
    #                                (0.169, 0.435, 0.223), (0.455, 0.478, 0.196), (0.757, 0.474, 0.435),\
    #                                (0.827, 0.561, 0.772), (0.761, 0.757, 0.949), (0.808, 0.921, 0.937)])
    #clevels = [0., 0.5, 1.0, 1.5, 2.0, 2.5, 3., 3.5, 4., 5.]
    clevels = [0., 0.2, 1.0, 1.5, 2.0, 2.5, 3., 3.5, 4., 5.]

else:
    raise SystemExit('varname {} not supported'.format(varname))
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
        modelfile = '{}/{}.mpassi.hist.am.timeSeriesStatsMonthly.{:04d}-{:02d}-01.nc'.format(modeldir,\
                    casename, year, month)
        print(modelfile)
        f = netcdf_dataset(modelfile, mode='r')
        fld = f.variables['timeMonthly_avg_{}'.format(varname)][:, :]
        f.close()
        fld = np.squeeze(fld)
        #if varname!='iceAreaCell':
        #    iceFrac = f.variables['timeMonthly_avg_iceAreaCell'][:, :]
        #    fld = ma.masked_less(iceFrac, 0.15)
        #else:
        #    fld = ma.masked_less(fld, 0.15)
        ##fld = ma.masked_less(fld, 0.1)

        plt.figure(figsize=figsize, dpi=figdpi)
        figtitle = '{}, Year={:04d}, Month={:02d}'.format(figtext, year, month)

        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
        #ax = plt.axes(projection=ccrs.Miller(central_longitude=143))

        _add_land_lakes_coastline(ax)

        data_crs = ccrs.PlateCarree()
        ax.set_extent([-180, 180, 50, 90], crs=data_crs)
        #ax.set_extent([132.0, 154.0, 70., 85.], crs=data_crs)
        gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=5)
        # This will work with cartopy 0.18:
        #gl.xlocator = mticker.FixedLocator(np.arange(132., 154., 2.))
        #gl.ylocator = mticker.FixedLocator(np.arange(75., 84., 1.))

        # Plot model field
        sc = ax.scatter(lon, lat, s=0.2, c=fld, cmap=colormap, norm=cnorm,
        #sc = ax.scatter(lon, lat, s=100, c=fld, cmap=colormap, norm=cnorm,
                        marker='o', transform=data_crs)
        cax, kwc = mpl.colorbar.make_axes(ax, location='bottom', pad=0.05, shrink=0.7)
        cbar = plt.colorbar(sc, cax=cax, ticks=clevels, boundaries=clevels, extend='max', **kwc)
        cbar.ax.tick_params(labelsize=22, labelcolor='black')
        cbar.set_label(units, fontsize=20, fontweight='bold')

        ax.set_title(figtitle, y=1.04, fontsize=22, fontweight='bold')
        figfile = '{}/{}NH_{}_{:04d}-{:02d}.png'.format(figdir, varname, runname, year, month)
        plt.savefig(figfile, bbox_inches='tight')
        plt.close()
