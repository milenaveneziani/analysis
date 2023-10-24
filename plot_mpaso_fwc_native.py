from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import glob
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import numpy.ma as ma
from scipy.ndimage.filters import gaussian_filter
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

from common_functions import add_land_lakes_coastline


#meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oARRM60to10/ocean.ARRM60to10.180715.nc'
#runname = 'E3SM-Arctic60to10'
meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oEC60to30v3/oEC60to30v3_60layer.170506.nc'
#runname = '20190509.A_WCYCL1950S_CMIP6_LRtunedHR.ne30_oECv3_ICG.anvil'
#runname = '20180215.DECKv1b_H1.ne30_oEC.edison'
runname = 'E3SM-LR-OSI'
#meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oRRS18to6v3/oRRS18to6v3.171116.nc'
#runname = 'theta.20180906.branch_noCNT.A_WCYCL1950S_CMIP6_HR.ne120_oRRS18v3_ICG'
#climoyearStart = 119
#climoyearEnd = 133
#climoyearStart = 148
#climoyearEnd = 157
#climoyearStart = 125
#climoyearEnd = 149
#climoyearStart = 1
#climoyearEnd = 15
climoyearStart = 166
climoyearEnd = 177
#climoyearStart = 1985
#climoyearEnd = 2014
#climoyearStart = 26
#climoyearEnd = 55
#modeldir = '/global/project/projectdirs/m1199/milena/analysis/mpas/ARRM60to10_new/clim/mpas/avg/unmasked_ARRM60to10'
modeldir = '/global/project/projectdirs/m1199/milena/analysis/mpas/E3SM60to30/clim/mpas/avg/unmasked_oEC60to30v3'
#modeldir = '/global/project/projectdirs/e3sm/milena/analysis/mpas/20190509.A_WCYCL1950S_CMIP6_LRtunedHR.ne30_oECv3_ICG.anvil/clim/mpas/avg/unmasked_oEC60to30v3'
#modeldir = '/global/project/projectdirs/e3sm/milena/analysis/mpas/20180215.DECKv1b_H1.ne30_oEC.edison/clim/mpas/avg/unmasked_oEC60to30v3'
#modeldir = '/global/project/projectdirs/e3sm/milena/analysis/mpas/theta.20180906.branch_noCNT.A_WCYCL1950S_CMIP6_HR.ne120_oRRS18v3_ICG/clim/mpas/avg/unmasked_oRRS18to6v3'
modelfile = '{}/mpaso_ANN_{:04d}01_{:04d}12_climo.nc'.format(
             modeldir, climoyearStart, climoyearEnd)
#modeldir = '/global/project/projectdirs/e3sm/milena/postprocessing/{}/means_years{}-{}'.format(
#            runname, climoyearStart, climoyearEnd)
#modelfile = '{}/activeTracers_salinity/activeTracers_salinity_mpaso_ANN_{:04d}_{:04d}_mean.nc'.format(
#             modeldir, climoyearStart, climoyearEnd)

figdir = './fwc_native'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figsize = [20, 20]
figdpi = 100

sref = 34.8

pi2deg = 180/np.pi

varname = 'activeTracers_salinity'
mpasvarname = 'timeMonthly_avg_{}'.format(varname)

colorIndices = [0, 28, 57, 85, 113, 142, 170, 198, 227, 242, 255]
clevels_z = [0., 50., 150., 250., 500., 750., 1000., 1500., 2000., 3000.]
clevels_fwc = [0., 5., 10., 15., 17.5, 20., 22.5, 25., 27.5, 30.]
colormap_z = cmocean.cm.deep_r
colormap_fwc = plt.get_cmap('Spectral_r')
colormap_z = cols.ListedColormap(colormap_z(colorIndices))
colormap_fwc = cols.ListedColormap(colormap_fwc(colorIndices))
cnorm_z = mpl.colors.BoundaryNorm(clevels_z, colormap_z.N)
cnorm_fwc = mpl.colors.BoundaryNorm(clevels_fwc, colormap_fwc.N)

# Info about MPAS mesh
f = netcdf_dataset(meshfile, mode='r')
lon = f.variables['lonCell'][:]
lat = f.variables['latCell'][:]
z = f.variables['refBottomDepth'][:]
f.close()
lon = pi2deg*lon
lat = pi2deg*lat
nCells = np.shape(lon)[0]
dz = np.ones(np.shape(z))
dz[0] = z[0]
dz[1:] = np.diff(z)

figtitle_z = 'Depth of S=Sref, ANN climatology over years={}-{}'.format(
              climoyearStart, climoyearEnd)
figtitle_fwc = 'FWC, ANN climatology over years={}-{}'.format(
                climoyearStart, climoyearEnd)
figfile_z = '{}/ArcticFWCdepth_{}_ANN_years{:04d}-{:04d}.png'.format(
             figdir, runname, climoyearStart, climoyearEnd)
figfile_fwc = '{}/ArcticFWC_{}_ANN_years{:04d}-{:04d}.png'.format(
               figdir, runname, climoyearStart, climoyearEnd)

f = netcdf_dataset(modelfile, mode='r')
salinity = f.variables[mpasvarname][0, :, :]
f.close()
salinity = np.squeeze(salinity)
salinity[salinity>1e15] = np.nan
salinity[salinity<-1e15] = np.nan
srefDepth = np.zeros(nCells)
fwc = np.zeros(nCells)
for icell in range(nCells):
    zindex = np.where(salinity[icell, :]<=sref)[0]
    if zindex.any():
        srefDepth[icell] = z[zindex[-1]]
        fwc_tmp = (sref - salinity[icell, zindex])/sref * dz[zindex]
        fwc[icell] = fwc_tmp.sum()
        #print(lon[icell], lat[icell], salinity[icell, zindex])

plt.figure(figsize=figsize, dpi=figdpi)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
add_land_lakes_coastline(ax)
data_crs = ccrs.PlateCarree()
ax.set_extent([-180, 180, 70, 90], crs=data_crs)
gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6)
# This will work with cartopy 0.18:
#gl.xlocator = mticker.FixedLocator(np.arange(-180., 181., 20.))
#gl.ylocator = mticker.FixedLocator(np.arange(-80., 81., 10.))
# for E3SM-Arctic mesh
#sc = ax.scatter(lon, lat, s=50.0, c=srefDepth, cmap=colormap_z, norm=cnorm_z,
# for EC60to30 mesh
sc = ax.scatter(lon, lat, s=75.0, c=srefDepth, cmap=colormap_z, norm=cnorm_z,
                marker='o', transform=data_crs)
cax, kw = mpl.colorbar.make_axes(ax, location='bottom', pad=0.05, shrink=0.7)
cbar = plt.colorbar(sc, cax=cax, ticks=clevels_z, boundaries=clevels_z, extend='both', **kw)
cbar.ax.tick_params(labelsize=22, labelcolor='black')
cbar.set_label('[m]', fontsize=20)
ax.set_title(figtitle_z, y=1.04, fontsize=22)
plt.savefig(figfile_z, bbox_inches='tight')
plt.close()

plt.figure(figsize=figsize, dpi=figdpi)
ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
add_land_lakes_coastline(ax)
data_crs = ccrs.PlateCarree()
ax.set_extent([-180, 180, 70, 90], crs=data_crs)
gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6)
# for E3SM-Arctic mesh
#sc = ax.scatter(lon, lat, s=50.0, c=fwc, cmap=colormap_fwc, norm=cnorm_fwc,
# for EC60to30 mesh
sc = ax.scatter(lon, lat, s=75.0, c=fwc, cmap=colormap_fwc, norm=cnorm_fwc,
                marker='o', transform=data_crs)
cax, kw = mpl.colorbar.make_axes(ax, location='bottom', pad=0.05, shrink=0.7)
cbar = plt.colorbar(sc, cax=cax, ticks=clevels_fwc, boundaries=clevels_fwc, extend='both', **kw)
cbar.ax.tick_params(labelsize=22, labelcolor='black')
cbar.set_label('[m]', fontsize=20)
ax.set_title(figtitle_fwc, y=1.04, fontsize=22)
plt.savefig(figfile_fwc, bbox_inches='tight')
plt.close()
