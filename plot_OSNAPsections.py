"""

Plot OSNAP sections (annual and seasonal climatologies)

"""

from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import glob
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cols
from matplotlib.pyplot import cm
from matplotlib.colors import BoundaryNorm
import cmocean
import xarray as xr
from netCDF4 import Dataset
import gsw

earthRadius = 6367.44

obsfile = '/compyfs/vene705/observations/OSNAP/OSNAP_Gridded_TS_201408_201604_2018.nc'

# Figure details
figdir = './OSNAPsections'
if not os.path.isdir(figdir):
    os.mkdir(figdir)
figsize = (10, 6)
figdpi = 300
colorIndices0 = [0, 10, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]
clevelsT = [-1.0, -0.5, 0.0, 0.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8., 10., 12.]
clevelsS = [31.0, 33.0, 33.5, 33.8, 34.2, 34.6, 34.8, 34.85, 34.9, 34.95, 35.0, 35.2, 35.5]
colormapT = plt.get_cmap('RdBu_r')
colormapS = cmocean.cm.haline
#
underColor = colormapT(colorIndices0[0])
overColor = colormapT(colorIndices0[-1])
if len(clevelsT) + 1 == len(colorIndices0):
    # we have 2 extra values for the under/over so make the colormap
    # without these values
    colorIndices = colorIndices0[1:-1]
elif len(clevelsT) - 1 != len(colorIndices0):
    # indices list must be either one element shorter
    # or one element longer than colorbarLevels list
    raise ValueError('length mismatch between indices and '
                     'T colorbarLevels')
colormapT = cols.ListedColormap(colormapT(colorIndices))
colormapT.set_under(underColor)
colormapT.set_over(overColor)
underColor = colormapS(colorIndices0[0])
overColor = colormapS(colorIndices0[-1])
if len(clevelsS) + 1 == len(colorIndices0):
    # we have 2 extra values for the under/over so make the colormap
    # without these values
    colorIndices = colorIndices0[1:-1]
elif len(clevelsS) - 1 != len(colorIndices0):
    # indices list must be either one element shorter
    # or one element longer than colorbarLevels list
    raise ValueError('length mismatch between indices and '
                     'S colorbarLevels')
colormapS = cols.ListedColormap(colormapS(colorIndices))
colormapS.set_under(underColor)
colormapS.set_over(overColor)
#
cnormT = mpl.colors.BoundaryNorm(clevelsT, colormapT.N)
cnormS = mpl.colors.BoundaryNorm(clevelsS, colormapS.N)

#sigma2contours = [35, 36, 36.5, 36.8, 37, 37.1, 37.2, 37.25, 37.44, 37.52, 37.6]
sigma2contours = None
sigma0contours = np.arange(26.0, 28.0, 0.2)
#sigma0contours = None
print(sigma0contours)

degtopi = np.pi/180.0

# Data goes from Aug 2014 to Apr 2016:
months = np.array([8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4])
indJFM = np.nonzero(np.logical_or.reduce((months==1, months==2, months==3)))
indJAS = np.nonzero(np.logical_or.reduce((months==7, months==8, months==9)))

ncid = Dataset(obsfile, 'r')
temp = ncid.variables['TEMP'][:]
salt = ncid.variables['PSAL'][:]
lon = ncid.variables['LONGITUDE'][:]
lat = ncid.variables['LATITUDE'][:]
depth = ncid.variables['DEPTH'][:]
ncid.close()

# Compute climatologies for temperature and salinity
tempJFM = np.nanmean(np.squeeze(temp[indJFM, :, :]), axis=0)
tempJAS = np.nanmean(np.squeeze(temp[indJAS, :, :]), axis=0)
tempANN = np.nanmean(temp, axis=0)
saltJFM = np.nanmean(np.squeeze(salt[indJFM, :, :]), axis=0)
saltJAS = np.nanmean(np.squeeze(salt[indJAS, :, :]), axis=0)
saltANN = np.nanmean(salt, axis=0)

# Compute sigma's
lonmean = np.nanmean(lon)
latmean = np.nanmean(lat)
pressure = gsw.p_from_z(-depth, latmean)

SA = gsw.SA_from_SP(saltJFM, pressure[:, np.newaxis], lonmean, latmean)
CT = gsw.CT_from_pt(SA, tempJFM)
sigma2JFM = gsw.density.sigma2(SA, CT)
sigma0JFM = gsw.density.sigma0(SA, CT)

SA = gsw.SA_from_SP(saltJAS, pressure[:, np.newaxis], lonmean, latmean)
CT = gsw.CT_from_pt(SA, tempJAS)
sigma2JAS = gsw.density.sigma2(SA, CT)
sigma0JAS = gsw.density.sigma0(SA, CT)

SA = gsw.SA_from_SP(saltANN, pressure[:, np.newaxis], lonmean, latmean)
CT = gsw.CT_from_pt(SA, tempANN)
sigma2ANN = gsw.density.sigma2(SA, CT)
sigma0ANN = gsw.density.sigma0(SA, CT)

# Plot OSNAP section West
k0 = 0
k1 = 77
dist = [0]
for k in range(k0+1, k1):
    dx = degtopi * (lon[k]-lon[k-1]) * np.cos(0.5*degtopi*(lat[k]+lat[k-1]))
    dy = degtopi * (lat[k]-lat[k-1])
    dist.append(earthRadius * np.sqrt(dx**2 + dy**2))
dist = np.cumsum(dist)
[x, y] = np.meshgrid(dist, depth)

#  T first
figtitle = 'Temperature (OSNAP West), JFM (years=2014-2016)'
figfile = '{}/Temp_OSNAPwest_JFM.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, tempJFM[:, k0:k1], cmap=colormapT, norm=cnormT, levels=clevelsT, extend='max')
#cf = ax.pcolormesh(x, y, tempJFM[:, k0:k1], cmap=colormapT, norm=cnormT)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsT, boundaries=clevelsT, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('C$^\circ$', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2JFM[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0JFM[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

figtitle = 'Temperature (OSNAP West), JAS (years=2014-2016)'
figfile = '{}/Temp_OSNAPwest_JAS.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, tempJAS[:, k0:k1], cmap=colormapT, norm=cnormT, levels=clevelsT, extend='max')
#cf = ax.pcolormesh(x, y, tempJAS[:, k0:k1], cmap=colormapT, norm=cnormT)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsT, boundaries=clevelsT, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('C$^\circ$', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2JAS[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0JAS[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

figtitle = 'Temperature (OSNAP West), ANN (years=2014-2016)'
figfile = '{}/Temp_OSNAPwest_ANN.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, tempANN[:, k0:k1], cmap=colormapT, norm=cnormT, levels=clevelsT, extend='max')
#cf = ax.pcolormesh(x, y, tempANN[:, k0:k1], cmap=colormapT, norm=cnormT)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsT, boundaries=clevelsT, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('C$^\circ$', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2ANN[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0ANN[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

# Then S
figtitle = 'Salinity (OSNAP West), JFM (years=2014-2016)'
figfile = '{}/Salt_OSNAPwest_JFM.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, saltJFM[:, k0:k1], cmap=colormapS, norm=cnormS, levels=clevelsS, extend='max')
#cf = ax.pcolormesh(x, y, saltJFM[:, k0:k1], cmap=colormapS, norm=cnormS)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsS, boundaries=clevelsS, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('psu', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2JFM[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0JFM[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

figtitle = 'Salinity (OSNAP West), JAS (years=2014-2016)'
figfile = '{}/Salt_OSNAPwest_JAS.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, saltJAS[:, k0:k1], cmap=colormapS, norm=cnormS, levels=clevelsS, extend='max')
#cf = ax.pcolormesh(x, y, saltJAS[:, k0:k1], cmap=colormapS, norm=cnormS)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsS, boundaries=clevelsS, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('psu', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2JAS[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0JAS[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

figtitle = 'Salinity (OSNAP West), ANN (years=2014-2016)'
figfile = '{}/Salt_OSNAPwest_ANN.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, saltANN[:, k0:k1], cmap=colormapS, norm=cnormS, levels=clevelsS, extend='max')
#cf = ax.pcolormesh(x, y, saltANN[:, k0:k1], cmap=colormapS, norm=cnormS)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsS, boundaries=clevelsS, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('psu', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2ANN[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0ANN[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

# Plot OSNAP section East
k0 = 77
k1 = 252
dist = [0]
for k in range(k0+1, k1):
    dx = degtopi * (lon[k]-lon[k-1]) * np.cos(0.5*degtopi*(lat[k]+lat[k-1]))
    dy = degtopi * (lat[k]-lat[k-1])
    dist.append(earthRadius * np.sqrt(dx**2 + dy**2))
dist = np.cumsum(dist)
[x, y] = np.meshgrid(dist, depth)

#  T first
figtitle = 'Temperature (OSNAP East), JFM (years=2014-2016)'
figfile = '{}/Temp_OSNAPeast_JFM.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, tempJFM[:, k0:k1], cmap=colormapT, norm=cnormT, levels=clevelsT, extend='max')
#cf = ax.pcolormesh(x, y, tempJFM[:, k0:k1], cmap=colormapT, norm=cnormT)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsT, boundaries=clevelsT, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('C$^\circ$', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2JFM[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0JFM[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

figtitle = 'Temperature (OSNAP East), JAS (years=2014-2016)'
figfile = '{}/Temp_OSNAPeast_JAS.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, tempJAS[:, k0:k1], cmap=colormapT, norm=cnormT, levels=clevelsT, extend='max')
#cf = ax.pcolormesh(x, y, tempJAS[:, k0:k1], cmap=colormapT, norm=cnormT)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsT, boundaries=clevelsT, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('C$^\circ$', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2JAS[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0JAS[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

figtitle = 'Temperature (OSNAP East), ANN (years=2014-2016)'
figfile = '{}/Temp_OSNAPeast_ANN.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, tempANN[:, k0:k1], cmap=colormapT, norm=cnormT, levels=clevelsT, extend='max')
#cf = ax.pcolormesh(x, y, tempANN[:, k0:k1], cmap=colormapT, norm=cnormT)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsT, boundaries=clevelsT, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('C$^\circ$', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2ANN[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0ANN[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

# Then S
figtitle = 'Salinity (OSNAP East), JFM (years=2014-2016)'
figfile = '{}/Salt_OSNAPeast_JFM.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, saltJFM[:, k0:k1], cmap=colormapS, norm=cnormS, levels=clevelsS, extend='max')
#cf = ax.pcolormesh(x, y, saltJFM[:, k0:k1], cmap=colormapS, norm=cnormS)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsS, boundaries=clevelsS, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('psu', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2JFM[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0JFM[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

figtitle = 'Salinity (OSNAP East), JAS (years=2014-2016)'
figfile = '{}/Salt_OSNAPeast_JAS.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, saltJAS[:, k0:k1], cmap=colormapS, norm=cnormS, levels=clevelsS, extend='max')
#cf = ax.pcolormesh(x, y, saltJAS[:, k0:k1], cmap=colormapS, norm=cnormS)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsS, boundaries=clevelsS, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('psu', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2JAS[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0JAS[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()

figtitle = 'Salinity (OSNAP East), ANN (years=2014-2016)'
figfile = '{}/Salt_OSNAPeast_ANN.png'.format(figdir)
fig = plt.figure(figsize=figsize, dpi=figdpi)
ax = fig.add_subplot()
ax.set_facecolor('darkgrey')
cf = ax.contourf(x, y, saltANN[:, k0:k1], cmap=colormapS, norm=cnormS, levels=clevelsS, extend='max')
#cf = ax.pcolormesh(x, y, saltANN[:, k0:k1], cmap=colormapS, norm=cnormS)
cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
cbar = plt.colorbar(cf, cax=cax, ticks=clevelsS, boundaries=clevelsS, **kw)
cbar.ax.tick_params(labelsize=12, labelcolor='black')
cbar.set_label('psu', fontsize=12, fontweight='bold')
if sigma2contours is not None:
    cs = ax.contour(x, y, sigma2ANN[:, k0:k1], sigma2contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
if sigma0contours is not None:
    cs = ax.contour(x, y, sigma0ANN[:, k0:k1], sigma0contours, colors='k', linewidths=1.5)
    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
#ax.set_ylim(0, zmax)
ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
ax.set_title(figtitle, fontsize=14, fontweight='bold')
ax.annotate('lat={:5.2f}'.format(lat[k0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lat={:5.2f}'.format(lat[k1-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
ax.annotate('lon={:5.2f}'.format(lon[k1-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
ax.invert_yaxis()
plt.savefig(figfile, bbox_inches='tight')
plt.close()
