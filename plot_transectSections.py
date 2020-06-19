#!/usr/bin/env python
"""

Plot vertical sections (Annual climatology),
given a certain transect mask

"""

# ensure plots are rendered on ICC
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
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
import glob
from mpas_analysis.shared.io.utility import decode_strings

earthRadius = 6367.44

#casename = 'E3SM-Arctic-OSI_60to10' # no spaces
#meshfile = '/global/project/projectdirs/m1199/diagnostics/mpas_analysis/meshes/ocean.ARRM60to10.180715.nc'
#maskfile = '/global/project/projectdirs/m1199/diagnostics/mpas_analysis/region_masks/ARRM60to10_transportTransects_masks.nc'
#climoyearStart = 166
#climoyearEnd = 177
#modeldir = '/global/project/projectdirs/m1199/milena/analysis/mpas/ARRM60to10_new/clim/mpas/avg/unmasked_ARRM60to10'
#
casename = 'E3SM-LRdeck-historical1' # no spaces
casename = 'E3SM-LRtunedHR' # no spaces
meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oEC60to30v3/oEC60to30v3_60layer.170506.nc'
maskfile = '/global/project/projectdirs/e3sm/diagnostics/mpas_analysis/region_masks/oEC60to30v3_transportTransects_masks.nc'
climoyearStart = 26
climoyearEnd = 55
modeldir = '/global/project/projectdirs/e3sm/milena/analysis/mpas/20190509.A_WCYCL1950S_CMIP6_LRtunedHR.ne30_oECv3_ICG.anvil/clim/mpas/avg/unmasked_oEC60to30v3'
#
#casename = 'E3SM-HR' # no spaces (this case gives an error because no climatology of normalVelocity was computed in MPAS-Analysis)
#meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oRRS18to6v3/oRRS18to6v3.171116.nc'
#maskfile = '/global/project/projectdirs/e3sm/diagnostics/mpas_analysis/region_masks/oRRS18to6v3_transportTransects_masks.nc'
#climoyearStart = 26
#climoyearEnd = 55
#modeldir = '/global/project/projectdirs/e3sm/milena/analysis/mpas/theta.20180906.branch_noCNT.A_WCYCL1950S_CMIP6_HR.ne120_oRRS18v3_ICG/clim/mpas/avg/unmasked_oRRS18to6v3'

modelfile = '{}/mpaso_ANN_{:04d}01_{:04d}12_climo.nc'.format(
             modeldir, climoyearStart, climoyearEnd)

# Options for transect names available in 'maskfile':
# "Africa-Ant", "Agulhas", "Antilles Inflow", "Barents Sea Opening", "Bering Strait", "Davis Strait",
# "Drake Passage", "Florida-Bahamas", "Florida-Cuba", "Fram Strait", "Indonesian Throughflow",
# "Lancaster Sound", "Mona Passage", "Mozambique Channel", "Nares Strait", "Tasmania-Ant", "Windward Passage"
#transectNames = ['all']
transectNames = ['Fram Strait']

# Figure details
figdir = './verticalSections'
if not os.path.isdir(figdir):
    os.mkdir(figdir)
figsize = (10, 6)
figdpi = 300
colorIndices = [0, 10, 28, 57, 85, 113, 142, 170, 198, 227, 242, 255]
clevelsT = [-2.0, -1.8, -1.5, -1.0, -0.5, 0.0, 0.5, 2.0, 4.0, 8.0, 12.]
clevelsS = [30.0, 31.0, 32.0, 33.0, 33.5, 34.0, 34.5, 34.8, 35.0, 35.5, 36.0]
clevelsV = [-0.2, -0.15, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
colormapT = plt.get_cmap('RdBu_r')
colormapS = cmocean.cm.haline
colormapV = plt.get_cmap('RdBu_r')
#colormapV = cmocean.cm.balance
colormapT = cols.ListedColormap(colormapT(colorIndices))
colormapS = cols.ListedColormap(colormapS(colorIndices))
colormapV = cols.ListedColormap(colormapV(colorIndices))
cnormT = mpl.colors.BoundaryNorm(clevelsT, colormapT.N)
cnormS = mpl.colors.BoundaryNorm(clevelsS, colormapS.N)
cnormV = mpl.colors.BoundaryNorm(clevelsV, colormapV.N)

# Load in MPAS mesh and transect mask file
mesh = xr.open_dataset(meshfile)
mask = xr.open_dataset(maskfile)

allTransects = decode_strings(mask.transectNames)
if transectNames[0]=='all' or transectNames[0]=='StandardTransportSectionsRegionsGroup':
    transectNames = allTransects

# Get depth
z = mesh.refBottomDepth.values
nlevels = len(z)

# Load in T, S, and normalVelocity for each transect, and plot them
nTransects = len(transectNames)
maxEdges = mask.dims['maxEdgesInTransect']
for n in range(nTransects):
    transectName = transectNames[n]
    transectIndex = allTransects.index(transectName)
    print('Plotting sections for transect: ', transectName)

    # Choose mask for this particular transect
    transectmask = mask.isel(nTransects=transectIndex).squeeze()
    # Get a list of edges for this transect
    transectEdges = transectmask.transectEdgeGlobalIDs.values
    transectEdges = transectEdges[np.where(transectEdges > 0)[0]]
    ntransectEdges = len(transectEdges)

    # Get a list of cellsOnEdge pairs for each transect edge
    cellsOnEdge = mesh.cellsOnEdge.sel(nEdges=transectEdges-1).values

    # Create a land/topo mask for cellsOnEdge
    cellsOnEdge1 = cellsOnEdge[:, 0]
    cellsOnEdge2 = cellsOnEdge[:, 1]
    maxLevelCell1 = mesh.maxLevelCell.sel(nCells=cellsOnEdge1-1).values
    maxLevelCell2 = mesh.maxLevelCell.sel(nCells=cellsOnEdge2-1).values
    # Initialize mask to True everywhere
    cellMask1 = np.ones((ntransectEdges, nlevels), bool)
    cellMask2 = np.ones((ntransectEdges, nlevels), bool)
    for iEdge in range(ntransectEdges):
        # These become False if the second expression is negated (land cells)
        cellMask1[iEdge, :] = np.logical_and(cellMask1[iEdge, :],
                                             cellsOnEdge1[iEdge, np.newaxis] > 0)
        cellMask2[iEdge, :] = np.logical_and(cellMask2[iEdge, :],
                                             cellsOnEdge2[iEdge, np.newaxis] > 0)
        # These become False if the second expression is negated (topography cells)
        cellMask1[iEdge, :] = np.logical_and(cellMask1[iEdge, :],
                                             range(1, nlevels+1) <= maxLevelCell1[iEdge])
        cellMask2[iEdge, :] = np.logical_and(cellMask2[iEdge, :],
                                             range(1, nlevels+1) <= maxLevelCell2[iEdge])

    # Create a land/topo mask for transectEdges
    maxLevelEdge = []
    for iEdge in range(ntransectEdges):
        if cellsOnEdge1[iEdge]==0:
            maxLevelEdge.append(maxLevelCell2[iEdge])
        elif cellsOnEdge2[iEdge]==0:
            maxLevelEdge.append(maxLevelCell1[iEdge])
        else:
            maxLevelEdge.append(np.min([maxLevelCell1[iEdge], maxLevelCell2[iEdge]]))
    # Initialize mask to True everywhere
    edgeMask = np.ones((ntransectEdges, nlevels), bool)
    for iEdge in range(ntransectEdges):
        # These become False if the second expression is negated (topography cells)
        edgeMask[iEdge, :] = np.logical_and(edgeMask[iEdge, :],
                                            range(1, nlevels+1) <= maxLevelEdge[iEdge])

    # Get edge signs for across-edge velocity direction
    edgeSigns = mask.transectEdgeMaskSigns.sel(nEdges=transectEdges-1, nTransects=transectIndex).values
    # Get coordinates of each edge center and compute approximate spherical distance
    lonEdges = mesh.lonEdge.sel(nEdges=transectEdges-1).values
    latEdges = mesh.latEdge.sel(nEdges=transectEdges-1).values
    lonEdges[lonEdges>np.pi] = lonEdges[lonEdges>np.pi] - 2*np.pi
    dist = [0]
    for iEdge in range(1, ntransectEdges):
        dx = (lonEdges[iEdge]-lonEdges[iEdge-1]) * np.cos(0.5*(latEdges[iEdge]+latEdges[iEdge-1]))
        dy = latEdges[iEdge]-latEdges[iEdge-1]
        dist.append(earthRadius * np.sqrt(dx**2 + dy**2))
    dist = np.cumsum(dist)
    [x, y] = np.meshgrid(dist, z)
    x = x.T
    y = y.T
    # Check lon,lat of edges to make sure we have the right edges
    #print(180.0/np.pi*lonEdges)
    #print(180.0/np.pi*latEdges)
    # Check lon,lat of cells to make sure we have the right cellsOnEdge
    #print('lonCell0=', 180/np.pi*mesh.lonCell.sel(nCells=cellsOnEdge1-1).values)
    #print('latCell0=', 180/np.pi*mesh.latCell.sel(nCells=cellsOnEdge1-1).values)
    #print('lonCell1=', 180/np.pi*mesh.lonCell.sel(nCells=cellsOnEdge2-1).values)
    #print('latCell1=', 180/np.pi*mesh.latCell.sel(nCells=cellsOnEdge2-1).values)

    ncid = Dataset(modelfile, 'r')
    # Load in normalVelocity (on edge centers)
    if 'timeMonthly_avg_normalTransportVelocity' in ncid.variables.keys():
        vel = ncid.variables['timeMonthly_avg_normalTransportVelocity'][0, transectEdges-1, :]
    elif 'timeMonthly_avg_normalVelocity' in ncid.variables.keys():
        vel = ncid.variables['timeMonthly_avg_normalVelocity'][0, transectEdges-1, :]
        if 'timeMonthly_avg_normalGMBolusVelocity' in ncid.variables.keys():
            vel += ncid.variables['timeMonthly_avg_normalGMBolusVelocity'][0, transectEdges-1, :]
    else:
        raise KeyError('no appropriate normalVelocity variable found')
    # Load in T and S (on cellsOnEdge centers)
    tempOnCell1 = ncid.variables['timeMonthly_avg_activeTracers_temperature'][0, cellsOnEdge1-1, :]
    tempOnCell2 = ncid.variables['timeMonthly_avg_activeTracers_temperature'][0, cellsOnEdge2-1, :]
    saltOnCell1 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge1-1, :]
    saltOnCell2 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge2-1, :]
    ncid.close()
    # Mask T,S values that fall on land and topography
    tempOnCell1 = np.ma.masked_array(tempOnCell1, ~cellMask1)
    tempOnCell2 = np.ma.masked_array(tempOnCell2, ~cellMask2)
    saltOnCell1 = np.ma.masked_array(saltOnCell1, ~cellMask1)
    saltOnCell2 = np.ma.masked_array(saltOnCell2, ~cellMask2)
    # Interpolate T,S values onto edges
    temp = np.nanmean(np.array([tempOnCell1, tempOnCell2]), axis=0)
    salt = np.nanmean(np.array([saltOnCell1, saltOnCell2]), axis=0)
    # Mask V values that fall on land and topography
    vel = np.ma.masked_array(vel, ~edgeMask)
    # Get normalVelocity direction
    normalVel = vel*edgeSigns[:, np.newaxis]

    zmax = z[np.max(maxLevelEdge)]

    # Plot sections
    #  T first
    figtitle = 'Temperature ({}), ANN (years={}-{})'.format(
               transectName, climoyearStart, climoyearEnd)
    figfile = '{}/Temp_{}_{}_ANN_years{:04d}-{:04d}.png'.format(
              figdir, transectName.replace(' ', ''), casename, climoyearStart, climoyearEnd)
    fig = plt.figure(figsize=figsize, dpi=figdpi)
    ax = fig.add_subplot()
    ax.set_facecolor('darkgrey')
    cf = ax.contourf(x, y, temp, cmap=colormapT, norm=cnormT, levels=clevelsT)
    #cf = ax.pcolormesh(x, y, temp, cmap=colormapT, norm=cnormT)
    cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
    cbar = plt.colorbar(cf, cax=cax, ticks=clevelsT, boundaries=clevelsT, **kw)
    cbar.ax.tick_params(labelsize=12, labelcolor='black')
    cbar.set_label('C$^\circ$', fontsize=12, fontweight='bold')
    ax.set_ylim(0, zmax)
    ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title(figtitle, fontsize=14, fontweight='bold')
    ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
    ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
    ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
    ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
    ax.invert_yaxis()
    plt.savefig(figfile, bbox_inches='tight')
    plt.close()

    #  then S
    figtitle = 'Salinity ({}), ANN (years={}-{})'.format(
               transectName, climoyearStart, climoyearEnd)
    figfile = '{}/Salt_{}_{}_ANN_years{:04d}-{:04d}.png'.format(
              figdir, transectName.replace(' ', ''), casename, climoyearStart, climoyearEnd)
    fig = plt.figure(figsize=figsize, dpi=figdpi)
    ax = fig.add_subplot()
    ax.set_facecolor('darkgrey')
    cf = ax.contourf(x, y, salt, cmap=colormapS, norm=cnormS, levels=clevelsS)
    cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
    cbar = plt.colorbar(cf, cax=cax, ticks=clevelsS, boundaries=clevelsS, **kw)
    cbar.ax.tick_params(labelsize=12, labelcolor='black')
    cbar.set_label('psu', fontsize=12, fontweight='bold')
    ax.set_ylim(0, zmax)
    ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title(figtitle, fontsize=14, fontweight='bold')
    ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
    ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
    ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
    ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
    ax.invert_yaxis()
    plt.savefig(figfile, bbox_inches='tight')
    plt.close()

    #  and finally normalVelocity
    figtitle = 'Velocity ({}), ANN (years={}-{})'.format(
               transectName, climoyearStart, climoyearEnd)
    figfile = '{}/Vel_{}_{}_ANN_years{:04d}-{:04d}.png'.format(
              figdir, transectName.replace(' ', ''), casename, climoyearStart, climoyearEnd)
    fig = plt.figure(figsize=figsize, dpi=figdpi)
    ax = fig.add_subplot()
    ax.set_facecolor('darkgrey')
    cf = ax.contourf(x, y, normalVel, cmap=colormapV, norm=cnormV, levels=clevelsV)
    cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
    cbar = plt.colorbar(cf, cax=cax, ticks=clevelsV, boundaries=clevelsV, **kw)
    cbar.ax.tick_params(labelsize=12, labelcolor='black')
    cbar.set_label('m/s', fontsize=12, fontweight='bold')
    ax.set_ylim(0, zmax)
    ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
    ax.set_title(figtitle, fontsize=14, fontweight='bold')
    ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
    ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
    ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
    ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
    ax.invert_yaxis()
    plt.savefig(figfile, bbox_inches='tight')
    plt.close()
