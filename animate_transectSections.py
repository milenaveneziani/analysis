#!/usr/bin/env python
"""

Plot vertical sections (annual and seasonal climatologies),
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
from mpas_analysis.shared.io.utility import decode_strings
import gsw

from geometric_features import FeatureCollection, read_feature_collection

from common_functions import add_inset

earthRadius = 6367.44

####### Settings for onyx
#   NOTE: make sure to use the same mesh file that is in streams.ocean!
#meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#maskfile = '/p/work/milena/mpas-region_masks/ARRM10to60E2r1_arcticBeringToNorwaySection20230523.nc'
#casename = 'E3SMv2.1B60to10rA02'

####### Settings for cori
featurefile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/arctic_atlantic_budget_regionsTransects.geojson'
#featurefile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/arcticSections20210323.geojson'
#featurefile = '/global/homes/m/milena/proj_e3sm/milena/mpas-region_masks/standardTransportSections.geojson'
#featurefile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/arcticTransectsFramToBeaufortEast20230901.geojson'
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_arctic_atlantic_budget_regionsTransects20230313.nc'
#maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arcticSections20220916.nc'
#maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_standardTransportSections20210323.nc'
casename = 'E3SMv2.1B60to10rA02'
cname = 'E3SM-Arctic'
modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{casename}/ocn/singleVarFiles'
#meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/EC30to60E2r2/ocean.EC30to60E2r2.210210.nc'
#maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/EC30to60E2r2_arctic_atlantic_budget_regionsTransects20230313.nc'
#maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/EC30to60E2r2_arcticSections20220914.nc'
#maskfile = '/global/cfs/cdirs/e3sm/milena/mpas-region_masks/EC30to60E2r2_standardTransportSections20210323.nc'
#maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/EC30to60E2r2_arcticTransectsFramToBeaufortEast20230901.nc'
#casename = 'GMPAS-JRA1p4_EC30to60E2r2_GM600_Redi600_perlmutter'
#cname = 'GM600_Redi600'
#modeldir = f'/global/cfs/cdirs/e3sm/maltrud/archive/onHPSS/{casename}/ocn/hist'
yearStart = 1
yearEnd = 30

#transectNames = ['all']
#transectNames = ['Smith Bay - Beaufort Shelf West', 'Kaktovik - Beaufort Shelf Central', 'Mackenzie Shelf - Beaufort Shelf Central', 'Banks Island - Beaufort Shelf East']
# For Baremts, Kara, and Laptev Seas:
#transectNames = ['Fram Strait', 'Barents Sea Opening', 'Novaya Zemlya to Gakkel Ridge', 'Severnaya Zemlya to Gakkel Ridge', 'Novosibirskiye Islands to Lomonosov Ridge']
# For East Siberian, Chukchi, and Canada Basin:
#transectNames = ['Wrangel Island to Chukchi Plateau', 'Bering Sea North', 'Enurmino to Point Hope - Chukchi South', 'Chukchi Central', 'Herald Canyon to Icy Cape - Chukchi North', 'Wrangel Island to Russian coast', 'Barrow Canyon', 'Smith Bay - Beaufort Shelf West', 'Kaktovik - Beaufort Shelf Central', 'Mackenzie Shelf - Beaufort Shelf Central', 'Banks Island - Beaufort Shelf East', 'Bering Strait']
#
#transectNames = ['Fram Strait', 'Denmark Strait', 'Iceland-Faroe-Scotland', 'OSNAP section East', 'OSNAP section West']
transectNames = ['Atlantic zonal 50N', 'Atlantic zonal 27.2N', 'South Atlantic Ocean 34S']
#transectNames = ['Drake Passage']

zmaxUpperPanel = 100.0

figdir = f'./animations_verticalSections/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)
framesdir = f'{figdir}/frames'
if not os.path.isdir(framesdir):
    os.makedirs(framesdir)
outdir = f'./verticalSections_data/{casename}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found for this region group')

years = range(yearStart, yearEnd+1)
months = range(1, 13)

# Figure details
figdpi = 300

clevelsT = [-1.8, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10., 12.]
clevelsS = [30.0, 31.0, 32.0, 33.0, 33.5, 33.8, 34.0, 34.2, 34.4, 34.6, 34.8, 34.82, 34.84, 34.86, 34.88, 34.9, 34.95, 35.0, 35.5]
clevelsV = [-0.5, -0.3, -0.25, -0.2, -0.15, -0.1, -0.02, 0.0, 0.02, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
colormapT = plt.get_cmap(cmocean.cm.thermal)
colormapS = plt.get_cmap(cmocean.cm.haline)
colormapV = plt.get_cmap('RdBu_r')
colorIndicesT = np.int_(np.linspace(0, 255, num=len(clevelsT)+1, endpoint=True))
colorIndicesS = np.int_(np.linspace(0, 255, num=len(clevelsS)+1, endpoint=True))
colorIndicesV = np.int_(np.linspace(0, 255, num=len(clevelsV)+1, endpoint=True))
#
underColorT = colormapT(colorIndicesT[0])
overColorT = colormapT(colorIndicesT[-1])
colorIndicesT = colorIndicesT[1:-1]
colormapT = cols.ListedColormap(colormapT(colorIndicesT))
colormapT.set_under(underColorT)
colormapT.set_over(overColorT)
#
underColorS = colormapS(colorIndicesS[0])
overColorS = colormapS(colorIndicesS[-1])
colorIndicesS = colorIndicesS[1:-1]
colormapS = cols.ListedColormap(colormapS(colorIndicesS))
colormapS.set_under(underColorS)
colormapS.set_over(overColorS)
#
underColorV = colormapV(colorIndicesV[0])
overColorV = colormapV(colorIndicesV[-1])
colorIndicesV = colorIndicesV[1:-1]
colormapV = cols.ListedColormap(colormapV(colorIndicesV))
colormapV.set_under(underColorV)
colormapV.set_over(overColorV)
#
cnormT = mpl.colors.BoundaryNorm(clevelsT, colormapT.N)
cnormS = mpl.colors.BoundaryNorm(clevelsS, colormapS.N)
cnormV = mpl.colors.BoundaryNorm(clevelsV, colormapV.N)

sigma2contours = None
#sigma2contours = [35, 36, 36.5, 36.8, 37, 37.1, 37.2, 37.25, 37.44, 37.52, 37.6]
#sigma0contours = None
sigma0contours = [24.0, 24.5, 25.0, 26.0, 27.0, 27.8, 28.0]
#sigma0contours = np.arange(26.0, 28.0, 0.2) # Good for OSNAP, but not for all arcticSections
#sigma0contours = [24.0, 25.0, 26.0, 27.0, 27.2, 27.4, 27.6, 27.8, 28.0]

# Load in MPAS mesh and transect mask file
mesh = xr.open_dataset(meshfile)
mask = xr.open_dataset(maskfile)

allTransects = decode_strings(mask.transectNames)
if transectNames[0]=='all':
    transectNames = allTransects

# Get depth
z = mesh.refBottomDepth.values
nlevels = len(z)

nTransects = len(transectNames)
maxEdges = mask.dims['maxEdgesInTransect']
for n in range(nTransects):
    # Identify transect
    transectName = transectNames[n]
    tname = transectName.replace(' ', '')
    transectIndex = allTransects.index(transectName)
    print(f'Plotting sections for transect: {transectName}')

    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == transectName:
            fc.add_feature(feature)
            break

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
    latmean = 180.0/np.pi*np.nanmean(latEdges)
    lonmean = 180.0/np.pi*np.nanmean(lonEdges)
    pressure = gsw.p_from_z(-z, latmean)

    kframe = 1
    for yr in years:
        for mo in months:
            print(f'year={yr}, month={mo}')

            outfile = f'{outdir}/{tname}_{casename}_{yr:04d}-{mo:02d}.nc'
            if not os.path.exists(outfile):
                modelfile = f'{modeldir}/{casename}.mpaso.hist.am.timeSeriesStatsMonthly.{yr:04d}-{mo:02d}-01.nc'
                if os.path.exists(modelfile):
                    modelfileT = modelfile
                    modelfileS = modelfile
                    modelfileV = modelfile
                else:
                    modelfileT = f'{modeldir}/activeTracers_temperature.{casename}.mpaso.hist.am.timeSeriesStatsMonthly.{yr:04d}-{mo:02d}-01.nc'
                    modelfileS = f'{modeldir}/activeTracers_salinity.{casename}.mpaso.hist.am.timeSeriesStatsMonthly.{yr:04d}-{mo:02d}-01.nc'
                    modelfileV = f'{modeldir}/normalVelocity.{casename}.mpaso.hist.am.timeSeriesStatsMonthly.{yr:04d}-{mo:02d}-01.nc'
                    if not os.path.exists(modelfileT) or not os.path.exists(modelfileS) or not os.path.exists(modelfileV):
                        raise IOError('No model file with temperature and/or salinity and/or velocity found')
                # Load in T (on cellsOnEdge centers)
                ncid = Dataset(modelfileT, 'r')
                tempOnCell1 = ncid.variables['timeMonthly_avg_activeTracers_temperature'][0, cellsOnEdge1-1, :]
                tempOnCell2 = ncid.variables['timeMonthly_avg_activeTracers_temperature'][0, cellsOnEdge2-1, :]
                ncid.close()
                # Load in S (on cellsOnEdge centers)
                ncid = Dataset(modelfileS, 'r')
                saltOnCell1 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge1-1, :]
                saltOnCell2 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge2-1, :]
                ncid.close()
                # Load in normalVelocity (on edge centers)
                ncid = Dataset(modelfileV, 'r')
                vel = ncid.variables['timeMonthly_avg_normalVelocity'][0, transectEdges-1, :]
                #vel += ncid.variables['timeMonthly_avg_normalGMBolusVelocity'][0, transectEdges-1, :]
                ncid.close()

                # Mask T,S values that fall on land and topography
                tempOnCell1 = np.ma.masked_array(tempOnCell1, ~cellMask1)
                tempOnCell2 = np.ma.masked_array(tempOnCell2, ~cellMask2)
                saltOnCell1 = np.ma.masked_array(saltOnCell1, ~cellMask1)
                saltOnCell2 = np.ma.masked_array(saltOnCell2, ~cellMask2)
                # Interpolate T,S values onto edges
                temp = 0.5 * (tempOnCell1 + tempOnCell2)
                salt = 0.5 * (saltOnCell1 + saltOnCell2)
                # Mask velocity values that fall on land and topography
                vel = np.ma.masked_array(vel, ~edgeMask)
                # Get normalVelocity direction
                normalVel = vel*edgeSigns[:, np.newaxis]
                #normalVel = None

                # Compute sigma's
                SA = gsw.SA_from_SP(salt, pressure[np.newaxis, :], lonmean, latmean)
                CT = gsw.CT_from_pt(SA, temp)
                sigma2 = gsw.density.sigma2(SA, CT)
                sigma0 = gsw.density.sigma0(SA, CT)

                # Save to file
                dsOut = xr.Dataset(
                        {
                               'Temp': (['nx', 'nz'], np.nan*np.ones([len(dist), len(z)])),
                               'Salt': (['nx', 'nz'], np.nan*np.ones([len(dist), len(z)])),
                               'sigma2': (['nx', 'nz'], np.nan*np.ones([len(dist), len(z)])),
                               'sigma0': (['nx', 'nz'], np.nan*np.ones([len(dist), len(z)])),
                               'normalVel': (['nx', 'nz'], np.nan*np.ones([len(dist), len(z)])),
                        },
                        coords={
                               'dist': (['nx'], dist),
                               'depth': (['nz'], z),
                        },
                )
                dsOut['Temp'][:, :] = temp
                dsOut['Temp'].attrs['units'] = 'degC'
                dsOut['Temp'].attrs['long_name'] = 'potential temperature'
                dsOut['Salt'][:, :] = salt
                dsOut['Salt'].attrs['units'] = 'psu'
                dsOut['Salt'].attrs['long_name'] = 'salinity'
                dsOut['sigma2'][:, :] = sigma2
                dsOut['sigma2'].attrs['units'] = 'kg/m3'
                dsOut['sigma2'].attrs['long_name'] = 'sigma2'
                dsOut['sigma0'][:, :] = sigma0
                dsOut['sigma0'].attrs['units'] = 'kg/m3'
                dsOut['sigma0'].attrs['long_name'] = 'sigma0'
                dsOut['normalVel'][:, :] = normalVel
                dsOut['normalVel'].attrs['units'] = 'm/s'
                dsOut['normalVel'].attrs['long_name'] = 'normal velocity'
                dsOut['dist'][:] = dist
                dsOut['dist'].attrs['units'] = 'km'
                dsOut['dist'].attrs['long_name'] = 'spherical distance from beginning of transect'
                dsOut['depth'][:] = z
                dsOut['depth'].attrs['units'] = 'm'
                dsOut['depth'].attrs['long_name'] = 'depth levels'
                dsOut.to_netcdf(outfile)
            else:
                print('Data file for transect already exists, reading from it...')
                ncid = Dataset(outfile, 'r')
                temp = ncid.variables['Temp'][:, :]
                salt = ncid.variables['Salt'][:, :]
                sigma2 = ncid.variables['sigma2'][:, :]
                sigma0 = ncid.variables['sigma0'][:, :]
                normalVel = ncid.variables['normalVel'][:, :]
                dist = ncid.variables['dist'][:]
                z = ncid.variables['depth'][:]

            zmax = z[np.max(maxLevelEdge)-1]

            # Plot temperature section
            figtitle = f'Temperature ({transectName}), {cname} (year={yr}, month={mo})'
            figfile = f'{framesdir}/Temp_{tname}_{cname}_{kframe:04d}.png'
            if zmax > zmaxUpperPanel:
                figsize = (10, 8)
                [fig, ax] = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 3])
                cf = ax[0].contourf(x, y, temp, cmap=colormapT, norm=cnormT, levels=clevelsT, extend='both')
                ax[0].set_ylim(0, zmaxUpperPanel)
                cf = ax[1].contourf(x, y, temp, cmap=colormapT, norm=cnormT, levels=clevelsT, extend='both')
                ax[1].set_ylim(zmaxUpperPanel, zmax)
                ax[0].set_facecolor('darkgrey')
                ax[1].set_facecolor('darkgrey')
                ax[0].invert_yaxis()
                ax[1].invert_yaxis()
                ax[0].set_xticklabels([])
                ax[1].set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
                ax[1].set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
                ax[0].set_title(figtitle, fontsize=12, fontweight='bold')
                ax[1].annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[0]), xy=(0, -0.12), xycoords='axes fraction', ha='center', va='bottom')
                ax[1].annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[0]), xy=(0, -0.17), xycoords='axes fraction', ha='center', va='bottom')
                ax[1].annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[-1]), xy=(1, -0.12), xycoords='axes fraction', ha='center', va='bottom')
                ax[1].annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[-1]), xy=(1, -0.17), xycoords='axes fraction', ha='center', va='bottom')
                fig.tight_layout(pad=0.5)
                cax, kw = mpl.colorbar.make_axes(ax[1], location='bottom', pad=0.12, shrink=0.9)
                cbar = fig.colorbar(cf, cax=cax, ticks=clevelsT, boundaries=clevelsT, **kw)
                cbar.ax.tick_params(labelsize=9, labelcolor='black')
                cbar.set_label('C$^\circ$', fontsize=12, fontweight='bold')
                if sigma2contours is not None:
                    cs = ax[0].contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                    cs = ax[1].contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                if sigma0contours is not None:
                    cs = ax[0].contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                    cs = ax[1].contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                add_inset(fig, fc, width=1.2, height=1.2, xbuffer=-0.8, ybuffer=0.4)
            else:
                figsize = (12, 6)
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot()
                cf = ax.contourf(x, y, temp, cmap=colormapT, norm=cnormT, levels=clevelsT, extend='both')
                ax.set_ylim(0, zmax)
                ax.set_facecolor('darkgrey')
                ax.invert_yaxis()
                ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
                ax.set_title(figtitle, fontsize=12, fontweight='bold')
                ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
                ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
                ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
                ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
                cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
                cbar = plt.colorbar(cf, cax=cax, ticks=clevelsT, boundaries=clevelsT, **kw)
                cbar.ax.tick_params(labelsize=9, labelcolor='black')
                cbar.set_label('C$^\circ$', fontsize=12, fontweight='bold')
                if sigma2contours is not None:
                    cs = ax.contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                if sigma0contours is not None:
                    cs = ax.contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                add_inset(fig, fc, width=1.2, height=1.2, xbuffer=0.7, ybuffer=0.4)
            plt.savefig(figfile, dpi=figdpi, bbox_inches='tight')
            plt.close()

            # Plot salinity section
            figtitle = f'Salinity ({transectName}), {cname} (year={yr}, month={mo})'
            figfile = f'{framesdir}/Salt_{tname}_{cname}_{kframe:04d}.png'
            if zmax > zmaxUpperPanel:
                figsize = (10, 8)
                [fig, ax] = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 3])
                cf = ax[0].contourf(x, y, salt, cmap=colormapS, norm=cnormS, levels=clevelsS, extend='both')
                ax[0].set_ylim(0, zmaxUpperPanel)
                cf = ax[1].contourf(x, y, salt, cmap=colormapS, norm=cnormS, levels=clevelsS, extend='both')
                ax[1].set_ylim(zmaxUpperPanel, zmax)
                ax[0].set_facecolor('darkgrey')
                ax[1].set_facecolor('darkgrey')
                ax[0].invert_yaxis()
                ax[1].invert_yaxis()
                ax[0].set_xticklabels([])
                ax[1].set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
                ax[1].set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
                ax[0].set_title(figtitle, fontsize=12, fontweight='bold')
                ax[1].annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[0]), xy=(0, -0.12), xycoords='axes fraction', ha='center', va='bottom')
                ax[1].annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[0]), xy=(0, -0.17), xycoords='axes fraction', ha='center', va='bottom')
                ax[1].annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[-1]), xy=(1, -0.12), xycoords='axes fraction', ha='center', va='bottom')
                ax[1].annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[-1]), xy=(1, -0.17), xycoords='axes fraction', ha='center', va='bottom')
                fig.tight_layout(pad=0.5)
                cax, kw = mpl.colorbar.make_axes(ax[1], location='bottom', pad=0.12, shrink=0.9)
                cbar = fig.colorbar(cf, cax=cax, ticks=clevelsS, boundaries=clevelsS, **kw)
                cbar.ax.tick_params(labelsize=9, labelcolor='black')
                cbar.set_label('psu', fontsize=12, fontweight='bold')
                if sigma2contours is not None:
                    cs = ax[0].contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                    cs = ax[1].contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                if sigma0contours is not None:
                    cs = ax[0].contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                    cs = ax[1].contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                add_inset(fig, fc, width=1.2, height=1.2, xbuffer=-0.8, ybuffer=0.4)
            else:
                figsize = (12, 6)
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot()
                cf = ax.contourf(x, y, salt, cmap=colormapS, norm=cnormS, levels=clevelsS, extend='both')
                ax.set_ylim(0, zmax)
                ax.set_facecolor('darkgrey')
                ax.invert_yaxis()
                ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
                ax.set_title(figtitle, fontsize=12, fontweight='bold')
                ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
                ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
                ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
                ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
                cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
                cbar = plt.colorbar(cf, cax=cax, ticks=clevelsS, boundaries=clevelsS, **kw)
                cbar.ax.tick_params(labelsize=9, labelcolor='black')
                cbar.set_label('psu', fontsize=12, fontweight='bold')
                if sigma2contours is not None:
                    cs = ax.contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                if sigma0contours is not None:
                    cs = ax.contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
                    cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                add_inset(fig, fc, width=1.2, height=1.2, xbuffer=0.7, ybuffer=0.4)
            plt.savefig(figfile, dpi=figdpi, bbox_inches='tight')
            plt.close()

            if normalVel is not None:
                # Plot normal velocity section

                figtitle = f'Cross-transect velocity ({transectName}), {cname} (year={yr}, month={mo})'
                figfile = f'{framesdir}/Vel_{tname}_{cname}_{kframe:04d}.png'
                if zmax > zmaxUpperPanel:
                    figsize = (10, 8)
                    [fig, ax] = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 3])
                    cf = ax[0].contourf(x, y, normalVel, cmap=colormapV, norm=cnormV, levels=clevelsV, extend='both')
                    ax[0].set_ylim(0, zmaxUpperPanel)
                    cf = ax[1].contourf(x, y, normalVel, cmap=colormapV, norm=cnormV, levels=clevelsV, extend='both')
                    ax[1].set_ylim(zmaxUpperPanel, zmax)
                    ax[0].set_facecolor('darkgrey')
                    ax[1].set_facecolor('darkgrey')
                    ax[0].invert_yaxis()
                    ax[1].invert_yaxis()
                    ax[0].set_xticklabels([])
                    ax[1].set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
                    ax[1].set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
                    ax[0].set_title(figtitle, fontsize=12, fontweight='bold')
                    ax[1].annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[0]), xy=(0, -0.12), xycoords='axes fraction', ha='center', va='bottom')
                    ax[1].annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[0]), xy=(0, -0.17), xycoords='axes fraction', ha='center', va='bottom')
                    ax[1].annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[-1]), xy=(1, -0.12), xycoords='axes fraction', ha='center', va='bottom')
                    ax[1].annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[-1]), xy=(1, -0.17), xycoords='axes fraction', ha='center', va='bottom')
                    fig.tight_layout(pad=0.5)
                    cax, kw = mpl.colorbar.make_axes(ax[1], location='bottom', pad=0.12, shrink=0.9)
                    cbar = fig.colorbar(cf, cax=cax, ticks=clevelsV, boundaries=clevelsV, **kw)
                    cbar.ax.tick_params(labelsize=9, labelcolor='black')
                    cbar.set_label('m/s', fontsize=12, fontweight='bold')
                    if sigma2contours is not None:
                        cs = ax[0].contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
                        cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                        cs = ax[1].contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
                        cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                    if sigma0contours is not None:
                        cs = ax[0].contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
                        cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                        cs = ax[1].contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
                        cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                    add_inset(fig, fc, width=1.2, height=1.2, xbuffer=-0.8, ybuffer=0.4)
                else:
                    figsize = (12, 6)
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot()
                    cf = ax.contourf(x, y, normalVel, cmap=colormapV, norm=cnormV, levels=clevelsV, extend='both')
                    ax.set_ylim(0, zmax)
                    ax.set_facecolor('darkgrey')
                    ax.invert_yaxis()
                    ax.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
                    ax.set_title(figtitle, fontsize=12, fontweight='bold')
                    ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[0]), xy=(0, -0.1), xycoords='axes fraction', ha='center', va='bottom')
                    ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[0]), xy=(0, -0.15), xycoords='axes fraction', ha='center', va='bottom')
                    ax.annotate('lat={:5.2f}'.format(180.0/np.pi*latEdges[-1]), xy=(1, -0.1), xycoords='axes fraction', ha='center', va='bottom')
                    ax.annotate('lon={:5.2f}'.format(180.0/np.pi*lonEdges[-1]), xy=(1, -0.15), xycoords='axes fraction', ha='center', va='bottom')
                    cax, kw = mpl.colorbar.make_axes(ax, location='right', pad=0.05, shrink=0.9)
                    cbar = plt.colorbar(cf, cax=cax, ticks=clevelsV, boundaries=clevelsV, **kw)
                    cbar.ax.tick_params(labelsize=9, labelcolor='black')
                    cbar.set_label('m/s', fontsize=12, fontweight='bold')
                    if sigma2contours is not None:
                        cs = ax.contour(x, y, sigma2, sigma2contours, colors='k', linewidths=1.5)
                        cb = plt.clabel(cs, levels=sigma2contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                    if sigma0contours is not None:
                        cs = ax.contour(x, y, sigma0, sigma0contours, colors='k', linewidths=1.5)
                        cb = plt.clabel(cs, levels=sigma0contours, inline=True, inline_spacing=2, fmt='%2.1f', fontsize=9)
                    add_inset(fig, fc, width=1.2, height=1.2, xbuffer=0.7, ybuffer=0.4)
                plt.savefig(figfile, dpi=figdpi, bbox_inches='tight')
                plt.close()

            kframe = kframe + 1
    os.system(f'ffmpeg -r 5 -i {framesdir}/Temp_{tname}_{cname}_%04d.png -vcodec mpeg4 -vb 40M -y {figdir}/movieTemp_{tname}_{cname}_{yearStart:04d}-{yearEnd:04d}.mp4')
    os.system(f'ffmpeg -r 5 -i {framesdir}/Salt_{tname}_{cname}_%04d.png -vcodec mpeg4 -vb 40M -y {figdir}/movieSalt_{tname}_{cname}_{yearStart:04d}-{yearEnd:04d}.mp4')
    os.system(f'ffmpeg -r 5 -i {framesdir}/Vel_{tname}_{cname}_%04d.png -vcodec mpeg4 -vb 40M -y {figdir}/movieVel_{tname}_{cname}_{yearStart:04d}-{yearEnd:04d}.mp4')
