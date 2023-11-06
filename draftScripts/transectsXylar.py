#!/usr/bin/env python
import os
import warnings

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import numpy as np
import xarray as xr
from geometric_features import FeatureCollection, read_feature_collection
from matplotlib.colors import BoundaryNorm
from matplotlib.tri import Triangulation
from mpas_tools.viz import mesh_to_triangles
from mpas_tools.viz.transects import (
    find_transect_cells_and_weights,
    make_triangle_tree
)
from mpas_tools.ocean.transects import (
    find_transect_levels_and_weights,
    get_outline_segments,
    interp_mpas_to_transect_triangle_nodes
)
from mpas_tools.ocean.viz.inset import add_inset


def _make_colormap(colormap, clevels):
    colorIndices = np.int_(np.linspace(0, 255, num=len(clevels)+1, endpoint=True))
    underColor = colormap(colorIndices[0])
    overColor = colormap(colorIndices[-1])
    colorIndices = colorIndices[1:-1]
    colormap = cols.ListedColormap(colormap(colorIndices))
    colormap.set_under(underColor)
    colormap.set_over(overColor)
    cnorm = BoundaryNorm(clevels, colormap.N)
    return colormap, cnorm


def _compute_transect(feature, ds_mesh, flip):
    """
    build a sequence of triangles showing the transect intersecting mpas cells
    """

    ds_tris = mesh_to_triangles(ds_mesh)

    triangle_tree = make_triangle_tree(ds_tris)

    transectName = feature['properties']['name']
    print(f'building transect dataset for {transectName}')
    assert feature['geometry']['type'] == 'LineString'

    coordinates = feature['geometry']['coordinates']
    transect_lon, transect_lat = zip(*coordinates)
    transect_lon = np.array(transect_lon)
    transect_lat = np.array(transect_lat)
    if flip:
        transect_lon = transect_lon[::-1]
        transect_lat = transect_lat[::-1]
    transect_lon = xr.DataArray(data=transect_lon,
                                dims=('nPoints',))
    transect_lat = xr.DataArray(data=transect_lat,
                                dims=('nPoints',))

    ds_mpas_transect = find_transect_cells_and_weights(
        transect_lon, transect_lat, ds_tris, ds_mesh,
        triangle_tree, degrees=True)

    ds_mpas_transect = find_transect_levels_and_weights(
        ds_mpas_transect, ds_mesh.layerThickness,
        ds_mesh.bottomDepth, ds_mesh.maxLevelCell - 1)

    ds_mpas_transect['x'] = ds_mpas_transect.dNode.isel(
        nSegments=ds_mpas_transect.segmentIndices,
        nHorizBounds=ds_mpas_transect.nodeHorizBoundsIndices)

    ds_mpas_transect['z'] = ds_mpas_transect.zTransectNode

    ds_mpas_transect.compute()

    return ds_mpas_transect


def _plot_transect(ds_transect, mpas_field, cmap, norm, levels, fc, figtitle,
                   zmaxUpperPanel, zmax, units):
    """
    plot a transect showing the field on the MPAS-Ocean mesh and save to a file
    """
    transect_field = interp_mpas_to_transect_triangle_nodes(ds_transect,
                                                            mpas_field)


    mask = transect_field.notnull().all(dim='nTriangleNodes')
    tri_mask = np.logical_not(mask.values)

    field_masked = transect_field.values.ravel()

    triangulation_args = _get_ds_triangulation_args(ds_transect)

    triangulation_args['mask'] = tri_mask

    tris = Triangulation(**triangulation_args)

    figsize = (10, 8)
    [fig, ax] = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 3])
    
    ax[0].tricontourf(tris, field_masked, cmap=cmap, norm=norm, levels=levels,
                      extend='both')

    ax[0].set_facecolor('darkgrey')
    ax[0].set_ylim(zmaxUpperPanel, 0)
    ax[0].set_xticklabels([])
    ax[0].set_title(figtitle, fontsize=12, fontweight='bold')

    cf = ax[1].tricontourf(tris, field_masked, cmap=cmap, norm=norm,
                           levels=levels, extend='both')
    ax[1].set_facecolor('darkgrey')
    ax[1].set_ylim(zmax, zmaxUpperPanel)
    ax[1].set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
    ax[1].set_ylabel('Depth (m)', fontsize=12, fontweight='bold')

    fig.tight_layout(pad=0.5)

    cax, kw = mpl.colorbar.make_axes(ax[1], location='bottom', pad=0.12,
                                     shrink=0.9)
    cbar = fig.colorbar(cf, cax=cax, ticks=levels, boundaries=levels, **kw)
    cbar.ax.tick_params(labelsize=9, labelcolor='black')
    cbar.set_label(units, fontsize=12, fontweight='bold')

    add_inset(fig, fc, width=1.2, height=1.2, xbuffer=-0.8, ybuffer=0.4)


def _get_ds_triangulation_args(ds_transect):
    """
    get arguments for matplotlib Triangulation from triangulation dataset
    """

    n_transect_triangles = ds_transect.sizes['nTransectTriangles']
    d_node = ds_transect.dNode.isel(
        nSegments=ds_transect.segmentIndices,
        nHorizBounds=ds_transect.nodeHorizBoundsIndices)
    x = 1e-3 * d_node.values.ravel()

    z_transect_node = ds_transect.zTransectNode
    y = z_transect_node.values.ravel()

    tris = np.arange(3 * n_transect_triangles).reshape(
        (n_transect_triangles, 3))
    triangulation_args = dict(x=x, y=y, triangles=tris)

    return triangulation_args


warnings.simplefilter(action='ignore', category=FutureWarning)

plt.switch_backend('Agg')

featurefile = 'arctic_atlantic_budget_regionsTransects.geojson'

meshfile = 'mpaso.ARRM10to60E2r1.220730.nc'

casename = 'E3SMv2.1B60to10rA02'
cname = 'E3SM-Arctic'
modeldir = 'singleVarFiles'

yearStart = 1
yearEnd = 30

zmaxUpperPanel = -100.0
zmax = -5500

transectNames = ['Atlantic zonal 50N', 'Atlantic zonal 27.2N', 'South Atlantic Ocean 34S']

if os.path.exists(featurefile):
    fc_all = read_feature_collection(featurefile)
    fc_transect = dict()
    for feature in fc_all.features:
        transectName = feature['properties']['name']
        if transectName in transectNames:
            fc_transect[transectName] = FeatureCollection(features=[feature])
else:
    raise IOError('No feature file found for this region group')

figdir = f'./animations_verticalSections/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)
framesdir = f'{figdir}/frames'
if not os.path.isdir(framesdir):
    os.makedirs(framesdir)


years = range(yearStart, yearStart+1)
months = range(1, 2)

# Figure details
figdpi = 300
clevelsT = [-1.8, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10., 12.]
clevelsS = [30.0, 31.0, 32.0, 33.0, 33.5, 33.8, 34.0, 34.2, 34.4, 34.6, 34.8, 34.82, 34.84, 34.86, 34.88, 34.9, 34.95, 35.0, 35.5]
clevelsV = [-0.5, -0.3, -0.25, -0.2, -0.15, -0.1, -0.02, 0.0, 0.02, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
colormapT = plt.get_cmap(cmocean.cm.thermal)
colormapS = plt.get_cmap(cmocean.cm.haline)
colormapV = plt.get_cmap('RdBu_r')
[colormapT, cnormT] = _make_colormap(colormapT, clevelsT)
[colormapS, cnormS] = _make_colormap(colormapS, clevelsS)
[colormapV, cnormV] = _make_colormap(colormapV, clevelsV)

sigma0contours = [24.0, 24.5, 25.0, 26.0, 27.0, 27.8, 28.0]

# Load in MPAS mesh file
ds_mesh = xr.open_dataset(meshfile).isel(Time=0)

for transectName in transectNames:
    tname = transectName.replace(' ', '')

    # build transect dataset
    fc = fc_transect[transectName]
    ds_transect = _compute_transect(fc.features[0], ds_mesh, flip=False)

    kframe = 1
    for yr in years:
        for mo in months:
            print(f'year={yr}, month={mo}')

            figtitle = f'Temperature ({transectName}), {cname} (year={yr}, month={mo})'

            figfile = f'{framesdir}/Temp_{tname}_{cname}_{kframe:04d}.png'

            modelfileT = f'{modeldir}/activeTracers_temperature.{casename}.mpaso.hist.am.timeSeriesStatsMonthly.{yr:04d}-{mo:02d}-01.nc'
            ds = xr.open_dataset(modelfileT)
            temp = ds.timeMonthly_avg_activeTracers_temperature.isel(Time=0)

            _plot_transect(ds_transect=ds_transect,
                           mpas_field=temp,
                           cmap=colormapT,
                           norm=cnormT,
                           levels=clevelsT,
                           fc=fc,
                           figtitle=figtitle,
                           zmaxUpperPanel=zmaxUpperPanel,
                           zmax=zmax,
                           units='C$^\circ$')

            plt.savefig(figfile, dpi=figdpi, bbox_inches='tight')
            plt.close()
            kframe = kframe + 1
