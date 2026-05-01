from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import numpy as np
import numpy.ma as ma
import xarray as xr
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import matplotlib.animation as animation
from matplotlib.pyplot import cm
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import BoundaryNorm
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cmocean

from common_functions import add_land_lakes_coastline


def save_with_progress(fig, update, nframes, out_path,
					   fps=24, dpi=110,
					   codec="libx264",
					   extra_args=("-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "22")):
	writer = FFMpegWriter(fps=fps, codec=codec, extra_args=list(extra_args))
	with writer.saving(fig, out_path, dpi):
		for i in tqdm(range(nframes), desc="Saving video", unit="frame"):
			update(i)  # your update function must modify the artists for frame i
			writer.grab_frame()


# Settings for compy
#meshfile = '/compyfs/inputdata/ocn/mpas-o/EC30to60E2r2/ocean.EC30to60E2r2.200908.nc'
#runname = '20201030.alpha5_v1p-1_target.piControl.ne30pg2_r05_EC30to60E2r2-1900_ICG.compy'
#modeldir = f'/compyfs/malt823/E3SM_simulations/{runname}/archive/ocn/hist'

#modelComp = 'mpaso'
#model = 'ocn'
modelComp = 'mpassi'
model = 'ice'

# Settings for erdc.hpc.mil
meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runname = 'E3SMv2.1B60to10rA02'
runname = 'E3SMv2.1G60to10_01'
modeldir = f'/p/global/milena/{runname}/archive/{model}/hist'

# Number of time records to animate
ntimes = 120

infiles = sorted(glob.glob(f'{modeldir}/{runname}.{modelComp}.hist.am.timeSeriesStatsMonthly.00*'))[0:ntimes]
#infiles = sorted(glob.glob(f'{modeldir}/{runname}.{modelComp}.hist.am.timeSeriesStatsMonthlyMax.00*'))[0:ntimes]
print(f'\ninfiles={infiles}\n')

# Check is choice of variable is 
variable = 'salinity'
variable = 'SSSrestoringTend'
variable = 'iceAreaCell'
#variable = 'iceVolumeCell'
#variable = 'icePressure'
#variable = 'mld'
#variable = 'maxmld'
#variable = 'temperatureSurfaceFluxTendency'
#variable = 'temperatureShortWaveTendency'
#variable = 'temperatureHorizontalAdvectionTendency'
#variable = 'temperatureVerticalAdvectionTendency'
#variable = 'temperatureHorMixTendency'
#variable = 'temperatureVertMixTendency'
#variable = 'temperatureNonLocalTendency'
#variable = 'temperatureTotalAdvectionTendency' # derived variable
#variable = 'temperatureForcingTendency' # derived variable
#variable = 'temperatureSumTendencyTerms' # derived variable
#variable = 'temperatureTendency' # derived variable

figdir = './animations_xymaps'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figsize = [20, 20]
figdpi = 150
data_crs = ccrs.PlateCarree()

# z levels [m] (relevant for 3d variables)
#dlevels = [50.0, 100.0, 250.0, 500.0, 3000.0]
dlevels = [0.]

colorIndices0 = [0, 10, 28, 57, 85, 113, 142, 170, 198, 227, 242, 255]

variables = [
             {'name': 'temperature',
              'title': 'Temperature',
              'units': '$^\circ$C',
              'mpas': 'timeMonthly_avg_activeTracers_temperature',
              'factor': 1,
              'colormap': cmocean.cm.balance,
              'clevels': [-2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5, 2.0],
              'plot_anomalies': True,
              'is3d': True},
             {'name': 'salinity',
              'title': 'Salinity',
              'units': 'psu',
              'mpas': 'timeMonthly_avg_activeTracers_salinity',
              'factor': 1,
              'colormap': cmocean.cm.haline,
              'clevels': [27., 28., 29., 29.5, 30., 30.5, 31., 32., 33., 34., 35.],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'SSSrestoringTend',
              'title': 'SSS restoring tendency',
              'units': 'm psu s$^{-1}$',
              'mpas': 'timeMonthly_avg_salinitySurfaceRestoringTendency',
              'factor': 1,
              'colormap': cmocean.cm.balance,
              'clevels': [-5e-6, -4e-6, -3e-6, -2e-6, -1e-6, 0.0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'potentialDensity',
              'title': 'Potential Density',
              'units': 'kg m$^{-3}$',
              'mpas': 'timeMonthly_avg_potentialDensity',
              'factor': 1,
              'colormap': cmocean.cm.dense,
              'clevels': [24., 25.5, 25.9, 26.2, 26.5, 26.7, 26.8, 26.85, 26.9, 27.1, 27.75],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'maxmld',
              'title': 'Maximum Mixed Layer Depth',
              'units': 'm',
              'mpas': 'timeMonthlyMax_max_dThreshMLD',
              'factor': 1,
              #'colormap': plt.get_cmap('viridis'),
              'colormap': cmocean.cm.balance,
              'clevels': [10, 30, 50, 70, 100, 150, 200, 300, 500, 800, 2000],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'mld',
              'title': 'Mixed Layer Depth',
              'units': 'm',
              'mpas': 'timeMonthly_avg_dThreshMLD',
              'factor': 1,
              #'colormap': plt.get_cmap('viridis'),
              'colormap': cmocean.cm.balance,
              'clevels': [10, 30, 50, 70, 100, 150, 200, 300, 500, 800, 2000],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'iceAreaCell',
              'title': 'Sea ice concentration',
              'units': '%',
              'mpas': 'timeMonthly_avg_iceAreaCell',
              'factor': 100,
              'colormap': cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
                                               (0,     0.4,   0.4),   (0.169, 0.435, 0.223), (0.455, 0.478, 0.196),\
                                               (0.757, 0.474, 0.435), (0.827, 0.561, 0.772), (0.761, 0.757, 0.949),\
                                               (0.808, 0.921, 0.937)]),
              'clevels': [10, 15, 30, 50, 80, 90, 95, 97, 98, 99, 100],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'iceVolumeCell',
              'title': 'Sea ice thickness',
              'units': 'm',
              'mpas': 'timeMonthly_avg_iceVolumeCell',
              'factor': 1,
              'colormap': plt.get_cmap('YlGnBu_r'),
              'clevels': [0.1, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'icePressure',
              'title': 'Sea ice pressure',
              'units': 'N m$^{-1}$',
              'mpas': 'timeMonthly_avg_icePressure',
              'factor': 1,
              'colormap': plt.get_cmap('YlGnBu_r'),
              'clevels': [0.1, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureSurfaceFluxTendency',
              'title': 'Surface flux tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': 'timeMonthly_avg_activeTracerSurfaceFluxTendency_temperatureSurfaceFluxTendency',
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureShortWaveTendency',
              'title': 'Penetrating shortwave flux tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': 'timeMonthly_avg_temperatureShortWaveTendency',
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureForcingTendency',
              'title': 'Total forcing tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': None,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureHorizontalAdvectionTendency',
              'title': 'Horizontal advection tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': 'timeMonthly_avg_activeTracerHorizontalAdvectionTendency_temperatureHorizontalAdvectionTendency',
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureVerticalAdvectionTendency',
              'title': 'Vertical advection tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': 'timeMonthly_avg_activeTracerVerticalAdvectionTendency_temperatureVerticalAdvectionTendency',
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureTotalAdvectionTendency',
              'title': 'Total advection tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': None,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureHorMixTendency',
              'title': 'Horizontal mixing tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': 'timeMonthly_avg_activeTracerHorMixTendency_temperatureHorMixTendency',
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureVertMixTendency',
              'title': 'Vertical mixing tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': 'timeMonthly_avg_activeTracerVertMixTendency_temperatureVertMixTendency',
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureNonLocalTendency',
              'title': 'Non-local kpp flux tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': 'timeMonthly_avg_activeTracerNonLocalTendency_temperatureNonLocalTendency',
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureSumTendencyTerms',
              'title': 'Sum of all tendency terms for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': None,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureTendency',
              'title': 'Temperature tendency (derived)',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': 'timeMonthly_avg_activeTracers_temperature',
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureForcingMLTend',
              'title': 'Mixed Layer avg forcing tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': 'timeMonthly_avg_activeTracerForcingMLTend_temperatureForcingMLTend',
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-6, -3e-6, -2e-6, -1e-6, -0.5e-6, 0.0, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureHorAdvectionMLTend',
              'title': 'Mixed Layer avg Hadv tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': 'timeMonthly_avg_activeTracerHorAdvectionMLTend_temperatureHorAdvectionMLTend',
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-6, -3e-6, -2e-6, -1e-6, -0.5e-6, 0.0, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureVertAdvectionMLTend',
              'title': 'Mixed Layer avg Vadv tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': 'timeMonthly_avg_activeTracerVertAdvectionMLTend_temperatureVertAdvectionMLTend',
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-6, -3e-6, -2e-6, -1e-6, -0.5e-6, 0.0, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureHorMixMLTend',
              'title': 'Mixed Layer avg Hmix tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': 'timeMonthly_avg_activeTracerHorMixMLTend_temperatureHorMixMLTend',
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-7, -3e-7, -2e-7, -1e-7, -0.5e-7, 0.0, 0.5e-7, 1e-7, 2e-7, 3e-7, 4e-7],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureVertMixMLTend',
              'title': 'Mixed Layer avg Vmix tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': 'timeMonthly_avg_activeTracerVertMixMLTend_temperatureVertMixMLTend',
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-6, -3e-6, -2e-6, -1e-6, -0.5e-6, 0.0, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureNonLocalMLTend',
              'title': 'Mixed Layer avg Vmix-nonlocal tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': 'timeMonthly_avg_activeTracerNonLocalMLTend_temperatureNonLocalMLTend',
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-7, -3e-7, -2e-7, -1e-7, -0.5e-7, 0.0, 0.5e-7, 1e-7, 2e-7, 3e-7, 4e-7],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureTendML',
              'title': 'Mixed Layer avg temporal tendency of temperature',
              'units': '$^\circ$C/s',
              'mpas': 'timeMonthly_avg_activeTracersTendML_temperatureTendML',
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-6, -3e-6, -2e-6, -1e-6, -0.5e-6, 0.0, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6],
              'plot_anomalies': False,
              'is3d': False}
            ]

# Identify dictionary for desired variable
vardict = next(item for item in variables if item['name'] == variable)

varname = vardict['name']
mpasvarname = vardict['mpas']
factor = vardict['factor']
plot_anomalies = vardict['plot_anomalies']
is3d = vardict['is3d']
vartitle = vardict['title']
varunits = vardict['units']
clevels = vardict['clevels']
colormap = vardict['colormap']
if varname!='iceAreaCell':
    if len(clevels)+1 == len(colorIndices0):
        # we have 2 extra values for the under/over so make the colormap
        # without these values
        colorIndices = colorIndices0[1:-1]
        underColor = colormap(colorIndices0[0])
        overColor = colormap(colorIndices0[-1])
    else:
        colorIndices = colorIndices0
        underColor = None
        overColor = None
    colormap = cols.ListedColormap(colormap(colorIndices))
    if underColor is not None:
        colormap.set_under(underColor)
    if overColor is not None:
        colormap.set_over(overColor)
cnorm = cols.BoundaryNorm(clevels, colormap.N)

mesh = xr.open_dataset(meshfile)
lat = mesh.latCell.values
lon = mesh.lonCell.values
lat = np.rad2deg(lat)
lon = np.rad2deg(lon)
z = mesh.refBottomDepth
# Find model levels for each depth level
zlevels = np.zeros(np.shape(dlevels), dtype=np.int64)
for id in range(len(dlevels)):
    dz = np.abs(z.values-dlevels[id])
    zlevels[id] = np.argmin(dz)
#print('Model levels = ', z[zlevels])

ds = xr.open_mfdataset(infiles, combine='nested', concat_dim='Time')

if plot_anomalies:
    figtitle0 = 'Anomaly'
else:
    figtitle0 = ''

if is3d:
    for iz in range(len(dlevels)):
        figfile = f'{figdir}/{varname}{figtitle0}_depth{int(dlevels[iz]):04d}_{runname}_nrecords{ntimes:d}.mp4'
        figtitle0 = f'{vartitle} {figtitle0} (z={z[zlevels[iz]]:5.1f} m), {runname},'

        if varname=='temperatureTotalAdvectionTendency':
            mpasvarname1 = 'timeMonthly_avg_activeTracerHorizontalAdvectionTendency_temperatureHorizontalAdvectionTendency'
            mpasvarname2 = 'timeMonthly_avg_activeTracerVerticalAdvectionTendency_temperatureVerticalAdvectionTendency'
            fld = np.add(ds[mpasvarname1].isel(nVertLevels=zlevels[iz]).values,
                         ds[mpasvarname2].isel(nVertLevels=zlevels[iz]).values)
        elif varname=='temperatureForcingTendency':
            mpasvarname1 = 'timeMonthly_avg_activeTracerSurfaceFluxTendency_temperatureSurfaceFluxTendency'
            mpasvarname2 = 'timeMonthly_avg_temperatureShortWaveTendency'
            fld = np.add(ds[mpasvarname1].isel(nVertLevels=zlevels[iz]).values,
                         ds[mpasvarname2].isel(nVertLevels=zlevels[iz]).values)
        elif varname=='temperatureTendency':
            temp = ds[mpasvarname].isel(nVertLevels=zlevels[iz]).values
            #fld = np.nan*np.ones(np.shape(temp))
            #fld[1:-1, :] = np.diff(temp, n=1, axis=0)
            fld = np.diff(temp, n=1, axis=0, prepend=np.nan*np.ones([1, np.shape(temp)[1]]))/86400. # assumes daily values
        elif varname=='temperatureSumTendencyTerms':
            mpasvarname1 = 'timeMonthly_avg_activeTracerSurfaceFluxTendency_temperatureSurfaceFluxTendency'
            mpasvarname2 = 'timeMonthly_avg_temperatureShortWaveTendency'
            mpasvarname3 = 'timeMonthly_avg_activeTracerHorizontalAdvectionTendency_temperatureHorizontalAdvectionTendency'
            mpasvarname4 = 'timeMonthly_avg_activeTracerVerticalAdvectionTendency_temperatureVerticalAdvectionTendency'
            mpasvarname5 = 'timeMonthly_avg_activeTracerHorMixTendency_temperatureHorMixTendency'
            mpasvarname6 = 'timeMonthly_avg_activeTracerVertMixTendency_temperatureVertMixTendency'
            mpasvarname7 = 'timeMonthly_avg_activeTracerNonLocalTendency_temperatureNonLocalTendency'
            fld = np.add(ds[mpasvarname1].isel(nVertLevels=zlevels[iz]).values,
                         ds[mpasvarname2].isel(nVertLevels=zlevels[iz]).values)
            fld = np.add(fld, ds[mpasvarname3].isel(nVertLevels=zlevels[iz]).values)
            fld = np.add(fld, ds[mpasvarname4].isel(nVertLevels=zlevels[iz]).values)
            fld = np.add(fld, ds[mpasvarname5].isel(nVertLevels=zlevels[iz]).values)
            fld = np.add(fld, ds[mpasvarname6].isel(nVertLevels=zlevels[iz]).values)
            fld = np.add(fld, ds[mpasvarname7].isel(nVertLevels=zlevels[iz]).values)
        else:
            fld = ds[mpasvarname].isel(nVertLevels=zlevels[iz])
        fld = factor*fld
        if plot_anomalies:
            fld = fld - fld.isel(Time=0)
        #print(varname, int(dlevels[iz]), np.min(fld), np.max(fld))

        fig = plt.figure(figsize=figsize, dpi=figdpi)
        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
        ax.set_extent([-180, 180, 50, 90], crs=data_crs)
        gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6, draw_labels=True)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 200, 20))
        gl.ylocator = mticker.FixedLocator(np.arange(55, 85, 5))
        gl.n_steps = 100
        gl.right_labels = False
        gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 16}
        gl.ylabel_style = {'size': 16}
        gl.rotate_labels = False

        sc = ax.scatter(lon, lat, s=0.25, c=fld.isel(Time=0), cmap=colormap, norm=cnorm, marker='o', transform=data_crs)
        cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.03, shrink=.4, extend='both')
        cbar.ax.tick_params(labelsize=20, labelcolor='black')
        cbar.set_label(varunits, fontsize=20)
        figtitle = f'{figtitle0} month={1:d}'
        add_land_lakes_coastline(ax)
        ax.set_title(figtitle, y=1.08, fontsize=22)
        #plt.savefig('tmp.png', bbox_inches='tight')

        def animate(i):
            sc = ax.scatter(lon, lat, s=0.25, c=fld.isel(Time=i), cmap=colormap, norm=cnorm, marker='o', transform=data_crs)
            figtitle = f'{figtitle0} month={i+1:d}'
            ax.set_title(figtitle, y=1.08, fontsize=22)

        interval = 100 #in seconds     
        ani = animation.FuncAnimation(fig, animate, frames=range(ntimes), interval=interval)
        ani.save(figfile)
else:
    figfile = f'{figdir}/{varname}{figtitle0}_{runname}_nrecords{ntimes:d}.mp4'
    figtitle0 = f'{vartitle} {figtitle0}, {runname},'

    fld = ds[mpasvarname]
    fld = factor*fld
    if varname=='iceAreaCell' or varname=='iceVolumeCell' or varname=='icePressure':
        fld[np.where(fld<1e-15)] = np.nan
    if plot_anomalies:
        fld = fld - fld.isel(Time=0)
    #print(varname, np.min(fld.isel(Time=0).values), np.max(fld.isel(Time=0).values))
    #print(varname, np.min(fld.isel(Time=100).values), np.max(fld.isel(Time=100).values))

    fig = plt.figure(figsize=figsize, dpi=figdpi)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))
    ax.set_extent([-180, 180, 50, 90], crs=data_crs)
    gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6, draw_labels=True)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 200, 20))
    gl.ylocator = mticker.FixedLocator(np.arange(55, 85, 5))
    gl.n_steps = 100
    gl.right_labels = False
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}
    gl.rotate_labels = False

    sc = ax.scatter(lon, lat, s=0.25, c=fld.isel(Time=0), cmap=colormap, norm=cnorm, marker='o', transform=data_crs)
    cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.03, shrink=.4, extend='both')
    cbar.ax.tick_params(labelsize=20, labelcolor='black')
    cbar.set_label(varunits, fontsize=20)
    figtitle = f'{figtitle0} month={1:d}'
    add_land_lakes_coastline(ax)
    ax.set_title(figtitle, y=1.08, fontsize=22)
    plt.savefig('tmp.png', bbox_inches='tight')
    boh

    def animate(i):
        sc = ax.scatter(lon, lat, s=1.0, c=fld.isel(Time=i), cmap=colormap, norm=cnorm, marker='o', transform=data_crs)
        figtitle = f'{figtitle0} month={i+1:d}'
        ax.set_title(figtitle, y=1.08, fontsize=22)

    interval = 100 #in seconds     
    ani = animation.FuncAnimation(fig, animate, frames=range(ntimes), interval=interval)
    save_with_progress(fig, animate, nframes=range(ntimes), fps=4, out_path=figfile)
    #ani.save(figfile)
