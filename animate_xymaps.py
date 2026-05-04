from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import numpy as np
import numpy.ma as ma
import xarray as xr
import netCDF4
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import matplotlib.animation as animation
import matplotlib.path as mpath
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


modelComp = 'mpaso'
model = 'ocn'
#modelComp = 'mpassi'
#model = 'ice'

#fileType = 'timeSeriesStatsMonthly'
#varType = 'timeMonthly_avg_'
#fileType = 'timeSeriesStatsMonthlyMax'
#varType = 'timeMonthlyMax_max_'
fileType = 'timeSeriesStatsDaily'
varType = 'timeDaily_avg_'

# Settings for compy
#meshfile = '/compyfs/inputdata/ocn/mpas-o/EC30to60E2r2/ocean.EC30to60E2r2.200908.nc'
#runname = '20201030.alpha5_v1p-1_target.piControl.ne30pg2_r05_EC30to60E2r2-1900_ICG.compy'
#modeldir = f'/compyfs/malt823/E3SM_simulations/{runname}/archive/ocn/hist'

# Settings for erdc.hpc.mil
meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runname = 'E3SMv2.1B60to10rA02'
#runname = 'E3SMv2.1G60to10_01'
runname = 'E3SMv3G60to10_01cd25'
#modeldir = f'/p/global/milena/{runname}/archive/{model}/hist'
modeldir = f'/p/global/osinski/archive/{runname}/{model}/hist'

# Settings for lanl
#meshfile = '/usr/projects/w25_acoustics/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runname = 'E3SM-Arcticv3.1_1950control'
#modeldir = f'/lustre/scratch5/milena/E3SM/archive/{runname}/{model}/hist'

#yearStart = 10
#yearEnd = 11
yearStart = 1
yearEnd = 9
years = range(yearStart, yearEnd + 1)
referenceDate = '0001-01-01'
calendar = 'noleap'

infiles = []
for year in years:
    for month in range(1, 13):
         infiles.append(f'{modeldir}/{runname}.{modelComp}.hist.am.{fileType}.{year:04d}-{month:02d}-01.nc')
print(f'\ninfiles={infiles}\n')

# Check is choice of variable is 
variable = 'salinity'
#variable = 'SSSrestoringTend'
#variable = 'iceAreaCell'
#variable = 'iceVolumeCell'
#variable = 'icePressure'
variable = 'iceAirStressMagnitude'
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
figdpi = 100
dotSize = 1.0
data_crs = ccrs.PlateCarree()
centralLon = 0.0
#centralLon = -90.0
lon1 = -180.0
lon2 = 180.0
dlon = 20.0
lat1 = 65.0
lat2 = 90.0
dlat = 5.0

# z levels [m] (relevant for 3d variables)
#dlevels = [50.0, 100.0, 250.0, 500.0, 3000.0]
dlevels = [0.]

colorIndices0 = [0, 15, 28, 57, 85, 113, 142, 170, 198, 227, 242, 255]

variables = [
             {'name': 'temperature',
              'title': 'Temperature',
              'units': '$^\circ$C',
              'mpas': f'{varType}temperature',
              'isOnVertices': False,
              'factor': 1,
              'colormap': cmocean.cm.balance,
              'clevels': [-2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5, 2.0],
              'plot_anomalies': True,
              'is3d': True},
             {'name': 'salinity',
              'title': 'Salinity',
              'units': 'psu',
              'mpas': f'{varType}salinity',
              'isOnVertices': False,
              'factor': 1,
              'colormap': cmocean.cm.haline,
              'clevels': [27., 28., 29., 29.5, 30., 30.5, 31., 32., 33., 34., 35.],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'SSSrestoringTend',
              'title': 'SSS restoring tendency',
              'units': 'm psu s$^{-1}$',
              'mpas': f'{varType}salinitySurfaceRestoringTendency',
              'isOnVertices': False,
              'factor': 1,
              'colormap': cmocean.cm.balance,
              'clevels': [-5e-6, -4e-6, -3e-6, -2e-6, -1e-6, 0.0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'potentialDensity',
              'title': 'Potential Density',
              'units': 'kg m$^{-3}$',
              'mpas': f'{varType}potentialDensity',
              'isOnVertices': False,
              'factor': 1,
              'colormap': cmocean.cm.dense,
              'clevels': [24., 25.5, 25.9, 26.2, 26.5, 26.7, 26.8, 26.85, 26.9, 27.1, 27.75],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'maxmld',
              'title': 'Maximum Mixed Layer Depth',
              'units': 'm',
              'mpas': f'{varType}dThreshMLD',
              'isOnVertices': False,
              'factor': 1,
              #'colormap': plt.get_cmap('viridis'),
              'colormap': cmocean.cm.balance,
              'clevels': [10, 30, 50, 70, 100, 150, 200, 300, 500, 800, 2000],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'mld',
              'title': 'Mixed Layer Depth',
              'units': 'm',
              'mpas': f'{varType}dThreshMLD',
              'isOnVertices': False,
              'factor': 1,
              #'colormap': plt.get_cmap('viridis'),
              'colormap': cmocean.cm.balance,
              'clevels': [10, 30, 50, 70, 100, 150, 200, 300, 500, 800, 2000],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'iceAreaCell',
              'title': 'Sea ice concentration',
              'units': '%',
              'isOnVertices': False,
              'mpas': f'{varType}iceAreaCell',
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
              'mpas': f'{varType}iceVolumeCell',
              'isOnVertices': False,
              'factor': 1,
              #'colormap': plt.get_cmap('YlGnBu_r'),
              'colormap': cmocean.cm.thermal,
              'clevels': [0.3, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.4, 3.8],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'icePressure',
              'title': 'Sea ice pressure',
              'units': 'N m$^{-1}$',
              'mpas': f'{varType}icePressure',
              'isOnVertices': False,
              'factor': 1,
              'colormap': cmocean.cm.speed_r,
              'clevels': [0.2e5, 0.25e5, 0.3e5, 0.35e5, 0.4e5, 0.5e5, 0.6e5, 0.7e5, 0.8e5, 0.9e5, 1e5],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'iceAirStressMagnitude',
              'title': 'Sea ice-air stress magnitude',
              'units': 'N m$^{-2}$',
              'mpas': None,
              'isOnVertices': True,
              'factor': 1,
              'colormap': cmocean.cm.speed_r,
              'clevels': [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.21, 0.24, 0.3],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureSurfaceFluxTendency',
              'title': 'Surface flux tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': f'{varType}activeTracerSurfaceFluxTendency_temperatureSurfaceFluxTendency',
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureShortWaveTendency',
              'title': 'Penetrating shortwave flux tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': f'{varType}temperatureShortWaveTendency',
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureForcingTendency',
              'title': 'Total forcing tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': None,
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureHorizontalAdvectionTendency',
              'title': 'Horizontal advection tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': f'{varType}activeTracerHorizontalAdvectionTendency_temperatureHorizontalAdvectionTendency',
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureVerticalAdvectionTendency',
              'title': 'Vertical advection tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': f'{varType}activeTracerVerticalAdvectionTendency_temperatureVerticalAdvectionTendency',
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureTotalAdvectionTendency',
              'title': 'Total advection tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': None,
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureHorMixTendency',
              'title': 'Horizontal mixing tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': f'{varType}activeTracerHorMixTendency_temperatureHorMixTendency',
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureVertMixTendency',
              'title': 'Vertical mixing tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': f'{varType}activeTracerVertMixTendency_temperatureVertMixTendency',
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureNonLocalTendency',
              'title': 'Non-local kpp flux tendency for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': f'{varType}activeTracerNonLocalTendency_temperatureNonLocalTendency',
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureSumTendencyTerms',
              'title': 'Sum of all tendency terms for temperature',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': None,
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureTendency',
              'title': 'Temperature tendency (derived)',
              'units': '$^\circ$C/s (x1e-6)',
              'mpas': f'{varType}activeTracers_temperature',
              'isOnVertices': False,
              'factor': 1e6,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4, -3, -2, -1, -0.5, 0.0, 0.5, 1, 2, 3, 4],
              'plot_anomalies': False,
              'is3d': True},
             {'name': 'temperatureForcingMLTend',
              'title': 'Mixed Layer avg forcing tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': f'{varType}activeTracerForcingMLTend_temperatureForcingMLTend',
              'isOnVertices': False,
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-6, -3e-6, -2e-6, -1e-6, -0.5e-6, 0.0, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureHorAdvectionMLTend',
              'title': 'Mixed Layer avg Hadv tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': f'{varType}activeTracerHorAdvectionMLTend_temperatureHorAdvectionMLTend',
              'isOnVertices': False,
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-6, -3e-6, -2e-6, -1e-6, -0.5e-6, 0.0, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureVertAdvectionMLTend',
              'title': 'Mixed Layer avg Vadv tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': f'{varType}activeTracerVertAdvectionMLTend_temperatureVertAdvectionMLTend',
              'isOnVertices': False,
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-6, -3e-6, -2e-6, -1e-6, -0.5e-6, 0.0, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureHorMixMLTend',
              'title': 'Mixed Layer avg Hmix tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': f'{varType}activeTracerHorMixMLTend_temperatureHorMixMLTend',
              'isOnVertices': False,
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-7, -3e-7, -2e-7, -1e-7, -0.5e-7, 0.0, 0.5e-7, 1e-7, 2e-7, 3e-7, 4e-7],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureVertMixMLTend',
              'title': 'Mixed Layer avg Vmix tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': f'{varType}activeTracerVertMixMLTend_temperatureVertMixMLTend',
              'isOnVertices': False,
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-6, -3e-6, -2e-6, -1e-6, -0.5e-6, 0.0, 0.5e-6, 1e-6, 2e-6, 3e-6, 4e-6],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureNonLocalMLTend',
              'title': 'Mixed Layer avg Vmix-nonlocal tendency for temperature',
              'units': '$^\circ$C/s',
              'mpas': f'{varType}activeTracerNonLocalMLTend_temperatureNonLocalMLTend',
              'isOnVertices': False,
              'factor': 1,
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-4e-7, -3e-7, -2e-7, -1e-7, -0.5e-7, 0.0, 0.5e-7, 1e-7, 2e-7, 3e-7, 4e-7],
              'plot_anomalies': False,
              'is3d': False},
             {'name': 'temperatureTendML',
              'title': 'Mixed Layer avg temporal tendency of temperature',
              'units': '$^\circ$C/s',
              'mpas': f'{varType}activeTracersTendML_temperatureTendML',
              'isOnVertices': False,
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
isOnVertices = vardict['isOnVertices']
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
if isOnVertices:
    lat = mesh.latVertex.values
    lon = mesh.lonVertex.values
else:
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

ds = xr.open_mfdataset(infiles, combine='nested', concat_dim='Time', decode_times=False)
datetimes = netCDF4.num2date(ds.Time, f'days since {referenceDate}', calendar=calendar)
nframes = ds.Time.sizes['Time']
print('Total number of frames = ', nframes)

if plot_anomalies:
    figtitle0 = 'Anomaly'
else:
    figtitle0 = ''

if is3d:
    for iz in range(len(dlevels)):
        figfile = f'{figdir}/{varname}{figtitle0}_depth{int(dlevels[iz]):04d}_{runname}_years{yearStart:d}-{yearEnd:d}.mp4'
        figtitle0 = f'{vartitle} {figtitle0} (z={z[zlevels[iz]]:5.1f} m) {runname}'

        if varname=='temperatureTotalAdvectionTendency':
            mpasvarname1 = f'{varType}activeTracerHorizontalAdvectionTendency_temperatureHorizontalAdvectionTendency'
            mpasvarname2 = f'{varType}activeTracerVerticalAdvectionTendency_temperatureVerticalAdvectionTendency'
            fld = ds[mpasvarname1].isel(nVertLevels=zlevels[iz]) + \
                  ds[mpasvarname2].isel(nVertLevels=zlevels[iz])
        elif varname=='temperatureForcingTendency':
            mpasvarname1 = f'{varType}activeTracerSurfaceFluxTendency_temperatureSurfaceFluxTendency'
            mpasvarname2 = f'{varType}temperatureShortWaveTendency'
            fld = ds[mpasvarname1].isel(nVertLevels=zlevels[iz]) + \
                  ds[mpasvarname2].isel(nVertLevels=zlevels[iz])
        elif varname=='temperatureTendency':
            temp = ds[mpasvarname].isel(nVertLevels=zlevels[iz]).values
            #fld = np.nan*np.ones(np.shape(temp))
            #fld[1:-1, :] = np.diff(temp, n=1, axis=0)
            fld = np.diff(temp, n=1, axis=0, prepend=np.nan*np.ones([1, np.shape(temp)[1]]))/86400. # assumes daily values
        elif varname=='temperatureSumTendencyTerms':
            mpasvarname1 = f'{varType}activeTracerSurfaceFluxTendency_temperatureSurfaceFluxTendency'
            mpasvarname2 = f'{varType}temperatureShortWaveTendency'
            mpasvarname3 = f'{varType}activeTracerHorizontalAdvectionTendency_temperatureHorizontalAdvectionTendency'
            mpasvarname4 = f'{varType}activeTracerVerticalAdvectionTendency_temperatureVerticalAdvectionTendency'
            mpasvarname5 = f'{varType}activeTracerHorMixTendency_temperatureHorMixTendency'
            mpasvarname6 = f'{varType}activeTracerVertMixTendency_temperatureVertMixTendency'
            mpasvarname7 = f'{varType}activeTracerNonLocalTendency_temperatureNonLocalTendency'
            fld = ds[mpasvarname1].isel(nVertLevels=zlevels[iz]) + \
                  ds[mpasvarname2].isel(nVertLevels=zlevels[iz]) + \
                  ds[mpasvarname3].isel(nVertLevels=zlevels[iz]) + \
                  ds[mpasvarname4].isel(nVertLevels=zlevels[iz]) + \
                  ds[mpasvarname5].isel(nVertLevels=zlevels[iz]) + \
                  ds[mpasvarname6].isel(nVertLevels=zlevels[iz]) + \
                  ds[mpasvarname7].isel(nVertLevels=zlevels[iz])
        else:
            fld = ds[mpasvarname].isel(nVertLevels=zlevels[iz])
        fld = factor*fld
        if plot_anomalies:
            fld = fld - fld.isel(Time=0)
        #print(varname, int(dlevels[iz]), np.min(fld), np.max(fld))

        fig = plt.figure(figsize=figsize, dpi=figdpi)
        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=centralLon))
        ax.set_extent([lon1, lon2, lat1, lat1], crs=data_crs)
        gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6, draw_labels=True)
        gl.xlocator = mticker.FixedLocator(np.arange(lon1, lon2+dlon, dlon))
        gl.ylocator = mticker.FixedLocator(np.arange(lat1, lat2-dlat, dlat))
        gl.n_steps = 100
        gl.right_labels = False
        gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
        gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 16}
        gl.ylabel_style = {'size': 16}
        gl.rotate_labels = False

        # Circular boundary of the map
        # (see https://scitools.org.uk/cartopy/docs/v0.15/examples/always_circular_stereo.html)
        #theta  = np.linspace(0, 2*np.pi, 100)
        #center = [0.5, 0.5]
        #radius =  0.5
        #verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
        #circle = mpath.Path(verts * radius + center)
        #ax.set_boundary(circle, transform=ax.transAxes)

        sc = ax.scatter(lon, lat, s=dotSize, c=fld.isel(Time=0), cmap=colormap, norm=cnorm, marker='o', transform=data_crs)
        cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.03, shrink=.4, extend='both')
        cbar.ax.tick_params(labelsize=20, labelcolor='black')
        cbar.set_label(varunits, fontsize=20)
        if fileType=='timeSeriesStatsMonthly' or fileType=='timeSeriesStatsMonthlyMax':
            figtitle = f'{figtitle0} year={yearStart:d}, month={1:d}'
        if fileType=='timeSeriesStatsDaily':
            figtitle = f'{figtitle0} year={yearStart:d}, month={1:d}, day={1:d}'
        add_land_lakes_coastline(ax)
        ax.set_title(figtitle, y=1.08, fontsize=22)
        #plt.savefig('tmp.png', bbox_inches='tight')

        def animate(i):
            sc = ax.scatter(lon, lat, s=dotSize, c=fld.isel(Time=i), cmap=colormap, norm=cnorm, marker='o', transform=data_crs)
            year = datetimes[i].year + yearStart - 1
            month = datetimes[i].month
            if fileType=='timeSeriesStatsMonthly' or fileType=='timeSeriesStatsMonthlyMax':
                figtitle = f'{figtitle0} year={year:d}, month={month:d}'
                print(f'Processing year={year:d}, month={month:d}...')
            if fileType=='timeSeriesStatsDaily':
                day = datetimes[i].day
                figtitle = f'{figtitle0} year={year:d}, month={month:d}, day={day:d}'
                print(f'Processing year={year:d}, month={month:d}, day={day:d}...')
            #figtitle = f'{figtitle0} year={year:d}, month={i+1:d}'
            ax.set_title(figtitle, y=1.08, fontsize=22)

        interval = 100 #in seconds
        ani = animation.FuncAnimation(fig, animate, frames=range(nframes), interval=interval)
        ani.save(figfile)
else:
    figfile = f'{figdir}/{varname}{figtitle0}_{runname}_years{yearStart:d}-{yearEnd:d}.mp4'
    figtitle0 = f'{vartitle} {figtitle0} {runname}'

    if varname=='iceAirStressMagnitude':
        mpasvarname1 = f'{varType}airStressVertexUGeo'
        mpasvarname2 = f'{varType}airStressVertexVGeo'
        fld = 0.5 * (ds[mpasvarname1]**2 + ds[mpasvarname2]**2)**0.5
    else:
        fld = ds[mpasvarname]
    fld = factor*fld
    if varname=='iceAreaCell' or varname=='iceVolumeCell' or varname=='icePressure' or varname=='iceAirStressMagnitude':
        mask = (fld < 1e-15)
        fld = fld.where(~mask, drop=False)
    if plot_anomalies:
        fld = fld - fld.isel(Time=0)
    #print(varname, np.nanmin(fld.isel(Time=0).values), np.nanmax(fld.isel(Time=0).values))
    #print(varname, np.nanmin(fld.isel(Time=100).values), np.nanmax(fld.isel(Time=100).values))

    fig = plt.figure(figsize=figsize, dpi=figdpi)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=centralLon))
    ax.set_extent([lon1, lon2, lat1, lat1], crs=data_crs)
    gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=6, draw_labels=True)
    gl.xlocator = mticker.FixedLocator(np.arange(lon1, lon2+dlon, dlon))
    gl.ylocator = mticker.FixedLocator(np.arange(lat1, lat2-dlat, dlat))
    gl.n_steps = 100
    gl.right_labels = False
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}
    gl.rotate_labels = False

    # Circular boundary of the map
    # (see https://scitools.org.uk/cartopy/docs/v0.15/examples/always_circular_stereo.html)
    #theta  = np.linspace(0, 2*np.pi, 100)
    #center = [0.5, 0.5]
    #radius =  0.5
    #verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    #circle = mpath.Path(verts * radius + center)
    #ax.set_boundary(circle, transform=ax.transAxes)

    sc = ax.scatter(lon, lat, s=dotSize, c=fld.isel(Time=0), cmap=colormap, norm=cnorm, marker='o', transform=data_crs)
    if varname!='iceAreaCell':
        cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.03, shrink=.4, extend='both')
    else:
        cbar = plt.colorbar(sc, ticks=clevels, boundaries=clevels, location='right', pad=0.03, shrink=.4)
    cbar.ax.tick_params(labelsize=20, labelcolor='black')
    cbar.set_label(varunits, fontsize=20)
    if fileType=='timeSeriesStatsMonthly' or fileType=='timeSeriesStatsMonthlyMax':
        figtitle = f'{figtitle0} year={yearStart:d}, month={1:d}'
    if fileType=='timeSeriesStatsDaily':
        figtitle = f'{figtitle0} year={yearStart:d}, month={1:d}, day={1:d}'
    add_land_lakes_coastline(ax)
    ax.set_title(figtitle, y=1.08, fontsize=22)
    #plt.savefig('tmp.png', bbox_inches='tight')

    def animate(i):
        year = datetimes[i].year + yearStart - 1
        month = datetimes[i].month
        if fileType=='timeSeriesStatsMonthly' or fileType=='timeSeriesStatsMonthlyMax':
            figtitle = f'{figtitle0} year={year:d}, month={month:d}'
            print(f'Processing year={year:d}, month={month:d}')
        if fileType=='timeSeriesStatsDaily':
            day = datetimes[i].day
            figtitle = f'{figtitle0} year={year:d}, month={month:d}, day={day:d}'
            print(f'Processing year={year:d}, month={month:d}, day={day:d}...')
        sc = ax.scatter(lon, lat, s=dotSize, c=fld.isel(Time=i), cmap=colormap, norm=cnorm, marker='o', transform=data_crs)
        ax.set_title(figtitle, y=1.08, fontsize=22)

    interval = 100 #in seconds
    ani = animation.FuncAnimation(fig, animate, frames=range(nframes), interval=interval)
    ani.save(figfile)
    #save_with_progress(fig, animate, nframes=range(nframes), fps=4, out_path=figfile)
