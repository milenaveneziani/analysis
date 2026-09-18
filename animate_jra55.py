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


runname = 'E3SMv2.1B60to10rA02'
modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runname}/archive/atm/regridded'
jra55dir = '/global/cfs/cdirs/m1199/jra55/v1.5_noleap/monthly_regridded'

yearStart = 1
yearEnd = 20
#yearEnd = 1
yearStart_jra55 = 1958
yearEnd_jra55 = 1977
#yearEnd_jra55 = 1958

#variable = 'TREFHT'
#variable_jra55 = 't_10'
variable = 'QREFHT'
variable_jra55 = 'q_10'

years = range(yearStart, yearEnd + 1)
referenceDate = '0001-01-01'
calendar = 'noleap'
years_jra55 = range(yearStart_jra55, yearEnd_jra55 + 1)

figdir = './animations_jra55'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figsize = [20, 20]
figdpi = 100
data_crs = ccrs.PlateCarree()
centralLon = 0.0
#centralLon = -90.0
lon1 = -180.0
lon2 = 180.0
dlon = 20.0
lat1 = 45.0
lat2 = 90.0
dlat = 5.0

colorIndices0 = [0, 15, 28, 57, 85, 113, 142, 170, 198, 227, 242, 255]

variables = [{'name': 'TREFHT',
              'name_jra55': 't_10',
              'title': 'Air temperature at 10 m',
              'units': 'K',
              'colormap': cmocean.cm.balance,
              'clevels': [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
              'plot_anomalies': True},
             {'name': 'QREFHT',
              'name_jra55': 'q_10',
              'title': 'Specific humidity at 10 m',
              'units': 'Kg/Kg',
              'colormap': cmocean.cm.balance,
              'clevels': [-0.005, -0.004, -0.003, -0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
              'plot_anomalies': True}
            ]

# Identify dictionary for desired variable
vardict = next(item for item in variables if item['name'] == variable)

varname = vardict['name']
varname_jra55 = vardict['name_jra55']
plot_anomalies = vardict['plot_anomalies']
vartitle = vardict['title']
varunits = vardict['units']
clevels = vardict['clevels']
colormap = vardict['colormap']
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

if plot_anomalies:
    figtitle0 = 'Model-JRA55'
else:
    figtitle0 = ''
figfile = f'{figdir}/{varname}_{figtitle0}_{runname}_years{yearStart:d}-{yearEnd:d}.mp4'
figtitle0 = f'{vartitle} {figtitle0} {runname}'

infiles = []
for year in years:
    for month in range(1, 13):
        infiles.append(f'{modeldir}/{variable}/regridded_fv180x360/{runname}.eam.h0.{variable}.{year:04d}-{month:02d}.gr.nc')
#print(f'\ninfiles={infiles}\n')

infiles_jra55 = []
for year in years_jra55:
    infiles_jra55.append(f'{jra55dir}/JRA.v1.5.{variable_jra55}.TL319.{year:04d}.gr.nc')
#print(f'\ninfiles={infiles_jra55}\n')

lat = xr.open_dataset(infiles[0]).lat.values
lon = xr.open_dataset(infiles[0]).lon.values
[lon, lat] = np.meshgrid(lon, lat)

ds = xr.open_mfdataset(infiles, combine='nested', concat_dim='time', decode_times=False)
ds_jra55 = xr.open_mfdataset(infiles_jra55, combine='nested', concat_dim='time', decode_times=False)
ds_jra55 = ds_jra55.rename({'longitude': 'lon'})
ds_jra55 = ds_jra55.rename({'latitude': 'lat'})
ds['time'] = ds.time - 14
datetimes = netCDF4.num2date(ds.time, f'days since {referenceDate}', calendar=calendar)
nframes = ds.time.sizes['time']
print('Total number of frames = ', nframes)

fld = ds[varname]
fld_jra55 = ds_jra55[varname_jra55]
# Align times:
fld_jra55['time'] = fld.time
if plot_anomalies:
    fld = fld - fld_jra55
print(fld.min().values, fld.max().values)

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

cf = ax.pcolormesh(lon, lat, fld.isel(time=0), cmap=colormap, norm=cnorm, transform=data_crs)
cbar = plt.colorbar(cf, ticks=clevels, boundaries=clevels, location='right', pad=0.03, shrink=.4, extend='both')
cbar.ax.tick_params(labelsize=20, labelcolor='black')
cbar.set_label(varunits, fontsize=20)
figtitle = f'{figtitle0} year={yearStart:d}, month={1:d}'
add_land_lakes_coastline(ax)
ax.set_title(figtitle, y=1.08, fontsize=22)
#plt.savefig('tmp.png', bbox_inches='tight')

def animate(i):
    year = datetimes[i].year + yearStart - 1
    month = datetimes[i].month
    figtitle = f'{figtitle0} year={year:d}, month={month:d}'
    print(f'Processing year={year:d}, month={month:d}')
    cf = ax.pcolormesh(lon, lat, fld.isel(time=i), cmap=colormap, norm=cnorm, transform=data_crs)
    ax.set_title(figtitle, y=1.08, fontsize=22)

interval = 100 #in seconds
ani = animation.FuncAnimation(fig, animate, frames=range(nframes), interval=interval)
ani.save(figfile)
