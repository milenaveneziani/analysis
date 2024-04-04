from __future__ import absolute_import, division, print_function, \
    unicode_literals
import xarray as xr
import numpy as np
import cmocean

from make_plots import make_streamline_plot


colorIndices = [0, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 250, 255]
colormap = cmocean.cm.speed_r
clevels = [0.5, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

ds = xr.open_dataset('~/u_remap_bilin.nc')
#ds = xr.open_dataset('~/u_remap_aave.nc')
u = ds.timeMonthly_avg_velocityZonal.isel(Time=0, nVertLevels=0).values
lon = ds.lon.values
lat = ds.lat.values
ds = xr.open_dataset('~/v_remap_bilin.nc')
#ds = xr.open_dataset('~/v_remap_aave.nc')
v = ds.timeMonthly_avg_velocityMeridional.isel(Time=0, nVertLevels=0).values
u = 100*u 
v = 100*v 
speed = 0.5*np.sqrt(u*u + v*v)

make_streamline_plot(lon, lat, u, v, speed, 4, colormap, clevels, colorIndices, 'm/s', 'NorthPolarStereo', 'layer 1 velocities', './u_bilinLatLonNH.png', lon0=-50, lon1=50, dlon=10, lat0=60, lat1=80, dlat=4)
make_streamline_plot(lon, lat, u, v, speed, 4, colormap, clevels, colorIndices, 'm/s', 'SouthPolarStereo', 'layer 1 velocities', './u_bilinLatLonSH.png', lon0=-180, lon1=180, dlon=20, lat0=-58, lat1=-90, dlat=-4)
