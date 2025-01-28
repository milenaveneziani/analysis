from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean

from make_plots import make_contourf_plot, make_pcolormesh_plot


indirPHC = '/lcrc/group/e3sm/ac.milena/PHCdata'
infilePHC = f'{indirPHC}/phc3.0_monthly.nc'
indirWOA = '/lcrc/group/e3sm/ac.milena/WOA2023'
infileWOA = f'{indirWOA}/woa23_decav_s*.nc'
indirEN4 = '/lcrc/group/e3sm/ac.milena/EN4/climatologies'
#years = '1971_2000'
#years = '1964_1988'
years = '1989_2013'
infileEN4 = f'{indirEN4}/EN.4.2.2.f.analysis.c14_climo_{years}.nc'

depthlevel = 0
#depthlevel = -1

colorIndices = [0, 10, 28, 57, 85, 113, 142, 170, 198, 227, 242, 255]
# for salinity:
colormap = cmocean.cm.haline
# for SSS:
clevelsNH = [27., 28., 29., 29.5, 30., 30.5, 31., 32., 33., 34., 35.]
clevelsSH = [32.6, 33.0, 33.4, 33.6, 33.8, 34., 34.1, 34.2, 34.3, 34.4, 34.5]
# for Sbot:
#clevelsNH = [34., 34.1, 34.2, 34.3, 34.3, 34.4, 34.5, 34.6, 34.7, 34.8, 35.]
#clevelsSH = [34.4, 34.5, 34.55, 34.6, 34.65, 34.7, 34.75, 34.8, 34.85, 34.9, 35.]

dsPHC = xr.open_dataset(infilePHC, decode_times=False)
dsWOA = xr.open_mfdataset(infileWOA, decode_times=False)
dsEN4 = xr.open_mfdataset(infileEN4, decode_times=False)

lonPHC = dsPHC.lon
latPHC = dsPHC.lat
lonWOA = dsWOA.lon
latWOA = dsWOA.lat
lonEN4 = dsEN4.lon
latEN4 = dsEN4.lat

for month in range(0, 12):
#for month in range(0, 1):
    sssPHC = dsPHC.salt.isel(time=month, depth=depthlevel)
    sssWOA = dsWOA.s_an.isel(time=month, depth=depthlevel)
    sssEN4 = dsEN4.salinity.isel(time=month, depth=depthlevel)

    figTitle = f'PHC3 Surface Salinity (month={month+1:02d})'
    figFile = f'{indirPHC}/phc3.0_sssNH_month{month+1:02d}_pmesh_z{depthlevel}.png'
    #make_contourf_plot(lonPHC, latPHC, sssPHC, colormap, clevelsNH, colorIndices, '[psu]', figTitle, figFile, contourFld=None, contourValues=None, projectionName='NorthPolarStereo', lon0=-180, lon1=180, dlon=20, lat0=50, lat1=90, dlat=10)
    make_pcolormesh_plot(lonPHC, latPHC, sssPHC, colormap, clevelsNH, colorIndices, '[psu]', figTitle, figFile, contourFld=None, contourValues=None, projectionName='NorthPolarStereo', lon0=-180, lon1=180, dlon=20, lat0=50, lat1=90, dlat=10)
    figFile = f'{indirPHC}/phc3.0_sssSH_month{month+1:02d}_pmesh_z{depthlevel}.png'
    #make_contourf_plot(lonPHC, latPHC, sssPHC, colormap, clevelsSH, colorIndices, '[psu]', figTitle, figFile, contourFld=None, contourValues=None, projectionName='SouthPolarStereo', lon0=-180, lon1=180, dlon=20, lat0=-55, lat1=-90, dlat=10)
    make_pcolormesh_plot(lonPHC, latPHC, sssPHC, colormap, clevelsSH, colorIndices, '[psu]', figTitle, figFile, contourFld=None, contourValues=None, projectionName='SouthPolarStereo', lon0=-180, lon1=180, dlon=20, lat0=-55, lat1=-90, dlat=10)

    figTitle = f'WOA2023 Surface Salinity (month={month+1:02d})'
    figFile = f'{indirWOA}/woa2023_sssNH_month{month+1:02d}_pmesh_z{depthlevel}.png'
    make_pcolormesh_plot(lonWOA, latWOA, sssWOA, colormap, clevelsNH, colorIndices, '[psu]', figTitle, figFile, contourFld=None, contourValues=None, projectionName='NorthPolarStereo', lon0=-180, lon1=180, dlon=20, lat0=50, lat1=90, dlat=10)
    figFile = f'{indirWOA}/woa2023_sssSH_month{month+1:02d}_pmesh_z{depthlevel}.png'
    make_pcolormesh_plot(lonWOA, latWOA, sssWOA, colormap, clevelsSH, colorIndices, '[psu]', figTitle, figFile, contourFld=None, contourValues=None, projectionName='SouthPolarStereo', lon0=-180, lon1=180, dlon=20, lat0=-55, lat1=-90, dlat=10)

    figTitle = f'EN4.2.2.c14 Surface Salinity (month={month+1:02d})'
    figFile = f'{indirEN4}/EN4.2.2.c14_sssNH_month{month+1:02d}_pmesh_z{depthlevel}_{years}.png'
    make_pcolormesh_plot(lonEN4, latEN4, sssEN4, colormap, clevelsNH, colorIndices, '[psu]', figTitle, figFile, contourFld=None, contourValues=None, projectionName='NorthPolarStereo', lon0=-180, lon1=180, dlon=20, lat0=50, lat1=90, dlat=10)
    figFile = f'{indirEN4}/EN4.2.2.c14_sssSH_month{month+1:02d}_pmesh_z{depthlevel}_{years}.png'
    make_pcolormesh_plot(lonEN4, latEN4, sssEN4, colormap, clevelsSH, colorIndices, '[psu]', figTitle, figFile, contourFld=None, contourValues=None, projectionName='SouthPolarStereo', lon0=-180, lon1=180, dlon=20, lat0=-55, lat1=-90, dlat=10)
