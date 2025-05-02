from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean

from make_plots import make_scatter_plot, make_pcolormesh_plot


# Settings for nersc
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runname = 'E3SM-Arcticv2.1_historical'
modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runname}/postprocessing'
EN4dir = '/global/cfs/cdirs/m1199/milena/Obs/EN4/climatologies'
WOAdir = '/global/cfs/cdirs/e3sm/diagnostics/observations/Ocean/WOA23'
MLDdir = '/global/cfs/cdirs/e3sm/diagnostics/observations/Ocean/MLD'

figdir = f'./ocean_native/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

climoyear1 = 2000
climoyear2 = 2014

months = ['03', '09', 'ANN']
#months = ['ANN']

depthlevels = [0., 50., 250.]

projection = 'NorthPolarStereo'
lon0 = -50.0
lon1 = 50.0
dlon = 10.0
lat0 = 60.0
lat1 = 80.0
dlat = 4.0
colorIndices = [0, 14, 28, 57, 85, 113, 125, 142, 155, 170, 198, 227, 242, 255]
variables = [
             {'varname': 'activeTracers_temperature',
              'mpasvarname': 'timeMonthly_avg_activeTracers_temperature',
              'EN4varname': 'temperature',
              'WOAvarname': 'pt_an',
              'MLDvarname': None,
              'isvar3d': True,
              'title': 'Temperature',
              'units': '$^\circ$C',
              'colormap': plt.get_cmap('RdBu_r'),
              'clevels': [-1.0, -0.5, 0.0, 0.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8., 10., 12.]},
             {'varname': 'activeTracers_salinity',
              'mpasvarname': 'timeMonthly_avg_activeTracers_salinity',
              'EN4varname': 'salinity',
              'WOAvarname': 's_an',
              'MLDvarname': None,
              'isvar3d': True,
              'title': 'Salinity',
              'units': 'psu',
              'colormap': cmocean.cm.haline,
              'clevels': [31.0, 33.0, 34.2,  34.4,  34.6, 34.7,  34.8,  34.87, 34.9, 34.95, 35.0, 35.2, 35.4]},
             {'varname': 'dThreshMLD',
              'mpasvarname': 'timeMonthly_avg_dThreshMLD',
              'EN4varname': None,
              'WOAvarname': None,
              'MLDvarname': 'mld_dt_mean',
              'isvar3d': False,
              'title': 'MLD',
              'units': 'm',
              'colormap': plt.get_cmap('viridis'),
              'clevels': [10, 20, 50, 80, 100, 120, 150, 180, 250, 300, 400, 500, 800]}
            ]
##############################################################################

# Info about MPAS mesh
dsMesh = xr.open_dataset(meshfile)
lonMod = 180/np.pi*dsMesh.lonCell
latMod = 180/np.pi*dsMesh.latCell
maxLevelCell = dsMesh.maxLevelCell
nVertLevels = dsMesh.sizes['nVertLevels']
vertIndex = xr.DataArray(data=np.arange(nVertLevels), dims=('nVertLevels',))
depthmask = (vertIndex < maxLevelCell).transpose('nCells', 'nVertLevels')
# Find model levels for each depth level
zMod = dsMesh.refBottomDepth
zlevMod = np.zeros(np.shape(depthlevels), dtype=np.int64)
for iz in range(len(depthlevels)):
    dz = np.abs(zMod.values-depthlevels[iz])
    zlevMod[iz] = np.argmin(dz)

for month in months:
    print(f'\nPlotting {climoyear1}-{climoyear2} climatology for month: {month}...')

    if month=='ANN':
        EN4file = f'{EN4dir}/EN.4.2.2.f.analysis.c14_climo_{climoyear1}_{climoyear2}.nc'
        WOAfile = f'{WOAdir}/woa23_ANN_decav_04_pt_s_z_vol.20241101.nc'
        dsWOA = xr.open_dataset(WOAfile)
    else:
        EN4file = f'{EN4dir}/EN.4.2.2.f.analysis.c14_{month}_{climoyear1}_{climoyear2}.nc'
        WOAfile = f'{WOAdir}/woa23_decav_04_pt_s_mon_ann.20241101.nc'
        MLDfile = f'{MLDdir}/holtetalley_mld_climatology_20180710.nc'
        dsWOA = xr.open_dataset(WOAfile, decode_times=False).isel(time=int(month)-1)
        dsMLD = xr.open_dataset(MLDfile, decode_times=False).isel(iMONTH=int(month)-1)
    dsEN4 = xr.open_dataset(EN4file, decode_times=False).isel(time=0)
    
    lonEN4 = dsEN4['lon']
    latEN4 = dsEN4['lat']
    zEN4 = dsEN4['depth']
    zlevEN4 = np.zeros(np.shape(depthlevels), dtype=np.int64)
    for iz in range(len(depthlevels)):
        dz = np.abs(zEN4.values-depthlevels[iz])
        zlevEN4[iz] = np.argmin(dz)

    lonWOA = dsWOA['lon']
    latWOA = dsWOA['lat']
    zWOA = dsWOA['depth']
    zlevWOA = np.zeros(np.shape(depthlevels), dtype=np.int64)
    for iz in range(len(depthlevels)):
        dz = np.abs(np.abs(zWOA.values)-depthlevels[iz])
        zlevWOA[iz] = np.argmin(dz)

    # make sure these are right:
    #print(zMod.values[zlevMod])
    #print(zEN4.values[zlevEN4])
    #print(zWOA.values[zlevWOA])

    for var in variables:
        varname = var['varname']
        varnameMod = var['mpasvarname']
        varnameEN4 = var['EN4varname']
        varnameWOA = var['WOAvarname']
        varnameMLD = var['MLDvarname']
        isvar3d = var['isvar3d']
        varunits = var['units']
        vartitle = var['title']
        colormap = var['colormap']
        clevels = var['clevels']
        print(f'  variable: {vartitle}...')

        modelfile = f'{modeldir}/{varname}_ensembleMean_{month}_{climoyear1}_{climoyear2}.nc'
        dsMod = xr.open_dataset(modelfile, decode_times=False).isel(Time=0)
        if isvar3d:
            # mask values below local depth
            dsMod = dsMod.where(depthmask, drop=False)

        if vartitle=='MLD' and month!='ANN':
            figfileMod = f'{figdir}/{varname}_ensembleMean_{month}_{climoyear1}_{climoyear2}.png'
            figfileMLD = f'{figdir}/MLDholtetalley_{varnameMLD}_{month}.png'

            figtitleMod = f'{runname} {vartitle} ({month} climatology,\nyears={climoyear1}-{climoyear2})'
            figtitleMLD = f'Holte-Talley {vartitle} ({month} climatology)'

            fldMod = dsMod[varnameMod]
            fldMLD = dsMLD[varnameMLD]

            dotSize = 1.2 # this should go up as resolution decreases
            make_scatter_plot(lonMod, latMod, dotSize, figtitleMod, figfileMod, projectionName=projection,
                              lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                              fld=fldMod, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)

            lonMLD = dsMLD['lon']
            latMLD = dsMLD['lat']
            make_pcolormesh_plot(lonMLD, latMLD, fldMLD, colormap, clevels, colorIndices, varunits, figtitleMLD,
                                 figfileMLD, contourFld=None, contourValues=None, projectionName=projection,
                                 lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)

        if isvar3d:
            for iz in range(len(depthlevels)):
                depthlevel = depthlevels[iz]
                print(f'    depth level: {depthlevel}...')

                figfileMod = f'{figdir}/{varname}_ensembleMean_{month}_{climoyear1}_{climoyear2}_z{np.int16(depthlevel)}.png'
                figfileEN4 = f'{figdir}/EN4_{varnameEN4}_{month}_{climoyear1}_{climoyear2}_z{np.int16(depthlevel)}.png'
                figfileWOA = f'{figdir}/WOA23_{varnameWOA}_{month}_z{np.int16(depthlevel)}.png'

                figtitleMod = f'{runname} {vartitle} ({month} climatology,\nyears={climoyear1}-{climoyear2}, z={np.round(zMod[zlevMod[iz]].values)} m)'
                figtitleEN4 = f'EN4 {vartitle} ({month} climatology,\nyears={climoyear1}-{climoyear2}, z={np.round(zEN4[zlevEN4[iz]].values)} m)'
                figtitleWOA = f'WOA23 {vartitle} ({month} climatology,\nz={np.round(zWOA[zlevWOA[iz]].values)} m)'

                fldMod = dsMod[varnameMod].isel(nVertLevels=zlevMod[iz])
                fldEN4 = dsEN4[varnameEN4].isel(depth=zlevEN4[iz])
                fldWOA = dsWOA[varnameWOA].isel(depth=zlevWOA[iz])
                if vartitle=='Temperature':
                    fldEN4 = fldEN4 - 273.15 # Kelvin to Celsius
                    #print(np.nanmin(fldEN4.values), np.nanmax(fldEN4.values))

                dotSize = 1.2 # this should go up as resolution decreases
                make_scatter_plot(lonMod, latMod, dotSize, figtitleMod, figfileMod, projectionName=projection,
                                  lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat,
                                  fld=fldMod, cmap=colormap, clevels=clevels, cindices=colorIndices, cbarLabel=varunits)

                make_pcolormesh_plot(lonEN4, latEN4, fldEN4, colormap, clevels, colorIndices, varunits, figtitleEN4,
                                     figfileEN4, contourFld=None, contourValues=None, projectionName=projection,
                                     lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)

                make_pcolormesh_plot(lonWOA, latWOA, fldWOA, colormap, clevels, colorIndices, varunits, figtitleWOA,
                                     figfileWOA, contourFld=None, contourValues=None, projectionName=projection,
                                     lon0=lon0, lon1=lon1, dlon=dlon, lat0=lat0, lat1=lat1, dlat=dlat)
