#
# Plots regional T,S profiles for ensemble members
# This breaks for more than one season or year/month conbination
#
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import glob
import gsw


def haversine(lon1, lat1, lon2, lat2):
    # lon, lat should be in radians
    earthRadius = 6367.44 # km
    #earthRadius = 6371
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * earthRadius * np.arcsin(np.sqrt(a))


plotClimos = True
plotMonthly = False
if plotClimos==plotMonthly:
    raise ValueError('Variables plotClimos and plotMonthly cannot be identical')
plotPHCWOA = True # only works for monthly seasons for now
plotHighresMIP = True

ensembleName = 'E3SM-Arcticv2.1_historical'
ensembleMemberNames = ['0101', '0151', '0201', '0251', '0301']
colors = ['mediumblue', 'dodgerblue', 'deepskyblue', 'lightseagreen', 'teal'] # same length as ensembleMemberNames
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'

# Coordinates of point where to plot profiles
# Barents Sea:
#lonPoint = 37.5
#latPoint = 70
#pointTitle = 'Barents Sea South, 70N,37.5E'
#latPoint = 75
#pointTitle = 'Barents Sea Central, 75N,37.5E'
#latPoint = 80
#pointTitle = 'Barents Sea North, 80N,37.5E'
latPoint = 75
#lonPoint = 27
#pointTitle = 'Barents Sea West, 75N,27E'
lonPoint = 48
pointTitle = 'Barents Sea East, 75N,48E'
#lonPoint = 35
#latPoint = 83
#pointTitle = 'Barents Sea Abyssal, 83N,35E'

# relevant if plotClimos=True
#climoyearStart = 2000
#climoyearEnd = 2014
climoyearStart = 1950
climoyearEnd = 1970
# seasons options: '01'-'12', 'ANN', 'JFM', 'JAS', 'MAJ', 'OND'
# (depending on what was set in mpas-analysis)
seasons = ['02', '08']
#seasons = ['ANN']
#seasons = ['JFM', 'JAS']
modelClimodir1 = f'/pscratch/sd/m/milena/e3sm_scratch/pm-cpu/{ensembleName}'
modelClimodir2 = f'mpas-analysis/Years{climoyearStart}-{climoyearEnd}/clim/mpas/avg/unmasked_ARRM10to60E2r1'

# relevant if plotMonthly=True
years = [1950]
months = [9]
modeldir1 = f'/pscratch/sd/m/milena/e3sm_scratch/pm-cpu/{ensembleName}'
modeldir2 = f'archive/ocn/hist'

# relevant if plotPHCWOA=True
PHCfilename = '/global/cfs/cdirs/e3sm/observations_with_original_data/Ocean/PHC3.0/phc3.0_monthly_accessed08-08-2019.nc'
WOAfilename = '/global/cfs/cdirs/e3sm/observations_with_original_data/Ocean/WOA18/decadeAll/0.25degGrid/woa18_decav_04_TS_mon.nc'

# relevant if plotHighresMIP=True
HighresMIPdir = '/pscratch/sd/m/milena/CMIP6monthlyclimos/NCAR/CESM1-CAM5-SE-HR/hist-1950/r1i1p1f1/ncclimoFiles'
HighresMIP2dir = '/pscratch/sd/m/milena/CMIP6monthlyclimos/NCAR/CESM1-CAM5-SE-HR/highres-future/r1i1p1f1/ncclimoFiles'

figdir = f'./TSprofiles/{ensembleName}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

outdir0 = f'./TSprofiles_data'
if not os.path.isdir(outdir0):
    os.makedirs(outdir0)

figsize = (10, 15)
figdpi = 150
fontsize_smallLabels = 18
fontsize_labels = 20
fontsize_titles = 22
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}

nEnsembles = len(ensembleMemberNames)
################

# Read in relevant global mesh information
if os.path.exists(meshfile):
    dsMesh = xr.open_dataset(meshfile)
else:
    raise IOError(f'MPAS restart/mesh file {meshfile} not found')
depth = dsMesh.refBottomDepth
# Identify index of selected ocean cell, by computing the minimum
# of the spherical distance between all points and lonPoint,latPoint
nCells = dsMesh.dims['nCells']
lonCell = dsMesh.lonCell
latCell = dsMesh.latCell
spherDist = haversine(lonCell, latCell, lonPoint*np.pi/180, latPoint*np.pi/180)
indices = xr.DataArray(data=np.arange(nCells).astype(int), dims='nCells')
iCell = indices.where(spherDist==np.min(spherDist), drop=True).values.astype(int)[0]

lon_icell = lonCell.values[iCell]*180/np.pi
lat_icell = latCell.values[iCell]*180/np.pi
print(lonPoint, latPoint)
print(lon_icell, lat_icell)
pres = gsw.conversions.p_from_z(-depth, lat_icell)
nLevels = dsMesh.dims['nVertLevels']
maxLevelCell = dsMesh.maxLevelCell.isel(nCells=iCell)
vertIndex = xr.DataArray.from_dict({'dims': ('nVertLevels',),
                                    'data': np.arange(nLevels)})
vertMask = vertIndex < maxLevelCell

if plotPHCWOA is True:
    # Read in PHC climo
    dsPHC = xr.open_dataset(PHCfilename, decode_times=False)
    # Identify index of selected ocean cell, by computing the minimum
    # of the spherical distance between all points and lonPoint,latPoint
    latPHC = dsPHC.lat.values
    lonPHC = dsPHC.lon.values
    [x, y] = np.meshgrid(lonPHC, latPHC)
    if lonPoint<0:
        spherDist = haversine(x*np.pi/180, y*np.pi/180, (lonPoint+360)*np.pi/180, latPoint*np.pi/180)
    else:
        spherDist = haversine(x*np.pi/180, y*np.pi/180, lonPoint*np.pi/180, latPoint*np.pi/180)
    x = x[np.where(spherDist==np.min(spherDist))][0]
    y = y[np.where(spherDist==np.min(spherDist))][0]
    dsPHC = dsPHC.sel(lat=y, lon=x)
    depthPHC = dsPHC.depth
    presPHC = gsw.conversions.p_from_z(-depthPHC, y)

    # Read in WOA climo
    dsWOA = xr.open_dataset(WOAfilename)
    # Identify index of selected ocean cell, by computing the minimum
    # of the spherical distance between all points and lonPoint,latPoint
    latWOA = dsWOA.lat.values
    lonWOA = dsWOA.lon.values
    [x, y] = np.meshgrid(lonWOA, latWOA)
    spherDist = haversine(x*np.pi/180, y*np.pi/180, lonPoint*np.pi/180, latPoint*np.pi/180)
    x = x[np.where(spherDist==np.min(spherDist))][0]
    y = y[np.where(spherDist==np.min(spherDist))][0]
    dsWOA = dsWOA.sel(lat=y, lon=x)
    depthWOA = dsWOA.depth
    presWOA = gsw.conversions.p_from_z(-depthWOA, y)

if plotHighresMIP is True:
    # Read in data
    Tfiles = []
    Sfiles = []
    for im in range(1, 13):
        Tfiles.append(f'{HighresMIPdir}/thetao_Omon_CESM1-CAM5-SE-HR_hist-1950_r1i1p1f1_gn_{im:02d}_{climoyearStart:04d}{im:02d}_{climoyearEnd:04d}{im:02d}_climo.nc')
        Sfiles.append(f'{HighresMIPdir}/so_Omon_CESM1-CAM5-SE-HR_hist-1950_r1i1p1f1_gn_{im:02d}_{climoyearStart:04d}{im:02d}_{climoyearEnd:04d}{im:02d}_climo.nc')
    dsHighresMIPtemp = xr.open_mfdataset(Tfiles, combine='nested', concat_dim='time', decode_times=False)
    dsHighresMIPsalt = xr.open_mfdataset(Sfiles, combine='nested', concat_dim='time', decode_times=False)
    # Identify index of selected ocean cell, by computing the minimum
    # of the spherical distance between all points and lonPoint,latPoint
    lat = dsHighresMIPtemp.coords['lat'].values
    lon = dsHighresMIPtemp.coords['lon'].values
    if lonPoint<0:
        spherDist = haversine(lon*np.pi/180, lat*np.pi/180, (lonPoint+360)*np.pi/180, latPoint*np.pi/180)
    else:
        spherDist = haversine(lon*np.pi/180, lat*np.pi/180, lonPoint*np.pi/180, latPoint*np.pi/180)
    [nlat, nlon] = np.argwhere(spherDist==np.min(spherDist))[0]
    dsHighresMIPtemp = dsHighresMIPtemp.sel(nlat=nlat, nlon=nlon)
    dsHighresMIPsalt = dsHighresMIPsalt.sel(nlat=nlat, nlon=nlon)
    HighresMIPdepth = 1e-2 * dsHighresMIPtemp['lev']
    HighresMIPpres = gsw.conversions.p_from_z(-HighresMIPdepth, lat[nlat, nlon])
    #
    Tfiles = []
    Sfiles = []
    for im in range(1, 13):
        Tfiles.append(f'{HighresMIP2dir}/thetao_Omon_CESM1-CAM5-SE-HR_highres-future_r1i1p1f1_gn_{im:02d}_2031{im:02d}_2050{im:02d}_climo.nc')
        Sfiles.append(f'{HighresMIP2dir}/so_Omon_CESM1-CAM5-SE-HR_highres-future_r1i1p1f1_gn_{im:02d}_2031{im:02d}_2050{im:02d}_climo.nc')
    dsHighresMIPtemp2 = xr.open_mfdataset(Tfiles, combine='nested', concat_dim='time', decode_times=False)
    dsHighresMIPsalt2 = xr.open_mfdataset(Sfiles, combine='nested', concat_dim='time', decode_times=False)
    # Identify index of selected ocean cell, by computing the minimum
    # of the spherical distance between all points and lonPoint,latPoint
    lat = dsHighresMIPtemp2.coords['lat'].values
    lon = dsHighresMIPtemp2.coords['lon'].values
    if lonPoint<0:
        spherDist = haversine(lon*np.pi/180, lat*np.pi/180, (lonPoint+360)*np.pi/180, latPoint*np.pi/180)
    else:
        spherDist = haversine(lon*np.pi/180, lat*np.pi/180, lonPoint*np.pi/180, latPoint*np.pi/180)
    [nlat, nlon] = np.argwhere(spherDist==np.min(spherDist))[0]
    dsHighresMIPtemp2 = dsHighresMIPtemp2.sel(nlat=nlat, nlon=nlon)
    dsHighresMIPsalt2 = dsHighresMIPsalt2.sel(nlat=nlat, nlon=nlon)
    HighresMIPdepth2 = 1e-2 * dsHighresMIPtemp2['lev']
    HighresMIPpres2 = gsw.conversions.p_from_z(-HighresMIPdepth2, lat[nlat, nlon])

if plotClimos is True:
    for season in seasons:
        # Initialize figure and axis objects
        fig_Tprofile = plt.figure(figsize=figsize, dpi=figdpi)
        ax_Tprofile = fig_Tprofile.add_subplot()
        for tick in ax_Tprofile.xaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        for tick in ax_Tprofile.yaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        ax_Tprofile.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
        ax_Tprofile.yaxis.get_offset_text().set_weight('bold')
        #
        fig_Sprofile = plt.figure(figsize=figsize, dpi=figdpi)
        ax_Sprofile = fig_Sprofile.add_subplot()
        for tick in ax_Sprofile.xaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        for tick in ax_Sprofile.yaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        ax_Sprofile.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
        ax_Sprofile.yaxis.get_offset_text().set_weight('bold')
        #
        fig_Cprofile = plt.figure(figsize=figsize, dpi=figdpi)
        ax_Cprofile = fig_Cprofile.add_subplot()
        for tick in ax_Cprofile.xaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        for tick in ax_Cprofile.yaxis.get_ticklabels():
            tick.set_fontsize(fontsize_smallLabels)
            tick.set_weight('bold')
        ax_Cprofile.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
        ax_Cprofile.yaxis.get_offset_text().set_weight('bold')

        Tfigtitle = f'Temperature ({pointTitle})\n{season} - years {climoyearStart:04d}-{climoyearEnd:04d}'
        Tfigfile = f'{figdir}/Tprofile_icell{iCell:d}_{ensembleName}_{season}_years{climoyearStart:04d}-{climoyearEnd:04d}.png'
        Sfigtitle = f'Salinity ({pointTitle})\n{season} - years {climoyearStart:04d}-{climoyearEnd:04d}'
        Sfigfile = f'{figdir}/Sprofile_icell{iCell:d}_{ensembleName}_{season}_years{climoyearStart:04d}-{climoyearEnd:04d}.png'
        Cfigtitle = f'Sound speed ({pointTitle})\n{season} - years {climoyearStart:04d}-{climoyearEnd:04d}'
        Cfigfile = f'{figdir}/Cprofile_icell{iCell:d}_{ensembleName}_{season}_years{climoyearStart:04d}-{climoyearEnd:04d}.png'

        ax_Tprofile.set_xlabel('Temperature ($^\circ$C)', fontsize=fontsize_labels, fontweight='bold')
        ax_Tprofile.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
        ax_Tprofile.set_title(Tfigtitle, fontsize=fontsize_titles, fontweight='bold')
        #ax_Tprofile.set_xlim(-1.85, 1.8)
        ax_Tprofile.set_ylim(-depth[maxLevelCell.values], 0)
        #ax_Tprofile.set_ylim(-800, 0)
        #
        ax_Sprofile.set_xlabel('Salinity (psu)', fontsize=fontsize_labels, fontweight='bold')
        ax_Sprofile.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
        ax_Sprofile.set_title(Sfigtitle, fontsize=fontsize_titles, fontweight='bold')
        #ax_Sprofile.set_xlim(27.8, 35)
        ax_Sprofile.set_ylim(-depth[maxLevelCell.values], 0)
        #ax_Sprofile.set_ylim(-800, 0)
        #
        ax_Cprofile.set_xlabel('C (m/s)', fontsize=fontsize_labels, fontweight='bold')
        ax_Cprofile.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
        ax_Cprofile.set_title(Cfigtitle, fontsize=fontsize_titles, fontweight='bold')
        #ax_Cprofile.set_xlim(1430., 1470.)
        ax_Cprofile.set_ylim(-depth[maxLevelCell.values], 0)
        #ax_Cprofile.set_ylim(-800, 0)

        for i in range(nEnsembles):
            ensembleMemberName = ensembleMemberNames[i]
            print(f'\nProcessing ensemble member {ensembleMemberName}, season {season}...')

            modelfile = f'{modelClimodir1}{ensembleMemberName}/{modelClimodir2}/mpaso_{season}_{climoyearStart:04d}{season}_{climoyearEnd:04d}{season}_climo.nc'

            dsIn = xr.open_dataset(modelfile).isel(Time=0, nCells=iCell)
            dsIn = dsIn.where(vertMask)
            # Drop all variables but T and S, and mask bathymetry
            allvars = dsIn.data_vars.keys()
            dropvars = set(allvars) - set(['timeMonthly_avg_activeTracers_temperature',
                                           'timeMonthly_avg_activeTracers_salinity'])
            dsIn = dsIn.drop(dropvars)

            Tprofile = dsIn.timeMonthly_avg_activeTracers_temperature.values
            Sprofile = dsIn.timeMonthly_avg_activeTracers_salinity.values
            SA = gsw.conversions.SA_from_SP(Sprofile, pres, lon_icell, lat_icell)
            CT = gsw.conversions.CT_from_pt(SA, Tprofile)
            #sigma0profile = gsw.density.sigma0(SA, CT)
            soundspeed = gsw.sound_speed(SA, CT, pres)

            ax_Tprofile.plot(Tprofile[::-1], -depth[::-1], '-', color=colors[i], linewidth=3, label=f'{ensembleMemberName}')
            ax_Sprofile.plot(Sprofile[::-1], -depth[::-1], '-', color=colors[i], linewidth=3, label=f'{ensembleMemberName}')
            ax_Cprofile.plot(soundspeed[::-1], -depth[::-1], '-', color=colors[i], linewidth=3, label=f'{ensembleMemberName}')

            # Write to file
            outdir = f'{outdir0}/{ensembleName}/{ensembleMemberName}'
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            outfile = f'{outdir}/icell{iCell:d}_profiles_{ensembleName}{ensembleMemberName}_{season}_years{climoyearStart:04d}-{climoyearEnd:04d}.nc'
            dsOut = xr.Dataset()
            dsOut['Tprofile'] = Tprofile
            dsOut['Tprofile'].attrs['units'] = 'degC'
            dsOut['Tprofile'].attrs['long_name'] = 'Potential temperature'
            dsOut['Sprofile'] = Sprofile
            dsOut['Sprofile'].attrs['units'] = 'psu'
            dsOut['Sprofile'].attrs['long_name'] = 'Salinity'
            dsOut['CTprofile'] = CT
            dsOut['CTprofile'].attrs['units'] = 'degC'
            dsOut['CTprofile'].attrs['long_name'] = 'Conservative temperature'
            dsOut['SAprofile'] = SA
            dsOut['SAprofile'].attrs['units'] = 'psu'
            dsOut['SAprofile'].attrs['long_name'] = 'Absolute salinity'
            dsOut['Cprofile'] = soundspeed
            dsOut['Cprofile'].attrs['units'] = 'm/s'
            dsOut['Cprofile'].attrs['long_name'] = 'Sound speed (computed with python gsw package)'
            dsOut['depth'] = depth
            dsOut['depth'].attrs['units'] = 'm'
            dsOut['depth'].attrs['long_name'] = 'depth levels'
            dsOut['lon'] = lon_icell
            dsOut['lon'].attrs['units'] = 'degrees_east'
            dsOut['lon'].attrs['long_name'] = 'point longitude'
            dsOut['lat'] = lat_icell
            dsOut['lat'].attrs['units'] = 'degrees_north'
            dsOut['lat'].attrs['long_name'] = 'point latitude'
            dsOut.to_netcdf(outfile)

        if plotPHCWOA is True:
            dsPHC_monthlyClimo = dsPHC.isel(time=int(season)-1)
            SA = gsw.conversions.SA_from_SP(dsPHC_monthlyClimo['salt'].values, presPHC, x, y)
            CT = gsw.conversions.CT_from_pt(SA, dsPHC_monthlyClimo['temp'].values)
            soundspeedPHC = gsw.sound_speed(SA, CT, presPHC)

            dsWOA_monthlyClimo = dsWOA.isel(month=int(season)-1)
            SA = gsw.conversions.SA_from_SP(dsWOA_monthlyClimo['s_an'].values, presWOA, x, y)
            CT = gsw.conversions.CT_from_pt(SA, dsWOA_monthlyClimo['t_an'].values)
            soundspeedWOA = gsw.sound_speed(SA, CT, presWOA)

            ax_Tprofile.plot(dsPHC_monthlyClimo['temp'][::-1], -depthPHC[::-1], '-', color='mediumvioletred',
                             linewidth=3, label='PHC climatology')
            ax_Sprofile.plot(dsPHC_monthlyClimo['salt'][::-1], -depthPHC[::-1], '-', color='mediumvioletred',
                             linewidth=3, label='PHC climatology')
            ax_Cprofile.plot(soundspeedPHC[::-1], -depthPHC[::-1], '-', color='mediumvioletred',
                             linewidth=3, label='PHC climatology')

            ax_Tprofile.plot(dsWOA_monthlyClimo['t_an'][::-1], -depthWOA[::-1], '-', color='salmon',
                             linewidth=3, label='WOA climatology')
            ax_Sprofile.plot(dsWOA_monthlyClimo['s_an'][::-1], -depthWOA[::-1], '-', color='salmon',
                             linewidth=3, label='WOA climatology')
            ax_Cprofile.plot(soundspeedWOA[::-1], -depthWOA[::-1], '-', color='salmon',
                             linewidth=3, label='WOA climatology')

        if plotHighresMIP is True:
            HighresMIPtemp = dsHighresMIPtemp['thetao'].isel(time=int(season)-1)
            HighresMIPsalt = dsHighresMIPsalt['so'].isel(time=int(season)-1)
            SA = gsw.conversions.SA_from_SP(HighresMIPsalt.values, HighresMIPpres, x, y)
            CT = gsw.conversions.CT_from_pt(SA, HighresMIPtemp.values)
            soundspeed = gsw.sound_speed(SA, CT, HighresMIPpres)

            ax_Tprofile.plot(HighresMIPtemp[::-1], -HighresMIPdepth[::-1], '-', color='gold',
                             linewidth=3, label='HighresMIP')
            ax_Sprofile.plot(HighresMIPsalt[::-1], -HighresMIPdepth[::-1], '-', color='gold',
                             linewidth=3, label='HighresMIP')
            ax_Cprofile.plot(soundspeed[::-1], -HighresMIPdepth[::-1], '-', color='gold',
                             linewidth=3, label='HighresMIP')

            # Write to file
            outdir = f'{outdir0}/HighresMIP/hist-1950'
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            outfile = f'{outdir}/icell{iCell:d}_profiles_HighresMIP_hist-1950_{season}_years{climoyearStart:04d}-{climoyearEnd:04d}.nc'
            dsOut = xr.Dataset()
            dsOut['Tprofile'] = HighresMIPtemp
            dsOut['Tprofile'].attrs['units'] = 'degC'
            dsOut['Tprofile'].attrs['long_name'] = 'Potential temperature'
            dsOut['Sprofile'] = HighresMIPsalt
            dsOut['Sprofile'].attrs['units'] = 'psu'
            dsOut['Sprofile'].attrs['long_name'] = 'Salinity'
            dsOut['CTprofile'] = CT
            dsOut['CTprofile'].attrs['units'] = 'degC'
            dsOut['CTprofile'].attrs['long_name'] = 'Conservative temperature'
            dsOut['SAprofile'] = SA
            dsOut['SAprofile'].attrs['units'] = 'psu'
            dsOut['SAprofile'].attrs['long_name'] = 'Absolute salinity'
            dsOut['Cprofile'] = soundspeed
            dsOut['Cprofile'].attrs['units'] = 'm/s'
            dsOut['Cprofile'].attrs['long_name'] = 'Sound speed (computed with python gsw package)'
            dsOut['depth'] = HighresMIPdepth
            dsOut['depth'].attrs['units'] = 'm'
            dsOut['depth'].attrs['long_name'] = 'depth levels'
            dsOut['lon'] = x
            dsOut['lon'].attrs['units'] = 'degrees_east'
            dsOut['lon'].attrs['long_name'] = 'point longitude'
            dsOut['lat'] = y
            dsOut['lat'].attrs['units'] = 'degrees_north'
            dsOut['lat'].attrs['long_name'] = 'point latitude'
            dsOut.to_netcdf(outfile)
            #
            HighresMIPtemp2 = dsHighresMIPtemp2['thetao'].isel(time=int(season)-1)
            HighresMIPsalt2 = dsHighresMIPsalt2['so'].isel(time=int(season)-1)
            SA = gsw.conversions.SA_from_SP(HighresMIPsalt2.values, HighresMIPpres2, x, y)
            CT = gsw.conversions.CT_from_pt(SA, HighresMIPtemp2.values)
            soundspeed = gsw.sound_speed(SA, CT, HighresMIPpres2)

            ax_Tprofile.plot(HighresMIPtemp2[::-1], -HighresMIPdepth2[::-1], '-', color='darkgoldenrod',
                             linewidth=3, label='HighresMIP 2031-2050')
            ax_Sprofile.plot(HighresMIPsalt2[::-1], -HighresMIPdepth2[::-1], '-', color='darkgoldenrod',
                             linewidth=3, label='HighresMIP 2031-2050')
            ax_Cprofile.plot(soundspeed[::-1], -HighresMIPdepth2[::-1], '-', color='darkgoldenrod',
                             linewidth=3, label='HighresMIP 2031-2050')

            # Write to file
            outdir = f'{outdir0}/HighresMIP/highres-future'
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            outfile = f'{outdir}/icell{iCell:d}_profiles_HighresMIP_highres-future_{season}_years2031-2050.nc'
            dsOut = xr.Dataset()
            dsOut['Tprofile'] = HighresMIPtemp2
            dsOut['Tprofile'].attrs['units'] = 'degC'
            dsOut['Tprofile'].attrs['long_name'] = 'Potential temperature'
            dsOut['Sprofile'] = HighresMIPsalt2
            dsOut['Sprofile'].attrs['units'] = 'psu'
            dsOut['Sprofile'].attrs['long_name'] = 'Salinity'
            dsOut['CTprofile'] = CT
            dsOut['CTprofile'].attrs['units'] = 'degC'
            dsOut['CTprofile'].attrs['long_name'] = 'Conservative temperature'
            dsOut['SAprofile'] = SA
            dsOut['SAprofile'].attrs['units'] = 'psu'
            dsOut['SAprofile'].attrs['long_name'] = 'Absolute salinity'
            dsOut['Cprofile'] = soundspeed
            dsOut['Cprofile'].attrs['units'] = 'm/s'
            dsOut['Cprofile'].attrs['long_name'] = 'Sound speed (computed with python gsw package)'
            dsOut['depth'] = HighresMIPdepth2
            dsOut['depth'].attrs['units'] = 'm'
            dsOut['depth'].attrs['long_name'] = 'depth levels'
            dsOut['lon'] = x
            dsOut['lon'].attrs['units'] = 'degrees_east'
            dsOut['lon'].attrs['long_name'] = 'point longitude'
            dsOut['lat'] = y
            dsOut['lat'].attrs['units'] = 'degrees_north'
            dsOut['lat'].attrs['long_name'] = 'point latitude'
            dsOut.to_netcdf(outfile)

        #ax_Tprofile.legend(prop=legend_properties)
        ax_Tprofile.legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1, 0.5))
        ax_Tprofile.grid(visible=True, which='both')
        fig_Tprofile.savefig(Tfigfile, bbox_inches='tight')
        plt.close(fig_Tprofile)

        #ax_Sprofile.legend(prop=legend_properties)
        ax_Sprofile.legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1, 0.5))
        ax_Sprofile.grid(visible=True, which='both')
        fig_Sprofile.savefig(Sfigfile, bbox_inches='tight')
        plt.close(fig_Sprofile)

        #ax_Cprofile.legend(prop=legend_properties)
        ax_Cprofile.legend(prop=legend_properties, loc='lower left', bbox_to_anchor=(1, 0.5))
        ax_Cprofile.grid(visible=True, which='both')
        fig_Cprofile.savefig(Cfigfile, bbox_inches='tight')
        plt.close(fig_Cprofile)

if plotMonthly is True:
    for year in years:
        for month in months:

            Tfigtitle = f'Temperature ({pointTitle}), year={year}, month={month}'
            Sfigtitle = f'Salinity ({pointTitle}), year={year}, month={month}'
            Tfigfile = f'{figdir}/Tprofile_icell{iCell:d}_{ensembleName}_{year:04d}-{month:02d}.png'
            Sfigfile = f'{figdir}/Sprofile_icell{iCell:d}_{ensembleName}_{year:04d}-{month:02d}.png'

            for i in range(nEnsembles):
                ensembleMemberName = ensembleMemberNames[i]
                print(f'\nProcessing ensemble member {ensembleMemberName}, year={year}, month={month}...')

                modelfile = f'{modeldir1}{ensembleMemberName}/{modeldir2}/{ensembleName}{ensembleMemberName}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

