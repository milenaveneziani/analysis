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
from mpas_analysis.shared.io.utility import decode_strings
import gsw

plotClimos = True
plotMonthly = False
if plotClimos==plotMonthly:
    raise ValueError('Variables plotClimos and plotMonthly cannot be identical')
plotPHCWOA = True # only works for monthly seasons for now (one season at a  time)
plotHighresMIP = True

ensembleName = 'E3SM-Arcticv2.1_historical'
ensembleMemberNames = ['0101', '0151', '0201', '0251', '0301']
colors = ['mediumblue', 'dodgerblue', 'deepskyblue', 'lightseagreen', 'teal'] # same length as ensembleMemberNames
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
#regionmaskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arcticRegions.nc'
regionmaskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arctic_regions_detailed.nc'
#regionName = 'Barents Sea'
#regionName = 'Eurasian Basin'
regionName = 'Canada Basin'
#regionName = 'Kara Sea'
#regionName = 'Greenland Sea'
#regionName = 'Norwegian Sea'

# relevant if plotClimos=True
climoyearStart = 2000
climoyearEnd = 2014
#climoyearStart = 1950
#climoyearEnd = 1970
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
nLevels = dsMesh.sizes['nVertLevels']
vertIndex = xr.DataArray.from_dict({'dims': ('nVertLevels',),
                                    'data': np.arange(nLevels)})
vertMask = vertIndex < dsMesh.maxLevelCell
areaCell = dsMesh.areaCell
lonCell = dsMesh.lonCell
latCell = dsMesh.latCell

# Read in regions information
rname = regionName.replace(' ', '')
if os.path.exists(regionmaskfile):
    dsRegionMask = xr.open_dataset(regionmaskfile)
else:
    raise IOError(f'Regional mask file {regionmaskfile} not found')
regions = decode_strings(dsRegionMask.regionNames)
regionIndex = regions.index(regionName)
dsMask = dsRegionMask.isel(nRegions=regionIndex)
cellMask = dsMask.regionCellMasks == 1
regionArea3d = (areaCell * vertMask).where(cellMask, drop=True)
regionArea = regionArea3d.sum('nCells')
lonMean = lonCell.where(cellMask, drop=True).mean('nCells')
latMean = latCell.where(cellMask, drop=True).mean('nCells')
lonMean = lonMean*180/np.pi
latMean = latMean*180/np.pi
pres = gsw.conversions.p_from_z(-depth, latMean)

if regionName=='Canada Basin':
    latRegion = [68, 82]
    lonRegion = [200, 235]
    #lonRegion = [-160, -125]
elif regionName=='Barents Sea':
    latRegion = [68, 82]
    lonRegion = [20, 65]
elif regionName=='Kara Sea':
    latRegion = [70, 82]
    lonRegion = [65, 100]
elif regionName=='Eurasian Basin':
    latRegion = [82, 89]
    lonRegion = [0, 140]
else:
    latRegion = None
    lonRegion = None

if plotPHCWOA is True and lonRegion is not None:
    latRegionMean = np.mean(latRegion)
    lonRegionMean = np.mean(lonRegion)

    # Read in PHC climo
    dsPHC = xr.open_dataset(PHCfilename, decode_times=False)
    # compute regional quanties
    lonRegionPHC = lonRegion.copy()
    if lonRegionPHC[0]<0:
        lonRegionPHC[0] = lonRegionPHC[0]+360
    if lonRegionPHC[1]<0:
        lonRegionPHC[1] = lonRegionPHC[1]+360
    dsPHC = dsPHC.sel(lat=slice(latRegion[0], latRegion[1]),
                      lon=slice(lonRegionPHC[0], lonRegionPHC[1]))
    dsPHC = dsPHC.mean(dim='lon').mean(dim='lat')
    depthPHC = dsPHC.depth
    presPHC = gsw.conversions.p_from_z(-depthPHC, latRegionMean)

    # Read in WOA climo
    dsWOA = xr.open_dataset(WOAfilename)
    # compute regional quanties
    dsWOA = dsWOA.sel(lat=slice(latRegion[0], latRegion[1]),
                      lon=slice(lonRegion[0], lonRegion[1]))
    dsWOA = dsWOA.mean(dim='lon').mean(dim='lat')
    depthWOA = dsWOA.depth
    presWOA = gsw.conversions.p_from_z(-depthWOA, latRegionMean)

if plotHighresMIP is True and lonRegion is not None:
    latRegionMean = np.mean(latRegion)
    lonRegionMean = np.mean(lonRegion)

    # Read in data
    Tfiles = []
    Sfiles = []
    for im in range(1, 13):
        Tfiles.append(f'{HighresMIPdir}/thetao_Omon_CESM1-CAM5-SE-HR_hist-1950_r1i1p1f1_gn_{im:02d}_{climoyearStart:04d}{im:02d}_{climoyearEnd:04d}{im:02d}_climo.nc')
        Sfiles.append(f'{HighresMIPdir}/so_Omon_CESM1-CAM5-SE-HR_hist-1950_r1i1p1f1_gn_{im:02d}_{climoyearStart:04d}{im:02d}_{climoyearEnd:04d}{im:02d}_climo.nc')
    dsHighresMIPtemp = xr.open_mfdataset(Tfiles, combine='nested', concat_dim='time', decode_times=False)
    dsHighresMIPsalt = xr.open_mfdataset(Sfiles, combine='nested', concat_dim='time', decode_times=False)
    # compute regional quanties
    lat = dsHighresMIPtemp.coords['lat']
    lon = dsHighresMIPtemp.coords['lon']
    mask = ((lat<=latRegion[1]) & (lat>=latRegion[0]) & (lon<=lonRegion[1]) & (lon>=lonRegion[0]))
    dsHighresMIPtemp = dsHighresMIPtemp.where(mask.compute(), drop=True)
    dsHighresMIPtemp = dsHighresMIPtemp.mean(dim='nlon').mean(dim='nlat')
    dsHighresMIPsalt = dsHighresMIPsalt.where(mask.compute(), drop=True)
    dsHighresMIPsalt = dsHighresMIPsalt.mean(dim='nlon').mean(dim='nlat')
    HighresMIPdepth = 1e-2 * dsHighresMIPtemp['lev']
    HighresMIPpres = gsw.conversions.p_from_z(-HighresMIPdepth, latRegionMean)
    #
    Tfiles = []
    Sfiles = []
    for im in range(1, 13):
        Tfiles.append(f'{HighresMIP2dir}/thetao_Omon_CESM1-CAM5-SE-HR_highres-future_r1i1p1f1_gn_{im:02d}_2031{im:02d}_2050{im:02d}_climo.nc')
        Sfiles.append(f'{HighresMIP2dir}/so_Omon_CESM1-CAM5-SE-HR_highres-future_r1i1p1f1_gn_{im:02d}_2031{im:02d}_2050{im:02d}_climo.nc')
    dsHighresMIPtemp2 = xr.open_mfdataset(Tfiles, combine='nested', concat_dim='time', decode_times=False)
    dsHighresMIPsalt2 = xr.open_mfdataset(Sfiles, combine='nested', concat_dim='time', decode_times=False)
    # compute regional quanties
    lat = dsHighresMIPtemp2.coords['lat']
    lon = dsHighresMIPtemp2.coords['lon']
    mask = ((lat<=latRegion[1]) & (lat>=latRegion[0]) & (lon<=lonRegion[1]) & (lon>=lonRegion[0]))
    dsHighresMIPtemp2 = dsHighresMIPtemp2.where(mask.compute(), drop=True)
    dsHighresMIPtemp2 = dsHighresMIPtemp2.mean(dim='nlon').mean(dim='nlat')
    dsHighresMIPsalt2 = dsHighresMIPsalt2.where(mask.compute(), drop=True)
    dsHighresMIPsalt2 = dsHighresMIPsalt2.mean(dim='nlon').mean(dim='nlat')
    HighresMIPdepth2 = 1e-2 * dsHighresMIPtemp2['lev']
    HighresMIPpres2 = gsw.conversions.p_from_z(-HighresMIPdepth2, latRegionMean)

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

        Tfigtitle = f'Temperature ({regionName})\n{season} - years {climoyearStart:04d}-{climoyearEnd:04d}'
        Tfigfile = f'{figdir}/Tprofile{rname}_{ensembleName}_{season}_years{climoyearStart:04d}-{climoyearEnd:04d}.png'
        Sfigtitle = f'Salinity ({regionName})\n{season} - years {climoyearStart:04d}-{climoyearEnd:04d}'
        Sfigfile = f'{figdir}/Sprofile{rname}_{ensembleName}_{season}_years{climoyearStart:04d}-{climoyearEnd:04d}.png'
        Cfigtitle = f'Sound speed ({regionName})\n{season} - years {climoyearStart:04d}-{climoyearEnd:04d}'
        Cfigfile = f'{figdir}/Cprofile{rname}_{ensembleName}_{season}_years{climoyearStart:04d}-{climoyearEnd:04d}.png'

        ax_Tprofile.set_xlabel('Temperature ($^\circ$C)', fontsize=fontsize_labels, fontweight='bold')
        ax_Tprofile.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
        ax_Tprofile.set_title(Tfigtitle, fontsize=fontsize_titles, fontweight='bold')
        #ax_Tprofile.set_xlim(-1.85, 1.8)
        ax_Tprofile.set_ylim(-800, 0)
        #
        ax_Sprofile.set_xlabel('Salinity (psu)', fontsize=fontsize_labels, fontweight='bold')
        ax_Sprofile.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
        ax_Sprofile.set_title(Sfigtitle, fontsize=fontsize_titles, fontweight='bold')
        #ax_Sprofile.set_xlim(27.8, 35)
        ax_Sprofile.set_ylim(-800, 0)
        #
        ax_Cprofile.set_xlabel('C (m/s)', fontsize=fontsize_labels, fontweight='bold')
        ax_Cprofile.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
        ax_Cprofile.set_title(Cfigtitle, fontsize=fontsize_titles, fontweight='bold')
        #ax_Cprofile.set_xlim(1430., 1470.)
        ax_Cprofile.set_ylim(-800, 0)

        for i in range(nEnsembles):
            ensembleMemberName = ensembleMemberNames[i]
            print(f'\nProcessing ensemble member {ensembleMemberName}, season {season}...')

            modelfile = f'{modelClimodir1}{ensembleMemberName}/{modelClimodir2}/mpaso_{season}_{climoyearStart:04d}{season}_{climoyearEnd:04d}{season}_climo.nc'

            dsIn = xr.open_dataset(modelfile).isel(Time=0)
            # Drop all variables but T and S, and mask bathymetry
            allvars = dsIn.data_vars.keys()
            dropvars = set(allvars) - set(['timeMonthly_avg_activeTracers_temperature',
                                           'timeMonthly_avg_activeTracers_salinity'])
            dsIn = dsIn.drop(dropvars)
            dsIn = dsIn.where(vertMask)

            dsInRegion = dsIn.where(cellMask, drop=True)
            dsInRegionProfile = (dsInRegion * regionArea3d).sum(dim='nCells') / regionArea

            Tprofile = dsInRegionProfile.timeMonthly_avg_activeTracers_temperature.values
            Sprofile = dsInRegionProfile.timeMonthly_avg_activeTracers_salinity.values
            SA = gsw.conversions.SA_from_SP(Sprofile, pres, lonMean, latMean)
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
            outfile = f'{outdir}/{rname}_profiles_{ensembleName}{ensembleMemberName}_{season}_years{climoyearStart:04d}-{climoyearEnd:04d}.nc'
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
            dsOut.to_netcdf(outfile)

        if plotPHCWOA is True:
            dsPHC_monthlyClimo = dsPHC.isel(time=int(season)-1)
            SA = gsw.conversions.SA_from_SP(dsPHC_monthlyClimo['salt'].values, presPHC, lonRegionMean, latRegionMean)
            CT = gsw.conversions.CT_from_pt(SA, dsPHC_monthlyClimo['temp'].values)
            soundspeedPHC = gsw.sound_speed(SA, CT, presPHC)

            dsWOA_monthlyClimo = dsWOA.isel(month=int(season)-1)
            SA = gsw.conversions.SA_from_SP(dsWOA_monthlyClimo['s_an'].values, presWOA, lonRegionMean, latRegionMean)
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
            SA = gsw.conversions.SA_from_SP(HighresMIPsalt.values, HighresMIPpres, lonRegionMean, latRegionMean)
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
            outfile = f'{outdir}/{rname}_profiles_HighresMIP_hist-1950_{season}_years{climoyearStart:04d}-{climoyearEnd:04d}.nc'
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
            dsOut.to_netcdf(outfile)
            #
            HighresMIPtemp2 = dsHighresMIPtemp2['thetao'].isel(time=int(season)-1)
            HighresMIPsalt2 = dsHighresMIPsalt2['so'].isel(time=int(season)-1)
            SA = gsw.conversions.SA_from_SP(HighresMIPsalt2.values, HighresMIPpres2, lonRegionMean, latRegionMean)
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
            outfile = f'{outdir}/{rname}_profiles_HighresMIP_highres-future_{season}_years2031-2050.nc'
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

            Tfigtitle = f'Temperature ({pointTitle})\nyear={year}, month={month}'
            Sfigtitle = f'Salinity ({pointTitle})\nyear={year}, month={month}'
            Tfigfile = f'{figdir}/Tprofile_icell{iCell:d}_{ensembleName}_{year:04d}-{month:02d}.png'
            Sfigfile = f'{figdir}/Sprofile_icell{iCell:d}_{ensembleName}_{year:04d}-{month:02d}.png'

            for i in range(nEnsembles):
                ensembleMemberName = ensembleMemberNames[i]
                print(f'\nProcessing ensemble member {ensembleMemberName}, year={year}, month={month}...')

                modelfile = f'{modeldir1}{ensembleMemberName}/{modeldir2}/{ensembleName}{ensembleMemberName}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
