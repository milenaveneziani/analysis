#
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import cmocean
from cm_xml_to_matplotlib import make_cmap
import os
import glob
from mpas_analysis.shared.io.utility import decode_strings
import gsw


# settings for nersc
runname = 'E3SM-Arcticv2.1_historical0301'
modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{runname}/archive'
oceandir = f'{modeldir}/ocn/hist'
seaicedir = f'{modeldir}/ice/hist'
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
regionmaskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arctic_regions_detailed.nc'
regionName = 'Canada Basin'
#regionName = 'Barents Sea'
#regionName = 'Eurasian Basin'
#regionName = 'Kara Sea'
#regionName = 'Greenland Sea'
#regionName = 'Norwegian Sea'

years = np.arange(1950, 1971) # try with 20 years at a time
#years = np.arange(1971, 1991)
years = np.arange(2000, 2015)
months = [2, 8]

plotPHCWOA = True

# relevant if plotPHCWOA=True
PHCfilename = '/global/cfs/cdirs/e3sm/observations_with_original_data/Ocean/PHC3.0/phc3.0_monthly_accessed08-08-2019.nc'
WOAfilename = '/global/cfs/cdirs/e3sm/observations_with_original_data/Ocean/WOA18/decadeAll/0.25degGrid/woa18_decav_04_TS_mon.nc'

figdir = f'./acoustics/{runname}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

outdir = f'./acoustics_data/{runname}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

if regionName=='Canada Basin':
    latRegion = [68, 82]
    lonRegion = [-160, -125]
    zmax = -1000.
elif regionName=='Barents Sea':
    latRegion = [68, 82]
    lonRegion = [20, 65]
    zmax = -300.
elif regionName=='Kara Sea':
    latRegion = [70, 82]
    lonRegion = [65, 100]
    zmax = -300.
elif regionName=='Eurasian Basin':
    latRegion = [82, 89]
    lonRegion = [0, 140]
    zmax = -1000.
else:
    latRegion = None
    lonRegion = None
    zmax = -1000.

figsize = (20, 10)
figdpi = 150
fontsize_smallLabels = 18
fontsize_labels = 20
fontsize_titles = 24
legend_properties = {'size':14, 'weight':'bold'}
#colormap = cm.viridis
#colormap = make_cmap('./ScientificColourMaps7/bukavu/bukavu_PARAVIEW.xml')
colormap = make_cmap('./ScientificColourMaps7/batlow/batlow_PARAVIEW.xml')
#########################################################################

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
lonMean = 180/np.pi*lonCell.where(cellMask, drop=True).mean('nCells')
latMean = 180/np.pi*latCell.where(cellMask, drop=True).mean('nCells')
pres = gsw.conversions.p_from_z(-depth, latMean)

# Read in PHC/WOA climo if plotPHCWOA is True
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

for month in months:
    print(f'Processing month {month}...')
    # Initialize plots
    color = iter(colormap(np.linspace(0, 1, np.size(years))))
    Tfigtitle = f'Temperature, {runname} ({regionName}, month={month})'
    Tfigfile = f'{figdir}/Tprofile{rname}_{years[0]:04d}-{years[-1]:04d}_M{month:02d}.png'
    figT = plt.figure(figsize=figsize, dpi=figdpi)
    axT = figT.add_subplot()
    for tick in axT.xaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    for tick in axT.yaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    axT.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
    axT.yaxis.get_offset_text().set_weight('bold')
    axT.set_xlabel('Temperature ($^\circ$C)', fontsize=fontsize_labels, fontweight='bold')
    axT.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
    axT.set_title(Tfigtitle, fontsize=fontsize_titles, fontweight='bold')
    axT.set_xlim(-1.8, 1.6)
    axT.set_ylim(zmax, 0)

    Sfigtitle = f'Salinity, {runname} ({regionName}, month={month})'
    Sfigfile = f'{figdir}/Sprofile{rname}_{years[0]:04d}-{years[-1]:04d}_M{month:02d}.png'
    figS = plt.figure(figsize=figsize, dpi=figdpi)
    axS = figS.add_subplot()
    for tick in axS.xaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    for tick in axS.yaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    axS.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
    axS.yaxis.get_offset_text().set_weight('bold')
    axS.set_xlabel('Salinity (psu)', fontsize=fontsize_labels, fontweight='bold')
    axS.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
    axS.set_title(Sfigtitle, fontsize=fontsize_titles, fontweight='bold')
    axS.set_xlim(25.4, 35.2)
    axS.set_ylim(zmax, 0)

    Cfigtitle = f'Sound speed, {runname} ({regionName}, month={month})'
    Cfigfile = f'{figdir}/Cprofile{rname}_{years[0]:04d}-{years[-1]:04d}_M{month:02d}.png'
    figC = plt.figure(figsize=figsize, dpi=figdpi)
    axC = figC.add_subplot()
    for tick in axC.xaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    for tick in axC.yaxis.get_ticklabels():
        tick.set_fontsize(fontsize_smallLabels)
        tick.set_weight('bold')
    axC.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
    axC.yaxis.get_offset_text().set_weight('bold')
    axC.set_xlabel('C (m/s)', fontsize=fontsize_labels, fontweight='bold')
    axC.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
    axC.set_title(Cfigtitle, fontsize=fontsize_titles, fontweight='bold')
    axC.set_xlim(1430., 1470.)
    axC.set_ylim(zmax, 0)

    for year in years:
        outfile = f'{outdir}/regionalProfiles_{rname}_{year:04d}-{month:02d}.nc'
        if not os.path.exists(outfile):
            print(f'   Processing year {year}...')
            modelfile = f'{oceandir}/{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'
            dsIn = xr.open_dataset(modelfile).isel(Time=0)
            # Drop all variables but T and S
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
            sigma0profile = gsw.density.sigma0(SA, CT)
            soundspeed = gsw.sound_speed(SA, CT, pres)

            # Write to file
            dsOut = xr.Dataset()
            dsOut['Tprofile'] = Tprofile
            dsOut['Tprofile'].attrs['units'] = 'degC'
            dsOut['Tprofile'].attrs['long_name'] = 'temperature profile'
            dsOut['Sprofile'] = Sprofile
            dsOut['Sprofile'].attrs['units'] = 'psu'
            dsOut['Sprofile'].attrs['long_name'] = 'salinity profile'
            dsOut['CTprofile'] = CT
            dsOut['CTprofile'].attrs['units'] = 'degC'
            dsOut['CTprofile'].attrs['long_name'] = 'conservative temperature profile'
            dsOut['SAprofile'] = SA
            dsOut['SAprofile'].attrs['units'] = 'psu'
            dsOut['SAprofile'].attrs['long_name'] = 'absolute salinity profile'
            dsOut['sigma0profile'] = sigma0profile
            dsOut['sigma0profile'].attrs['units'] = 'kg/m^3'
            dsOut['sigma0profile'].attrs['long_name'] = 'sigma0 profile'
            dsOut['Cprofile'] = soundspeed
            dsOut['Cprofile'].attrs['units'] = 'm/s'
            dsOut['Cprofile'].attrs['long_name'] = 'sound speed profile (computed with python gsw package)'
            dsOut['depth'] = depth
            dsOut['depth'].attrs['units'] = 'm'
            dsOut['depth'].attrs['long_name'] = 'depth levels'
            dsOut.to_netcdf(outfile)
        else:
            print(f'   Profiles for year {year} was computed previously. Plotting only...')
            profiles = xr.open_dataset(outfile)
            Tprofile = profiles.Tprofile.values
            Sprofile = profiles.Sprofile.values
            soundspeed = profiles.Cprofile.values
            depth = profiles.depth.values

        # Update plots
        c = next(color)
        axT.plot(Tprofile[::-1], -depth[::-1], '-', color=c, linewidth=3, label=f'{year}')
        axS.plot(Sprofile[::-1], -depth[::-1], '-', color=c, linewidth=3, label=f'{year}')
        axC.plot(soundspeed[::-1], -depth[::-1], '-', color=c, linewidth=3, label=f'{year}')

    if plotPHCWOA is True:
        dsPHC_monthlyClimo = dsPHC.isel(time=month-1)
        SA = gsw.conversions.SA_from_SP(dsPHC_monthlyClimo['salt'].values, presPHC, lonRegionMean, latRegionMean)
        CT = gsw.conversions.CT_from_pt(SA, dsPHC_monthlyClimo['temp'].values)
        soundspeedPHC = gsw.sound_speed(SA, CT, presPHC)

        dsWOA_monthlyClimo = dsWOA.isel(month=month-1)
        SA = gsw.conversions.SA_from_SP(dsWOA_monthlyClimo['s_an'].values, presWOA, lonRegionMean, latRegionMean)
        CT = gsw.conversions.CT_from_pt(SA, dsWOA_monthlyClimo['t_an'].values)
        soundspeedWOA = gsw.sound_speed(SA, CT, presWOA)

        axT.plot(dsPHC_monthlyClimo['temp'][::-1], -depthPHC[::-1], '-', color='lightseagreen',
                 linewidth=3, label='PHC climatology')
        axS.plot(dsPHC_monthlyClimo['salt'][::-1], -depthPHC[::-1], '-', color='lightseagreen',
                 linewidth=3, label='PHC climatology')
        axC.plot(soundspeedPHC[::-1], -depthPHC[::-1], '-', color='lightseagreen',
                 linewidth=3, label='PHC climatology')

        axT.plot(dsWOA_monthlyClimo['t_an'][::-1], -depthWOA[::-1], '-', color='aqua',
                 linewidth=3, label='WOA climatology')
        axS.plot(dsWOA_monthlyClimo['s_an'][::-1], -depthWOA[::-1], '-', color='aqua',
                 linewidth=3, label='WOA climatology')
        axC.plot(soundspeedWOA[::-1], -depthWOA[::-1], '-', color='aqua',
                 linewidth=3, label='WOA climatology')

    axT.legend(prop=legend_properties)
    plt.grid(True)
    figT.savefig(Tfigfile, bbox_inches='tight')
    plt.close(figT)

    axS.legend(prop=legend_properties)
    plt.grid(True)
    figS.savefig(Sfigfile, bbox_inches='tight')
    plt.close(figS)

    axC.legend(prop=legend_properties)
    plt.grid(True)
    figC.savefig(Cfigfile, bbox_inches='tight')
    plt.close(figC)
