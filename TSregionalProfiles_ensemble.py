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

ensembleName = 'E3SM-Arcticv2.1_historical'
ensembleMemberNames = ['0101', '0151', '0201', '0251', '0301']
colors = ['mediumblue', 'dodgerblue', 'deepskyblue', 'lightseagreen', 'teal'] # same length as ensembleMemberNames
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
#regionmaskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arcticRegions.nc'
regionmaskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arctic_regions_detailed.nc'
#regionNames = ['all']
#regionNames = ['Canada Basin'] # only works for one region at a time for now
#regionNames = ['Eurasian Basin']
#regionNames = ['Barents Sea']
#regionNames = ['Kara Sea']
#regionNames = ['Greenland Sea']
regionNames = ['Norwegian Sea']

# relevant if plotClimos=True
climoyearStart = 2000
climoyearEnd = 2014
#climoyearStart = 1950
#climoyearEnd = 1970
# seasons options: '01'-'12', 'ANN', 'JFM', 'JAS', 'MAJ', 'OND'
# (depending on what was set in mpas-analysis)
seasons = ['03']
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

figdir = f'./TSprofiles/{ensembleName}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)

figsize = (10, 15)
figdpi = 150
fontsize_smallLabels = 18
fontsize_labels = 20
fontsize_titles = 22
legend_properties = {'size':fontsize_smallLabels, 'weight':'bold'}

nEnsembles = len(ensembleMemberNames)
################

if plotPHCWOA is True:
    if regionNames[0]=='Canada Basin':
        latClimo = [68, 82]
        lonClimo = [-160, -125]
    elif regionNames[0]=='Barents Sea':
        latClimo = [68, 82]
        lonClimo = [20, 65]
    elif regionNames[0]=='Kara Sea':
        latClimo = [70, 82]
        lonClimo = [65, 100]
    elif regionNames[0]=='Eurasian Basin':
        latClimo = [82, 89]
        lonClimo = [0, 140]
    else:
        latClimo = None
        lonClimo = None
        plotPHCWOA = False

    if lonClimo is not None:
        latClimoMean = np.mean(latClimo)
        lonClimoMean = np.mean(lonClimo)

        # Read in PHC climo
        dsPHC = xr.open_dataset(PHCfilename, decode_times=False)
        dsPHC_monthlyClimo = dsPHC.isel(time=int(seasons[0])-1)
        depthPHC = dsPHC.depth
        presPHC = gsw.conversions.p_from_z(-depthPHC, latClimoMean)
        lonClimoPHC = lonClimo.copy()
        if lonClimoPHC[0]<0:
            lonClimoPHC[0] = lonClimoPHC[0]+360
        if lonClimoPHC[1]<0:
            lonClimoPHC[1] = lonClimoPHC[1]+360
        # compute regional quanties
        dsPHC_monthlyClimo = dsPHC_monthlyClimo.sel(lat=slice(latClimo[0], latClimo[1]),
                                                    lon=slice(lonClimoPHC[0], lonClimoPHC[1]))
        dsPHC_monthlyClimo = dsPHC_monthlyClimo.mean(dim='lon').mean(dim='lat')

        # Read in WOA climo
        dsWOA = xr.open_dataset(WOAfilename)
        #dsWOA_seasonalClimo = dsWOA.groupby_bins('month', month_bins, labels=['JFM', 'AMJ', 'JAS', 'OND']).mean()
        #dsWOA_seasonalClimo = dsWOA_seasonalClimo.rename({'month_bins':'season'})
        dsWOA_monthlyClimo = dsWOA.isel(month=int(seasons[0])-1)
        depthWOA = dsWOA.depth
        presWOA = gsw.conversions.p_from_z(-depthWOA, latClimoMean)
        # compute regional quanties
        dsWOA_monthlyClimo = dsWOA_monthlyClimo.sel(lat=slice(latClimo[0], latClimo[1]),
                                                    lon=slice(lonClimo[0], lonClimo[1]))
        dsWOA_monthlyClimo = dsWOA_monthlyClimo.mean(dim='lon').mean(dim='lat')

# Read in regions information
if os.path.exists(regionmaskfile):
    dsRegionMask = xr.open_dataset(regionmaskfile)
else:
    raise IOError(f'Regional mask file {regionmaskfile} not found')
regions = decode_strings(dsRegionMask.regionNames)
if regionNames[0]=='all':
    regionNames = regions
nRegions = np.size(regionNames)

# Read in relevant global mesh information
if os.path.exists(meshfile):
    dsMesh = xr.open_dataset(meshfile)
else:
    raise IOError(f'MPAS restart/mesh file {meshfile} not found')
depth = dsMesh.refBottomDepth
nVertLevels = dsMesh.sizes['nVertLevels']
vertIndex = xr.DataArray.from_dict({'dims': ('nVertLevels',),
                                    'data': np.arange(nVertLevels)})
vertMask = vertIndex < dsMesh.maxLevelCell
areaCell = dsMesh.areaCell
lonCell = dsMesh.lonCell
latCell = dsMesh.latCell

for i in range(nEnsembles):
    ensembleMemberName = ensembleMemberNames[i]

    modelfiles = []
    if plotClimos is True:
        for season in seasons:
            modelfiles.append(glob.glob(f'{modelClimodir1}{ensembleMemberName}/{modelClimodir2}/mpaso_{season}_{climoyearStart:04d}??_{climoyearEnd:04d}??_climo.nc')[0])
    elif plotMonthly is True:
        txt1 = []
        txt2 = []
        for year in years:
            for month in months:
                modelfiles.append(glob.glob(f'{modeldir1}{ensembleMemberName}/{modeldir2}/{ensembleName}{ensembleMemberName}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc')[0])
                txt1.append(f'year={year}, month={month}')
                txt2.append(f'{year:04d}-{month:02d}')

    for nTime in range(len(modelfiles)):

        if plotClimos is True:
            print(f'\nProcessing ensemble member {ensembleMemberName}, season {seasons[nTime]}...')
        elif plotMonthly is True:
            print(f'\nProcessing ensemble member {ensembleMemberName}, {txt1[nTime]}...')

        # Initialize figure and axis objects
        if i==0:
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

        dsIn = xr.open_dataset(modelfiles[nTime])

        # Drop all variables but T and S, and mask bathymetry
        allvars = dsIn.data_vars.keys()
        dropvars = set(allvars) - set(['timeMonthly_avg_activeTracers_temperature',
                                       'timeMonthly_avg_activeTracers_salinity'])
        dsIn = dsIn.drop(dropvars)
        dsIn = dsIn.isel(Time=0)
        dsIn = dsIn.where(vertMask)

        for n in range(nRegions):
            regionName = regionNames[n]
            rname = regionName.replace(' ', '')
            regionIndex = regions.index(regionName)
            print(f'Region: {regionName}')

            if i==0:
                if plotClimos is True:
                    Tfigtitle = f'Temperature ({regionName}), {seasons[nTime]} - years {climoyearStart:04d}-{climoyearEnd:04d}'
                    Tfigfile = f'{figdir}/Tprofile{rname}_{ensembleName}_{seasons[nTime]}_years{climoyearStart:04d}-{climoyearEnd:04d}.png'
                    Sfigtitle = f'Salinity ({regionName}), {seasons[nTime]} - years {climoyearStart:04d}-{climoyearEnd:04d}'
                    Sfigfile = f'{figdir}/Sprofile{rname}_{ensembleName}_{seasons[nTime]}_years{climoyearStart:04d}-{climoyearEnd:04d}.png'
                elif plotMonthly is True:
                    txt = txt1[nTime]
                    Tfigtitle = f'Temperature ({regionName}), {txt}'
                    Sfigtitle = f'Salinity ({regionName}), {txt}'
                    txt = txt2[nTime]
                    Tfigfile = f'{figdir}/Tprofile{rname}_{ensembleName}_{txt}.png'
                    Sfigfile = f'{figdir}/Sprofile{rname}_{ensembleName}_{txt}.png'
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

            dsMask = dsRegionMask.isel(nRegions=regionIndex)
            cellMask = dsMask.regionCellMasks == 1
            regionArea3d = (areaCell * vertMask).where(cellMask, drop=True)
            regionArea = regionArea3d.sum('nCells')
            lonMean = lonCell.where(cellMask, drop=True).mean('nCells')
            latMean = latCell.where(cellMask, drop=True).mean('nCells')
            lonMean = lonMean*180/np.pi
            latMean = latMean*180/np.pi
            pres = gsw.conversions.p_from_z(-depth, latMean)
            #print(lonMean, latMean)

            dsInRegion = dsIn.where(cellMask, drop=True)

            dsInRegionProfile = (dsInRegion * regionArea3d).sum(dim='nCells') / regionArea

            Tprofile = dsInRegionProfile.timeMonthly_avg_activeTracers_temperature.values
            Sprofile = dsInRegionProfile.timeMonthly_avg_activeTracers_salinity.values
            SA = gsw.conversions.SA_from_SP(Sprofile, pres, lonMean, latMean)
            CT = gsw.conversions.CT_from_pt(SA, Tprofile)
            sigma0profile = gsw.density.sigma0(SA, CT)

            # Make plots
            ax_Tprofile.plot(Tprofile[::-1], -depth[::-1], '-', color=colors[i], linewidth=3, label=f'{ensembleMemberName}')
            #ax_Tprofile.plot(dsObs_seasonalClimo['CCB_temperature'].sel(season=season)[::-1], -depth[::-1], '-', color='limegreen',
            #                 linewidth=3, label='obs (Central Canada Basin)')
            #ax_Tprofile.plot(dsObs_seasonalClimo['SCB_temperature'].sel(season=season)[::-1], -depth[::-1], '-', color='mediumspringgreen',
            #                 linewidth=3, label='obs (Southern Canada Basin)')
            if plotPHCWOA is True and i==nEnsembles-1:
                ax_Tprofile.plot(dsPHC_monthlyClimo['temp'][::-1], -depthPHC[::-1], '-', color='mediumvioletred',
                                 linewidth=3, label='PHC climatology')
                ax_Tprofile.plot(dsWOA_monthlyClimo['t_an'][::-1], -depthWOA[::-1], '-', color='salmon',
                                 linewidth=3, label='WOA climatology')
            ax_Tprofile.legend(prop=legend_properties)
            ax_Tprofile.grid(visible=True, which='both')
            fig_Tprofile.savefig(Tfigfile, bbox_inches='tight')
            plt.close(fig_Tprofile)
            #
            ax_Sprofile.plot(Sprofile[::-1], -depth[::-1], '-', color=colors[i], linewidth=3, label=f'{ensembleMemberName}')
            #ax_Sprofile.plot(dsObs_seasonalClimo['CCB_salinity'].sel(season=season)[::-1], -depth[::-1], '-', color='limegreen',
            #                 linewidth=3, label='obs (Central Canada Basin)')
            #ax_Sprofile.plot(dsObs_seasonalClimo['SCB_salinity'].sel(season=season)[::-1], -depth[::-1], '-', color='mediumspringgreen',
            #                 linewidth=3, label='obs (Southern Canada Basin)')
            if plotPHCWOA is True and i==nEnsembles-1:
                ax_Sprofile.plot(dsPHC_monthlyClimo['salt'][::-1], -depthPHC[::-1], '-', color='mediumvioletred',
                                 linewidth=3, label='PHC climatology')
                ax_Sprofile.plot(dsWOA_monthlyClimo['s_an'][::-1], -depthWOA[::-1], '-', color='salmon',
                                 linewidth=3, label='WOA climatology')
            ax_Sprofile.legend(prop=legend_properties)
            ax_Sprofile.grid(visible=True, which='both')
            fig_Sprofile.savefig(Sfigfile, bbox_inches='tight')
            plt.close(fig_Sprofile)
#ax_Tprofile.legend(prop=legend_properties)
#ax_Tprofile.grid(visible=True, which='both')
#fig_Tprofile.savefig(Tfigfile, bbox_inches='tight')
#plt.close(fig_Tprofile)
##
#ax_Sprofile.legend(prop=legend_properties)
#ax_Sprofile.grid(visible=True, which='both')
#fig_Sprofile.savefig(Sfigfile, bbox_inches='tight')
#plt.close(fig_Sprofile)



#    s0figtitle = 'sigma0 profile (Canada Basin), {}'.format(season)
#    s0figfile = '{}/sigma0profileCanadaBasin_multimodel_{}.png'.format(figdir, season)
#    fig_s0profile = plt.figure(figsize=figsize, dpi=figdpi)
#    ax_s0profile = fig_s0profile.add_subplot()
#    for tick in ax_s0profile.xaxis.get_ticklabels():
#        tick.set_fontsize(fontsize_smallLabels)
#        tick.set_weight('bold')
#    for tick in ax_s0profile.yaxis.get_ticklabels():
#        tick.set_fontsize(fontsize_smallLabels)
#        tick.set_weight('bold')
#    ax_s0profile.yaxis.get_offset_text().set_fontsize(fontsize_smallLabels)
#    ax_s0profile.yaxis.get_offset_text().set_weight('bold')
#
#    ax_s0profile.plot(sigma0profile1[::-1], -depth1[::-1], '-', color='mediumblue', linewidth=3, 
#                      label='{}, years={}-{}'.format(runname1, climoyearStart[0], climoyearEnd[0]))
#    ax_s0profile.plot(sigma0profile2[::-1], -depth2[::-1], '-', color='tomato', linewidth=3, 
#                      label='{}, years={}-{}'.format(runname2, climoyearStart[1], climoyearEnd[1]))
#    ax_s0profile.plot(sigma0profile3[::-1], -depth3[::-1], '-', color='slategrey', linewidth=3, 
#                      label='{}, years={}-{}'.format(runname3, climoyearStart[2], climoyearEnd[2]))
#    SA = gsw.conversions.SA_from_SP(dsObs_seasonalClimo['CCB_salinity'].sel(season=season), pres, lonCB, latCB)
#    CT = gsw.conversions.CT_from_pt(SA, dsObs_seasonalClimo['CCB_temperature'].sel(season=season))
#    CCB_sigma0 = gsw.density.sigma0(SA, CT)
#    ax_s0profile.plot(CCB_sigma0[::-1], -depth[::-1], '-', color='limegreen',
#                      linewidth=3, label='obs (Central Canada Basin)')
#    #SA = gsw.conversions.SA_from_SP(dsObs_seasonalClimo['SCB_salinity'].sel(season=season), pres, lonCB, latCB)
#    #CT = gsw.conversions.CT_from_pt(SA, dsObs_seasonalClimo['SCB_temperature'].sel(season=season))
#    #SCB_sigma0 = gsw.density.sigma0(SA, CT)
#    #ax_s0profile.plot(SCB_sigma0[::-1], -depth[::-1], '-', color='mediumspringgreen',
#    #                  linewidth=3, label='obs (Southern Canada Basin)')
#    #SA = gsw.conversions.SA_from_SP(dsPHC_seasonalClimoCB['salt'].sel(season=season), presPHC, lonCB, latCB)
#    #CT = gsw.conversions.CT_from_pt(SA, dsPHC_seasonalClimoCB['temp'].sel(season=season))
#    #PHC_sigma0 = gsw.density.sigma0(SA, CT)
#    #ax_s0profile.plot(PHC_sigma0[::-1], -depthPHC[::-1], '-', color='mediumvioletred',
#    #                  linewidth=3, label='PHC climatology')
#    SA = gsw.conversions.SA_from_SP(dsWOA_seasonalClimoCB['s_an'].sel(season=season), presWOA, lonCB, latCB)
#    CT = gsw.conversions.CT_from_pt(SA, dsWOA_seasonalClimoCB['t_an'].sel(season=season))
#    WOA_sigma0 = gsw.density.sigma0(SA, CT)
#    ax_s0profile.plot(WOA_sigma0[::-1], -depthWOA[::-1], '-', color='blueviolet',
#                      linewidth=3, label='WOA climatology')
#    ax_s0profile.set_xlabel('sigma0', fontsize=fontsize_labels, fontweight='bold')
#    ax_s0profile.set_ylabel('Depth (m)', fontsize=fontsize_labels, fontweight='bold')
#    ax_s0profile.set_title(s0figtitle, fontsize=fontsize_titles, fontweight='bold')
#    ax_s0profile.legend(prop=legend_properties)
#    ax_s0profile.set_xlim(22.3, 28.2)
#    ax_s0profile.set_ylim(-800, 0)
#    plt.grid(True)
#    fig_s0profile.savefig(s0figfile, bbox_inches='tight')
#    plt.close(fig_s0profile)
