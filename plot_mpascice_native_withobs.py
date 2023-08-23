from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import glob
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import numpy.ma as ma
from scipy.ndimage.filters import gaussian_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cols
from matplotlib.pyplot import cm
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import cmocean

def _add_land_lakes_coastline(ax):
    land_50m = cfeature.NaturalEarthFeature(
            'physical', 'land', '50m', edgecolor='face',
            facecolor='lightgray', linewidth=0.5)
    lakes_50m = cfeature.NaturalEarthFeature(
            'physical', 'lakes', '50m', edgecolor='k',
            facecolor='aliceblue',
            linewidth=0.5)
    coast_50m = cfeature.NaturalEarthFeature(
            'physical', 'coastline', '50m', edgecolor='k',
            facecolor='None', linewidth=0.5)
    ax.add_feature(land_50m, zorder=2)
    ax.add_feature(lakes_50m, zorder=3)
    ax.add_feature(coast_50m, zorder=4)

projdir = '/global/project/projectdirs/e3sm'
meshname = '60to10'
#modeldir = '/global/cscratch1/sd/milena/E3SM_simulations/ARRM60to10_JRA_GM_ramp/run'
#modeldir = '/global/cscratch1/sd/milena/SeaIceIC'
modeldir = '/global/cfs/projectdirs/m1199/e3sm-arrm-simulations/20210416.GMPAS-JRA1p4.TL319_oARRM60to10.cori-knl/run'
meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oARRM60to10/ocean.ARRM60to10.180715.nc'
figdir = './seaice_native'
if not os.path.isdir(figdir):
    os.mkdir(figdir)
#runname = 'E3SM-Arctic-OSI{}'.format(meshname)
#runname = 'E3SMv2_Arc_G60to10_cf34'
runname = '20210416.GMPAS-JRA1p4.TL319_oARRM60to10.cori-knl'

#varname = 'iceAreaCell' # ice concentration in fraction units (0-1)
varname = 'iceVolumeCell' # ice thickness in m

# Observation availability is as follows:
#   - SSM/I is avaliable monthly from Oct-1978 until Dec-2018 (present-day data available, but needs to be downloaded)
#   - IceSat is available for Mar,Oct-2003; Mar,Oct-2004; Mar,Nov-2005; Mar,Nov-2006; Mar,Oct-2007; Mar,Oct-2008
#            (note that IceSat-2 data started becoming available in late 2018)
#   - CryoSat-2 is available for Oct-Dec-2010; Jan-Apr and Oct-Dec 2011-2018; and Jan-Apr 2019 (more data will be available later)
# Therefore, years/months can be chosen here, but in some cases (especially for IceSat) observations won't be available
# and won't be plotted
#years = [1980, 1990, 2000, 2010, 2014, 2016]
#months = [3, 9, 10] # Sep (Oct) best for ice concentration (thickness)
#years = [1989, 1990, 1994]
#years = [1993]
years = [1988]
months = [1] # Sep (Oct) best for ice concentration (thickness)

#modelJRAcycle = 3 # choose JRA cycle from which to plot
modelJRAcycle = 1 # choose JRA cycle from which to plot
JRAyear1 = 1958
JRAyear2 = 2016
cycleYears = JRAyear2 - JRAyear1 + 1

pi2deg = 180/np.pi

# Info about polar stereographic projection on which obs were interpolated
dx = 25000
dy = 25000
xobs = np.arange(-3850000, +3750000, +dx)
yobs = np.arange(+5850000, -5350000, -dy)
kw = dict(central_latitude=90, central_longitude=-45, true_scale_latitude=70)

if varname=='iceAreaCell':
    figtext = 'Sea-ice concentration ({})'.format(meshname)
    units = 'ice fraction'
    clevels_obs = [0.15, 0.80]
    #clevels_obs = [0.15, 0.80, 0.95]
    # Colormap for model field
    #colormap = cmocean.cm.ice
    #colorIndices = [0, 40, 80, 120, 160, 180, 200, 240, 255]
    #colormap = cols.ListedColormap(colormap(colorIndices))
    colormap = cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
                                    (0.169, 0.435, 0.223), (0.455, 0.478, 0.196), (0.757, 0.474, 0.435),\
                                    (0.827, 0.561, 0.772), (0.761, 0.757, 0.949), (0.808, 0.921, 0.937)])
    clevels_mod = [0.15, 0.3, 0.45, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1]
elif varname=='iceVolumeCell':
    figtext = 'Sea-ice thickness ({})'.format(meshname)
    units = 'meters'
    #clevels_obs = [2, 3.5]
    clevels_obs = [2]
    # Colormap for model field
    #colormap = cmocean.cm.deep_r
    #colormap = cmocean.cm.thermal
    colormap = plt.get_cmap('YlGnBu_r')
    colorIndices = [0, 40, 80, 120, 160, 180, 200, 240, 255]
    colormap = cols.ListedColormap(colormap(colorIndices))
    #colormap = cols.ListedColormap([(0.102, 0.094, 0.204), (0.07, 0.145, 0.318),  (0.082, 0.271, 0.306),\
    #                                (0.169, 0.435, 0.223), (0.455, 0.478, 0.196), (0.757, 0.474, 0.435),\
    #                                (0.827, 0.561, 0.772), (0.761, 0.757, 0.949), (0.808, 0.921, 0.937)])
    #clevels_mod = [0., 0.5, 1.0, 1.5, 2.0, 2.5, 3., 3.5, 4., 5.]
    clevels_mod = [0., 0.2, 1.0, 1.5, 2.0, 2.5, 3., 3.5, 4., 5.]

else:
    raise SystemExit('varname {} not supported'.format(varname))
cnorm = mpl.colors.BoundaryNorm(clevels_mod, colormap.N)

# Info about MPAS mesh
f = netcdf_dataset(meshfile, mode='r')
lon = f.variables['lonCell'][:]
lat = f.variables['latCell'][:]
f.close()
lon = pi2deg*lon
lat = pi2deg*lat

figsize = [20, 20]
figdpi = 150

for year in years:
    if meshname=='60to10':
        modelYear = year - JRAyear1 + 1 + (modelJRAcycle-1)*cycleYears
    else:
        modelYear = year + (modelJRAcycle-1)*cycleYears
    for month in months:
        # Read in observations
        if varname=='iceAreaCell':
            sigma_filter = None
            obsdir = '{}/observations_with_original_data/SeaIce/SSMI/NASATeam_NSIDC0051/north/monthly'.format(projdir)
            if year<1979:
                print('Warning: observations for ice concentration not available for year {:d}. Skipping obs plotting...'.format(year))
                fld_obs = None
            else:
                if year>=2007:
                    obsfilecode = 'f17'
                elif year>=1996 or (year==1995 and month>=10):
                    obsfilecode = 'f13'
                elif year>=1992 or (year==1995 and month<10):
                    obsfilecode = 'f11'
                elif year>=1988 or (year==1987 and month>=9):
                    obsfilecode = 'f08'
                else:
                    obsfilecode = 'n07'
                if year>=2015:
                    obsfile = '{}/nt_{:04d}{:02d}_{}_v1.1_n.bin'.format(obsdir, year, month, obsfilecode)
                else:
                    obsfile = '{}/nt_{:04d}{:02d}_{}_v01_n.bin'.format(obsdir, year, month, obsfilecode)
                print(obsfile)
                with open(obsfile, 'rb') as f:
                    hdr = f.read(300)
                    fld_obs = np.fromfile(f, dtype=np.uint8)
                f.close()
                fld_obs = fld_obs/250.
                fld_obs = ma.masked_greater(fld_obs, 1.0)
                #fld_obs = ma.masked_less(fld_obs, 0.1) # useful only if plotting in colors
        else:
            if year<2003 or year==2009:
                print('Warning: observations for ice thickness not available for year {:d}. Skipping obs plotting...'.format(year))
                fld_obs = None
            else:
                if year<2009: # IceSat data
                    sigma_filter = 1
                    obsdir = '{}/observations_with_original_data/SeaIce/ICESat/Arctic/NSIDC0393_GLAS_SI_Freeboard_v01/glas_seaice_grids'.format(projdir)
                    if month!=3 and month!=10 and month!=11:
                        print('Warning: observations for ice thickness not available for year {:d}, month {:d}. Skipping obs plotting...'.format(year, month))
                        fld_obs = None
                    else:
                        if year==2003 and month==3:
                            obsfilecode = '1'
                        elif year==2003 and month==10:
                            obsfilecode = '2a'
                        elif year==2004 and month==3:
                            obsfilecode = '2b'
                        elif year==2004 and month==10:
                            obsfilecode = '3a'
                        elif year==2005 and month==3:
                            obsfilecode = '3b'
                        elif year==2005 and month==11:
                            obsfilecode = '3d'
                        elif year==2006 and month==3:
                            obsfilecode = '3e'
                        elif year==2006 and month==11:
                            obsfilecode = '3g'
                        elif year==2007 and month==3:
                            obsfilecode = '3h'
                        elif year==2007 and month==10:
                            obsfilecode = '3i'
                        elif year==2008 and month==3:
                            obsfilecode = '3j'
                        elif year==2008 and month==10:
                            obsfilecode = '3k'
                        obsfile = '{}/laser{}_thickness_mskd.img'.format(obsdir, obsfilecode)
                        print(obsfile)
                        with open(obsfile, 'rb') as f:
                            #hdr = f.read(300)
                            fld_obs = np.fromfile(f, dtype=np.float32)
                        f.close()
                else: # CryoSat-2 data
                    sigma_filter = 0.5
                    obsdir = '{}/observations_with_original_data/SeaIce/CryoSat-2'.format(projdir)
                    if (year==2010 and month<10) or (month>4 and month<10):
                        print('Warning: observations for ice thickness not available for year {:d}, month {:d}. Skipping obs plotting...'.format(year, month))
                        fld_obs = None
                    else:
                        obsfile = glob.glob('{}/RDEFT4_{:04d}{:02d}*.nc'.format(obsdir, year, month))[0]
                        #obsfile = '{}/RDEFT4_{:04d}{:02d}*.nc'.format(obsdir, year, month)
                        print(obsfile)
                        f = netcdf_dataset(obsfile, mode='r')
                        fld_obs = f.variables['sea_ice_thickness'][:, :]
                        f.close()

        # Read in model data
        # v1:
        #modelfile = '{}/mpascice.hist.am.timeSeriesStatsMonthly.{:04d}-{:02d}-01.nc'.format(modeldir,\
        #            modelYear, month)
        # v2:
        modelfile = '{}/{}.mpassi.hist.am.timeSeriesStatsMonthly.{:04d}-{:02d}-01.nc'.format(modeldir,\
                    runname, modelYear, month)
        print(modelfile)
        f = netcdf_dataset(modelfile, mode='r')
        fld_mod = f.variables['timeMonthly_avg_{}'.format(varname)][:, :]
        f.close()
        fld_mod = np.squeeze(fld_mod)
        if varname=='iceVolumeCell':
            fld_mod = ma.masked_less(fld_mod, 0.01)
        else:
            fld_mod = ma.masked_less(fld_mod, 0.15)

        plt.figure(figsize=figsize, dpi=figdpi)
        figtitle = '{}, Year={:04d}, Month={:02d}'.format(figtext, year, month)

        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=0))

        _add_land_lakes_coastline(ax)

        data_crs = ccrs.PlateCarree()
        ax.set_extent([-180, 180, 50, 90], crs=data_crs)
        gl = ax.gridlines(crs=data_crs, color='k', linestyle=':', zorder=5)
        # This will work with cartopy 0.18:
        #gl.xlocator = mticker.FixedLocator(np.arange(-180., 181., 20.))
        #gl.ylocator = mticker.FixedLocator(np.arange(-80., 81., 10.))

        # Plot model field
        sc = ax.scatter(lon, lat, s=0.2, c=fld_mod, cmap=colormap, norm=cnorm,
                        marker='o', transform=data_crs)
        cax, kwc = mpl.colorbar.make_axes(ax, location='bottom', pad=0.05, shrink=0.7)
        cbar = plt.colorbar(sc, cax=cax, ticks=clevels_mod, boundaries=clevels_mod, extend='max', **kwc)
        cbar.ax.tick_params(labelsize=22, labelcolor='black')
        cbar.set_label(units, fontsize=20, fontweight='bold')

        if fld_obs is not None:
            fld_obs = fld_obs.reshape(448, 304)
            if sigma_filter is not None:
                fld_obs = gaussian_filter(fld_obs, sigma_filter)
            # Plot obs contours
            cs = ax.contour(xobs, yobs, fld_obs, clevels_obs, colors='firebrick', linewidths=2,
                           linestyles='solid', transform=ccrs.Stereographic(**kw))
            if len(clevels_obs)>1:
                ax.clabel(cs, inline=1, fontsize=10, fmt='%1.2f')
            #cs = ax.pcolormesh(xobs, yobs, fld_obs, cmap=plt.cm.Blues,
            #                   transform=ccrs.Stereographic(**kw))
            #cs = ax.contourf(xobs, yobs, fld_obs, cmap=plt.cm.Blues,
            #                   transform=ccrs.Stereographic(**kw))
            #figtitle = '{} (color=model, contours=obs)'.format(figtitle)

        ax.set_title(figtitle, y=1.04, fontsize=22, fontweight='bold')
        figfile = '{}/{}NH_{}_{:04d}-{:02d}.png'.format(figdir, varname, runname, year, month)
        plt.savefig(figfile, bbox_inches='tight')
        plt.close()
