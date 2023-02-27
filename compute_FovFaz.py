#!/usr/bin/env python
"""
Name: compute_FovFaz.py
Author: Milena Veneziani

Computes Fov (meridional freshwater flux due to overturning circulation)
and Faz (meridional freshwater flux due to the azonal or gyre circulation
component) as defined in de Vries and Weber (2005), their Eqs. 2,3.

"""

# ensure plots are rendered on ICC
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from netCDF4 import Dataset
import glob
import platform
import gsw

szero = 35.0
m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
#
rhoRef = 1027.0 # kg/m^3
m3ps_to_km3py = 1e-9*86400*365.25  # m^3/s FW flux to km^3/year

# years for transport time series (empty if plotting full time series)
#years = '001[0-2]'
years = '0001'
#years = ''

meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
maskfile = '/global/project/projectdirs/e3sm/milena/mpas-region_masks/ARRM10to60E2r1_atlantic34S.nc'
casename = 'fullyRRM_lat-dep-bd-submeso'
historyFileList = '/global/cscratch1/sd/milena/e3sm_scratch/cori-knl/20221201.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.lat-dep-bd-submeso.cori-knl/run/20221201.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.lat-dep-bd-submeso.cori-knl.mpaso.hist.am.timeSeriesStatsMonthly.{}*nc'.format(years)
#
#meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/oRRS18to6v3/oRRS18to6v3.171116.nc'
#maskfile = '/global/project/projectdirs/e3sm/diagnostics/mpas_analysis/region_masks/RRS18to6v3_atlantic34S.nc'
#casename = 'E3SM-HR'
#historyFileList = '/global/cscratch1/sd/milena/E3SM_simulations/theta.20180906.branch_noCNT.A_WCYCL1950S_CMIP6_HR.ne120_oRRS18v3_ICG/run/mpaso.hist.am.timeSeriesStatsMonthly.{}*nc'.format(years)
#
#meshfile = '/global/project/projectdirs/e3sm/inputdata/ocn/mpas-o/EC30to60E2r2/ocean.EC30to60E2r2.210210.nc'
#maskfile = '/global/project/projectdirs/e3sm/milena/mpas-region_masks/EC30to60E2r2_atlantic34S.nc'
#casename = 'v2.LR.piControl'
#historyFileList = '/global/cscratch1/sd/dcomeau/e3sm_scratch/cori-knl/v2.LR.piControl/archive/ocn/hist/{}.mpaso.hist.am.timeSeriesStatsMonthly.{}*nc'.format(casename, years)
#

outfile = 'FovFoz_{}.nc'.format(casename)
transectName = 'all'
#transectName = 'Atlantic34S'
figdir = './FovFaz/{}'.format(casename)
if not os.path.isdir(figdir):
    os.makedirs(figdir)

fileList = sorted(glob.glob(historyFileList))
nTime = len(fileList)
print('Computing time series from the following files ', fileList)

# Read in transect information
mask = get_mask_short_names(xr.open_dataset(maskfile))
if transectName=='all':
    transectList = mask.shortNames[:].values
else:
    transectList = transectName.split(',')
    if platform.python_version()[0]=='3':
        for i in range(len(transectList)):
            transectList[i] = "b'" + transectList[i]
nTransects = len(transectList)
maxEdges = mask.dims['maxEdgesInTransect']
print('  for these transects ', transectList)

# Create a list of edges and total edges in each transect
nEdgesInTransect = np.zeros(nTransects)
edgeVals = np.zeros((nTransects, maxEdges))
for j in range(nTransects):
    amask = mask.sel(shortNames=transectList[j]).squeeze()
    transectEdges = amask.transectEdgeGlobalIDs.values
    inds = np.where(transectEdges > 0)[0]
    nEdgesInTransect[j] = len(inds)
    transectEdges = transectEdges[inds]
    edgeVals[j, :len(inds)] = np.asarray(transectEdges-1, dtype='i')
nEdgesInTransect = np.asarray(nEdgesInTransect, dtype='i')
edgesToRead = edgeVals[0, :nEdgesInTransect[0]]
for j in range(1, nTransects):
    edgesToRead = np.hstack([edgesToRead, edgeVals[j, :nEdgesInTransect[j]]])
edgesToRead = np.asarray(edgesToRead, dtype='i')

# Create a list with the start and stop for transect bounds
nTransectStartStop = np.zeros(nTransects+1)
for j in range(1, nTransects+1):
    nTransectStartStop[j] = nTransectStartStop[j-1] + nEdgesInTransect[j-1]

# Read in relevant mesh information
mesh = xr.open_dataset(meshfile)
dvEdge = mesh.dvEdge.sel(nEdges=edgesToRead).values
cellsOnEdge = mesh.cellsOnEdge.sel(nEdges=edgesToRead).values
edgeSigns = np.zeros((nTransects, len(edgesToRead)))
for j in range(nTransects):
    edgeSigns[j, :] = mask.sel(nEdges=edgesToRead, shortNames=transectList[j]).squeeze().transectEdgeMaskSigns.values

# Read in velocity, salinity, and layerThickness from history files
vTransect = np.zeros((nTime, nTransects))
sTransect = np.zeros((nTime, nTransects))
moc = np.zeros((nTime, nTransects))
Fov = np.zeros((nTime, nTransects))
Faz = np.zeros((nTime, nTransects))
t = np.zeros(nTime)
for i,fname in enumerate(fileList):
    ncid = Dataset(fname, 'r')
    if 'timeMonthly_avg_normalTransportVelocity' in ncid.variables.keys():
        vel = ncid.variables['timeMonthly_avg_normalTransportVelocity'][0, edgesToRead, :]
    elif 'timeMonthly_avg_normalVelocity' in ncid.variables.keys():
        vel = ncid.variables['timeMonthly_avg_normalVelocity'][0, edgesToRead, :]
        if 'timeMonthly_avg_normalGMBolusVelocity' in ncid.variables.keys():
            vel += ncid.variables['timeMonthly_avg_normalGMBolusVelocity'][0, edgesToRead, :]
    else:
        raise KeyError('no appropriate normalVelocity variable found')
    saltOnCells1 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge[:, 0]-1, :]
    saltOnCells2 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge[:, 1]-1, :]
    dzOnCells1 = ncid.variables['timeMonthly_avg_layerThickness'][0, cellsOnEdge[:, 0]-1, :]
    dzOnCells2 = ncid.variables['timeMonthly_avg_layerThickness'][0, cellsOnEdge[:, 1]-1, :]
    t[i] = ncid.variables['timeMonthly_avg_daysSinceStartOfSim'][:]/365.
    ncid.close()
    # Mask values that fall on land
    saltOnCells1[cellsOnEdge[:, 0]==0, :] = np.nan
    saltOnCells2[cellsOnEdge[:, 1]==0, :] = np.nan
    dzOnCells1[cellsOnEdge[:, 0]==0, :] = np.nan
    dzOnCells2[cellsOnEdge[:, 1]==0, :] = np.nan
    # Interpolate values onto edges
    saltOnEdges = np.nanmean(np.array([saltOnCells1, saltOnCells2]), axis=0)
    dzOnEdges = np.nanmean(np.array([dzOnCells1, dzOnCells2]), axis=0)
    # Mask values that fall onto topography
    saltOnEdges[np.logical_or(saltOnEdges> 1e15, saltOnEdges<-1e15)] = np.nan

    # Compute moc, Fov, and Faz for each transect
    for j in range(nTransects):
        start = int(nTransectStartStop[j])
        stop = int(nTransectStartStop[j+1])
        dx = dvEdge[start:stop]
        dz = dzOnEdges[start:stop, :]
        dArea = dx[:, np.newaxis] * dz
        transectLength = np.nansum(dx)
        transectArea = np.nansum(dArea) 

        normalVel = vel[start:stop, :] * edgeSigns[j, start:stop, np.newaxis]
        salt = saltOnEdge[start:stop, :]

        vTransect[i, j] = np.nansum(np.nansum(normalVel * dArea)) / transectArea
        sTransect[i, j] = np.nansum(np.nansum(salt * dArea)) / transectArea
        if use_fixedSref:
            sRef = sZero
        else:
            sRef = sTransect[i, j]

        vZonalAvg = np.nansum(normalVel * dx[:, np.newaxis], axis=0) / transectLength
        sZonalAvg = np.nansum(salt * dx[:, np.newaxis], axis=0) / transectLength
        vAzonal = normalVel - vZonalAvg
        sAzonal = salt - sZonalAvg

        vres = normalVel - vTransect[i, j]
        moc[i, j] = np.nansum(np.nansum(vres * dArea))
        Fov[i, j] = - np.nansum(np.nansum(normalVel * dArea, axis=0) * (sZonalAvg-sRef)) / sRef
        Faz[i, j] = - np.nansum(np.nansum(vazonal * dArea, axis=0) * (sAzonal-sRef)) / sRef
moc = m3ps_to_Sv*moc
Fov = m3ps_to_Sv*Fov
Faz = m3ps_to_Sv*Faz

## Define some dictionaries for transect plotting
#obsDict = {'Drake Passage':[120, 175], 'Tasmania-Ant':[147, 167], 'Africa-Ant':None, 'Antilles Inflow':[-23.1, -13.7], \
#           'Mona Passage':[-3.8, -1.4],'Windward Passage':[-7.2, -6.8], 'Florida-Cuba':[30, 33], 'Florida-Bahamas':[30, 33], \
#           'Indonesian Throughflow':[-21, -11], 'Agulhas':[-90, -50], 'Mozambique Channel':[-20, -8], \
#           'Bering Strait':[0.6, 1.0], 'Lancaster Sound':[-1.0, -0.5], 'Fram Strait':[-4.7, 0.7], \
#           'Robeson Channel':None, 'Davis Strait':[-1.6, -3.6], 'Barents Sea Opening':[1.4, 2.6], \
#           'Nares Strait':[-1.8, 0.2], 'Denmark Strait':None, 'Iceland-Faroe-Scotland':None}
#labelDict = {'Drake Passage':'drake', 'Tasmania-Ant':'tasmania', 'Africa-Ant':'africaAnt', 'Antilles Inflow':'antilles', \
#             'Mona Passage':'monaPassage', 'Windward Passage':'windwardPassage', 'Florida-Cuba':'floridaCuba', \
#             'Florida-Bahamas':'floridaBahamas', 'Indonesian Throughflow':'indonesia', 'Agulhas':'agulhas', \
#             'Mozambique Channel':'mozambique', 'Bering Strait':'beringStrait', 'Lancaster Sound':'lancasterSound', \
#             'Fram Strait':'framStrait', 'Robeson Channel':'robeson', 'Davis Strait':'davisStrait', 'Barents Sea Opening':'barentsSea', \
#             'Nares Strait':'naresStrait', 'Denmark Strait':'denmarkStrait', 'Iceland-Faroe-Scotland':'icelandFaroeScotland'}
figsize = (20, 50)
figdpi = 150

for j in range(nTransects):
    if platform.python_version()[0]=='3':
        searchString = transectList[j][2:]
    else:
        searchString = transectList[j]

    #if searchString in labelDict:
    #    transectName_forfigfile = labelDict[searchString]
    #else:
    #    transectName_forfigfile = searchString.replace(" ", "")
    transectName_forfigfile = searchString.replace(" ", "")

    #if searchString in obsDict:
    #    bounds = obsDict[searchString]
    #else:
    #    bounds = None

    # Plot MOC streamfunction, Fov, and Faz
    figfile = '{}/mocFovFaz_{}_{}.png'.format(figdir, transectName_forfigfile, casename)
    fig, ax = plt.subplots(5, 1, figsize=figsize)
    ax[0].plot(t, vTransect[:, j], 'k', linewidth=2)
    ax[1].plot(t, sTransect[:, j], 'k', linewidth=2)
    ax[2].plot(t, moc[:, j], 'k', linewidth=2)
    ax[3].plot(t, Fov[:, j], 'k', linewidth=2)
    ax[4].plot(t, Faz[:, j], 'k', linewidth=2)

    ax[0].set_title('mean={:5.2f} $\pm$ {:5.2f})'.format(np.nanmean(vTransect[:, j]), \
                    np.nanstd(vTransect[:, j]), fontsize=16, fontweight='bold'))
    ax[1].set_title('mean={:5.2f} $\pm$ {:5.2f})'.format(np.nanmean(sTransect[:, j]), \
                    np.nanstd(sTransect[:, j]), fontsize=16, fontweight='bold'))
    ax[2].set_title('mean={:5.2f} $\pm$ {:5.2f})'.format(np.nanmean(moc[:, j]), \
                    np.nanstd(moc[:, j]), fontsize=16, fontweight='bold'))
    ax[3].set_title('mean={:5.2f} $\pm$ {:5.2f})'.format(np.nanmean(Fov[:, j]), \
                    np.nanstd(Fov[:, j]), fontsize=16, fontweight='bold'))
    ax[4].set_title('mean={:5.2f} $\pm$ {:5.2f})'.format(np.nanmean(Faz[:, j]), \
                    np.nanstd(Faz[:, j]), fontsize=16, fontweight='bold'))
    #if bounds is not None:
    #    plt.gca().fill_between(t, bounds[0]*np.ones_like(t), bounds[1]*np.ones_like(t), alpha=0.3, label='observations (net)')

    ax[0].set_ylabel('Avg cross-transect velocity (m/s)', fontsize=12, fontweight='bold')
    ax[1].set_ylabel('Avg transect salinity (psu)', fontsize=12, fontweight='bold')
    ax[2].set_ylabel('MOC streamfunction (Sv)', fontsize=12, fontweight='bold')
    ax[3].set_ylabel('Fov (Sv)', fontsize=12, fontweight='bold')
    ax[4].set_ylabel('Faz (Sv)', fontsize=12, fontweight='bold')
    ax[4].set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    fig.tight_layout(pad=0.5)
    fig.suptitle('Transect = {} ({})'.format(searchString, casename), fontsize=20, fontweight='bold', y=1.1)
    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')

# Add calls to save transport and then can build up
ncid = Dataset(outfile, mode='w', clobber=True, format='NETCDF3_CLASSIC')

# Save to file
ncid = Dataset(outfile, mode='w', clobber=True, format='NETCDF3_CLASSIC')
ncid.createDimension('Time', None)
ncid.createDimension('nTransects', nTransects)
ncid.createDimension('StrLen', 64)

transectNames = ncid.createVariable('TransectNames', 'c', ('nTransects', 'StrLen'))
times = ncid.createVariable('Time', 'f8', 'Time')
vTransectVar = ncid.createVariable('vTransect', 'f8', ('Time', 'nTransects'))
sTransectVar = ncid.createVariable('sTransect', 'f8', ('Time', 'nTransects'))
mocVar = ncid.createVariable('moc', 'f8', ('Time', 'nTransects'))
FovVar = ncid.createVariable('Fov', 'f8', ('Time', 'nTransects'))
FazVar = ncid.createVariable('Faz', 'f8', ('Time', 'nTransects'))

vTransectVar.units = 'm/s'
sTransectVar.units = 'psu'
mocVar.units = 'Sv'
FovVar.units = 'Sv'
FazVar.units = 'Sv'

vTransectVar.description = 'Cross-transect averaged velocity'
sTransectVar.description = 'Transect averaged salinity'
mocVar.description = 'Meridional Overturning Streamfunction across transect'
FovVar.description = 'Meridional Freshwater transport due to overturning circulation'
FazVar.description = 'Meridional Freshwater transport due to azonal (gyre) circulation'

times[:] = t
vTransectVar[:, :] = vTransect
sTransectVar[:, :] = sTransect
mocVar[:, :] = moc
FovVar[:, :] = Fov
FazVar[:, :] = Faz
for j in range(nTransects):
    nLetters = len(transectList[j])
    transectNames[j, :nLetters] = transectList[j]
ncid.close()


def get_mask_short_names(mask):
    # This is the right way to handle transects defined in the main transect file mask:
    shortnames = [str(aname.values)[:str(aname.values).find(',')].strip()
                  for aname in mask.transectNames]
    #shortnames = mask.transectNames.values
    mask['shortNames'] = xr.DataArray(shortnames, dims='nTransects')
    mask = mask.set_index(nTransects=['transectNames', 'shortNames'])
    return mask
