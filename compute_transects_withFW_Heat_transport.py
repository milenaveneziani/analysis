#!/usr/bin/env python
"""
Name: compute_transects.py
Author: Phillip J. Wolfram, Mark Petersen, Luke Van Roekel, Milena Veneziani

Computes transport through sections.

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

saltRef = 34.8
rhoRef = 1027.0 # kg/m^3
cp = 3.987*1e3 # J/(kg*degK) 
m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
m3ps_to_km3py = 1e-9*86400*365.25  # m^3/s FW flux to km^3/year
W_to_TW = 1e-12

def get_mask_short_names(mask):
    # This is the right way to handle transects defined in the main transect file mask:
    shortnames = [str(aname.values)[:str(aname.values).find(',')].strip()
                  for aname in mask.transectNames]
    #shortnames = mask.transectNames.values
    mask['shortNames'] = xr.DataArray(shortnames, dims='nTransects')
    mask = mask.set_index(nTransects=['transectNames', 'shortNames'])
    return mask

def compute_transport(historyFileList, casename, meshfile, maskfile, figdir,\
                      transectName='Drake Passage', outfile='transport.nc'):
    mesh = xr.open_dataset(meshfile)
    mask = get_mask_short_names(xr.open_dataset(maskfile))

    if transectName=='all' or transectName=='StandardTransportSectionsRegionsGroup':
        transectList = mask.shortNames[:].values
        condition = transectList != "Atlantic Transec"
        transectList = np.extract(condition, transectList)
    else:
        transectList = transectName.split(',')
        if platform.python_version()[0]=='3':
            for i in range(len(transectList)):
                transectList[i] = "b'" + transectList[i]

    print('Computing Transport for the following transects ', transectList)
    nTransects = len(transectList)
    maxEdges = mask.dims['maxEdgesInTransect']
    # Compute refLayerThickness to avoid need for hist file
    refBottom = mesh.refBottomDepth.values
    nz = mesh.dims['nVertLevels']
    h = np.zeros(nz)
    h[0] = refBottom[0]
    for i in range(1, nz):
        h[i] = refBottom[i] - refBottom[i-1]

    # Get a list of edges and total edges in each transect
    nEdgesInTransect = np.zeros(nTransects)
    edgeVals = np.zeros((nTransects, maxEdges))
    for i in range(nTransects):
        amask = mask.sel(shortNames=transectList[i]).squeeze()
        transectEdges = amask.transectEdgeGlobalIDs.values
        inds = np.where(transectEdges > 0)[0]
        nEdgesInTransect[i] = len(inds)
        transectEdges = transectEdges[inds]
        edgeVals[i, :len(inds)] = np.asarray(transectEdges-1, dtype='i')

    nEdgesInTransect = np.asarray(nEdgesInTransect, dtype='i')

    # Create a list with the start and stop for transect bounds
    nTransectStartStop = np.zeros(nTransects+1)
    for j in range(1, nTransects+1):
        nTransectStartStop[j] = nTransectStartStop[j-1] + nEdgesInTransect[j-1]

    edgesToRead = edgeVals[0, :nEdgesInTransect[0]]
    for i in range(1, nTransects):
        edgesToRead = np.hstack([edgesToRead, edgeVals[i, :nEdgesInTransect[i]]])

    edgesToRead = np.asarray(edgesToRead, dtype='i')
    dvEdge = mesh.dvEdge.sel(nEdges=edgesToRead).values
    cellsOnEdge = mesh.cellsOnEdge.sel(nEdges=edgesToRead).values
    edgeSigns = np.zeros((nTransects, len(edgesToRead)))
    for i in range(nTransects):
        edgeSigns[i, :] = mask.sel(nEdges=edgesToRead, shortNames=transectList[i]).squeeze().transectEdgeMaskSigns.values

    # Read history files one at a time and slice
    fileList = sorted(glob.glob(historyFileList))
    vol_transport = np.zeros((len(fileList), nTransects))
    heat_transport = np.zeros((len(fileList), nTransects))
    salt_transport = np.zeros((len(fileList), nTransects))
    t = np.zeros(len(fileList))
    for i,fname in enumerate(fileList):
        ncid = Dataset(fname,'r')
        if 'timeMonthly_avg_normalTransportVelocity' in ncid.variables.keys():
            vel = ncid.variables['timeMonthly_avg_normalTransportVelocity'][0, edgesToRead, :]
        elif 'timeMonthly_avg_normalVelocity' in ncid.variables.keys():
            vel = ncid.variables['timeMonthly_avg_normalVelocity'][0, edgesToRead, :]
            if 'timeMonthly_avg_normalGMBolusVelocity' in ncid.variables.keys():
                vel += ncid.variables['timeMonthly_avg_normalGMBolusVelocity'][0, edgesToRead, :]
        else:
            raise KeyError('no appropriate normalVelocity variable found')
        tempOnEdge = 0.5 * (ncid.variables['timeMonthly_avg_activeTracers_temperature'][0, cellsOnEdge[:, 0]-1, :] +
                            ncid.variables['timeMonthly_avg_activeTracers_temperature'][0, cellsOnEdge[:, 1]-1, :])
        saltOnEdge = 0.5 * (ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge[:, 0]-1, :] +
                            ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge[:, 1]-1, :])
        t[i] = ncid.variables['timeMonthly_avg_daysSinceStartOfSim'][:]/365.
        ncid.close()
        tempOnEdge[np.logical_or(tempOnEdge> 1e15, tempOnEdge<-1e15)] = np.nan
        saltOnEdge[np.logical_or(saltOnEdge> 1e15, saltOnEdge<-1e15)] = np.nan
        # Compute transport for each transect
        for j in range(nTransects):
            start = int(nTransectStartStop[j])
            stop = int(nTransectStartStop[j+1])
            dArea = dvEdge[start:stop, np.newaxis]*h[np.newaxis, :]
            normalVel = vel[start:stop, :]*edgeSigns[j, start:stop, np.newaxis]
            vol_transport[i, j]  = np.nansum(np.nansum(normalVel * dArea))
            heat_transport[i, j] = np.nansum(np.nansum(tempOnEdge[start:stop, :] * normalVel * dArea))
            salt_transport[i, j] = np.nansum(np.nansum(saltOnEdge[start:stop, :] * normalVel * dArea))
            salt_transport[i, j] = vol_transport[i, j] - salt_transport[i, j]/saltRef
    vol_transport = m3ps_to_Sv*vol_transport
    heat_transport = W_to_TW*rhoRef*cp*heat_transport
    salt_transport = m3ps_to_km3py*salt_transport

    # Define some dictionaries for transect plotting
    obsDict = {'Drake Passage':[120, 175], 'Tasmania-Ant':[147, 167], 'Africa-Ant':None, 'Antilles Inflow':[-23.1, -13.7], \
               'Mona Passage':[-3.8, -1.4],'Windward Passage':[-7.2, -6.8], 'Florida-Cuba':[30, 33], 'Florida-Bahamas':[30, 33], \
               'Indonesian Throughflow':[-21, -11], 'Agulhas':[-90, -50], 'Mozambique Channel':[-20, -8], \
               'Bering Strait':[0.6, 1.0], 'Lancaster Sound':[-1.0, -0.5], 'Fram Strait':[-4.7, 0.7], \
               'Robeson Channel':None, 'Davis Strait':[-1.6, -3.6], 'Barents Sea Opening':[1.4, 2.6], \
               'Nares Strait':[-1.8, 0.2]}
    labelDict = {'Drake Passage':'drake', 'Tasmania-Ant':'tasmania', 'Africa-Ant':'africaAnt', 'Antilles Inflow':'antilles', \
                 'Mona Passage':'monaPassage', 'Windward Passage':'windwardPassage', 'Florida-Cuba':'floridaCuba', \
                 'Florida-Bahamas':'floridaBahamas', 'Indonesian Throughflow':'indonesia', 'Agulhas':'agulhas', \
                 'Mozambique Channel':'mozambique', 'Bering Strait':'beringStrait', 'Lancaster Sound':'lancasterSound', \
                 'Fram Strait':'framStrait', 'Robeson Channel':'robeson', 'Davis Strait':'davisStrait', 'Barents Sea Opening':'BarentsSea', \
                 'Nares Strait':'naresStrait'}
    figsize = (20, 10)
    figdpi = 80

    for i in range(nTransects):
        if platform.python_version()[0]=='3':
            searchString = transectList[i][2:]
        else:
            searchString = transectList[i]

        # Plot Volume Transport
        figfile = '{}/volTransport_{}_{}.png'.format(figdir, labelDict[searchString], casename)
        plt.figure(figsize=figsize, dpi=figdpi)
        bounds = obsDict[searchString]
        plt.plot(t, vol_transport[:,i], 'k', linewidth=2)
        if bounds is not None:
            plt.gca().fill_between(t, bounds[0]*np.ones_like(t), bounds[1]*np.ones_like(t), alpha=0.3, label='observations')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylabel('Volume transport (Sv)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (Years)', fontsize=12, fontweight='bold')
        plt.title('Volume transport for {} ({}, mean={:5.2f} $\pm$ {:5.2f})'.format(searchString, casename, \
                  np.nanmean(vol_transport[:,i])), np.nanstd(vol_transport[:,i])), fontsize=16, fontweight='bold')
        plt.savefig(figfile, bbox_inches='tight')

        # Plot Heat Transport
        figfile = '{}/heatTransport_{}_{}.png'.format(figdir, labelDict[searchString], casename)
        plt.figure(figsize=figsize, dpi=figdpi)
        plt.plot(t, heat_transport[:,i], 'k', linewidth=2)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylabel('Heat transport (TW)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (Years)', fontsize=12, fontweight='bold')
        plt.title('Heat transport for {} ({}, mean={:5.2f}) $\pm$ {:5.2f}'.format(searchString, casename, \
                  np.nanmean(heat_transport[:,i])), np.nanstd(heat_transport[:,i])), fontsize=16, fontweight='bold')
        plt.savefig(figfile, bbox_inches='tight')

        # Plot FW Transport
        figfile = '{}/fwTransport_{}_{}.png'.format(figdir, labelDict[searchString], casename)
        plt.figure(figsize=figsize, dpi=figdpi)
        plt.plot(t, salt_transport[:,i], 'k', linewidth=2)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylabel('FW transport (km$^3$/year)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (Years)', fontsize=12, fontweight='bold')
        plt.title('FW transport for {} ({}, mean={:5.2f} $\pm$ {:5.2f})'.format(searchString, casename, \
                  np.nanmean(salt_transport[:,i])), np.nanstd(salt_transport[:,i])), fontsize=16, fontweight='bold')
        plt.savefig(figfile, bbox_inches='tight')

    # Add calls to save transport and then can build up
    ncid = Dataset(outfile, mode='w', clobber=True, format='NETCDF3_CLASSIC')

    # Add calls to save transport and then can build up
    ncid = Dataset(outfile, mode='w', clobber=True, format='NETCDF3_CLASSIC')
    ncid.createDimension('Time', None)
    ncid.createDimension('nTransects', nTransects)
    ncid.createDimension('StrLen', 64)
    transectNames = ncid.createVariable('TransectNames', 'c', ('nTransects', 'StrLen'))
    times = ncid.createVariable('Time', 'f8', 'Time')
    vol_transportOut = ncid.createVariable('volTransport', 'f8', ('Time', 'nTransects'))
    heat_transportOut = ncid.createVariable('heatTransport', 'f8', ('Time', 'nTransects'))
    salt_transportOut = ncid.createVariable('FWTransport', 'f8', ('Time', 'nTransects'))

    times[:] = t
    vol_transportOut[:, :] = vol_transport
    heat_transportOut[:, :] = heat_transport
    salt_transportOut[:, :] = salt_transport

    for i in range(nTransects):
        nLetters = len(transectList[i])
        transectNames[i, :nLetters] = transectList[i]

    ncid.close()
#######################################################

casename = 'E3SM-Arctic-OSI_60to10' # no spaces
meshfile = '/global/project/projectdirs/m1199/diagnostics/mpas_analysis/meshes/ocean.ARRM60to10.180715.nc'
# years for transport time series (empty if plotting full time series)
#years = '000[12]*'
years = ''
historyFileList = '/global/cscratch1/sd/milena/E3SM_simulations/ARRM60to10_JRA_GM_ramp/run/mpaso.hist.am.timeSeriesStatsMonthly.{}*nc'.format(years)
print(historyFileList)
maskfile = '/global/project/projectdirs/m1199/diagnostics/mpas_analysis/region_masks/ARRM60to10_transportTransects_masks.nc'
outfile = 'transectsTransport_{}.nc'.format(casename) # not very useful at the moment
transectName = 'StandardTransportSectionsRegionsGroup'
#transectName = 'all'
figdir = './transects'
if not os.path.isdir(figdir):
    os.mkdir(figdir)

compute_transport(historyFileList, casename, meshfile,
                  maskfile, figdir, transectName, outfile)
