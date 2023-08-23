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
import gsw

saltRef = 34.8
tempRef = -2.0
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

    latmean = 180.0/np.pi*np.nanmean(mesh.latEdge.sel(nEdges=edgesToRead).values)
    lonmean = 180.0/np.pi*np.nanmean(mesh.lonEdge.sel(nEdges=edgesToRead).values)
    pressure = gsw.p_from_z(-refBottom, latmean)

    # Read history files one at a time and slice
    fileList = sorted(glob.glob(historyFileList))
    vol_transport = np.zeros((len(fileList), nTransects))
    vol_transportIn = np.zeros((len(fileList), nTransects))
    vol_transportOut = np.zeros((len(fileList), nTransects))
    heat_transport = np.zeros((len(fileList), nTransects))  # Tref = 0degC
    heat_transportIn = np.zeros((len(fileList), nTransects))
    heat_transportOut = np.zeros((len(fileList), nTransects))
    heat_transportTfp = np.zeros((len(fileList), nTransects))  # Tref = T freezing point computed below (Tfp)
    heat_transportTfpIn = np.zeros((len(fileList), nTransects))
    heat_transportTfpOut = np.zeros((len(fileList), nTransects))
    salt_transport = np.zeros((len(fileList), nTransects)) # Sref = saltRef
    salt_transportIn = np.zeros((len(fileList), nTransects))
    salt_transportOut = np.zeros((len(fileList), nTransects))
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
        tempOnCell1 = ncid.variables['timeMonthly_avg_activeTracers_temperature'][0, cellsOnEdge[:, 0]-1, :]
        tempOnCell2 = ncid.variables['timeMonthly_avg_activeTracers_temperature'][0, cellsOnEdge[:, 1]-1, :]
        saltOnCell1 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge[:, 0]-1, :]
        saltOnCell2 = ncid.variables['timeMonthly_avg_activeTracers_salinity'][0, cellsOnEdge[:, 1]-1, :]
        t[i] = ncid.variables['timeMonthly_avg_daysSinceStartOfSim'][:]/365.
        ncid.close()
        # Mask T,S values that fall on land
        tempOnCell1[cellsOnEdge[:, 0]==0, :] = np.nan
        tempOnCell2[cellsOnEdge[:, 1]==0, :] = np.nan
        saltOnCell1[cellsOnEdge[:, 0]==0, :] = np.nan
        saltOnCell2[cellsOnEdge[:, 1]==0, :] = np.nan
        # Interpolate T,S values onto edges
        tempOnEdge = np.nanmean(np.array([tempOnCell1, tempOnCell2]), axis=0)
        saltOnEdge = np.nanmean(np.array([saltOnCell1, saltOnCell2]), axis=0)
        # Mask values that fall onto topography
        tempOnEdge[np.logical_or(tempOnEdge> 1e15, tempOnEdge<-1e15)] = np.nan
        saltOnEdge[np.logical_or(saltOnEdge> 1e15, saltOnEdge<-1e15)] = np.nan

        # Compute freezing temperature
        SA = gsw.SA_from_SP(saltOnEdge, pressure, lonmean, latmean)
        CTfp = gsw.CT_freezing(SA, pressure, 0.)
        Tfp = gsw.pt_from_CT(SA, CTfp)

        # Compute transports for each transect
        for j in range(nTransects):
            start = int(nTransectStartStop[j])
            stop = int(nTransectStartStop[j+1])
            dArea = dvEdge[start:stop, np.newaxis]*h[np.newaxis, :]
            normalVel = vel[start:stop, :]*edgeSigns[j, start:stop, np.newaxis]
            temp = tempOnEdge[start:stop, :]
            salt = saltOnEdge[start:stop, :]
            tfreezing = Tfp[start:stop, :]
            indVelP = np.where(normalVel>0)
            indVelM = np.where(normalVel<0)
            vol_transport[i, j]  = np.nansum(np.nansum(normalVel * dArea))
            vol_transportIn[i, j]  = np.nansum(np.nansum(normalVel[indVelP] * dArea[indVelP]))
            vol_transportOut[i, j]  = np.nansum(np.nansum(normalVel[indVelM] * dArea[indVelM]))
            heat_transport[i, j] = np.nansum(np.nansum(temp * normalVel * dArea))
            heat_transportIn[i, j] = np.nansum(np.nansum(temp[indVelP] * normalVel[indVelP] * dArea[indVelP]))
            heat_transportOut[i, j] = np.nansum(np.nansum(temp[indVelM] * normalVel[indVelM] * dArea[indVelM]))
            heat_transportTfp[i, j] = np.nansum(np.nansum((temp - tfreezing) * normalVel * dArea))
            heat_transportTfpIn[i, j] = np.nansum(np.nansum((temp[indVelP] - tfreezing[indVelP]) * normalVel[indVelP] * dArea[indVelP]))
            heat_transportTfpOut[i, j] = np.nansum(np.nansum((temp[indVelM] - tfreezing[indVelM]) * normalVel[indVelM] * dArea[indVelM]))
            salt_transport[i, j] = np.nansum(np.nansum(salt * normalVel * dArea))
            salt_transport[i, j] = vol_transport[i, j] - salt_transport[i, j]/saltRef
            salt_transportIn[i, j] = np.nansum(np.nansum(salt[indVelP] * normalVel[indVelP] * dArea[indVelP]))
            salt_transportIn[i, j] = vol_transportIn[i, j] - salt_transportIn[i, j]/saltRef
            salt_transportOut[i, j] = np.nansum(np.nansum(salt[indVelM] * normalVel[indVelM] * dArea[indVelM]))
            salt_transportOut[i, j] = vol_transportOut[i, j] - salt_transportOut[i, j]/saltRef
    vol_transport = m3ps_to_Sv*vol_transport
    vol_transportIn = m3ps_to_Sv*vol_transportIn
    vol_transportOut = m3ps_to_Sv*vol_transportOut
    heat_transport = W_to_TW*rhoRef*cp*heat_transport
    heat_transportIn = W_to_TW*rhoRef*cp*heat_transportIn
    heat_transportOut = W_to_TW*rhoRef*cp*heat_transportOut
    heat_transportTfp = W_to_TW*rhoRef*cp*heat_transportTfp
    heat_transportTfpIn = W_to_TW*rhoRef*cp*heat_transportTfpIn
    heat_transportTfpOut = W_to_TW*rhoRef*cp*heat_transportTfpOut
    salt_transport = m3ps_to_km3py*salt_transport
    salt_transportIn = m3ps_to_km3py*salt_transportIn
    salt_transportOut = m3ps_to_km3py*salt_transportOut

    # Define some dictionaries for transect plotting
    obsDict = {'Drake Passage':[120, 175], 'Tasmania-Ant':[147, 167], 'Africa-Ant':None, 'Antilles Inflow':[-23.1, -13.7], \
               'Mona Passage':[-3.8, -1.4],'Windward Passage':[-7.2, -6.8], 'Florida-Cuba':[30, 33], 'Florida-Bahamas':[30, 33], \
               'Indonesian Throughflow':[-21, -11], 'Agulhas':[-90, -50], 'Mozambique Channel':[-20, -8], \
               'Bering Strait':[0.6, 1.0], 'Lancaster Sound':[-1.0, -0.5], 'Fram Strait':[-4.7, 0.7], \
               'Robeson Channel':None, 'Davis Strait':[-1.6, -3.6], 'Barents Sea Opening':[1.4, 2.6], \
               'Nares Strait':[-1.8, 0.2], 'Denmark Strait':None, 'Iceland-Faroe-Scotland':None}
    labelDict = {'Drake Passage':'drake', 'Tasmania-Ant':'tasmania', 'Africa-Ant':'africaAnt', 'Antilles Inflow':'antilles', \
                 'Mona Passage':'monaPassage', 'Windward Passage':'windwardPassage', 'Florida-Cuba':'floridaCuba', \
                 'Florida-Bahamas':'floridaBahamas', 'Indonesian Throughflow':'indonesia', 'Agulhas':'agulhas', \
                 'Mozambique Channel':'mozambique', 'Bering Strait':'beringStrait', 'Lancaster Sound':'lancasterSound', \
                 'Fram Strait':'framStrait', 'Robeson Channel':'robeson', 'Davis Strait':'davisStrait', 'Barents Sea Opening':'barentsSea', \
                 'Nares Strait':'naresStrait', 'Denmark Strait':'denmarkStrait', 'Iceland-Faroe-Scotland':'icelandFaroeScotland'}
    figsize = (20, 10)
    figdpi = 80

    for i in range(nTransects):
        if platform.python_version()[0]=='3':
            searchString = transectList[i][2:]
        else:
            searchString = transectList[i]

        if searchString in labelDict:
            transectName_forfigfile = labelDict[searchString]
        else:
            transectName_forfigfile = searchString.replace(" ", "")

        if searchString in obsDict:
            bounds = obsDict[searchString]
        else:
            bounds = None

        # Plot Volume Transport
        figfile = '{}/volTransport_{}_{}.png'.format(figdir, transectName_forfigfile, casename)
        plt.figure(figsize=figsize, dpi=figdpi)
        plt.plot(t, vol_transport[:,i], 'k', linewidth=2, label='model (net)')
        plt.plot(t, vol_transportIn[:,i], 'r', linewidth=2, label='model (inflow)')
        plt.plot(t, vol_transportOut[:,i], 'b', linewidth=2, label='model (outflow)')
        if bounds is not None:
            plt.gca().fill_between(t, bounds[0]*np.ones_like(t), bounds[1]*np.ones_like(t), alpha=0.3, label='observations (net)')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylabel('Volume transport (Sv)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (Years)', fontsize=12, fontweight='bold')
        plt.title('Volume transport for {} ({}, mean (net)={:5.2f} $\pm$ {:5.2f})'.format(searchString, casename, \
                  np.nanmean(vol_transport[:,i]), np.nanstd(vol_transport[:,i]), fontsize=16, fontweight='bold'))
        plt.legend()
        plt.savefig(figfile, bbox_inches='tight')

        # Plot Heat Transport wrt Tref=0
        figfile = '{}/heatTransport_{}_{}.png'.format(figdir, transectName_forfigfile, casename)
        plt.figure(figsize=figsize, dpi=figdpi)
        plt.plot(t, heat_transport[:,i], 'k', linewidth=2, label='model (net)')
        plt.plot(t, heat_transportIn[:,i], 'r', linewidth=2, label='model (inflow)')
        plt.plot(t, heat_transportOut[:,i], 'b', linewidth=2, label='model (outflow)')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylabel('Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (Years)', fontsize=12, fontweight='bold')
        plt.title('Heat transport for {} ({}, mean (net)={:5.2f} $\pm$ {:5.2f})'.format(searchString, casename, \
                  np.nanmean(heat_transport[:,i]), np.nanstd(heat_transport[:,i]), fontsize=16, fontweight='bold'))
        plt.legend()
        plt.savefig(figfile, bbox_inches='tight')

        # Plot Heat Transport wrt Tref=tempRef
        figfile = '{}/heatTransportTfp_{}_{}.png'.format(figdir, transectName_forfigfile, casename)
        plt.figure(figsize=figsize, dpi=figdpi)
        plt.plot(t, heat_transportTfp[:,i], 'k', linewidth=2, label='model (net)')
        plt.plot(t, heat_transportTfpIn[:,i], 'r', linewidth=2, label='model (inflow)')
        plt.plot(t, heat_transportTfpOut[:,i], 'b', linewidth=2, label='model (outflow)')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylabel('Heat transport wrt freezing point (TW)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (Years)', fontsize=12, fontweight='bold')
        plt.title('Heat transport for {} ({}, mean (net)={:5.2f} $\pm$ {:5.2f})'.format(searchString, casename, \
                  np.nanmean(heat_transportTfp[:,i]), np.nanstd(heat_transportTfp[:,i]), fontsize=16, fontweight='bold'))
        plt.legend()
        plt.savefig(figfile, bbox_inches='tight')

        # Plot FW Transport
        figfile = '{}/fwTransport_{}_{}.png'.format(figdir, transectName_forfigfile, casename)
        plt.figure(figsize=figsize, dpi=figdpi)
        plt.plot(t, salt_transport[:,i], 'k', linewidth=2, label='model (net)')
        plt.plot(t, salt_transportIn[:,i], 'r', linewidth=2, label='model (inflow)')
        plt.plot(t, salt_transportOut[:,i], 'b', linewidth=2, label='model (outflow)')
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.ylabel('FW transport (km$^3$/year)', fontsize=12, fontweight='bold')
        plt.xlabel('Time (Years)', fontsize=12, fontweight='bold')
        plt.title('FW transport for {} ({}, mean (net)={:5.2f} $\pm$ {:5.2f})'.format(searchString, casename, \
                  np.nanmean(salt_transport[:,i]), np.nanstd(salt_transport[:,i]), fontsize=16, fontweight='bold'))
        plt.legend()
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
    vol_transportVar = ncid.createVariable('volTransport', 'f8', ('Time', 'nTransects'))
    vol_transportInVar = ncid.createVariable('volTransportIn', 'f8', ('Time', 'nTransects'))
    vol_transportOutVar = ncid.createVariable('volTransportOut', 'f8', ('Time', 'nTransects'))
    heat_transportVar = ncid.createVariable('heatTransport', 'f8', ('Time', 'nTransects'))
    heat_transportInVar = ncid.createVariable('heatTransportIn', 'f8', ('Time', 'nTransects'))
    heat_transportOutVar = ncid.createVariable('heatTransportOut', 'f8', ('Time', 'nTransects'))
    heat_transportTfpVar = ncid.createVariable('heatTransportTfp', 'f8', ('Time', 'nTransects'))
    heat_transportTfpInVar = ncid.createVariable('heatTransportTfpIn', 'f8', ('Time', 'nTransects'))
    heat_transportTfpOutVar = ncid.createVariable('heatTransportTfpOut', 'f8', ('Time', 'nTransects'))
    salt_transportVar = ncid.createVariable('FWTransport', 'f8', ('Time', 'nTransects'))
    salt_transportInVar = ncid.createVariable('FWTransportIn', 'f8', ('Time', 'nTransects'))
    salt_transportOutVar = ncid.createVariable('FWTransportOut', 'f8', ('Time', 'nTransects'))

    vol_transportVar.units = 'Sv'
    vol_transportInVar.units = 'Sv'
    vol_transportOutVar.units = 'Sv'
    heat_transportVar.units = 'TW'
    heat_transportInVar.units = 'TW'
    heat_transportOutVar.units = 'TW'
    heat_transportTfpVar.units = 'TW'
    heat_transportTfpInVar.units = 'TW'
    heat_transportTfpOutVar.units = 'TW'
    salt_transportVar.units = 'km^3/year'
    salt_transportInVar.units = 'km^3/year'
    salt_transportOutVar.units = 'km^3/year'

    vol_transportVar.description = 'Net volume transport across transect'
    vol_transportInVar.description = 'Inflow volume transport across transect (in/out determined by edgeSign)'
    vol_transportOutVar.description = 'Outflow volume transport across transect (in/out determined by edgeSign)'
    heat_transportVar.description = 'Net heat transport (wrt 0degC) across transect'
    heat_transportInVar.description = 'Inflow heat transport (wrt 0degC) across transect (in/out determined by edgeSign)'
    heat_transportOutVar.description = 'Outflow heat transport (wrt 0degC) across transect (in/out determined by edgeSign)'
    heat_transportTfpVar.description = 'Net heat transport (wrt freezing point) across transect'
    heat_transportTfpInVar.description = 'Inflow heat transport (wrt freezing point) across transect (in/out determined by edgeSign)'
    heat_transportTfpOutVar.description = 'Outflow heat transport (wrt freezing point) across transect (in/out determined by edgeSign)'
    salt_transportVar.description = 'Net FW transport (wrt {:4.1f} psu) across transect'.format(saltRef)
    salt_transportInVar.description = 'Inflow FW transport (wrt {:4.1f} psu) across transect (in/out determined by edgeSign)'.format(saltRef)
    salt_transportOutVar.description = 'Outflow FW transport (wrt {:4.1f} psu) across transect (in/out determined by edgeSign)'.format(saltRef)

    times[:] = t
    vol_transportVar[:, :] = vol_transport
    vol_transportInVar[:, :] = vol_transportIn
    vol_transportOutVar[:, :] = vol_transportOut
    heat_transportVar[:, :] = heat_transport
    heat_transportInVar[:, :] = heat_transportIn
    heat_transportOutVar[:, :] = heat_transportOut
    heat_transportTfpVar[:, :] = heat_transportTfp
    heat_transportTfpInVar[:, :] = heat_transportTfpIn
    heat_transportTfpOutVar[:, :] = heat_transportTfpOut
    salt_transportVar[:, :] = salt_transport
    salt_transportInVar[:, :] = salt_transportIn
    salt_transportOutVar[:, :] = salt_transportOut

    for i in range(nTransects):
        nLetters = len(transectList[i])
        transectNames[i, :nLetters] = transectList[i]

    ncid.close()
#######################################################

# years for transport time series (empty if plotting full time series)
#years = '010[1-2]'
years = ''

meshfile = '/global/cfs/projectdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.220730.nc'
maskfile = '/global/cfs/projectdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arcticSections20220916.nc'
casename = 'ne30pg2_ARRM10to60E2r1.baseline_bdvslat'
historyFileList = '/global/cscratch1/sd/milena/e3sm_scratch/cori-knl/20220810.WCYCL1950.ne30pg2_ARRM10to60E2r1.baseline_bdvslat.cori-knl/run/20220810.WCYCL1950.ne30pg2_ARRM10to60E2r1.baseline_bdvslat.cori-knl.mpaso.hist.am.timeSeriesStatsMonthly.{}*nc'.format(years)
#casename = 'arcticx4v1pg2_ARRM10to60E2r1.baseline_bdvslat'
#historyFileList = '/global/cscratch1/sd/milena/e3sm_scratch/cori-knl/20220810.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.baseline_bdvslat.cori-knl/run/20220810.WCYCL1950.arcticx4v1pg2_ARRM10to60E2r1.baseline_bdvslat.cori-knl.mpaso.hist.am.timeSeriesStatsMonthly.{}*nc'.format(years)

#meshfile = '/global/cfs/projectdirs/e3sm/inputdata/ocn/mpas-o/oARRM60to10/ocean.ARRM60to10.180715.nc'
##maskfile = '/global/cfs/projectdirs/m1199/diagnostics/mpas_analysis/region_masks/ARRM60to10_transportTransects_masks.nc'
#maskfile = '/global/cfs/projectdirs/m1199/milena/mpas-region_masks/ARRM60to10_standardTransportSections20210323.nc'
#casename = 'E3SM-Arctic-OSI_60to10' # no spaces
#casename = 'E3SM-Arctic-OSIv2'
#casename = 'E3SM-Arctic-v2beta1'
#historyFileList = '/global/cfs/projectdirs/m1199/e3sm-arrm-simulations/ARRM60to10_JRA_GM_ramp/run/mpaso.hist.am.timeSeriesStatsMonthly.{}*nc'.format(years)
#historyFileList = '/global/cfs/projectdirs/m1199/e3sm-arrm-simulations/20210416.GMPAS-JRA1p4.TL319_oARRM60to10.cori-knl/run/20210416.GMPAS-JRA1p4.TL319_oARRM60to10.cori-knl.mpaso.hist.am.timeSeriesStatsMonthly.{}*nc'.format(years)
#historyFileList = '/global/cfs/projectdirs/m1199/e3sm-arrm-simulations/20210204.A_WCYCL1850S_CMIP6.ne30pg2_oARRM60to10_ICG.beta1.cori-knl/run/20210204.A_WCYCL1850S_CMIP6.ne30pg2_oARRM60to10_ICG.beta1.cori-knl.mpaso.hist.am.timeSeriesStatsMonthly.{}*nc'.format(years)
#
#meshfile = '/global/cfs/projectdirs/e3sm/inputdata/ocn/mpas-o/EC30to60E2r2/ocean.EC30to60E2r2.210210.nc'
#maskfile = '/global/cfs/projectdirs/e3sm/milena/mpas-region_masks/EC30to60E2r2_arcticSections20220914.nc'
#
#meshfile = '/global/cfs/projectdirs/e3sm/inputdata/ocn/mpas-o/WC14to60E2r3/ocean.WC14to60E2r3.200714.nc'
#maskfile = '/global/cfs/projectdirs/e3sm/milena/mpas-region_masks/WC14to60E2r3_arcticSections20220914.nc'
##maskfile = '/global/cfs/projectdirs/e3sm/milena/mpas-region_masks/WC14_arcticTransects.nc'
#casename = 'northamericax4v1pg2_WC14.piControl'

print(historyFileList)

outfile = 'transectsTransport_{}.nc'.format(casename)
transectName = 'all'
#transectName = 'StandardTransportSectionsRegionsGroup'
figdir = './transects/{}'.format(casename)
if not os.path.isdir(figdir):
    os.makedirs(figdir)

compute_transport(historyFileList, casename, meshfile,
                  maskfile, figdir, transectName, outfile)
