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
import platform
import gsw

from common_functions import add_inset
from geometric_features import FeatureCollection, read_feature_collection
from mpas_analysis.shared.io.utility import decode_strings


def get_mask_short_names(mask):
    # This is the right way to handle transects defined in the main transect file mask:
    shortnames = [str(aname.values)[:str(aname.values).find(',')].strip()
                  for aname in mask.transectNames]
    #shortnames = mask.transectNames.values
    mask['shortNames'] = xr.DataArray(shortnames, dims='nTransects')
    mask = mask.set_index(nTransects=['transectNames', 'shortNames'])
    return mask


meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arcticSections20220916.nc'
featurefile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/arcticSections20210323.geojson'
casenameFull = 'E3SM-Arcticv2.1_historical0251'
casename = 'E3SM-Arcticv2.1_historical0251'
modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{casenameFull}/archive/ocn/hist'

# Choose years
year1 = 1950
year2 = 1951
#year2 = 2014
years = range(year1, year2+1)
nTime = 12*len(years)

use_fixeddz = False

figdir = f'./transports/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)
outdir = f'./transports_data/{casename}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
outfile = f'{outdir}/arcticSectionsTransports_{casename}_years{year1:04d}-{year2:04d}.nc'

if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found')

saltRef = 34.8
rhoRef = 1027.0 # kg/m^3
cp = 3.987*1e3 # J/(kg*degK) 
m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
m3ps_to_km3py = 1e-9*86400*365.25  # m^3/s FW flux to km^3/year
W_to_TW = 1e-12
###################################################################

transectName = 'all'
# Read in transect information
dsMask = get_mask_short_names(xr.open_dataset(maskfile))

if transectName=='all' or transectName=='StandardTransportSectionsRegionsGroup':
    transectList = dsMask.shortNames[:].values
else:
    transectList = transectName.split(',')
    if platform.python_version()[0]=='3':
        for i in range(len(transectList)):
            transectList[i] = "b'" + transectList[i]
nTransects = len(transectList)
maxEdges = dsMask.dims['maxEdgesInTransect']
print(f'Computing/plotting time series for these transects: {transectList}\n')

# Create a list of edges and total edges in each transect
nEdgesInTransect = np.zeros(nTransects)
edgeVals = np.zeros((nTransects, maxEdges))
for i in range(nTransects):
    amask = dsMask.sel(shortNames=transectList[i]).squeeze()
    transectEdges = amask.transectEdgeGlobalIDs.values
    inds = np.where(transectEdges > 0)[0]
    nEdgesInTransect[i] = len(inds)
    transectEdges = transectEdges[inds]
    edgeVals[i, :len(inds)] = np.asarray(transectEdges-1, dtype='i')
nEdgesInTransect = np.asarray(nEdgesInTransect, dtype='i')
edgesToRead = edgeVals[0, :nEdgesInTransect[0]]
for i in range(1, nTransects):
    edgesToRead = np.hstack([edgesToRead, edgeVals[i, :nEdgesInTransect[i]]])
edgesToRead = np.asarray(edgesToRead, dtype='i')

# Create a list with the start and stop for transect bounds
nTransectStartStop = np.zeros(nTransects+1)
for i in range(1, nTransects+1):
    nTransectStartStop[i] = nTransectStartStop[i-1] + nEdgesInTransect[i-1]

# Read in relevant mesh information
dsMesh = xr.open_dataset(meshfile)
dvEdge = dsMesh.dvEdge.sel(nEdges=edgesToRead).values
coe0 = dsMesh.cellsOnEdge.isel(TWO=0, nEdges=edgesToRead)
coe1 = dsMesh.cellsOnEdge.isel(TWO=1, nEdges=edgesToRead)
#dcoe = coe0-coe1
landmask1 = coe0.values==0
landmask2 = coe1.values==0
# convert to python indexing
coe0 = coe0 - 1
coe1 = coe1 - 1
kmaxOnCells1 = dsMesh.maxLevelCell.isel(nCells=coe0).values
kmaxOnCells2 = dsMesh.maxLevelCell.isel(nCells=coe1).values
edgeSigns = np.zeros((nTransects, len(edgesToRead)))
# Note to self: when I have some time, I need to make the script
# use transectNames everywhere.
transectNames = decode_strings(dsMask.transectNames)
for i in range(nTransects):
    edgeSigns[i, :] = dsMask.sel(nEdges=edgesToRead, shortNames=transectList[i]).squeeze().transectEdgeMaskSigns.values
    # WARNING: The following is a quick hack valid only for the arcticSections mask file!
    # I will need to change the geojson files to make *all* transects go from south to north
    # or west to east, so that I can have the correct edgeSigns for all of them.
    if transectNames[i]!='Bering Strait' and transectNames[i]!='Hudson Bay-Labrador Sea':
        edgeSigns[i, :] = -edgeSigns[i, :]
refBottom = dsMesh.refBottomDepth.values
latmean = 180.0/np.pi*np.nanmean(dsMesh.latEdge.sel(nEdges=edgesToRead).values)
lonmean = 180.0/np.pi*np.nanmean(dsMesh.lonEdge.sel(nEdges=edgesToRead).values)
pressure = gsw.p_from_z(-refBottom, latmean)
nLevels = dsMesh.dims['nVertLevels']
if use_fixeddz:
    dz = np.zeros(nLevels)
    dz[0] = refBottom[0]
    for k in range(1, nLevels):
        dz[k] = refBottom[k] - refBottom[k-1]

# Compute transports if outfile does not exist
if not os.path.exists(outfile):
    t = np.zeros(nTime)
    vol_transport = np.zeros((nTime, nTransects))
    vol_transportIn = np.zeros((nTime, nTransects))
    vol_transportOut = np.zeros((nTime, nTransects))
    heat_transport = np.zeros((nTime, nTransects))  # Tref = 0degC
    heat_transportIn = np.zeros((nTime, nTransects))
    heat_transportOut = np.zeros((nTime, nTransects))
    heat_transportTfp = np.zeros((nTime, nTransects))  # Tref = T freezing point computed below (Tfp)
    heat_transportTfpIn = np.zeros((nTime, nTransects))
    heat_transportTfpOut = np.zeros((nTime, nTransects))
    salt_transport = np.zeros((nTime, nTransects)) # Sref = saltRef
    salt_transportIn = np.zeros((nTime, nTransects))
    salt_transportOut = np.zeros((nTime, nTransects))

    ktime = 0
    for year in years:
        print(f'Year = {year:04d} out of {len(years)} years total')
        for month in range(1, 13):
            print(f'  Month= {month:02d}')
            modelfile = f'{modeldir}/{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

            ds = xr.open_dataset(modelfile, decode_times=False)

            #t[ktime] = ds.Time.isel(Time=0).values
            t[ktime] = ds.timeMonthly_avg_daysSinceStartOfSim.isel(Time=0).values/365.

            if 'timeMonthly_avg_normalTransportVelocity' in ds.keys():
                vel = ds.timeMonthly_avg_normalTransportVelocity.isel(Time=0, nEdges=edgesToRead).values
            elif 'timeMonthly_avg_normalVelocity' in ds.keys():
                vel = ds.timeMonthly_avg_normalVelocity.isel(Time=0, nEdges=edgesToRead).values
                if 'timeMonthly_avg_normalGMBolusVelocity' in ds.keys():
                    vel = vel + ds.timeMonthly_avg_normalGMBolusVelocity.isel(Time=0, nEdges=edgesToRead).values
                if 'timeMonthly_avg_normalMLEvelocity' in ds.keys():
                    vel = vel + ds.timeMonthly_avg_normalMLEvelocity.isel(Time=0, nEdges=edgesToRead).values
            else:
                raise KeyError('no appropriate normalVelocity variable found')
            # Note that the following is incorrect when coe is zero (cell straddling the
            # transect edge is on land), but that is OK because the value will be masked
            # during land-sea masking below
            tempOnCells1 = ds.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe0).values
            tempOnCells2 = ds.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe1).values
            saltOnCells1 = ds.timeMonthly_avg_activeTracers_salinity.isel(Time=0, nCells=coe0).values
            saltOnCells2 = ds.timeMonthly_avg_activeTracers_salinity.isel(Time=0, nCells=coe1).values

            # Mask values that fall on land
            tempOnCells1[landmask1, :] = np.nan
            tempOnCells2[landmask2, :] = np.nan
            saltOnCells1[landmask1, :] = np.nan
            saltOnCells2[landmask2, :] = np.nan
            # Mask values that fall onto topography
            for k in range(len(kmaxOnCells1)):
                tempOnCells1[k, kmaxOnCells1[k]:] = np.nan
                saltOnCells1[k, kmaxOnCells1[k]:] = np.nan
            for k in range(len(kmaxOnCells2)):
                tempOnCells2[k, kmaxOnCells2[k]:] = np.nan
                saltOnCells2[k, kmaxOnCells2[k]:] = np.nan
            # The following should *not* happen at this point:
            if np.any(tempOnCells1[np.logical_or(tempOnCells1> 1e15, tempOnCells1<-1e15)]) or \
               np.any(tempOnCells2[np.logical_or(tempOnCells2> 1e15, tempOnCells2<-1e15)]):
                print('WARNING: something is wrong with land and/or topography masking!')
            if np.any(saltOnCells1[np.logical_or(saltOnCells1> 1e15, saltOnCells1<-1e15)]) or \
               np.any(saltOnCells2[np.logical_or(saltOnCells2> 1e15, saltOnCells2<-1e15)]):
                print('WARNING: something is wrong with land and/or topography masking!')

            if not use_fixeddz:
                dzOnCells1 = ds.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe0).values
                dzOnCells2 = ds.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe1).values
                dzOnCells1[landmask1, :] = np.nan
                dzOnCells2[landmask2, :] = np.nan
                for k in range(len(kmaxOnCells1)):
                    dzOnCells1[k, kmaxOnCells1[k]:] = np.nan
                for k in range(len(kmaxOnCells2)):
                    dzOnCells2[k, kmaxOnCells2[k]:] = np.nan

            # Interpolate values onto edges
            tempOnEdges = np.nanmean(np.array([tempOnCells1, tempOnCells2]), axis=0)
            saltOnEdges = np.nanmean(np.array([saltOnCells1, saltOnCells2]), axis=0)
            if not use_fixeddz:
                dzOnEdges = np.nanmean(np.array([dzOnCells1, dzOnCells2]), axis=0)

            # Compute freezing temperature
            SA = gsw.SA_from_SP(saltOnEdges, pressure, lonmean, latmean)
            CTfp = gsw.CT_freezing(SA, pressure, 0.)
            Tfp = gsw.pt_from_CT(SA, CTfp)

            # Compute transports for each transect
            for i in range(nTransects):
                start = int(nTransectStartStop[i])
                stop = int(nTransectStartStop[i+1])

                normalVel = vel[start:stop, :] * edgeSigns[i, start:stop, np.newaxis]
                temp = tempOnEdges[start:stop, :]
                salt = saltOnEdges[start:stop, :]
                maskOnEdges = np.isnan(salt)
                normalVel[maskOnEdges] = np.nan

                dx = dvEdge[start:stop]
                dx2d = np.transpose(np.tile(dx, (nLevels, 1)))
                dx2d[maskOnEdges] = np.nan
                if use_fixeddz:
                    dArea = dx2d * dz[np.newaxis, :]
                else:
                    dz = dzOnEdges[start:stop, :]
                    dArea = dx2d * dz

                tfreezing = Tfp[start:stop, :]
                indVelP = np.where(normalVel>0)
                indVelM = np.where(normalVel<0)
                #
                vol_transport[ktime, i]    = np.nansum(np.nansum(normalVel * dArea))
                vol_transportIn[ktime, i]  = np.nansum(np.nansum(normalVel[indVelP] * dArea[indVelP]))
                vol_transportOut[ktime, i] = np.nansum(np.nansum(normalVel[indVelM] * dArea[indVelM]))
                #
                heat_transport[ktime, i]    = np.nansum(np.nansum(temp * normalVel * dArea))
                heat_transportIn[ktime, i]  = np.nansum(np.nansum(temp[indVelP] * normalVel[indVelP] * dArea[indVelP]))
                heat_transportOut[ktime, i] = np.nansum(np.nansum(temp[indVelM] * normalVel[indVelM] * dArea[indVelM]))
                heat_transportTfp[ktime, i]    = np.nansum(np.nansum((temp - tfreezing) * normalVel * dArea))
                heat_transportTfpIn[ktime, i]  = np.nansum(np.nansum((temp[indVelP] - tfreezing[indVelP]) * normalVel[indVelP] * dArea[indVelP]))
                heat_transportTfpOut[ktime, i] = np.nansum(np.nansum((temp[indVelM] - tfreezing[indVelM]) * normalVel[indVelM] * dArea[indVelM]))
                #
                salt_transport[ktime, i]    = np.nansum(np.nansum(salt * normalVel * dArea))
                salt_transport[ktime, i]    = vol_transport[ktime, i] - salt_transport[ktime, i]/saltRef
                salt_transportIn[ktime, i]  = np.nansum(np.nansum(salt[indVelP] * normalVel[indVelP] * dArea[indVelP]))
                salt_transportIn[ktime, i]  = vol_transportIn[ktime, i] - salt_transportIn[ktime, i]/saltRef
                salt_transportOut[ktime, i] = np.nansum(np.nansum(salt[indVelM] * normalVel[indVelM] * dArea[indVelM]))
                salt_transportOut[ktime, i] = vol_transportOut[ktime, i] - salt_transportOut[ktime, i]/saltRef

            ktime = ktime + 1

    vol_transport    = m3ps_to_Sv * vol_transport
    vol_transportIn  = m3ps_to_Sv * vol_transportIn
    vol_transportOut = m3ps_to_Sv * vol_transportOut
    heat_transport    = W_to_TW * rhoRef * cp * heat_transport
    heat_transportIn  = W_to_TW * rhoRef * cp * heat_transportIn
    heat_transportOut = W_to_TW * rhoRef * cp * heat_transportOut
    heat_transportTfp    = W_to_TW * rhoRef * cp * heat_transportTfp
    heat_transportTfpIn  = W_to_TW * rhoRef * cp * heat_transportTfpIn
    heat_transportTfpOut = W_to_TW * rhoRef * cp * heat_transportTfpOut
    salt_transport    = m3ps_to_km3py * salt_transport
    salt_transportIn  = m3ps_to_km3py * salt_transportIn
    salt_transportOut = m3ps_to_km3py * salt_transportOut

    # Save to file
    ncid = Dataset(outfile, mode='w', clobber=True, format='NETCDF3_CLASSIC')
    ncid.createDimension('Time', None)
    ncid.createDimension('nTransects', nTransects)
    ncid.createDimension('StrLen', 64)

    times = ncid.createVariable('Time', 'f8', 'Time')
    transectNames = ncid.createVariable('TransectNames', 'c', ('nTransects', 'StrLen'))
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
    salt_transportVar.description = f'Net FW transport (wrt {saltRef:4.1f} psu) across transect'
    salt_transportInVar.description = f'Inflow FW transport (wrt {saltRef:4.1f} psu) across transect (in/out determined by edgeSign)'
    salt_transportOutVar.description = f'Outflow FW transport (wrt {saltRef:4.1f} psu) across transect (in/out determined by edgeSign)'

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
else:
    print(f'\nFile {outfile} already exists. Plotting only...\n')

# Read in previously computed transport quantities
ncid = Dataset(outfile, mode='r')
t = ncid.variables['Time'][:]
vol_transport = ncid.variables['volTransport'][:, :]
vol_transportIn = ncid.variables['volTransportIn'][:, :]
vol_transportOut = ncid.variables['volTransportOut'][:, :]
heat_transport = ncid.variables['heatTransport'][:, :]
heat_transportIn = ncid.variables['heatTransportIn'][:, :]
heat_transportOut = ncid.variables['heatTransportOut'][:, :]
heat_transportTfp = ncid.variables['heatTransportTfp'][:, :]
heat_transportTfpIn = ncid.variables['heatTransportTfpIn'][:, :]
heat_transportTfpOut = ncid.variables['heatTransportTfpOut'][:, :]
salt_transport = ncid.variables['FWTransport'][:, :]
salt_transportIn = ncid.variables['FWTransportIn'][:, :]
salt_transportOut = ncid.variables['FWTransportOut'][:, :]
ncid.close()

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

figsize = (8, 12)
figdpi = 300
for i in range(nTransects):
    if platform.python_version()[0]=='3':
        searchString = transectList[i][2:]
    else:
        searchString = transectList[i]

    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == searchString:
            fc.add_feature(feature)
            break

    if searchString in labelDict:
        transectName_forfigfile = labelDict[searchString]
    else:
        transectName_forfigfile = searchString.replace(" ", "")

    if searchString in obsDict:
        bounds = obsDict[searchString]
    else:
        bounds = None

    #vol_runavg = pd.Series.rolling(pd.DataFrame(vol_transport[:, i]), 12, center=True).mean()
    #heat_runavg = pd.Series.rolling(pd.DataFrame(heat_transport[:, i]), 12, center=True).mean()
    #heatTfp_runavg = pd.Series.rolling(pd.DataFrame(heat_transportTfp[:, i]), 12, center=True).mean()
    #salt_runavg = pd.Series.rolling(pd.DataFrame(salt_transport[:, i]), 12, center=True).mean()

    # Plot Volume Transport
    figfile = f'{figdir}/transports_{transectName_forfigfile}_{casename}.png'
    fig, ax = plt.subplots(3, 1, figsize=figsize)
    ax[0].plot(t, vol_transport[:,i], 'k', linewidth=2, label=f'net ({np.nanmean(vol_transport[:,i]):5.2f} $\pm$ {np.nanstd(vol_transport[:,i]):5.2f})')
    ax[0].plot(t, vol_transportIn[:,i], 'r', linewidth=2, label=f'inflow ({np.nanmean(vol_transportIn[:,i]):5.2f} $\pm$ {np.nanstd(vol_transportIn[:,i]):5.2f})')
    ax[0].plot(t, vol_transportOut[:,i], 'b', linewidth=2, label=f'outflow ({np.nanmean(vol_transportOut[:,i]):5.2f} $\pm$ {np.nanstd(vol_transportOut[:,i]):5.2f})')
    if bounds is not None:
        ax[0].fill_between(t, np.full_like(t, bounds[0]), np.full_like(t, bounds[1]), alpha=0.3, label='obs (net)')
    ax[0].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[0].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[0].autoscale(enable=True, axis='x', tight=True)
    ax[0].set_ylabel('Volume transport (Sv)', fontsize=12, fontweight='bold')
    ax[0].legend()

    # Plot Heat Transport wrt Tref=0
    #ax[1].plot(t, heat_transport[:,i], 'k', linewidth=2, label='model (net)')
    #ax[1].plot(t, heat_transportIn[:,i], 'r', linewidth=2, label='model (inflow)')
    #ax[1].plot(t, heat_transportOut[:,i], 'b', linewidth=2, label='model (outflow)')
    #ax[1].plot(t, np.zeros_like(t), 'k', linewidth=1)
    #ax[1].grid(color='k', linestyle=':', linewidth = 0.5)
    #ax[1].autoscale(enable=True, axis='x', tight=True)
    #ax[1].set_ylabel('Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')
    #ax[1].legend()

    # Plot Heat Transport wrt Tref=TfreezingPoint
    ax[1].plot(t, heat_transportTfp[:,i], 'k', linewidth=2, label=f'net ({np.nanmean(heat_transportTfp[:,i]):5.2f} $\pm$ {np.nanstd(heat_transportTfp[:,i]):5.2f})')
    ax[1].plot(t, heat_transportTfpIn[:,i], 'r', linewidth=2, label=f'inflow ({np.nanmean(heat_transportTfpIn[:,i]):5.2f} $\pm$ {np.nanstd(heat_transportTfpIn[:,i]):5.2f})')
    ax[1].plot(t, heat_transportTfpOut[:,i], 'b', linewidth=2, label=f'outflow ({np.nanmean(heat_transportTfpOut[:,i]):5.2f} $\pm$ {np.nanstd(heat_transportTfpOut[:,i]):5.2f})')
    ax[1].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[1].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[1].autoscale(enable=True, axis='x', tight=True)
    ax[1].set_ylabel('Heat transport wrt freezing point (TW)', fontsize=12, fontweight='bold')
    ax[1].legend()

    # Plot FW Transport
    ax[2].plot(t, salt_transport[:,i], 'k', linewidth=2, label=f'net ({np.nanmean(salt_transport[:,i]):5.2f} $\pm$ {np.nanstd(salt_transport[:,i]):5.2f})')
    ax[2].plot(t, salt_transportIn[:,i], 'r', linewidth=2, label=f'inflow ({np.nanmean(salt_transportIn[:,i]):5.2f} $\pm$ {np.nanstd(salt_transportIn[:,i]):5.2f})')
    ax[2].plot(t, salt_transportOut[:,i], 'b', linewidth=2, label=f'outflow ({np.nanmean(salt_transportOut[:,i]):5.2f} $\pm$ {np.nanstd(salt_transportOut[:,i]):5.2f})')
    ax[2].plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax[2].grid(color='k', linestyle=':', linewidth = 0.5)
    ax[2].autoscale(enable=True, axis='x', tight=True)
    ax[2].set_ylabel('FW transport (km$^3$/year)', fontsize=12, fontweight='bold')
    ax[2].set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax[2].legend()

    fig.tight_layout(pad=0.5)
    fig.suptitle(f'Transect = {searchString}\nrunname = {casename}', fontsize=14, fontweight='bold', y=1.025)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=-0.5, ybuffer=-1)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
