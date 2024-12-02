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
import gsw
import cmocean

from common_functions import add_inset
from geometric_features import FeatureCollection, read_feature_collection
from mpas_analysis.shared.io.utility import decode_strings
from mpas_analysis.shared.io import write_netcdf_with_fill


def get_mask_short_names(mask):
    # This is the right way to handle transects defined in the main transect file mask:
    shortnames = [str(aname.values)[:str(aname.values).find(',')].strip()
                  for aname in mask.transectNames]
    #shortnames = mask.transectNames.values
    mask['shortNames'] = xr.DataArray(shortnames, dims='nTransects')
    mask = mask.set_index(nTransects=['transectNames', 'shortNames'])
    return mask


# Settings for nersc
meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20240910.nc'
#featurefile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/atlanticZonal_sections20240910.geojson'
#outfile0 = 'atlanticZonalSectionsTransportsvsdepth'
maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arcticSections20220916.nc'
featurefile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/arcticSections20210323.geojson'
outfile0 = 'arcticSectionsTransportsvsdepth'
casenameFull = 'E3SM-Arcticv2.1_historical0301'
casename = 'E3SM-Arcticv2.1_historical0301'
modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{casenameFull}/archive/ocn/hist'

# Settings for erdc.hpc.mil
#meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#maskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20240910.nc'
#featurefile = '/p/home/milena/mpas-region_masks/atlanticZonal_sections20240910.geojson'
#outfile0 = 'atlanticZonalSectionsTransportsvsdepth'
#maskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_arcticSections20220916.nc'
#featurefile = '/p/home/milena/mpas-region_masks/arcticSections20210323.geojson'
#outfile0 = 'arcticSectionsTransportsvsdepth'
#casenameFull = 'E3SMv2.1B60to10rA02'
#casename = 'E3SMv2.1B60to10rA02'
#modeldir = f'/p/cwfs/milena/{casenameFull}/archive/ocn/hist'
#casenameFull = 'E3SMv2.1B60to10rA07'
#casename = 'E3SMv2.1B60to10rA07'
#modeldir = f'/p/work/milena/{casenameFull}/archive/ocn/hist'

# Choose years
year1 = 1950
year2 = 2014
#year1 = 1
#year2 = 386 # rA02
#year2 = 246 # rA07
years = range(year1, year2+1)

figdir = f'./transports/{casename}'
if not os.path.isdir(figdir):
    os.makedirs(figdir)
outdir = f'./transports_data/{casename}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

if os.path.exists(featurefile):
    fcAll = read_feature_collection(featurefile)
else:
    raise IOError('No feature file found')

saltRef = 34.8
rhoRef = 1027.0 # kg/m^3
cp = 3.987*1e3 # J/(kg*degK) 
m3ps_to_Sv = 1e-6 # m^3/s flux to Sverdrups
m3ps_to_mSv = 1e-3 # m^3/s flux to 10^-3 Sverdrups (mSv)
m3ps_to_km3py = 1e-9*86400*365.25  # m^3/s FW flux to km^3/year
W_to_TW = 1e-12
###################################################################

# Read in transect information
dsMask = get_mask_short_names(xr.open_dataset(maskfile))
transectNames = decode_strings(dsMask.transectNames)
transectList = dsMask.shortNames.values
nTransects = len(transectList)
maxEdges = dsMask.dims['maxEdgesInTransect']
print(f'\nComputing/plotting time series for these transects: {transectNames}\n')

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
dvEdge = dsMesh.dvEdge.sel(nEdges=edgesToRead)
coe0 = dsMesh.cellsOnEdge.isel(TWO=0, nEdges=edgesToRead)
coe1 = dsMesh.cellsOnEdge.isel(TWO=1, nEdges=edgesToRead)
# Build land-sea mask and topomask for the two cells surrounding each edge of the transect 
landmask1 = ~(coe0==0)
landmask2 = ~(coe1==0)
# convert to python indexing
coe0 = coe0 - 1
coe1 = coe1 - 1
#
maxLevelCell = dsMesh.maxLevelCell
nVertLevels = dsMesh.sizes['nVertLevels']
vertIndex = xr.DataArray(data=np.arange(nVertLevels), dims=('nVertLevels',))
#vertIndex = xr.DataArray.from_dict({'dims': ('nVertLevels',), 'data': np.arange(nVertLevels)})
depthmask = (vertIndex < maxLevelCell).transpose('nCells', 'nVertLevels')
depthmask1 = depthmask.isel(nCells=coe0)
depthmask2 = depthmask.isel(nCells=coe1)
#
edgeSigns = np.zeros((nTransects, len(edgesToRead)))
for i in range(nTransects):
    edgeSigns[i, :] = dsMask.sel(nEdges=edgesToRead, shortNames=transectList[i]).squeeze().transectEdgeMaskSigns.values
    # WARNING: The following is a quick hack valid only for the arcticSections mask file!
    # I will need to change the geojson files to make *all* transects go from south to north
    # or west to east, so that I can have the correct edgeSigns for all of them.
    if transectNames[i]!='Bering Strait' and transectNames[i]!='Hudson Bay-Labrador Sea':
        edgeSigns[i, :] = -edgeSigns[i, :]
edgeSigns = xr.DataArray(data=edgeSigns, dims=('nTransect', 'nEdges'))
refBottom = dsMesh.refBottomDepth
latmean = 180.0/np.pi * dsMesh.latEdge.sel(nEdges=edgesToRead).mean()
lonmean = 180.0/np.pi * dsMesh.lonEdge.sel(nEdges=edgesToRead).mean()
pressure = gsw.p_from_z(-refBottom, latmean)

kyear = 0
for year in years:
    kyear = kyear + 1
    print(f'Year = {year:04d} ({kyear} out of {len(years)} years total)')

    outfile = f'{outdir}/{outfile0}_{casename}_year{year:04d}.nc'
    # Compute transports if outfile does not exist
    if not os.path.exists(outfile):
        dsOut = []
        for month in range(1, 13):
            print(f'   Month= {month:02d}')
            modelfile = f'{modeldir}/{casenameFull}.mpaso.hist.am.timeSeriesStatsMonthly.{year:04d}-{month:02d}-01.nc'

            dsIn = xr.open_dataset(modelfile, decode_times=False)
            dsOutMonthly = xr.Dataset()

            if 'timeMonthly_avg_normalTransportVelocity' in dsIn.keys():
                vel = dsIn.timeMonthly_avg_normalTransportVelocity.isel(Time=0, nEdges=edgesToRead)
            elif 'timeMonthly_avg_normalVelocity' in dsIn.keys():
                vel = dsIn.timeMonthly_avg_normalVelocity.isel(Time=0, nEdges=edgesToRead)
                if 'timeMonthly_avg_normalGMBolusVelocity' in dsIn.keys():
                    vel = vel + dsIn.timeMonthly_avg_normalGMBolusVelocity.isel(Time=0, nEdges=edgesToRead)
                if 'timeMonthly_avg_normalMLEvelocity' in dsIn.keys():
                    vel = vel + dsIn.timeMonthly_avg_normalMLEvelocity.isel(Time=0, nEdges=edgesToRead)
            else:
                raise KeyError('no appropriate normalVelocity variable found')
            # Note that the following is incorrect when coe is zero (cell straddling the
            # transect edge is on land), but that is OK because the value will be masked
            # during land-sea masking below
            tempOnCells1 = dsIn.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe0)
            tempOnCells2 = dsIn.timeMonthly_avg_activeTracers_temperature.isel(Time=0, nCells=coe1)
            saltOnCells1 = dsIn.timeMonthly_avg_activeTracers_salinity.isel(Time=0, nCells=coe0)
            saltOnCells2 = dsIn.timeMonthly_avg_activeTracers_salinity.isel(Time=0, nCells=coe1)

            # Mask values that fall on land
            tempOnCells1 = tempOnCells1.where(landmask1, drop=False)
            tempOnCells2 = tempOnCells2.where(landmask2, drop=False)
            saltOnCells1 = saltOnCells1.where(landmask1, drop=False)
            saltOnCells2 = saltOnCells2.where(landmask2, drop=False)
            # Mask values that fall onto topography
            tempOnCells1 = tempOnCells1.where(depthmask1, drop=False)
            tempOnCells2 = tempOnCells2.where(depthmask2, drop=False)
            saltOnCells1 = saltOnCells1.where(depthmask1, drop=False)
            saltOnCells2 = saltOnCells2.where(depthmask2, drop=False)
            # The following should *not* happen at this point:
            if np.any(tempOnCells1.values[np.logical_or(tempOnCells1.values> 1e15, tempOnCells1.values<-1e15)]) or \
               np.any(tempOnCells2.values[np.logical_or(tempOnCells2.values> 1e15, tempOnCells2.values<-1e15)]):
                print('WARNING: something is wrong with land and/or topography masking!')
            if np.any(saltOnCells1.values[np.logical_or(saltOnCells1.values> 1e15, saltOnCells1.values<-1e15)]) or \
               np.any(saltOnCells2.values[np.logical_or(saltOnCells2.values> 1e15, saltOnCells2.values<-1e15)]):
                print('WARNING: something is wrong with land and/or topography masking!')

            dzOnCells1 = dsIn.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe0)
            dzOnCells2 = dsIn.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe1)
            dzOnCells1 = dzOnCells1.where(landmask1, drop=False)
            dzOnCells2 = dzOnCells2.where(landmask2, drop=False)
            dzOnCells1 = dzOnCells1.where(depthmask1, drop=False)
            dzOnCells2 = dzOnCells2.where(depthmask2, drop=False)

            # Interpolate values onto edges
            tempOnEdges = np.nanmean(np.array([tempOnCells1.values, tempOnCells2.values]), axis=0)
            saltOnEdges = np.nanmean(np.array([saltOnCells1.values, saltOnCells2.values]), axis=0)
            dzOnEdges = np.nanmean(np.array([dzOnCells1.values, dzOnCells2.values]), axis=0)
            # The following doesn't do a proper nansum.. (and couldn't find anything online
            # about nansumming two *separate* xarray datasets):
            #tempOnEdges = 0.5 * (tempOnCells1 + tempOnCells2)
            #saltOnEdges = 0.5 * (saltOnCells1 + saltOnCells2)
            #dzOnEdges = 0.5 * (dzOnCells1 + dzOnCells2)
            tempOnEdges = xr.DataArray(data=tempOnEdges, dims=('nEdges', 'nVertLevels'), name='tempOnEdges')
            saltOnEdges = xr.DataArray(data=saltOnEdges, dims=('nEdges', 'nVertLevels'), name='saltOnEdges')
            dzOnEdges = xr.DataArray(data=dzOnEdges, dims=('nEdges', 'nVertLevels'), name='dzOnEdges')

            # Compute freezing temperature
            SA = gsw.SA_from_SP(saltOnEdges, pressure, lonmean, latmean)
            CTfp = gsw.CT_freezing(SA, pressure, 0.)
            Tfp = gsw.pt_from_CT(SA, CTfp)

            # Compute transports for each transect
            #  Initialize to nan's (note that np.empty does *not* work properly)
            vol_transport = np.nan*np.ones((nTransects, nVertLevels))
            heat_transport = np.nan*np.ones((nTransects, nVertLevels)) # Tref = 0degC
            heat_transportTfp = np.nan*np.ones((nTransects, nVertLevels)) # Tref = T freezing point computed below (Tfp)
            FW_transport = np.nan*np.ones((nTransects, nVertLevels)) # uses absolute salinity (FW=(1-1e-3*Sabs))
            FW_transportSref = np.nan*np.ones((nTransects, nVertLevels)) # Sref = saltRef
            temp_transect = np.nan*np.ones((nTransects, nVertLevels))
            salt_transect = np.nan*np.ones((nTransects, nVertLevels))
            depth_transect = np.nan*np.ones((nTransects, len(edgesToRead)))
            for i in range(nTransects):
                start = int(nTransectStartStop[i])
                stop = int(nTransectStartStop[i+1])

                normalVel = vel.isel(nEdges=range(start, stop)) * edgeSigns.isel(nTransect=i, nEdges=range(start, stop))
                temp = tempOnEdges.isel(nEdges=range(start, stop))
                salt = saltOnEdges.isel(nEdges=range(start, stop))
                Sabs = SA.isel(nEdges=range(start, stop)) # FW=volTransport-(volTransport-int(1e-3*Sabs*v)dV)
                maskOnEdges = salt.notnull()
                normalVel = normalVel.where(maskOnEdges, drop=False)

                dx = dvEdge.isel(nEdges=range(start, stop)).expand_dims({'nVertLevels':nVertLevels}, axis=1)
                dx = dx.where(maskOnEdges, drop=False)
                dz = dzOnEdges.isel(nEdges=range(start, stop))
                dArea = dx * dz
                area_transect = dArea.sum(dim='nEdges').sum(dim='nVertLevels')
                length_transect = dx.sum(dim='nEdges')

                tfreezing = Tfp.isel(nEdges=range(start, stop))

                vol_transport[i, :] = (normalVel * dArea).sum(dim='nEdges')
                heat_transport[i, :] = (temp * normalVel * dArea).sum(dim='nEdges')
                heat_transportTfp[i, :] = ((temp-tfreezing) * normalVel * dArea).sum(dim='nEdges')
                FW_transportSref[i, :] = vol_transport[i] - (salt * normalVel * dArea).sum(dim='nEdges')/saltRef
                FW_transport[i, :] = (0.001*Sabs * normalVel * dArea).sum(dim='nEdges')
                temp_transect[i, :] = (temp * dx).sum(dim='nEdges')/length_transect
                salt_transect[i, :] = (salt * dx).sum(dim='nEdges')/length_transect
                depth_transect[i, start:stop] = dz.sum(dim='nVertLevels')

            dsOutMonthly['volTransport'] = xr.DataArray(
                    data=m3ps_to_Sv * vol_transport,
                    dims=('nTransects', 'nVertLevels', ),
                    attrs=dict(description='Net volume transport across transect', units='Sv', )
                    )
            dsOutMonthly['heatTransport'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * heat_transport,
                    dims=('nTransects', 'nVertLevels', ),
                    attrs=dict(description='Net heat transport (wrt 0degC) across transect', units='TW', )
                    )
            dsOutMonthly['heatTransportTfp'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * heat_transportTfp,
                    dims=('nTransects', 'nVertLevels', ),
                    attrs=dict(description='Net heat transport (wrt freezing point) across transect', units='TW', )
                    )
            dsOutMonthly['FWTransportSref'] = xr.DataArray(
                    data=m3ps_to_mSv * FW_transportSref,
                    dims=('nTransects', 'nVertLevels', ),
                    attrs=dict(description=f'Net FW transport (wrt {saltRef:4.1f} psu) across transect', units='mSv', )
                    )
            dsOutMonthly['FWTransport'] = xr.DataArray(
                    data=m3ps_to_mSv * FW_transport,
                    dims=('nTransects', 'nVertLevels', ),
                    attrs=dict(description='Net FW transport (FW=(1 - 1e-3*Sabs)) across transect', units='mSv', )
                    )
            dsOutMonthly['tempTransect'] = xr.DataArray(
                    data=temp_transect,
                    dims=('nTransects', 'nVertLevels', ),
                    attrs=dict(description='Mean temperature across transect', units='degree C', )
                    )
            dsOutMonthly['saltTransect'] = xr.DataArray(
                    data=salt_transect,
                    dims=('nTransects', 'nVertLevels', ),
                    attrs=dict(description='Mean salinity across transect', units='psu', )
                    )
            dsOutMonthly['depthTransect'] = xr.DataArray(
                    data=depth_transect,
                    dims=('nTransects', 'nEdges', ),
                    attrs=dict(description='Depth along transect', units='m', )
                    )
            dsOutMonthly['transectNames'] = xr.DataArray(data=transectNames, dims=('nTransects', ))
            dsOutMonthly['Time'] = xr.DataArray(
                    data=[dsIn.timeMonthly_avg_daysSinceStartOfSim.isel(Time=0)/365.], 
                    dims=('Time', ), 
                    attrs=dict(description='days since start of simulation (assumes 365-day year)',
                               units='days', )
                    )
            #        data=[dsIn.Time.isel(Time=0)], 
            #        dims=('Time', ), 
            #        attrs=dict(description='days since start of simulation (assumes 365-day year)',
            #                   units='days', )
            #        )

            dsOut.append(dsOutMonthly)

        dsOut = xr.concat(dsOut, dim='Time')
        dsOut['refBottomDepth'] = refBottom
        write_netcdf_with_fill(dsOut, outfile)
    else:
        print(f'  Outfile for year {year} already exists. Proceed...')

print(f'\nPlotting...')
# Read in previously computed transport quantities
nyears = len(years)
infiles = []
for year in years:
    infiles.append(f'{outdir}/{outfile0}_{casename}_year{year:04d}.nc')
dsIn = xr.open_mfdataset(infiles, decode_times=False)
t = dsIn['Time']
#z = dsIn['refBottomDepth']
z = dsIn['refBottomDepth'].isel(Time=0)
volTransport = dsIn['volTransport']
heatTransport = dsIn['heatTransport']
heatTransportTfp = dsIn['heatTransportTfp']
FWTransportSref = dsIn['FWTransportSref']
FWTransport = dsIn['FWTransport']
tempTransect = dsIn['tempTransect']
saltTransect = dsIn['saltTransect']
#depthTransect = dsIn['depthTransect']
depthTransect = dsIn['depthTransect'].isel(Time=0)
t_annual = t.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
volTransport_annual = volTransport.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
heatTransport_annual = heatTransport.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
heatTransportTfp_annual = heatTransportTfp.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
FWTransportSref_annual = FWTransportSref.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
FWTransport_annual = FWTransport.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
tempTransect_annual = tempTransect.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})
saltTransect_annual = saltTransect.groupby_bins('Time', nyears).mean().rename({'Time_bins': 'Time'})

#[x, y] = np.meshgrid(t.values, z.values)
[x, y] = np.meshgrid(t_annual.values, z.values)
x = x.T
y = y.T

# Define some dictionaries for transect plotting
labelDict = {'Drake Passage':'drake', 'Tasmania-Ant':'tasmania', 'Africa-Ant':'africaAnt', 'Antilles Inflow':'antilles', \
             'Mona Passage':'monaPassage', 'Windward Passage':'windwardPassage', 'Florida-Cuba':'floridaCuba', \
             'Florida-Bahamas':'floridaBahamas', 'Indonesian Throughflow':'indonesia', 'Agulhas':'agulhas', \
             'Mozambique Channel':'mozambique', 'Bering Strait':'beringStrait', 'Lancaster Sound':'lancasterSound', \
             'Fram Strait':'framStrait', 'Robeson Channel':'robeson', 'Davis Strait':'davisStrait', 'Barents Sea Opening':'barentsSea', \
             'Nares Strait':'naresStrait', 'Denmark Strait':'denmarkStrait', 'Iceland-Faroe-Scotland':'icelandFaroeScotland'}

figsize = (16, 16)
figdpi = 300
for i in range(nTransects):
    transectName = transectNames[i]
    print(transectName)
    zmax = depthTransect.isel(nTransects=i).max().values
    zmin = depthTransect.isel(nTransects=i).min().values
    print('zmin, zmax = ', zmin, zmax)

    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == transectName:
            fc.add_feature(feature)

    if transectName in labelDict:
        transectName_forfigfile = labelDict[transectName]
    else:
        transectName_forfigfile = transectName.replace(" ", "")

    #vol_runavg = pd.Series.rolling(pd.DataFrame(volTransport[:, i]), 12, center=True).mean()
    #heat_runavg = pd.Series.rolling(pd.DataFrame(heatTransport[:, i]), 12, center=True).mean()
    #heatTfp_runavg = pd.Series.rolling(pd.DataFrame(heatTransportTfp[:, i]), 12, center=True).mean()
    #salt_runavg = pd.Series.rolling(pd.DataFrame(saltTransport[:, i]), 12, center=True).mean()

    # Plot Volume Transport
    figfile = f'{figdir}/transportsvsdepth_{transectName_forfigfile}_{casename}.png'
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(421)
    fld = np.squeeze(volTransport_annual.values[:, i, :])
    colormap = cmocean.cm.balance
    cf = ax1.contourf(x, y, fld, cmap=colormap, extend='both')
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label('Volume transport (Sv)', fontsize=10, fontweight='bold')
    #ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylim(0, zmax)
    ax1.invert_yaxis()
    ax1.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')

    # Plot Heat Transport wrt Tref=0
    ax2 = plt.subplot(422)
    fld = np.squeeze(heatTransport_annual.values[:, i, :])
    cf = ax2.contourf(x, y, fld, cmap=colormap, extend='both')
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label('Heat transport (0$^\circ$C; TW)', fontsize=10, fontweight='bold')
    #ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_ylim(0, zmax)
    ax2.invert_yaxis()
    ax2.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')

    # Plot Heat Transport wrt Tref=TfreezingPoint
    ax3 = plt.subplot(423)
    fld = np.squeeze(heatTransportTfp_annual.values[:, i, :])
    cf = ax3.contourf(x, y, fld, cmap=colormap, extend='both')
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label('Heat transport (Tfp; TW)', fontsize=10, fontweight='bold')
    #ax3.autoscale(enable=True, axis='x', tight=True)
    ax3.set_ylim(0, zmax)
    ax3.invert_yaxis()
    ax3.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')

    # Plot transect mean temperature
    ax4 = plt.subplot(424)
    fld = np.squeeze(tempTransect_annual.values[:, i, :])
    colormap = cmocean.cm.thermal
    cf = ax4.contourf(x, y, fld, cmap=colormap, extend='both')
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label('Temperature ($^\circ$C)', fontsize=10, fontweight='bold')
    #ax4.autoscale(enable=True, axis='x', tight=True)
    ax4.set_ylim(0, zmax)
    ax4.invert_yaxis()
    ax4.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')

    # Plot FW Transport wrt Sref
    ax5 = plt.subplot(425)
    fld = np.squeeze(FWTransportSref_annual.values[:, i, :])
    colormap = cmocean.cm.balance
    cf = ax5.contourf(x, y, fld, cmap=colormap, extend='both')
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    #cbar.set_label(f'FW transport wrt {saltRef:4.1f} (mSv)', fontsize=10, fontweight='bold')
    cbar.set_label(f'FW transport (Sref; mSv)', fontsize=10, fontweight='bold')
    #ax5.autoscale(enable=True, axis='x', tight=True)
    ax5.set_ylim(0, zmax)
    ax5.invert_yaxis()
    ax5.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')

    # Plot FW Transport using absolute salinity
    ax6 = plt.subplot(426)
    fld = np.squeeze(FWTransport_annual.values[:, i, :])
    colormap = cmocean.cm.balance
    cf = ax6.contourf(x, y, fld, cmap=colormap, extend='both')
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label(f'FW transport (Sabs; mSv)', fontsize=10, fontweight='bold')
    #ax6.autoscale(enable=True, axis='x', tight=True)
    ax6.set_ylim(0, zmax)
    ax6.invert_yaxis()
    ax6.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Time (Years)', fontsize=10, fontweight='bold')

    # Plot transect mean salinity
    ax7 = plt.subplot(427)
    fld = np.squeeze(saltTransect_annual.values[:, i, :])
    colormap = cmocean.cm.haline
    cf = ax7.contourf(x, y, fld, cmap=colormap, extend='both')
    cbar = plt.colorbar(cf, location='right', pad=0.05, shrink=0.9, extend='both')
    cbar.ax.tick_params(labelsize=10, labelcolor='black')
    cbar.set_label(f'Salinity (psu)', fontsize=10, fontweight='bold')
    #ax7.autoscale(enable=True, axis='x', tight=True)
    ax7.set_ylim(0, zmax)
    ax7.invert_yaxis()
    ax7.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')
    ax7.set_xlabel('Time (Years)', fontsize=10, fontweight='bold')

    fig.suptitle(f'Transect = {transectName}\nrunname = {casename}', fontsize=12, fontweight='bold', y=0.93)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=1.5, ybuffer=0.0)
    #fig.tight_layout(pad=0.5)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
