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
import pandas as pd
import gsw

from common_functions import add_inset
from geometric_features import FeatureCollection, read_feature_collection
from mpas_analysis.shared.io.utility import decode_strings
from mpas_analysis.shared.io import write_netcdf_with_fill
from mpas_analysis.ocean.utility import compute_zmid


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
maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_arctic_subarctic_transects4transports20250918.nc'
featurefile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/arctic_subarctic_transects4transports20250918.geojson'
outfile0 = 'arcticSubarcticSectionsTransports'
#maskfile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20240910.nc'
#featurefile = '/global/cfs/cdirs/m1199/milena/mpas-region_masks/atlanticZonal_sections20240910.geojson'
#outfile0 = 'atlanticZonalSectionsTransports'
casenameFull = 'E3SM-Arcticv2.1_historical0301'
casename = 'E3SM-Arcticv2.1_historical0301'
modeldir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations/{casenameFull}/archive/ocn/hist'

# Settings for erdc.hpc.mil
#meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#maskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_arctic_subarctic_transects4transports20250918.nc'
#featurefile = '/p/home/milena/mpas-region_masks/arctic_subarctic_transects4transports20250918.geojson'
#outfile0 = 'arcticSubarcticSectionsTransports'
#maskfile = '/p/home/milena/mpas-region_masks/ARRM10to60E2r1_atlanticZonal_sections20240910.nc'
#featurefile = '/p/home/milena/mpas-region_masks/atlanticZonal_sections20240910.geojson'
#outfile0 = 'atlanticZonalSectionsTransports'
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

zmin = -400.0
zmax = 10.0

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
maxLevelCell1 = dsMesh.maxLevelCell.isel(nCells=coe0) - 1 # now compute_zmid uses 0-based indexing
maxLevelCell2 = dsMesh.maxLevelCell.isel(nCells=coe1) - 1 # now compute_zmid uses 0-based indexing
depth1 = dsMesh.bottomDepth.isel(nCells=coe0)
depth2 = dsMesh.bottomDepth.isel(nCells=coe1)
nVertLevels = dsMesh.sizes['nVertLevels']
#
edgeSigns = np.zeros((nTransects, len(edgesToRead)))
for i in range(nTransects):
    edgeSigns[i, :] = dsMask.sel(nEdges=edgesToRead, shortNames=transectList[i]).squeeze().transectEdgeMaskSigns.values
edgeSigns = xr.DataArray(data=edgeSigns, dims=('nTransect', 'nEdges'))
refBottom = dsMesh.refBottomDepth
latmean = 180.0/np.pi * dsMesh.latEdge.sel(nEdges=edgesToRead).mean()
lonmean = 180.0/np.pi * dsMesh.lonEdge.sel(nEdges=edgesToRead).mean()
pressure = gsw.p_from_z(-refBottom, latmean)

kyear = 0
for year in years:
    kyear = kyear + 1
    print(f'Year = {year:04d} ({kyear} out of {len(years)} years total)')

    if zmax>0:
        outfile = f'{outdir}/{outfile0}_z0000-{np.abs(np.int32(zmin)):04d}_{casename}_year{year:04d}.nc'
    else:
        outfile = f'{outdir}/{outfile0}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{casename}_year{year:04d}.nc'
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
            dzOnCells1 = dsIn.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe0)
            dzOnCells2 = dsIn.timeMonthly_avg_layerThickness.isel(Time=0, nCells=coe1)

            # Compute depthmask
            zMid1 = compute_zmid(depth1, maxLevelCell1, dzOnCells1)
            zMid2 = compute_zmid(depth2, maxLevelCell2, dzOnCells2)
            depthmask1 = np.logical_and(zMid1 >= zmin, zMid1 <= zmax)
            depthmask2 = np.logical_and(zMid2 >= zmin, zMid2 <= zmax)

            # Mask values that fall on land
            tempOnCells1 = tempOnCells1.where(landmask1, drop=False)
            tempOnCells2 = tempOnCells2.where(landmask2, drop=False)
            saltOnCells1 = saltOnCells1.where(landmask1, drop=False)
            saltOnCells2 = saltOnCells2.where(landmask2, drop=False)
            dzOnCells1 = dzOnCells1.where(landmask1, drop=False)
            dzOnCells2 = dzOnCells2.where(landmask2, drop=False)
            # Mask values that fall onto topography
            tempOnCells1 = tempOnCells1.where(depthmask1, drop=False)
            tempOnCells2 = tempOnCells2.where(depthmask2, drop=False)
            saltOnCells1 = saltOnCells1.where(depthmask1, drop=False)
            saltOnCells2 = saltOnCells2.where(depthmask2, drop=False)
            dzOnCells1 = dzOnCells1.where(depthmask1, drop=False)
            dzOnCells2 = dzOnCells2.where(depthmask2, drop=False)
            # Not sure why, but this is still necessary due to the depthmasks computed above:
            mask = ~np.logical_or(tempOnCells1> 1e15, tempOnCells1<-1e15)
            tempOnCells1 = tempOnCells1.where(mask, drop=False)
            mask = ~np.logical_or(tempOnCells2> 1e15, tempOnCells2<-1e15)
            tempOnCells2 = tempOnCells2.where(mask, drop=False)
            mask = ~np.logical_or(saltOnCells1> 1e15, saltOnCells1<-1e15)
            saltOnCells1 = saltOnCells1.where(mask, drop=False)
            mask = ~np.logical_or(saltOnCells2> 1e15, saltOnCells2<-1e15)
            saltOnCells2 = saltOnCells2.where(mask, drop=False)

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

            # Compute spiciness
            CT = gsw.CT_from_pt(SA, tempOnEdges)
            spicinessOnEdges = gsw.spiciness0(SA, CT)

            # Compute transports for each transect
            vol_transport = np.zeros(nTransects)
            heat_transport = np.zeros(nTransects) # Tref = 0degC
            heat_transportTfp = np.zeros(nTransects) # Tref = T freezing point computed below (Tfp)
            FW_transport = np.zeros(nTransects) # uses absolute salinity (FW=(1-1e-3*Sabs))
            FW_transportSref = np.zeros(nTransects) # Sref = saltRef
            temp_transect = np.zeros(nTransects)
            salt_transect = np.zeros(nTransects)
            spice_transect = np.zeros(nTransects)
            for i in range(nTransects):
                start = int(nTransectStartStop[i])
                stop = int(nTransectStartStop[i+1])

                normalVel = vel.isel(nEdges=range(start, stop)) * edgeSigns.isel(nTransect=i, nEdges=range(start, stop))
                temp = tempOnEdges.isel(nEdges=range(start, stop))
                salt = saltOnEdges.isel(nEdges=range(start, stop))
                spice = spicinessOnEdges.isel(nEdges=range(start, stop))
                Sabs = SA.isel(nEdges=range(start, stop)) # FW=volTransport-(volTransport-int(1e-3*Sabs*v)dV)
                maskOnEdges = salt.notnull()
                normalVel = normalVel.where(maskOnEdges, drop=False)

                dArea = dvEdge.isel(nEdges=range(start, stop)) * dzOnEdges.isel(nEdges=range(start, stop))
                area_transect = dArea.sum(dim='nEdges').sum(dim='nVertLevels')

                tfreezing = Tfp.isel(nEdges=range(start, stop))
                #
                vol_transport[i]    = (normalVel * dArea).sum(dim='nEdges').sum(dim='nVertLevels')
                heat_transport[i]    = (temp * normalVel * dArea).sum(dim='nEdges').sum(dim='nVertLevels')
                heat_transportTfp[i]    = ((temp-tfreezing) * normalVel * dArea).sum(dim='nEdges').sum(dim='nVertLevels')
                FW_transportSref[i]    = vol_transport[i] - (salt * normalVel * dArea).sum(dim='nEdges').sum(dim='nVertLevels')/saltRef
                FW_transport[i]    = (0.001*Sabs * normalVel * dArea).sum(dim='nEdges').sum(dim='nVertLevels')
                temp_transect[i] = (temp * dArea).sum(dim='nEdges').sum(dim='nVertLevels')/area_transect
                salt_transect[i] = (salt * dArea).sum(dim='nEdges').sum(dim='nVertLevels')/area_transect
                spice_transect[i] = (spice * dArea).sum(dim='nEdges').sum(dim='nVertLevels')/area_transect

            dsOutMonthly['volTransport'] = xr.DataArray(
                    data=m3ps_to_Sv * vol_transport,
                    dims=('nTransects', ),
                    attrs=dict(description='Net volume transport across transect', units='Sv', )
                    )
            dsOutMonthly['heatTransport'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * heat_transport,
                    dims=('nTransects', ),
                    attrs=dict(description='Net heat transport (wrt 0degC) across transect', units='TW', )
                    )
            dsOutMonthly['heatTransportTfp'] = xr.DataArray(
                    data=W_to_TW * rhoRef * cp * heat_transportTfp,
                    dims=('nTransects', ),
                    attrs=dict(description='Net heat transport (wrt freezing point) across transect', units='TW', )
                    )
            dsOutMonthly['FWTransportSref'] = xr.DataArray(
                    data=m3ps_to_mSv * FW_transportSref,
                    dims=('nTransects', ),
                    attrs=dict(description=f'Net FW transport (wrt {saltRef:4.1f} psu) across transect', units='mSv', )
                    )
            dsOutMonthly['FWTransport'] = xr.DataArray(
                    data=m3ps_to_mSv * FW_transport,
                    dims=('nTransects', ),
                    attrs=dict(description='Net FW transport (FW=(1 - 1e-3*Sabs)) across transect', units='mSv', )
                    )
            dsOutMonthly['tempTransect'] = xr.DataArray(
                    data=temp_transect,
                    dims=('nTransects', ),
                    attrs=dict(description='Mean temperature across transect', units='degree C', )
                    )
            dsOutMonthly['saltTransect'] = xr.DataArray(
                    data=salt_transect,
                    dims=('nTransects', ),
                    attrs=dict(description='Mean salinity across transect', units='psu', )
                    )
            dsOutMonthly['spiceTransect'] = xr.DataArray(
                    data=spice_transect,
                    dims=('nTransects', ),
                    attrs=dict(description='Mean spiciness (wrt sea level pressure) across transect', units='kg/m^3', )
                    )
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
        dsOut['transectNames'] = xr.DataArray(data=transectNames, dims=('nTransects', ))
        dsOut['zbounds'] = ('nbounds', [zmin, zmax])
        dsOut.zbounds.attrs['units'] = 'm'
        write_netcdf_with_fill(dsOut, outfile)
    else:
        print(f'  Outfile for year {year} and depth range {zmax}, {zmin} already exists. Proceed...')

print(f'\nPlotting...')
# Read in previously computed transport quantities
infiles = []
for year in years:
    if zmax>0:
        infiles.append(f'{outdir}/{outfile0}_z0000-{np.abs(np.int32(zmin)):04d}_{casename}_year{year:04d}.nc')
    else:
        infiles.append(f'{outdir}/{outfile0}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{casename}_year{year:04d}.nc')
dsIn = xr.open_mfdataset(infiles, decode_times=False)
t = dsIn['Time'].values
volTransport = dsIn['volTransport'].values
heatTransport = dsIn['heatTransport'].values
heatTransportTfp = dsIn['heatTransportTfp'].values
FWTransportSref = dsIn['FWTransportSref'].values
FWTransport = dsIn['FWTransport'].values
tempTransect = dsIn['tempTransect'].values
saltTransect = dsIn['saltTransect'].values
spiceTransect = dsIn['spiceTransect'].values

# Define some dictionaries for transect plotting
labelDict = {'Drake Passage':'drake', 'Tasmania-Ant':'tasmania', 'Africa-Ant':'africaAnt', 'Antilles Inflow':'antilles', \
             'Mona Passage':'monaPassage', 'Windward Passage':'windwardPassage', 'Florida-Cuba':'floridaCuba', \
             'Florida-Bahamas':'floridaBahamas', 'Indonesian Throughflow':'indonesia', 'Agulhas':'agulhas', \
             'Mozambique Channel':'mozambique', 'Bering Strait':'beringStrait', 'Lancaster Sound':'lancasterSound', \
             'Fram Strait':'framStrait', 'Robeson Channel':'robeson', 'Davis Strait':'davisStrait', 'Barents Sea Opening':'barentsSeaOpening', \
             'Nares Strait':'naresStrait', 'Denmark Strait':'denmarkStrait', 'Iceland-Faroe-Scotland':'icelandFaroeScotland'}

figsize = (16, 16)
figdpi = 300
for i in range(nTransects):
    transectName = transectNames[i]

    fc = FeatureCollection()
    for feature in fcAll.features:
        if feature['properties']['name'] == transectName:
            fc.add_feature(feature)

    if transectName in labelDict:
        transectName_forfigfile = labelDict[transectName]
    else:
        transectName_forfigfile = transectName.replace(" ", "")

    vol_runavg = pd.Series.rolling(pd.DataFrame(volTransport[:, i]), 12, center=True).mean()
    heat_runavg = pd.Series.rolling(pd.DataFrame(heatTransport[:, i]), 12, center=True).mean()
    heatTfp_runavg = pd.Series.rolling(pd.DataFrame(heatTransportTfp[:, i]), 12, center=True).mean()
    fw_runavg = pd.Series.rolling(pd.DataFrame(FWTransport[:, i]), 12, center=True).mean()
    fwSref_runavg = pd.Series.rolling(pd.DataFrame(FWTransportSref[:, i]), 12, center=True).mean()
    salt_runavg = pd.Series.rolling(pd.DataFrame(saltTransect[:, i]), 12, center=True).mean()
    temp_runavg = pd.Series.rolling(pd.DataFrame(tempTransect[:, i]), 12, center=True).mean()
    spice_runavg = pd.Series.rolling(pd.DataFrame(spiceTransect[:, i]), 12, center=True).mean()

    if zmax>0:
        figfile = f'{figdir}/transports_{transectName_forfigfile}_z0000-{np.abs(np.int32(zmin)):04d}_{casename}.png'
    else:
        figfile = f'{figdir}/transports_{transectName_forfigfile}_z{np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}_{casename}.png'

    # Plot Volume Transport
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(421)
    ax1.plot(t, volTransport[:, i], 'k', linewidth=1.2, alpha=0.5, label=f'net ({np.nanmean(volTransport[:,i]):5.2f} $\pm$ {np.nanstd(volTransport[:,i]):5.2f})')
    ax1.plot(t, vol_runavg, 'r', linewidth=2.2)
    ax1.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax1.grid(color='k', linestyle=':', linewidth = 0.5)
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel('Volume transport (Sv)', fontsize=12, fontweight='bold')
    ax1.legend()

    # Plot Heat Transport wrt Tref=0
    ax2 = plt.subplot(422)
    ax2.plot(t, heatTransport[:, i], 'k', linewidth=1.2, alpha=0.5, label=f'net ({np.nanmean(heatTransport[:,i]):5.2f} $\pm$ {np.nanstd(heatTransport[:,i]):5.2f})')
    ax2.plot(t, heat_runavg, 'r', linewidth=2.2)
    ax2.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax2.grid(color='k', linestyle=':', linewidth = 0.5)
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_ylabel('Heat transport wrt 0$^\circ$C (TW)', fontsize=12, fontweight='bold')
    ax2.legend()

    # Plot Heat Transport wrt Tref=TfreezingPoint
    ax3 = plt.subplot(423)
    ax3.plot(t, heatTransportTfp[:, i], 'k', linewidth=1.2, alpha=0.5, label=f'net ({np.nanmean(heatTransportTfp[:,i]):5.2f} $\pm$ {np.nanstd(heatTransportTfp[:,i]):5.2f})')
    ax3.plot(t, heatTfp_runavg, 'r', linewidth=2.2)
    ax3.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax3.grid(color='k', linestyle=':', linewidth = 0.5)
    ax3.autoscale(enable=True, axis='x', tight=True)
    ax3.set_ylabel('Heat transport wrt freezing point (TW)', fontsize=12, fontweight='bold')
    ax3.legend()

    # Plot transect mean temperature
    ax4 = plt.subplot(424)
    ax4.plot(t, tempTransect[:, i], 'k', linewidth=1.2, alpha=0.5, label=f'temp ({np.nanmean(tempTransect[:,i]):5.2f} $\pm$ {np.nanstd(tempTransect[:,i]):5.2f})')
    ax4.plot(t, temp_runavg, 'r', linewidth=2.2)
    ax4.grid(color='k', linestyle=':', linewidth = 0.5)
    ax4.autoscale(enable=True, axis='x', tight=True)
    ax4.set_ylabel('Transect mean temperature ($^\circ$C)', fontsize=12, fontweight='bold')
    ax4.legend()

    # Plot FW Transport wrt Sref
    ax5 = plt.subplot(425)
    ax5.plot(t, FWTransportSref[:, i], 'k', linewidth=1.2, alpha=0.5, label=f'net ({np.nanmean(FWTransportSref[:,i]):5.2f} $\pm$ {np.nanstd(FWTransportSref[:,i]):5.2f})')
    ax5.plot(t, fwSref_runavg, 'r', linewidth=2.2)
    ax5.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax5.grid(color='k', linestyle=':', linewidth = 0.5)
    ax5.autoscale(enable=True, axis='x', tight=True)
    ax5.set_ylabel(f'FW transport wrt {saltRef:4.1f} (mSv)', fontsize=12, fontweight='bold')
    #ax5.set_ylabel('FW transport wrt {saltRef:4.1f} (km$^3$/year)', fontsize=12, fontweight='bold')
    #ax5.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax5.legend()

    # Plot FW Transport using absolute salinity
    ax6 = plt.subplot(426)
    ax6.plot(t, FWTransport[:, i], 'k', linewidth=1.2, alpha=0.5, label=f'net ({np.nanmean(FWTransport[:,i]):5.2f} $\pm$ {np.nanstd(FWTransport[:,i]):5.2f})')
    ax6.plot(t, fw_runavg, 'r', linewidth=2.2)
    ax6.plot(t, np.zeros_like(t), 'k', linewidth=1)
    ax6.grid(color='k', linestyle=':', linewidth = 0.5)
    ax6.autoscale(enable=True, axis='x', tight=True)
    ax6.set_ylabel('FW transport wrt Sabs (mSv)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax6.legend()

    # Plot transect mean salinity
    ax7 = plt.subplot(427)
    ax7.plot(t, saltTransect[:, i], 'k', linewidth=1.2, alpha=0.5, label=f'salt ({np.nanmean(saltTransect[:,i]):5.2f} $\pm$ {np.nanstd(saltTransect[:,i]):5.2f})')
    ax7.plot(t, salt_runavg, 'r', linewidth=2.2)
    ax7.grid(color='k', linestyle=':', linewidth = 0.5)
    ax7.autoscale(enable=True, axis='x', tight=True)
    ax7.set_ylabel('Transect mean salinity (psu)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax7.legend()

    # Plot transect mean spiciness0
    ax8 = plt.subplot(428)
    ax8.plot(t, spiceTransect[:, i], 'k', linewidth=1.2, alpha=0.5, label=f'spice0 ({np.nanmean(spiceTransect[:,i]):5.2f} $\pm$ {np.nanstd(spiceTransect[:,i]):5.2f})')
    ax8.plot(t, spice_runavg, 'r', linewidth=2.2)
    ax8.grid(color='k', linestyle=':', linewidth = 0.5)
    ax8.autoscale(enable=True, axis='x', tight=True)
    ax8.set_ylabel('Transect mean spiciness0 (kg/m^3)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
    ax8.legend()

    fig.tight_layout(pad=0.5)
    if zmax>0:
        fig.suptitle(f'Transect = {transectName}\nrunname = {casename}, depth range=0000-{np.abs(np.int32(zmin)):04d}', fontsize=14, fontweight='bold', y=1.045)
    else:
        fig.suptitle(f'Transect = {transectName}\nrunname = {casename}, depth range={np.abs(np.int32(zmax)):04d}-{np.abs(np.int32(zmin)):04d}', fontsize=14, fontweight='bold', y=1.045)
    add_inset(fig, fc, width=1.5, height=1.5, xbuffer=-0.5, ybuffer=-1.65)

    fig.savefig(figfile, dpi=figdpi, bbox_inches='tight')
