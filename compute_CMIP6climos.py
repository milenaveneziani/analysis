from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


indir = '/pscratch/sd/g/gennaro/ACOU/CMIP6/HighResMIP'
modelCenter = 'NCAR'
simName = 'CESM1-CAM5-SE-HR'
#simType = 'hist-1950'
#modelVersion = 'v20200810'
simType = 'highres-future'
modelVersion = 'v20200731'
simMember = 'r1i1p1f1'
gridType = 'gn' # native

#climoYear1 = 1950
#climoYear2 = 1970
#climoYear1 = 2000
#climoYear2 = 2014
climoYear1 = 2031
climoYear2 = 2050

indir = f'{indir}/{modelCenter}/{simName}/{simType}/{simMember}/Omon'
outdir = f'/pscratch/sd/m/milena/CMIP6monthlyclimos/{modelCenter}/{simName}/{simType}/{simMember}'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

vars = ['so', 'thetao']

for var in vars:
    #outfile = f'{outdir}/{var}_Omon_{simName}_{simType}_{simMember}_{gridType}_climoYears{climoYear1}-{climoYear2}.nc'
    infiles = []
    for year in range(climoYear1, climoYear2+1):
        infiles.append(f'{indir}/{var}/{gridType}/{modelVersion}/{var}_Omon_{simName}_{simType}_{simMember}_{gridType}_{year}01-{year}12.nc')
    ds = xr.open_mfdataset(infiles, combine='nested', concat_dim='time', parallel=True, chunks={'time': 1, 'lev': 8})
    dsMonthlyClimo = ds.groupby('time.month').mean('time')
    for im in range(1, 13):
        outfile = f'{outdir}/{var}_Omon_{simName}_{simType}_{simMember}_{gridType}_climoYears{climoYear1}-{climoYear2}_M{im:02d}.nc'
        dsMonthlyClimo.isel(month=im-1).to_netcdf(outfile)
    #dsMonthlyClimo.to_netcdf(outfile, compute=False)
    #with ProgressBar():
    #    print(f"Writing to {outfile}")
    #    dsMonthlyClimo.compute()
