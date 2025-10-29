#

from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import xarray as xr
import numpy as np

from barotropicStreamfunction import compute_barotropic_streamfunction_vertex


# Settings for nersc
#meshfile = '/global/cfs/cdirs/e3sm/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
#runname = 'E3SM-Arcticv2.1_historical0301'
# Directories where fields for step 2) are stored:
#maindir = f'/global/cfs/cdirs/m1199/e3sm-arrm-simulations'

# Settings for erdc.hpc.mil
meshfile = '/p/app/unsupported/RASM/acme/inputdata/ocn/mpas-o/ARRM10to60E2r1/mpaso.ARRM10to60E2r1.rstFrom1monthG-chrys.220802.nc'
runname = 'E3SMv2.1B60to10rA02'
maindir = f'/p/cwfs/milena'

postprocmaindir = maindir
isShortTermArchive = True

if isShortTermArchive:
    rundir = f'{maindir}/{runname}/archive/ocn/hist'
    postprocdir = f'{postprocmaindir}/{runname}/archive/ocn/postproc'
else:
    rundir = f'{maindir}/{runname}/run'
    postprocdir = f'{postprocmaindir}/{runname}/run'
if not os.path.isdir(postprocdir):
    os.makedirs(postprocdir)

#startYear = 2000
#endYear = 2014
startYear = 1
endYear = 386
########################################################################

dsMesh = xr.open_dataset(meshfile)

min_lat = -45.0
min_depth = -10000.0
max_depth = 10.0

for iy in np.arange(startYear, endYear + 1):
    print(f'Processing year: {iy}...')
    for im in range(1, 13):
        datafile = f'{rundir}/{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{int(iy):04d}-{int(im):02d}-01.nc'
        if not os.path.isfile(datafile):
            raise SystemExit(f'File {datafile} not found. Exiting...\n')
        outfile = f'{postprocdir}/barotropicStreamfunction.{runname}.mpaso.hist.am.timeSeriesStatsMonthly.{int(iy):04d}-{int(im):02d}-01.nc'
        if not os.path.isfile(outfile):
            print(f'  month: {im}...')
            dsIn = xr.open_dataset(datafile)
            fld = compute_barotropic_streamfunction_vertex(dsMesh, dsIn, min_lat, min_depth, max_depth)
            dsOut = xr.Dataset()
            dsOut['barotropicStreamfunction'] = fld
            dsOut['barotropicStreamfunction'].attrs['long_name'] = 'Barotropic streamfunction'
            dsOut['barotropicStreamfunction'].attrs['units'] = 'Sv'
            dsOut.to_netcdf(outfile)
        else:
            print(f'  File {outfile} already exists. Skipping it...')
