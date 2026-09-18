#!/usr/bin/env python
"""     
Name: jra55_timeavg.py
Purpose: Monthly average 3-hourly JRA55 atmosphere fields
Author: Milena Veneziani
        
        
"""     

# ensure plots are rendered on ICC
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import os
import cartopy
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
mpl.use('Agg')

indir = '/global/cfs/cdirs/e3sm/inputdata/ocn/jra55/v1.5_noleap'
outdir = '/global/cfs/cdirs/m1199/jra55/v1.5_noleap/monthly'
infile_header = 'JRA.v1.5'
infile_tail = '210504'

year1 = 1958
year2 = 1977

vars = ['t_10', 'q_10']

if not os.path.isdir(outdir):
    os.makedirs(outdir)
    
for year in range(year1, year2+1):
    print(year)
    for var in vars:
        print(f'    var={var}')
        outfile = f'{outdir}/{infile_header}.{var}.TL319.{year}.nc'
        if not os.path.exists(outfile):
           infile = f'{indir}/{infile_header}.{var}.TL319.{year}.{infile_tail}.nc'
           ds = xr.open_dataset(infile)
           ds_monthly = ds.resample(time="MS").mean()
           ds_monthly.to_netcdf(outfile)
        else:   
            print(f'  File for year {year} and var {var} already exists. Moving to the next one...') 
