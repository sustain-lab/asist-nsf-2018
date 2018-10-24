#!/usr/bin/env python3
"""
Processes raw hotfilm voltages to velocities
and stores them into NetCDF file. This is for Exp1.
"""
from asist_nsf_2018.experiments import experiments
from asist_nsf_2018.process_level2 import clean_hotfilm_exp1
from asist.utility import binavg, limit_to_percentile_range, running_mean
from asist.hotfilm import effective_velocity, hotfilm_velocity, read_hotfilm_from_netcdf
from asist.pitot import read_pitot_from_netcdf
from datetime import datetime, timedelta
import numpy as np
import os
from netCDF4 import Dataset

np.warnings.filterwarnings('ignore') # ignore numpy warnings

L2_DATA_PATH = os.environ['L2_DATA_PATH']

exp_name = 'asist-windonly-fresh'
exp = experiments[exp_name]

hotfilm_filename = 'hotfilm_' + exp_name + '.nc'

origin, hotfilm_seconds, fan, ch1, ch2 = read_hotfilm_from_netcdf(hotfilm_filename)
origin, pitot_seconds, fan, u_pitot = read_pitot_from_netcdf(L2_DATA_PATH + '/pitot_' + exp_name + '.nc')

ch1, ch2 = clean_hotfilm_exp1(exp, ch1, ch2, hotfilm_seconds)

# start and end time of fitting period
t0 = exp.runs[1].start_time + timedelta(seconds=60)
t1 = exp.runs[-2].end_time

# start and end seconds of fitting period
t0_seconds = (t0 - origin).total_seconds()
t1_seconds = (t1 - origin).total_seconds()

# start index of pitot and hotfilm time series
n0 = np.argmin((pitot_seconds - t0_seconds)**2)
n1 = np.argmin((pitot_seconds - t1_seconds)**2)
pitot = u_pitot[n0-1:n1] # special case to handle dropped records in pressure files
    
n0 = np.argmin((hotfilm_seconds - t0_seconds)**2)
n1 = np.argmin((hotfilm_seconds - t1_seconds)**2)
ch1_binavg = binavg(ch1[n0:n1], 100)
ch2_binavg = binavg(ch2[n0:n1], 100)
    
# 4-th order polynomial -- ordered highest to lowest degree
p1 = np.polyfit(ch1_binavg, effective_velocity(pitot), 4)
p2 = np.polyfit(ch2_binavg, effective_velocity(pitot), 4)

# compute effective velocities
veff1 = np.polyval(p1, ch1)
veff2 = np.polyval(p2, ch2)

veff1[veff1 < 0] = 0
veff2[veff2 < 0] = 0

u, w = hotfilm_velocity(veff1, veff2)

nc = Dataset(hotfilm_filename, 'r+')

var = nc.createVariable('ch1_clean', 'f4', dimensions=('Time'))
var[:] = ch1[:]
var.setncattr('name', 'Channel 1 voltage, clean')
var.setncattr('units', 'V')

var = nc.createVariable('ch2_clean', 'f4', dimensions=('Time'))
var[:] = ch2[:]
var.setncattr('name', 'Channel 2 voltage, clean')
var.setncattr('units', 'V')

var = nc.createVariable('veff1', 'f4', dimensions=('Time'))
var[:] = veff1[:]
var.setncattr('name', 'Channel 1 effective velocity')
var.setncattr('units', 'm/s')

var = nc.createVariable('veff2', 'f4', dimensions=('Time'))
var[:] = veff2[:]
var.setncattr('name', 'Channel 2 effective velocity')
var.setncattr('units', 'm/s')

var = nc.createVariable('u', 'f4', dimensions=('Time'))
var[:] = u[:]
var.setncattr('name', 'x-component of velocity')
var.setncattr('units', 'm/s')

var = nc.createVariable('w', 'f4', dimensions=('Time'))
var[:] = w[:]
var.setncattr('name', 'z-component of velocity')
var.setncattr('units', 'm/s')

nc.close()
