#!/usr/bin/env python3
"""
Processes raw hotfilm voltages to velocities.
"""
from asist_nsf_2018.experiments import experiments
from asist.utility import binavg
from asist.hotfilm import hotfilm_velocity, read_hotfilm_from_netcdf
from asist.pitot import read_pitot_from_netcdf
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib.pyplot as plt

L2_DATA_PATH = os.environ['L2_DATA_PATH']

exp_name = 'asist-windonly-fresh'
exp = experiments[exp_name]

origin, hotfilm_seconds, fan, ch1, ch2 = read_hotfilm_from_netcdf(L2_DATA_PATH + '/hotfilm_' + exp_name + '.nc')
origin, pitot_seconds, fan, u = read_pitot_from_netcdf(L2_DATA_PATH + '/pitot_' + exp_name + '.nc')

#TODO clean out hotfilm data

# start and end time of fitting period
t0 = exp.runs[1].start_time + timedelta(seconds=60)
t1 = exp.runs[-2].end_time

# start and end seconds of fitting period
t0_seconds = (t0 - origin).total_seconds()
t1_seconds = (t1 - origin).total_seconds()

# start index of pitot and hotfilm time series
n0 = np.argmin((pitot_seconds - t0_seconds)**2)
n1 = np.argmin((pitot_seconds - t1_seconds)**2)
pitot = u[n0-1:n1] # special case to handle dropped records in pressure files
    
n0 = np.argmin((hotfilm_seconds - t0_seconds)**2)
n1 = np.argmin((hotfilm_seconds - t1_seconds)**2)
ch1_binavg = binavg(ch1[n0:n1], 100)
ch2_binavg = binavg(ch2[n0:n1], 100)
    
# 4-th order polynomial -- orderd highest to lowest degree
p1 = np.polyfit(ch1_binavg, pitot, 4)
p2 = np.polyfit(ch2_binavg, pitot, 4)

u, w = hotfilm_velocity(veff1, veff2)
