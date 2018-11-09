#!/usr/bin/env python3
"""
Processes raw hotfilm voltages to velocities.
"""
from asist_nsf_2018.experiments import experiments
from asist_nsf_2018.process_level2 import clean_hotfilm_exp1
from asist.utility import binavg, limit_to_percentile_range, running_mean
from asist.hotfilm import effective_velocity, hotfilm_velocity, read_hotfilm_from_netcdf
from asist.pitot import read_pitot_from_netcdf
from asist.irgason import read_irgason_from_netcdf
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib.pyplot as plt

def hotfilm_velocity(veff1, veff2, k1=0.3, k2=0.3):
    """For a pair effective velocities from wire 1 and 2,
    calculates u and w components."""
    un = np.sqrt((veff1**2 - k1**2 * veff2**2) / (1 - k1**2 * k2**2))
    ut = np.sqrt((veff2**2 - k2**2 * veff1**2) / (1 - k1**2 * k2**2))
    u = (ut + un) / np.sqrt(2.)
    w = (ut - un) / np.sqrt(2.)
    return u, w

def smooth_ust(u, z):
    """Given input velocity u at height z,
    returns friction velocity of the smooth flow."""
    z0 = 1e-3
    kappa = 0.4
    nu_air = 1.56e-5
    for i in range(20):
        ust = kappa * u / np.log(z / z0)
        z0 = 0.132 * nu_air / ust
    return ust

def rotate(u, w, th):
    """Rotates the vector (u, w) by angle th."""
    ur =  np.cos(th) * u + np.sin(th) * w
    wr = -np.sin(th) * u + np.cos(th) * w
    return ur, wr

plt.rcParams.update({'font.size': 16}) # global font size setting
np.warnings.filterwarnings('ignore') # ignore numpy warnings

L2_DATA_PATH = os.environ['L2_DATA_PATH']

exp_name = 'asist-windonly-fresh'
exp = experiments[exp_name]

origin, hotfilm_seconds, fan, ch1, ch2 = read_hotfilm_from_netcdf(L2_DATA_PATH + '/hotfilm_' + exp_name + '.nc')
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

veff1[veff1 < 0.2] = 0.2
veff2[veff2 < 0.2] = 0.2
u_hf, w_hf = hotfilm_velocity(veff1, veff2)

##########################################

# read IRGASON data
data = read_irgason_from_netcdf(L2_DATA_PATH + '/irgason_' + exp_name + '.nc')
t, u, v, w, fan, flag = data['time'], data['u'], data['v'], data['w'], data['fan'], data['flag']

# first clean out velocities that exceed percentile range
for run in exp.runs[:-1]:
    mask = (t >= run.start_time) & (t <= run.end_time)
    try:
        u[mask] = limit_to_percentile_range(u[mask], 0.1, 99.9)
        v[mask] = limit_to_percentile_range(v[mask], 0.1, 99.9)
        w[mask] = limit_to_percentile_range(w[mask], 0.1, 99.9)
    except:
        pass
u_irg, w_irg = u, w

fan_settings = np.arange(0, 65, 5)

U, W, uw = [], [], []
for run in exp.runs[1:-1]:
    mask = (t > run.start_time + timedelta(seconds=30))\
         & (t < run.end_time - timedelta(seconds=30))
    um = np.mean(u[mask])
    wm = np.mean(w[mask])
    th = np.arctan2(wm, um)
    ur, wr = rotate(u[mask], w[mask], th)
    um, wm = np.mean(ur), np.mean(wr)
    up, wp = ur - um, wr - wm
    U.append(um)
    W.append(wm)
    uw.append(np.mean(up * wp))
U_irg, W_irg, uw_irg = map(np.array, [U, W, uw])


# averaging over full runs (5 minutes)
U, W, uw = [], [], []
for run in exp.runs[1:-1]:
    t0 = run.start_time + timedelta(seconds=30)
    t1 = run.end_time - timedelta(seconds=30)
    t0_seconds = (t0 - origin).total_seconds()
    t1_seconds = (t1 - origin).total_seconds()
    mask = (hotfilm_seconds > t0_seconds) & (hotfilm_seconds < t1_seconds)
    um, wm = np.mean(u_hf[mask]), np.mean(w_hf[mask])
    U.append(um)
    W.append(wm)
    th = np.arctan2(wm, um)
    ur, wr = rotate(u_hf[mask], w_hf[mask], th)
    uw.append(np.mean((ur - um) * (wr - wm)))
U_hf, W_hf, uw_hf = map(np.array, [U, W, uw])
