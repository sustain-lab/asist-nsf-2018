#!/usr/bin/env python3
"""
Processes raw hotfilm voltages to velocities.
"""
from asist_nsf_2018.experiments import experiments
from asist_nsf_2018.process_level2 import clean_hotfilm_exp3
from asist.utility import binavg, limit_to_percentile_range, running_mean
from asist.hotfilm import hotfilm_velocity, read_hotfilm_from_netcdf
from asist.pitot import read_pitot_from_netcdf
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16}) # global font size setting
np.warnings.filterwarnings('ignore') # ignore numpy warnings

L2_DATA_PATH = os.environ['L2_DATA_PATH']

exp_name = 'asist-windonly-salt'
exp = experiments[exp_name]

origin, hotfilm_seconds, fan, ch1, ch2 = read_hotfilm_from_netcdf(L2_DATA_PATH + '/hotfilm_' + exp_name + '.nc')
origin, pitot_seconds, fan, u = read_pitot_from_netcdf(L2_DATA_PATH + '/pitot_' + exp_name + '.nc')

ch1, ch2 = clean_hotfilm_exp3(exp, ch1, ch2)

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

# compute effective velocities
veff1 = p1[4] + p1[3] * ch1 + p1[2] * ch1**2 + p1[1] * ch1**3 + p1[0] * ch1**4
veff2 = p2[4] + p2[3] * ch2 + p2[2] * ch2**2 + p2[1] * ch2**3 + p2[0] * ch2**4 

u, w = hotfilm_velocity(veff2, veff1, k1=0., k2=0.)
u, w = u.filled(0), w.filled(0) # masked values should be zero


# averaging over full runs (5 minutes)
U, W, uw = [], [], []
for run in exp.runs[1:-1]:
    t0 = run.start_time + timedelta(seconds=30)
    t1 = run.end_time - timedelta(seconds=30)
    t0_seconds = (t0 - origin).total_seconds()
    t1_seconds = (t1 - origin).total_seconds()
    mask = (hotfilm_seconds > t0_seconds) & (hotfilm_seconds < t1_seconds)
    um, wm = np.mean(u[mask]), np.mean(w[mask])
    U.append(um)
    W.append(wm)
    uw.append(np.mean((u[mask] - um) * (w[mask] - wm)))
U = np.array(U)
W = np.array(W)
uw = np.array(uw)

# averaging over 60-s bins:
U, W, uw = [], [], []
for run in exp.runs[1:-1]:
    t0 = run.start_time + timedelta(seconds=30)
    t1 = run.end_time - timedelta(seconds=30)
    t0_seconds = (t0 - origin).total_seconds()
    t1_seconds = (t1 - origin).total_seconds()
    binsize = 60.
    for tt in np.arange(t0_seconds, t1_seconds, binsize):
        start, stop = tt, tt + binsize
        mask = (hotfilm_seconds > start) & (hotfilm_seconds < stop)
        um, wm = np.mean(u[mask]), np.mean(w[mask])
        U.append(um)
        W.append(wm)
        uw.append(np.mean((u[mask] - um) * (w[mask] - wm)))
U = np.array(U)
W = np.array(W)
uw = np.array(uw)

ust = np.sqrt(- uw)
Cd = ust**2 / U**2

# u* vs U
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 35))
plt.plot(U, ust, 'k.', ms=12)
plt.xlabel(r'$U_z$ [m/s]')
plt.ylabel(r'$u^*$ [m/s]')
plt.grid()
plt.title(r'Hotfilm $u*$, salt water')
plt.savefig('ust_hotfilm_salt.png', dpi=100)
plt.close(fig)

# u'w' vs U
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 35))
plt.plot(U, -uw, 'k.', ms=12)
plt.xlabel(r'$U_z$ [m/s]')
plt.ylabel(r"$\overline{u'w'}$ [m/s]")
plt.grid()
plt.title(r"Hotfilm $\overline{u'w'}$, salt water")
plt.savefig('uw_hotfilm_salt.png', dpi=100)
plt.close(fig)

# Cd vs U
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 35), ylim=(0, 3e-3))
plt.plot(U, Cd, 'k.', ms=12)
plt.xlabel(r'$U_z$ [m/s]')
plt.ylabel(r"$C_D$")
plt.grid()
plt.title(r"Hotfilm $C_D$, salt water")
plt.savefig('cd_hotfilm_salt.png', dpi=100)
plt.close(fig)
