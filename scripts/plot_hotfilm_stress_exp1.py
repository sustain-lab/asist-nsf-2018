#!/usr/bin/env python3
"""
Processes raw hotfilm voltages to velocities.
"""
from asist_nsf_2018.experiments import experiments
from asist_nsf_2018.process_level2 import clean_hotfilm_exp1
from asist.utility import binavg, limit_to_percentile_range, running_mean
from asist.hotfilm import effective_velocity, hotfilm_velocity, read_hotfilm_from_netcdf
from asist.pitot import read_pitot_from_netcdf
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

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
plt.plot(hotfilm_seconds, ch1, 'b-', lw=0.5, label='Channel 1')
plt.plot(hotfilm_seconds, ch2, 'r-', lw=0.5, label='Channel 2')
plt.legend(loc='upper left', fancybox=True, shadow=True)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Hot film output [V]')
plt.title('Raw hot film output [V], cleaned, ' + exp_name)
plt.savefig('hotfilm_output_' + exp_name + '.png', dpi=100)
plt.close(fig)

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

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
plt.plot(ch1_binavg, effective_velocity(pitot), 'k.', ms=0.1)
plt.plot(ch1, veff1, 'r.', ms=0.1)
plt.grid(True)
plt.xlabel('Input voltage [V]')
plt.ylabel('Velocity [m/s]')
plt.title('Polyfit of Ch1 -> pitot, ' + exp_name)
plt.savefig('hotfilm_ch1_polyfit_' + exp_name + '.png', dpi=100)
plt.close(fig)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
plt.plot(ch2_binavg, effective_velocity(pitot), 'k.', ms=0.1)
plt.plot(ch2, veff2, 'r.', ms=0.1)
plt.grid(True)
plt.xlabel('Input voltage [V]')
plt.ylabel('Velocity [m/s]')
plt.title('Polyfit of Ch2 -> pitot, ' + exp_name)
plt.savefig('hotfilm_ch2_polyfit_' + exp_name + '.png', dpi=100)
plt.close(fig)

veff1[veff1 < 0.2] = 0.2
veff2[veff2 < 0.2] = 0.2
u, w = hotfilm_velocity(veff1, veff2)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
plt.plot(hotfilm_seconds, u, 'b-', lw=0.1)
plt.plot(hotfilm_seconds, w, 'r-', lw=0.1)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Hot film velocity, ' + exp_name)
plt.savefig('hotfilm_velocity_' + exp_name + '.png', dpi=100)
plt.close(fig)

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
    th = np.arctan2(wm, um)
    up, wp = rotate(u[mask] - um, w[mask] - wm, th)
    uw.append(np.mean(up * wp))
U = np.array(U)
W = np.array(W)
uw = np.array(uw)

ustc = smooth_ust(U, 0.31)
uwc = ustc**2


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
plt.title(r'Hotfilm $u*$, fresh water')
plt.savefig('ust_hotfilm_fresh.png', dpi=100)
plt.close(fig)

# u'w' vs U
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 35))
plt.plot(U, -uw, 'k.', ms=12)
plt.xlabel(r'$U_z$ [m/s]')
plt.ylabel(r"$\overline{u'w'}$ [m/s]")
plt.grid()
plt.title(r"Hotfilm $\overline{u'w'}$, fresh water")
plt.savefig('uw_hotfilm_fresh.png', dpi=100)
plt.close(fig)

# Cd vs U
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 35), ylim=(0, 3e-3))
plt.plot(U, Cd, 'k.', ms=12)
plt.xlabel(r'$U_z$ [m/s]')
plt.ylabel(r"$C_D$")
plt.grid()
plt.title(r"Hotfilm $C_D$, fresh water")
plt.savefig('cd_hotfilm_fresh.png', dpi=100)
plt.close(fig)
