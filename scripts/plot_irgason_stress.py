#!/usr/bin/env python3
"""
Processes IRGASON velocities and plot drag coefficient.
"""
from asist.irgason import read_irgason_from_netcdf
from asist.utility import limit_to_percentile_range, running_mean
from asist_nsf_2018.experiments import experiments
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib.pyplot as plt

def rotate(u, w, th):
    """Rotates the vector (u, w) by angle th."""
    ur =  np.cos(th) * u + np.sin(th) * w
    wr = -np.sin(th) * u + np.cos(th) * w
    return ur, wr

# path to L2 data
L2_DATA_PATH = os.environ['L2_DATA_PATH']

# experiments to process
exp_names = [
    'asist-windonly-fresh',
    'asist-wind-swell-fresh',
    'asist-windonly-salt',
    'asist-wind-swell-salt'
    ]

# loop over experiments
for exp_name in exp_names:

    exp = experiments[exp_name]

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

    fan_settings = np.arange(0, 65, 5)

    U, W, uw = [], [], []
    for run in exp.runs[:-1]:
        mask = (t > run.start_time + timedelta(seconds=60))\
             & (t < run.end_time)
        um = np.mean(u[mask])
        wm = np.mean(w[mask])
        th = np.arctan2(wm, um)
        ur, wr = rotate(u[mask], w[mask], th)
        um, wm = np.mean(ur), np.mean(wr)
        up, wp = ur - um, wr - wm
        U.append(um)
        W.append(wm)
        uw.append(np.mean(up * wp))
    U, W, uw = map(np.array, [U, W, uw])

    cd = uw / um**2

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, xlim=(0, 60), ylim=(0, 3e-3))
    plt.plot(fan_settings, cd, 'k.', ms=8)
    plt.plot(fan_settings, cd, 'k-', lw=1)
    plt.grid(True)
    plt.xlabel('Fan [Hz]', fontsize=16)
    plt.ylabel(r'$C_D$', fontsize=16)
    plt.title('Drag coefficient, ' + exp_name, fontsize=16)
    plt.savefig('cd_' + exp_name + '.png', dpi=100)
    plt.close(fig)
