#!/usr/bin/env python3
"""
Processes raw hotfilm voltages to velocities.
"""
from asist_nsf_2018.experiments import experiments
from asist.utility import binavg, limit_to_percentile_range, running_mean
from asist.pressure import read_pressure_from_netcdf
from datetime import datetime, timedelta
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16}) # global font size setting

L2_DATA_PATH = os.environ['L2_DATA_PATH']

exp_names = [
    'asist-windonly-fresh',
    'asist-wind-swell-fresh',
    'asist-windonly-salt',
    'asist-wind-swell-salt'
    ]

# distance between 2 pressure ports
dx = 8 * 0.77

col = ['b', 'g', 'r', 'm']

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 60), ylim=(-40, 20))

for n, exp_name in enumerate(exp_names):
    exp = experiments[exp_name]
    origin, seconds, fan, dp = read_pressure_from_netcdf(L2_DATA_PATH + '/pitot_' + exp_name + '.nc')

    dp[dp > 600] = np.nan
    dp[dp < -600] = np.nan

    dpm, f = [], []
    for run in exp.runs[:-1]:
        t0 = run.start_time + timedelta(seconds=60)
        t1 = run.end_time - timedelta(seconds=60)
        t0_seconds = (t0 - origin).total_seconds()
        t1_seconds = (t1 - origin).total_seconds()
        mask = (seconds > t0_seconds) & (seconds < t1_seconds)
        dpm.append(np.mean(dp[mask]))
        f.append(run.fan)
    dpm = np.array(dpm)

    plt.plot(f, dpm / dx, col[n] + '-', label=exp_name)
    plt.plot(f, dpm / dx, col[n] + '.', ms=12)

plt.plot([0, 60], [0, 0], 'k--')
plt.grid(True)
plt.xlabel('Fan [Hz]')
plt.ylabel('Pressure difference [Pa]')
plt.legend(loc='lower left', fancybox=True, shadow=True)
plt.title('Along-tank dp [Pa], ' + exp_name)
plt.savefig('dpdx.png', dpi=100)
plt.close(fig)
