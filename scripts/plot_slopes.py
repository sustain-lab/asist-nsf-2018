#!/usr/bin/env python3

from asist_nsf_2018.experiments import experiments
from asist.wave_probe import read_wave_probe_csv
from asist.utility import running_mean
from asist.pressure import read_pressure_from_netcdf
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import detrend
from datetime import datetime, timedelta

plt.rcParams.update({'font.size': 16})

def get_run_elevations(eta, start_index, n, run_length=360, frequency=100, offset=30):
    n0 = start_index + n * run_length * frequency + offset * frequency
    n1 = n0 + run_length * frequency - 2 * offset * frequency
    return eta[n0:n1]

def demean(x):
    return x - np.mean(x)

def variance_spectrum(eta, sampling_rate):
    e = demean(detrend(eta))
    n = e.size
    f = np.fft.fftfreq(n, 1 / sampling_rate)[:n//2]
    df = 2 * sampling_rate / e.size
    ai = 2 * np.abs(np.fft.fft(e)[:n//2]) / n
    F = ai**2 / 2 / df
    return F, f, df

def sig_wave_height(F, df):
    """Significant wave height [m]."""
    return 4 * np.sqrt(np.sum(F * df))

def mean_wave_period(F, f, df):
    """First-order mean wave period [s]."""
    return np.sum(F * df) / np.sum(F * f * df)

def wave_energy(F, df, rhow=1030, g=9.8):
    """Returns total wave energy."""
    return rhow * g * np.sum(F * df)

def mean_water_height(eta, exp, start_index):
    fan, h = [], []
    for n, run in enumerate(exp.runs[:-1]):
        fan.append(run.fan)
        e = get_run_elevations(eta, start_index, n)
        if n == 0:
            offset = np.mean(e)
            h.append(0)
        else:
            h.append(np.mean(e) - offset)
    return np.array(fan), np.array(h)

def mean_slope(h1, h2, dx, rhow=1030, g=9.8, depth=0.42):
    h1, h2 = np.array(h1), np.array(h2)
    hmean = 0.5 * (h1 + h2) + depth
    return rhow * g * hmean * (h2 - h1) / dx

def get_wave_properties(eta, exp, start_index):
    fan, swh, mwp, Sxx = [], [], [], []
    for n, run in enumerate(exp.runs[:-1]):
        e = get_run_elevations(eta, start_index, n)
        F, f, df = variance_spectrum(e, 100)
        fan.append(run.fan)
        swh.append(sig_wave_height(F, df))
        mwp.append(mean_wave_period(F, f, df))
        Sxx.append(0.5 * wave_energy(F, df))
    return np.array(fan), np.array(swh), np.array(mwp), np.array(Sxx)

path = os.environ['WAVEPROBE_DATA_PATH']
L2_DATA_PATH = os.environ['L2_DATA_PATH']

# experiments to process
exp_names = [
    'asist-windonly-fresh',
    'asist-wind-swell-fresh',
    'asist-windonly-salt',
    'asist-wind-swell-salt'
    ]

frequency = 100 # Hz
run_length = 360 # s
dx = 2.7 # distance between probes
dx_pressure = 6.12 # distance between pressure ports

for exp_name in exp_names:

    if exp_name == 'asist-windonly-fresh':
        known_index = 241500
        start_index_fan = 10
    elif exp_name == 'asist-wind-swell-fresh':
        known_index = 430000
        start_index_fan = 60
    elif exp_name == 'asist-windonly-salt':
        known_index = 449000
        start_index_fan = 60
    elif exp_name == 'asist-wind-swell-salt':
        known_index = 426000
        start_index_fan = 60
    else:
        raise NotImplementedError()

    exp = experiments[exp_name]

    start_time, time, eta3 = read_wave_probe_csv(path + '/' + exp_name + '/ch3.csv')
    start_time, time, eta4 = read_wave_probe_csv(path + '/' + exp_name + '/ch4.csv')
    start_time, time, eta6 = read_wave_probe_csv(path + '/' + exp_name + '/ch6.csv')

    start_index = known_index - (start_index_fan // 5)\
                              * run_length * frequency
    start_index = 0 if start_index < 0 else start_index

    fan, swh3, mwp3, Sxx3 = get_wave_properties(eta3, exp, start_index)
    fan, swh4, mwp4, Sxx4 = get_wave_properties(eta4, exp, start_index)
    fan, swh6, mwp6, Sxx6 = get_wave_properties(eta6, exp, start_index)

    fan, h3 = mean_water_height(eta3, exp, start_index)
    fan, h4 = mean_water_height(eta4, exp, start_index)
    fan, h6 = mean_water_height(eta6, exp, start_index)

    # mean slopes
    s3 = mean_slope(h3, h6, dx)
    s4 = mean_slope(h4, h6, dx)

    # radiation stress
    Sxx3, Sxx4, Sxx6 = map(np.array, [Sxx3, Sxx4, Sxx6])

    # radiation stress gradient
    dSdx3 = (Sxx6 - Sxx3) / dx
    dSdx4 = (Sxx6 - Sxx4) / dx

    # pressure gradient
    origin, seconds, _, dp = read_pressure_from_netcdf(L2_DATA_PATH + '/pitot_' + exp_name + '.nc')

    dp[dp > 600] = np.nan
    dp[dp < -600] = np.nan

    dpm, f = [], []
    for run in exp.runs[:-1]:
        t0 = run.start_time + timedelta(seconds=30)
        t1 = run.end_time - timedelta(seconds=30)
        t0_seconds = (t0 - origin).total_seconds()
        t1_seconds = (t1 - origin).total_seconds()
        mask = (seconds > t0_seconds) & (seconds < t1_seconds)
        dpm.append(np.mean(dp[mask]))
        f.append(run.fan)
    dpm = np.array(dpm)

    dpdx = dpm / dx_pressure

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, xlim=(0, 60), ylim=(0, 0.35))
    plt.plot(fan, swh3, color='b', marker='.', ms=12, label='Fetch 6 m, left')
    plt.plot(fan, swh4, color='g', marker='.', ms=12, label='Fetch 6 m, right')
    plt.plot(fan, swh6, color='r', marker='.', ms=12, label='Fetch 8.7 m')
    plt.legend(loc='upper left', fancybox=True, shadow=True)
    plt.grid()
    plt.xlabel('Fan speed [Hz]')
    plt.ylabel('Sig. wave height [m]')
    plt.title(exp_name)
    plt.savefig('swh_' + exp_name + '.png', dpi=100)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, xlim=(0, 60), ylim=(0, 1))
    plt.plot(fan, mwp3, color='b', marker='.', ms=12, label='Fetch 6 m, left')
    plt.plot(fan, mwp4, color='g', marker='.', ms=12, label='Fetch 6 m, right')
    plt.plot(fan, mwp6, color='r', marker='.', ms=12, label='Fetch 8.7 m')
    plt.legend(loc='lower right', fancybox=True, shadow=True)
    plt.grid()
    plt.xlabel('Fan speed [Hz]')
    plt.ylabel('Mean wave period [s]')
    plt.title(exp_name)
    plt.savefig('mwp_' + exp_name + '.png', dpi=100)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, xlim=(0, 60), ylim=(-20, 60))
    plt.plot(fan, s3, color='b', marker='.', ms=12, label='Fetch 6 m, left')
    plt.plot(fan, s4, color='g', marker='.', ms=12, label='Fetch 6 m, right')
    plt.plot(fan, dpdx, color='k', marker='.', ms=12, label='dp/dx')
    plt.legend(loc='upper left', fancybox=True, shadow=True)
    plt.grid()
    plt.xlabel('Fan speed [Hz]')
    plt.ylabel('Mean slope')
    plt.title(exp_name)
    plt.savefig('slope_' + exp_name + '.png', dpi=100)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111,xlim=(0, 60),  ylim=(-8, 4))
    plt.plot(fan, dSdx3, color='b', marker='.', ms=12, label='Fetch 6 m, left')
    plt.plot(fan, dSdx4, color='g', marker='.', ms=12, label='Fetch 6 m, right')
    plt.plot([0, 60], [0, 0], 'k--')
    plt.legend(loc='upper left', fancybox=True, shadow=True)
    plt.grid()
    plt.xlabel('Fan speed [Hz]')
    plt.ylabel(r'$\dfrac{\partial S_{xx}}{\partial x}$')
    plt.title(exp_name)
    plt.savefig('Sxx_gradient_' + exp_name + '.png', dpi=100)
    plt.close(fig)


    tau3 = s3 + dSdx3 + 0.5 * (h3 + h6) * dpdx
    tau4 = s4 + dSdx4 + 0.5 * (h4 + h6) * dpdx

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111,xlim=(0, 60),  ylim=(-20, 60))
    plt.plot(fan, tau3, color='b', marker='.', ms=12, label='Fetch 6 m, left')
    plt.plot(fan, tau4, color='g', marker='.', ms=12, label='Fetch 6 m, right')
    plt.legend(loc='upper left', fancybox=True, shadow=True)
    plt.grid()
    plt.xlabel('Fan speed [Hz]')
    plt.ylabel(r'$\tau$')
    plt.title(exp_name)
    plt.savefig('tau_' + exp_name + '.png', dpi=100)
    plt.close(fig)

    rhoa = 1.2
    cd3 = tau3 / (rhoa * fan**2)
    cd4 = tau4 / (rhoa * fan**2)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, xlim=(0, 60), ylim=(-2e-2, 2e-2))
    plt.plot(fan, cd3, color='b', marker='.', ms=12, label='Fetch 6 m, left')
    plt.plot(fan, cd4, color='g', marker='.', ms=12, label='Fetch 6 m, right')
    plt.plot([0, 60], [0, 0], 'k--')
    plt.legend(loc='lower right', fancybox=True, shadow=True)
    plt.grid()
    plt.xlabel('Fan speed [Hz]')
    plt.ylabel(r'$C_D$')
    plt.title(exp_name)
    plt.savefig('cd_' + exp_name + '.png', dpi=100)
    plt.close(fig)
