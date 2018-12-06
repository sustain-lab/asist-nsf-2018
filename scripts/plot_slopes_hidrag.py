#!/usr/bin/env python3

from asist_nsf_2018.experiments import experiments
from asist.wave_probe import read_wave_probe_csv
from asist.utility import running_mean
from asist.pressure import read_pressure_from_netcdf
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, date2num
import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import detrend
from datetime import datetime, timedelta
from dispersion import w2k
from netCDF4 import Dataset
from process_leg import leg1, leg2, leg_slope

plt.rcParams.update({'font.size': 12})

def get_run_elevations(eta, start_index, n, run_length=360, frequency=100, offset=30):
    n0 = start_index + n * run_length * frequency + offset * frequency
    n1 = n0 + run_length * frequency - 2 * offset * frequency
    return eta[n0:n1]

def demean(x):
    return x - np.mean(x)

def variance_spectrum(eta, sampling_rate, fmin=0.1, fmax=100):
    e = demean(detrend(eta))
    n = e.size
    f = np.fft.fftfreq(n, 1 / sampling_rate)[:n//2]
    #df = 2 * sampling_rate / e.size
    df = sampling_rate / e.size
    ai = 2 * np.abs(np.fft.fft(e)[:n//2]) / n
    F = ai**2 / 2 / df
    mask = (f >= fmin) & (f < fmax)
    return F[mask], f[mask], df

def cp_cg(F, f, df, depth):
    w = 2 * np.pi * f
    k = w2k(w, depth)[0]
    cp = w[1:] / k[1:]
    cg = np.diff(w) / np.diff(k)
    return cp, cg

def sig_wave_height(F, df):
    """Significant wave height [m]."""
    return 4 * np.sqrt(np.sum(F * df))

def mean_wave_period(F, f, df):
    """First-order mean wave period [s]."""
    return np.sum(F * df) / np.sum(F * f * df)

def wave_energy(F, df, rhow=1000, g=9.8):
    """Returns total wave energy."""
    return rhow * g * np.sum(F * df)

def radiation_stress(F, f, df, depth, rhow=1000, g=9.8):
    """Returns radiation stress."""
    cp, cg = cp_cg(F, f, df, depth)
    rad_stress_fac = 2 * cg / cp - 0.5
    return rhow * g * np.sum(rad_stress_fac * F[1:] * df)

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

def mean_slope(h1, h2, dx, rhow=1000, g=9.8, depth=0.42):
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
        Sxx.append(radiation_stress(F, f, df, 0.42))
    return np.array(fan), np.array(swh), np.array(mwp), np.array(Sxx)

path = os.environ['WAVEPROBE_DATA_PATH']
L2_DATA_PATH = os.environ['L2_DATA_PATH']
hidrag_path = '/home/milan/Work/sustain/data/hidrag'

frequency = 100 # Hz
run_length = 360 # s

fetch1_c18 = 6.02
fetch2_c18 = 8.71
fetch1_d04 = 4.592
fetch2_d04 = 8.991

dx_c18 = fetch2_c18 - fetch1_c18
dx_d04 = fetch2_d04 - fetch1_d04

# experiments to process
exp_name = 'asist-windonly-fresh'
exp = experiments[exp_name]
known_index = 241500
start_index_fan = 10

_, time, eta3 = read_wave_probe_csv(path + '/' + exp_name + '/ch3.csv')
_, time, eta4 = read_wave_probe_csv(path + '/' + exp_name + '/ch4.csv')
_, time, eta6 = read_wave_probe_csv(path + '/' + exp_name + '/ch6.csv')

start_index = known_index - (start_index_fan // 5)\
                          * run_length * frequency
start_index = 0 if start_index < 0 else start_index
    
fan, swh3, mwp3, Sxx3 = get_wave_properties(eta3, exp, start_index)
fan, swh4, mwp4, Sxx4 = get_wave_properties(eta4, exp, start_index)
fan, swh6, mwp6, Sxx6 = get_wave_properties(eta6, exp, start_index)

# radiation stress
Sxx3, Sxx4, Sxx6 = map(np.array, [Sxx3, Sxx4, Sxx6])

# radiation stress gradient
dSdx3 = (Sxx6 - Sxx3) / dx_c18
dSdx4 = (Sxx6 - Sxx4) / dx_c18

fan, h3 = mean_water_height(eta3, exp, start_index)
fan, h4 = mean_water_height(eta4, exp, start_index)
fan, h6 = mean_water_height(eta6, exp, start_index)

# mean slopes
s3 = (h6 - h3) / dx_c18
s4 = (h6 - h4) / dx_c18

# air pressure gradient
with Dataset(L2_DATA_PATH + '/air-pressure_asist-christian-shadowgraph.nc') as nc:
    seconds = nc.variables['Time'][:]
    seconds -= seconds[0]
    origin = datetime.strptime(nc.variables['Time'].origin, '%Y-%m-%dT%H:%M:%S')
    time_air = np.array([origin + timedelta(seconds=s) for s in seconds])
    dpdx_air = nc.variables['dpdx'][:]
    fan_air = nc.variables['fan'][:]

exp = experiments['asist-christian-shadowgraph']

dpdx_c18 = []
for run in exp.runs[:-1]:
    t0 = run.start_time + timedelta(seconds=30)
    t1 = run.end_time - timedelta(seconds=30)
    mask = (time_air > t0) & (time_air < t1)
    dpdx_c18.append(np.mean(dpdx_air[mask]))

rhow = 1000
rhoa = 1.15
g = 9.8
depth = 0.42

dpdx_c18 = - np.array(dpdx_c18) / (rhow * g)

### HIDRAG data

# Location of probes from entrance to tank
ps1 = 3.014
ps2 = 7.012
ps3 = 11.009

mat = loadmat(hidrag_path + '/uwvsu2-24.mat')
U = mat['ups'][0]
uw = mat['uw'][0]
LEG3 = mat['LEG3'][0] * 1e-2
LEG1 = mat['LEG1'][0] * 1e-2
M1 = mat['M1'][0]
M3 = mat['M3'][0]

ps13 = - mat['ps13'][0] * 1e-2
ps12 = - mat['ps12'][0] * 1e-2
dpdx = ps13 / (ps3 - ps1)
dpdx -= dpdx[0]

LEG3 -= LEG3[0]
LEG1 -= LEG1[0]

U = np.array([0] + list(U))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 25))
plt.plot(U[1:], LEG1, 'b--', marker='o', ms=5, label='D04, #1, 4.6 m')
plt.plot(U[1:], LEG3, 'b-', marker='o', ms=5, label='D04, #2, 9.0 m')
plt.plot(U, h4, 'r--', marker='o', ms=5, label='C18, #1, 6.0 m')
plt.plot(U, h6, 'r-', marker='o', ms=5, label='C18, #2, 8.7 m')
plt.plot(U, leg1, 'r--', marker='*', ms=10, label='C18, LEG1')
plt.plot(U, leg2, 'r-', marker='*', ms=10, label='C18, LEG2')
plt.plot([0, 50], [0, 0], 'k--')
plt.legend(loc='lower left', fancybox=True, shadow=True)
plt.grid()
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Mean surface elevation [m]')
plt.title('Mean elevation as function of wind speed')
plt.savefig('HIDRAG_elevation.png', dpi=100)
plt.close(fig)

slope_d04 = (LEG3 - LEG1) / dx_d04
slope_c18 = (h6 - h4) / dx_c18

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 25))
plt.plot(U[1:], slope_d04, 'b-', marker='o', ms=5, label='D04 dh/dx')
plt.plot(U[1:], dpdx, 'b-', marker='*', ms=8, label='D04 dp/dx')
plt.plot(U, slope_c18, 'r-', marker='o', ms=5, label='C18 dh/dx')
plt.plot(U, leg_slope, 'r-', marker='s', ms=5, label='C18 LEG dh/dx')
plt.plot(U[1:], dpdx_c18[1:], 'r-', marker='*', ms=8, label='C18 dp/dx')
plt.plot([0, 50], [0, 0], 'k--')
plt.legend(loc='upper left', fancybox=True, shadow=True)
plt.grid()
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Elevation and pressure slope')
plt.title('Elevation and pressure slope vs wind speed')
plt.savefig('HIDRAG_slope.png', dpi=100)
plt.close(fig)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 25))
plt.plot(U[1:], M1, 'b--', marker='o', ms=5, label='D04, #1, 4.6 m')
plt.plot(U[1:], M3, 'b-', marker='o', ms=5, label='D04, #2, 9.0 m')
plt.plot(U, Sxx4, 'r--', marker='o', ms=5, label='C18, #1, 6.0 m')
plt.plot(U, Sxx6, 'r-', marker='o', ms=5, label='C18, #2, 8.7 m')
plt.legend(loc='upper left', fancybox=True, shadow=True)
plt.grid()
plt.plot([0, 50], [0, 0], 'k--')
plt.xlabel('Wind speed [m/s]')
plt.ylabel(r'$S_{xx}$ [$kg/s^3$]')
plt.title('Radiation stress $S_{xx}$ vs wind speed')
plt.savefig('HIDRAG_Sxx.png', dpi=100)
plt.close(fig)

dSdx_d04 = (M3 - M1) / dx_d04
dSdx_c18 = (Sxx6 - Sxx4) / dx_c18

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 25))
plt.plot(U[1:], dSdx_d04, 'b-', marker='o', ms=5, label='D04')
plt.plot(U, dSdx_c18, 'r-', marker='o', ms=5, label='C18')
plt.legend(loc='upper left', fancybox=True, shadow=True)
plt.grid()
plt.plot([0, 50], [0, 0], 'k--')
plt.xlabel('Wind speed [m/s]')
plt.ylabel(r'$dS_{xx}/dx$ [$N/m^2$]')
plt.title('Radiation stress gradient $dS_{xx}/dx$ vs wind speed')
plt.savefig('HIDRAG_dSdx.png', dpi=100)
plt.close(fig)

# Bottom stress from Brian
taub = rhow * np.array([.0007, .0014, .0013, .0025, .0030, .0038, .0054, .0040, .0061, .01, .0052, 0.0046])**2
taub_c18 = np.array([0] + list(taub))

cd_d04 = (rhow * g * depth * (slope_d04 + dpdx) + dSdx_d04 - taub) / rhoa / (2 * U[1:])**2
cd_c18 = (rhow * g * depth * (slope_c18 + dpdx_c18) + dSdx_c18 - taub_c18) / rhoa / (2 * U)**2
cd_leg = (rhow * g * depth * (leg_slope + dpdx_c18) + dSdx_c18 - taub_c18) / rhoa / (2 * U)**2


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, ylim=(-5e-3, 1.5e-2), xlim=(0, 50))
plt.plot(2 * U[1:], cd_d04, 'b-', marker='o', ms=5, label='D04')
plt.plot(2 * U, cd_c18, 'r-', marker='o', ms=5, label='C18')
plt.plot(2 * U, cd_leg, 'r-', marker='*', ms=10, label='C18 LEG')
plt.legend(loc='lower right', fancybox=True, shadow=True)
plt.grid()
plt.xlabel('Wind speed [m/s]')
plt.ylabel(r'$C_{D10}$')
plt.plot([0, 50], [0, 0], 'k--')
plt.title('Drag coefficient vs wind speed')
plt.savefig('HIDRAG_cd.png', dpi=100)
plt.close(fig)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, xlim=(0, 25))
plt.plot(U[1:], rhow * g * depth * slope_d04, 'b-', marker='o', ms=5, label='D04, dh/dx')
plt.plot(U[1:], rhow * g * depth * dpdx, 'b-', marker='*', ms=8, label='D04, dp/dx')
plt.plot(U[1:], taub, 'b-', marker='s', ms=8, label=r'D04, $\tau_b$')
plt.plot(U[1:], dSdx_d04, 'b-', marker='v', ms=8, label='D04, dSxx/dx')
plt.plot(U, rhow * g * depth * slope_c18, 'r-', marker='o', ms=5, label='C18, dh/dx')
plt.plot(U, dSdx_c18, 'r-', marker='v', ms=8, label='C18, dSxx/dx')
plt.legend(loc='upper left', fancybox=True, shadow=True)
plt.grid()
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Stress [$N/m^2$]')
plt.plot([0, 50], [0, 0], 'k--')
plt.title('Stress components vs wind speed')
plt.savefig('HIDRAG_mom_budget.png', dpi=100)
plt.close(fig)
