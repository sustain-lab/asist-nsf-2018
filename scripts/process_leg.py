#!/usr/bin/env python3

from asist_nsf_2018.experiments import experiments
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat
from datetime import datetime, timedelta

LEG_DATA_PATH = '/home/milan/Work/sustain/data/asist-nsf-2018/L1/LEG'
dt = 1 / 60

mat = loadmat(LEG_DATA_PATH + '/LEG_asist-windonly-fresh-warmup-1.mat')
eta1 = mat['eta_scaled'][0,:] * 1e-2
start_time = datetime(2018, 9, 24, 19, 10, 12, 330000)
time1 = np.array([start_time + n * timedelta(seconds=dt) 
    for n in range(eta1.size)])

mat = loadmat(LEG_DATA_PATH + '/LEG_asist-windonly-fresh-warmup-2.mat')
eta2 = mat['eta_interp'][0,:] * 1e-2
start_time = datetime(2018, 9, 24, 19, 10, 12, 330000)
time2 = np.array([start_time + n * timedelta(seconds=dt) 
    for n in range(eta2.size)])

def clean_leg_data(a):
    """Clean local minima and maxima that exceed a threshold."""
    x = np.array(a)
    for i in range(1, x.size-1):
        if (a[i] > a[i-1] and a[i] > a[i+1])\
            or (a[i] < a[i-1] and a[i] < a[i+1]):
                x[i] = 0.5 * (a[i-1] + a[i+1])
    return x

eta1 = clean_leg_data(eta1)
eta2 = clean_leg_data(eta2)

exp = experiments['asist-windonly-fresh_warmup']

# compute offsets
run = exp.runs[0]
t0 = run.start_time + timedelta(seconds=30)
t1 = run.end_time - timedelta(seconds=30)
mask = (time1 > t0) & (time1 < t1)
offset1 = np.nanmean(eta1[mask])
mask = (time2 > t0) & (time2 < t1)
offset2 = np.nanmean(eta2[mask])

leg1 = []
for run in exp.runs[:-1]:
    t0 = run.start_time + timedelta(seconds=30)
    t1 = run.end_time - timedelta(seconds=30)
    mask = (time1 > t0) & (time1 < t1)
    leg1.append(np.nanmean(eta1[mask]) - offset1)

leg2 = []
for run in exp.runs[:-1]:
    t0 = run.start_time + timedelta(seconds=30)
    t1 = run.end_time - timedelta(seconds=30)
    mask = (time2 > t0) & (time2 < t1)
    leg2.append(np.nanmean(eta2[mask]) - offset2)

# fix 5 Hz run at LEG2:
leg2[1] = 0.5 * (leg2[0] + leg2[2])

dx_leg = 5 * 0.77
leg_slope = (np.array(leg2) - np.array(leg1)) / dx_leg
