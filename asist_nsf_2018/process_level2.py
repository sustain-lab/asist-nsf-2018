"""
process_level2.py
"""
from datetime import datetime, timedelta
import numpy as np

def clean_hotfilm_exp1(exp, ch1, ch2):
    """Cleans channels 1 and 2 for asist-windonly-fresh experiment."""

    # treat first spike on channel 1;
    # it occurs between these indices
    i0, i1 = 3252000, 3253000

    # 10-s pre and post spike
    pre, post = ch1[i0-10000:i0], ch1[i1:i1+10000]
    offset = np.mean(post) - np.mean(pre)
    nspike = np.argmax(ch1[i0:i1])
    ch1[i0+nspike:] -= offset # remove main offset caused by spike

    # clean out the rest by fan; min and max bounds by fan speed
    # bounds determined by hand and eye
    ch1_max = {45: 2.24, 50: 2.37, 55: 2.41, 60: 2.51}
    ch1_min = {45: 1.71, 50: 1.80, 55: 1.85, 60: 1.92}
    ch2_max = {45: 2.15, 50: 2.23, 55: 2.31, 60: 2.35}
    ch2_min = {45: 1.75, 50: 1.80, 55: 1.88, 60: 1.93}

    for run in exp.runs[9:13]:
        t0 = run.start_time + timedelta(seconds=1)
        t1 = run.end_time - timedelta(seconds=1)
        origin = datetime(t0.year, t0.month, t0.day)
        t0_seconds = (t0 - origin).total_seconds()
        t1_seconds = (t1 - origin).total_seconds()
        n0 = np.argmin((hotfilm_seconds - t0_seconds)**2)
        n1 = np.argmin((hotfilm_seconds - t1_seconds)**2)
        ch1[n0:n1][ch1[n0:n1] > ch1_max[run.fan]] = ch1_max[run.fan]
        ch2[n0:n1][ch2[n0:n1] > ch2_max[run.fan]] = ch2_max[run.fan]
        ch1[n0:n1][ch1[n0:n1] < ch1_min[run.fan]] = ch1_min[run.fan]
        ch2[n0:n1][ch2[n0:n1] < ch2_min[run.fan]] = ch2_min[run.fan]

    return ch1, ch2


def clean_hotfilm_exp3(exp, ch1, ch2):
    """Cleans channels 1 and 2 for asist-windonly-salt experiment."""

    # treat first spike on channel 1;
    # it occurs between these indices
    i0, i1 = 3622000, 3624000

    # 10-s pre and post spike
    pre, post = ch1[i0-10000:i0], ch1[i1:i1+10000]
    offset = np.mean(post) - np.mean(pre)
    nspike = np.argmax(ch1[i0:i1])
    ch1[i0+nspike:] -= offset # remove main offset caused by spike

    # clean out the rest by fan; min and max bounds by fan speed
    # bounds determined by hand and eye
    ch1_max = {45: 2.26, 50: 2.37, 55: 2.42, 60: 2.52}
    ch1_min = {45: 1.71, 50: 1.80, 55: 1.85, 60: 1.92}
    ch2_max = {45: 2.18, 50: 2.27, 55: 2.33, 60: 2.38}
    ch2_min = {45: 1.71, 50: 1.76, 55: 1.84, 60: 1.88}

    for run in exp.runs[9:13]:
        t0 = run.start_time + timedelta(seconds=1)
        t1 = run.end_time - timedelta(seconds=1)
        origin = datetime(t0.year, t0.month, t0.day)
        t0_seconds = (t0 - origin).total_seconds()
        t1_seconds = (t1 - origin).total_seconds()
        n0 = np.argmin((hotfilm_seconds - t0_seconds)**2)
        n1 = np.argmin((hotfilm_seconds - t1_seconds)**2)
        ch1[n0:n1][ch1[n0:n1] > ch1_max[run.fan]] = ch1_max[run.fan]
        ch2[n0:n1][ch2[n0:n1] > ch2_max[run.fan]] = ch2_max[run.fan]
        ch1[n0:n1][ch1[n0:n1] < ch1_min[run.fan]] = ch1_min[run.fan]
        ch2[n0:n1][ch2[n0:n1] < ch2_min[run.fan]] = ch2_min[run.fan]

    return ch1, ch2
