#!/usr/bin/env python3

from asist.wave_probe import read_wave_probe_csv
from asist.utility import running_mean
import matplotlib.pyplot as plt
import os
from scipy.signal import detrend

path = os.environ['WAVEPROBE_DATA_PATH']

# experiments to process
exp_names = [
    'asist-windonly-fresh',
    'asist-wind-swell-fresh',
    'asist-windonly-salt',
    'asist-wind-swell-salt'
    ]

# loop over experiments
for exp_name in exp_names:

    print(exp_name)

    start_time, time, eta3 = read_wave_probe_csv(path + '/' + exp_name + '/ch3.csv')
    start_time, time, eta4 = read_wave_probe_csv(path + '/' + exp_name + '/ch4.csv')
    start_time, time, eta6 = read_wave_probe_csv(path + '/' + exp_name + '/ch6.csv')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, ylim=(-0.05, 0.05))
    plt.plot(time, running_mean(eta3, 6000), label='Ch3')
    plt.plot(time, running_mean(eta4, 6000), label='Ch4')
    plt.plot(time, running_mean(eta6, 6000), label='Ch6')
    plt.legend(loc='upper left', fancybox=True, shadow=True)
    plt.grid()
    plt.title(exp_name)
    plt.savefig('wave_probe_' + exp_name + '.png', dpi=100)
    plt.close(fig)
