#!/usr/bin/env python3
"""
Prints run fan speed and start and end time 
in UTC for each experiment.
"""
from asist_nsf_2018.experiments import experiments

for exp_name in experiments:
    print(exp_name)
    exp = experiments[exp_name]
    for run in exp.runs:
        print('%2.2i' % run.fan + ' Hz', run.start_time, run.end_time)
