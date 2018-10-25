#!/usr/bin/env python3
"""
Prints run fan speed and start and end time 
in UTC for each experiment.
"""
from asist_nsf_2018.experiments import experiments

for exp_name in experiments:
    print(exp_name)
    exp = experiments[exp_name]
    print(' | '.join(['Fan [Hz]', 'Start time', 'End time']))
    print(' | '.join(['--------'] * 3))
    for run in exp.runs:
        print(' | '.join(['%2.2i' % run.fan, 
            run.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            run.end_time.strftime('%Y-%m-%d %H:%M:%S')]))
