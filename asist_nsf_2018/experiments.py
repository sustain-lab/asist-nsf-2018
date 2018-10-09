"""
experiments.py
"""
from datetime import datetime, timedelta
from asist.experiment import Experiment, Run

fan = list(range(0, 65, 5)) + [0]
run_length = timedelta(seconds=360)

# Exp0 (pre-experiment) on 2018-09-24
exp0 = Experiment('asist-windonly-fresh-warmup')
exp_start_time = datetime(2018, 9, 24, 19, 10, 0)
for n, f in enumerate(fan):
    start_time = exp_start_time + run_length * n
    end_time = start_time + run_length
    exp0.add_run(Run(start_time, end_time, fan=f))

# Exp1 (fresh water, wind only) on 2018-09-26
exp1 = Experiment('asist-windonly-fresh')
exp_start_time = datetime(2018, 9, 26, 19, 18, 0)
for n, f in enumerate(fan):
    start_time = exp_start_time + run_length * n
    end_time = start_time + run_length
    exp1.add_run(Run(start_time, end_time, fan=f))

# Exp2 (fresh water, wind and swell) on 2018-09-27
exp2 = Experiment('asist-wind-swell-fresh')
exp_start_time = datetime(2018, 9, 27, 14, 42, 0)
for n, f in enumerate(fan):
    start_time = exp_start_time + run_length * n
    end_time = start_time + run_length
    exp2.add_run(Run(start_time, end_time, fan=f, paddle_amplitude=3., paddle_frequency=1.))

# Exp3 (salt water, wind only) on 2018-10-01
exp3 = Experiment('asist-windonly-salt')
exp_start_time = datetime(2018, 10, 1, 17, 20, 0)
for n, f in enumerate(fan):
    start_time = exp_start_time + run_length * n
    end_time = start_time + run_length
    exp3.add_run(Run(start_time, end_time, fan=f))

# Exp4 (salt water, wind and swell) on 2018-10-01
exp4 = Experiment('asist-wind-swell-salt')
exp_start_time = datetime(2018, 10, 1, 19, 54, 0)
for n, f in enumerate(fan):
    start_time = exp_start_time + run_length * n
    end_time = start_time + run_length
    exp4.add_run(Run(start_time, end_time, fan=f, paddle_amplitude=3., paddle_frequency=1.))

# Exp5 (fresh water, flow distortion experiment) on 2018-10-02
exp5 = Experiment('asist-flow-distortion')
exp_start_time = datetime(2018, 10, 2, 17, 56, 0)
run_length = timedelta(seconds=120)
for n, f in enumerate(fan):
    start_time = exp_start_time + run_length * n
    end_time = start_time + run_length
    exp5.add_run(Run(start_time, end_time, fan=f))

# Exp6 (fresh water, vertical profile, fan = 30 Hz) on 2018-10-02
exp6 = Experiment('asist-vertical-profile')
exp_start_time = datetime(2018, 10, 2, 20, 50, 0)
run_length = timedelta(seconds=120)
heights = [0.07 + (0.05 * n) for n in range(10)]
for n, h in enumerate(heights):
    start_time = exp_start_time + run_length * n
    end_time = start_time + run_length
    run = Run(start_time, end_time, fan=30)
    run.pitot_height = h
    exp6.add_run(run)

# Exp7 (fresh water, cross-tank profile, fan = 30 Hz) on 2018-10-04
exp7 = Experiment('asist-crosstank-profile')
exp_start_time = datetime(2018, 10, 4, 19, 0, 0)
run_length = timedelta(seconds=60)
# from left to right looking downwind
cross_distances = [0.075, 0.090, 0.110, 0.125, 0.150, 0.180, 0.235, 0.290,
                   0.345, 0.400, 0.455, 0.505, 0.565, 0.625, 0.675, 0.730,
                   0.785, 0.845, 0.870, 0.895, 0.915, 0.930, 0.950]
for n, d in enumerate(cross_distances):
    start_time = exp_start_time + 2 * run_length * n
    end_time = start_time + run_length
    run = Run(start_time, end_time, fan=30)
    run.pitot_height = 0.42
    run.pitot_cross_distance = d
    exp7.add_run(run)

experiments = {
    'asist-windonly-fresh_warmup': exp0,
    'asist-windonly-fresh': exp1,
    'asist-wind-swell-fresh': exp2,
    'asist-windonly-salt': exp3,
    'asist-wind-swell-salt': exp4,
    'asist-flow-distortion': exp5,
    'asist-vertical-profile': exp6,
    'asist-crosstank-profile': exp7,
}
