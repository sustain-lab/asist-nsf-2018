"""
process_level1.py
"""
from asist.irgason import read_irgason_from_toa5
from asist.hotfilm import read_hotfilm_from_lvm
from asist.pressure import read_pressure_from_toa5
from asist.pitot import pitot_velocity
from asist_nsf_2018.experiments import experiments
from datetime import datetime, timedelta
import glob
from matplotlib.dates import date2num, num2date
from netCDF4 import Dataset
import numpy as np
import os


def get_data_path(data_name):
    """Gets the data path from the env variable."""
    assert data_name in ['LEG', 'HOTFILM', 'IRGASON', 'PRESSURE'],\
        data_name + ' is not available'
    try:
        return os.environ[data_name + '_DATA_PATH']
    except KeyError:
        raise KeyError('Set ' + data_name + '_DATA_PATH env variable to the path with ' + data_name + ' data')


def get_experiment_time_series(time, data, exp):
    """Returns time and data slice between 
    experiment start and end times."""
    t0, t1 = exp.runs[0].start_time, exp.runs[-1].end_time
    mask = (time >= t0) & (time <= t1)
    return time[mask], data[mask]


def process_dp_to_level2():
    """Processes pressure gradient into a NetCDF file."""
    PRESSURE_DATA_PATH = get_data_path('PRESSURE')

    exp_name = 'asist-christian-shadowgraph'
    exp = experiments[exp_name]
    
    files = glob.glob(PRESSURE_DATA_PATH + '/TOA5_SUSTAINpresX4X2.pressure_229*.dat')
    files.sort()
    time, dp1, dp2 = read_pressure_from_toa5(files)

    # remove offset from pressure
    t0 = exp.runs[0].start_time
    t1 = exp.runs[0].end_time - timedelta(seconds=60)
    mask = (time >= t0) & (time <= t1)
    dp2_offset = np.mean(dp2[mask])
    for run in exp.runs:
        run_mask = (time >= run.start_time) & (time <= run.end_time)
        dp2[run_mask] -= dp2_offset

    time, dp = get_experiment_time_series(time, dp2, exp)
        
    # fan frequency
    fan = np.zeros(time.size)
    for run in exp.runs:
        run_mask = (time >= run.start_time) & (time <= run.end_time)
        fan[run_mask] = run.fan

    print('Writing ' + ncfile)

    # distance between air pressure ports
    # 14 panels, 0.77 m each, minus 2 cm on each end
    dx = 14 * 0.77 - 0.04 

    seconds = (date2num(time) - int(date2num(t0))) * 86400

    ncfile = 'air-pressure_' + exp_name + '.nc'
    with Dataset(ncfile, 'w', format='NETCDF4') as nc:

        nc.createDimension('Time', size=0)
        var = nc.createVariable('Time', 'f8', dimensions=('Time'))
        var[:] = seconds
        var.setncattr('name', 'Time in seconds of the day')
        var.setncattr('units', 's')
        var.setncattr('origin', t0.strftime('%Y-%m-%dT%H:%M:%S'))
        var.setncattr('dx', dx)

        var = nc.createVariable('fan', 'f4', dimensions=('Time'))
        var[:] = fan
        var.setncattr('name', 'Fan frequency')
        var.setncattr('units', 'Hz')

        var = nc.createVariable('dp', 'f4', dimensions=('Time'))
        var[:] = dp
        var.setncattr('name', 'Along-tank air pressure difference')
        var.setncattr('units', 'Pa')

        var = nc.createVariable('dpdx', 'f4', dimensions=('Time'))
        var[:] = dp / dx
        var.setncattr('name', 'Along-tank air pressure gradient')
        var.setncattr('units', 'Pa / m')


def process_hotfilm_to_level2():
    """Processes Hot film Labview files into NetCDF."""
    HOTFILM_DATA_PATH = get_data_path('HOTFILM')

    experiments_to_process = [
        'asist-windonly-fresh_warmup',
        'asist-windonly-fresh', 
        'asist-windonly-salt' 
    ]

    for exp_name in experiments_to_process:
        exp = experiments[exp_name]

        filename = HOTFILM_DATA_PATH + '/hot_film_'\
            + exp.runs[0].start_time.strftime('%Y%m%d')  + '.lvm'

        if exp_name == 'asist-windonly-fresh_warmup':
            start_time, seconds, ch1, ch2 = read_hotfilm_from_lvm(filename, dt=2e-3)
        else:
            start_time, seconds, ch1, ch2 = read_hotfilm_from_lvm(filename, dt=1e-3)

        origin = datetime(start_time.year, start_time.month, start_time.day)
        seconds = np.array(seconds) + (start_time - origin).total_seconds()
        ch1 = np.array(ch1)
        ch2 = np.array(ch2)

        t0 = date2num(exp.runs[0].start_time)
        t1 = date2num(exp.runs[-1].end_time)

        t0 = (t0 - int(t0)) * 86400
        t1 = (t1 - int(t1)) * 86400

        mask = (seconds >= t0) & (seconds <= t1)

        exp_seconds = seconds[mask]

        # fan frequency
        fan = np.zeros(exp_seconds.size)
        for run in exp.runs:
            run_mask = (exp_seconds >= t0) & (exp_seconds <= t1)
            fan[run_mask] = run.fan

        ncfile = 'hotfilm_' + exp_name + '.nc'
        print('Writing ' + ncfile)

        nc = Dataset(ncfile, 'w', format='NETCDF4')

        nc.createDimension('Time', size=0)
        var = nc.createVariable('Time', 'f8', dimensions=('Time'))
        var[:] = exp_seconds
        var.setncattr('name', 'Time in seconds of the day')
        var.setncattr('units', 's')
        var.setncattr('origin', origin.strftime('%Y-%m-%dT%H:%M:%S'))

        var = nc.createVariable('fan', 'f4', dimensions=('Time'))
        var[:] = fan
        var.setncattr('name', 'Fan frequency')
        var.setncattr('units', 'Hz')

        var = nc.createVariable('ch1', 'f4', dimensions=('Time'))
        var[:] = ch1[mask]
        var.setncattr('name', 'Channel 1 voltage')
        var.setncattr('units', 'V')

        var = nc.createVariable('ch2', 'f4', dimensions=('Time'))
        var[:] = ch2[mask]
        var.setncattr('name', 'Channel 2 voltage')
        var.setncattr('units', 'V')

        nc.close()


def process_irgason_to_level2():
    """Processes IRGASON TOA5 files into NetCDF."""

    IRGASON_DATA_PATH = get_data_path('IRGASON')

    experiments_to_process = [
        'asist-windonly-fresh_warmup', 
        'asist-windonly-fresh', 
        'asist-wind-swell-fresh', 
        'asist-windonly-salt', 
        'asist-wind-swell-salt',
        'asist-flow-distortion'
    ]

    files = glob.glob(IRGASON_DATA_PATH + '/TOA5*.dat')
    files.sort()
    time, u, v, w, Ts, Tc, Pc, RH = read_irgason_from_toa5(files)

    for exp_name in experiments_to_process:

        exp = experiments[exp_name]
        t0 = exp.runs[0].start_time
        t1 = exp.runs[-1].end_time
        mask = (time >= t0) & (time <= t1)
  
        exp_time = time[mask]

        # time in seconds of the day; save origin in nc attribute
        exp_seconds = (date2num(exp_time) - int(date2num(t0))) * 86400

        # fan frequency
        fan = np.zeros(exp_time.size)
        for run in exp.runs:
            run_mask = (exp_time >= run.start_time) & (exp_time <= run.end_time)
            fan[run_mask] = run.fan

        # status flag (0: good; 1: fan spin-up; 2: bad)
        flag = np.zeros(exp_time.size)
        for run in exp.runs:
            run_mask = (exp_time >= run.start_time) & (exp_time < run.start_time + timedelta(seconds=60))
            flag[run_mask] = 1

        ncfile = 'irgason_' + exp_name + '.nc'
        print('Writing ' + ncfile)

        nc = Dataset(ncfile, 'w', format='NETCDF4')

        nc.createDimension('Time', size=0)
        var = nc.createVariable('Time', 'f8', dimensions=('Time'))
        var[:] = exp_seconds
        var.setncattr('name', 'Time in seconds of the day')
        var.setncattr('units', 's')
        var.setncattr('origin', num2date(int(date2num(t0))).strftime('%Y-%m-%dT%H:%M:%S'))

        var = nc.createVariable('flag', 'i4', dimensions=('Time'))
        var[:] = flag
        var.setncattr('name', 'Status flag')
        var.setncattr('description', '0: good; 1: fan spin-up; 2: bad')
        var.setncattr('units', '')

        var = nc.createVariable('fan', 'f4', dimensions=('Time'))
        var[:] = fan
        var.setncattr('name', 'Fan frequency')
        var.setncattr('units', 'Hz')

        var = nc.createVariable('u', 'f4', dimensions=('Time'))
        var[:] = u[mask]
        var.setncattr('name', 'x- component of velocity')
        var.setncattr('units', 'm/s')

        var = nc.createVariable('v', 'f4', dimensions=('Time'))
        var[:] = v[mask]
        var.setncattr('name', 'y- component of velocity')
        var.setncattr('units', 'm/s')

        var = nc.createVariable('w', 'f4', dimensions=('Time'))
        var[:] = w[mask]
        var.setncattr('name', 'z- component of velocity')
        var.setncattr('units', 'm/s')

        var = nc.createVariable('Ts', 'f4', dimensions=('Time'))
        var[:] = Ts[mask]
        var.setncattr('name', 'Sonic temperature')
        var.setncattr('units', 'deg. C')

        var = nc.createVariable('Tc', 'f4', dimensions=('Time'))
        var[:] = Tc[mask]
        var.setncattr('name', 'Cell temperature')
        var.setncattr('units', 'deg. C')

        var = nc.createVariable('Pc', 'f4', dimensions=('Time'))
        var[:] = Pc[mask]
        var.setncattr('name', 'Cell pressure')
        var.setncattr('units', 'hPa')

        var = nc.createVariable('RH', 'f4', dimensions=('Time'))
        var[:] = RH[mask]
        var.setncattr('name', 'Relative humidity')
        var.setncattr('units', '%')

        nc.close()


def process_pitot_to_level2():
    """Processes MKS pressure difference from TOA5 files 
    into pitot tube velocity and writes it to NetCDF."""

    PRESSURE_DATA_PATH = get_data_path('PRESSURE')

    experiments_to_process = [
        'asist-windonly-fresh_warmup',
        'asist-windonly-fresh', 
        'asist-wind-swell-fresh', 
        'asist-windonly-salt', 
        'asist-wind-swell-salt'
    ]

    files = glob.glob(PRESSURE_DATA_PATH + '/TOA5*.dat')
    files.sort()
    time, dp1, dp2 = read_pressure_from_toa5(files)

    # remove offset from pressure before computing velocity
    for exp_name in experiments_to_process:
        exp = experiments[exp_name]
        t0 = exp.runs[0].start_time
        t1 = exp.runs[0].end_time - timedelta(seconds=60)
        mask = (time >= t0) & (time <= t1)
        dp1_offset = np.mean(dp1[mask])
        dp2_offset = np.mean(dp2[mask])
        for run in exp.runs:
            run_mask = (time >= run.start_time) & (time <= run.end_time)
            dp1[run_mask] -= dp1_offset
            dp2[run_mask] -= dp2_offset

    dp1[dp1 < 0] = 0
    air_density = 1.1554 # at 30 deg. C and 90% RH
    u = pitot_velocity(dp1, air_density)

    for exp_name in experiments_to_process:

        exp = experiments[exp_name]
        t0 = exp.runs[0].start_time
        t1 = exp.runs[-1].end_time
        mask = (time >= t0) & (time <= t1)
  
        exp_time = time[mask]

        # time in seconds of the day; save origin in nc attribute
        exp_seconds = (date2num(exp_time) - int(date2num(t0))) * 86400

        # fan frequency
        fan = np.zeros(exp_time.size)
        for run in exp.runs:
            run_mask = (exp_time >= run.start_time) & (exp_time <= run.end_time)
            fan[run_mask] = run.fan

        # status flag (0: good; 1: fan spin-up; 2: bad)
        flag = np.zeros(exp_time.size)
        for run in exp.runs:
            run_mask = (exp_time >= run.start_time) & (exp_time < run.start_time + timedelta(seconds=60))
            flag[run_mask] = 1

        ncfile = 'pitot_' + exp_name + '.nc'
        print('Writing ' + ncfile)

        nc = Dataset(ncfile, 'w', format='NETCDF4')

        nc.createDimension('Time', size=0)
        var = nc.createVariable('Time', 'f8', dimensions=('Time'))
        var[:] = exp_seconds
        var.setncattr('name', 'Time in seconds of the day')
        var.setncattr('units', 's')
        var.setncattr('origin', num2date(int(date2num(t0))).strftime('%Y-%m-%dT%H:%M:%S'))

        var = nc.createVariable('fan', 'f4', dimensions=('Time'))
        var[:] = fan
        var.setncattr('name', 'Fan frequency')
        var.setncattr('units', 'Hz')

        var = nc.createVariable('u', 'f4', dimensions=('Time'))
        var[:] = u[mask]
        var.setncattr('name', 'Pitot velocity')
        var.setncattr('units', 'm/s')

        var = nc.createVariable('dp_pitot', 'f4', dimensions=('Time'))
        var[:] = dp1[mask]
        var.setncattr('name', 'Pitot pressure difference')
        var.setncattr('units', 'Pa')

        var = nc.createVariable('dp_alongtank', 'f4', dimensions=('Time'))
        var[:] = dp2[mask]
        var.setncattr('name', 'Along-tank pressure difference')
        var.setncattr('units', 'Pa')

        nc.close()
