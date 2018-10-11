"""
process_level1.py
"""
from asist.irgason import read_irgason_from_toa5
from asist_nsf_2018.experiments import experiments
from datetime import datetime, timedelta
import glob
from matplotlib.dates import date2num, num2date
from netCDF4 import Dataset
import numpy as np
import os

try:
    IRGASON_DATA_PATH = os.environ['IRGASON_DATA_PATH']
except KeyError:
    raise KeyError('Set IRGASON_DATA_PATH env variable to the path with IRGASON data')

def process_irgason_to_level2():
    """Processes IRGASON TOA5 files into NetCDF."""

    experiments_to_process = [
        'asist-windonly-fresh', 
        'asist-wind-swell-fresh', 
        'asist-windonly-salt', 
        'asist-wind-swell-salt',
        'asist-flow-distortion'
    ]

    irgason_files = glob.glob(IRGASON_DATA_PATH + '/TOA5*.dat')
    irgason_files.sort()
    time, u, v, w, Ts, Tc, Pc, RH = read_irgason_from_toa5(irgason_files)

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
        var.setncattr('origin', num2date(int(date2num(t0))).strftime('%Y-%m-%d %H:%M:%S UTC'))

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
