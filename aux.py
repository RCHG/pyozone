# (C) R. Checa-Garcia, Department of Meteorology, University of Reading.
#     email: r.checa-garcia@reading.ac.uk
#
# This file is part of a set of Python Modules to process and
# perform data-analysis of the CMIP6 ozone dataset. Some of the methods 
# here included were developed for SMURPHS project.
#
# This software is licensed with GPLv3. Please see <http://www.gnu.org/licenses/>
"""
Python Module aux.py

Purpose ----------------------------------------------------------------------
    aux.py :
        auxiliary module created in the support of CMIP6 and SMURPHS
        database products.

Author -----------------------------------------------------------------------
   R. Checa-Garcia, Department of Meteorology, University of Reading.
   email: r.checa-garcia@reading.ac.uk

CODE INFO --------------------------------------------------------------------

__author__       = "R. Checa-Garcia"
__organization__ = ["University of Reading"]
__license__      = "GPLv3"
__version__      = "First: 0.7 - April 2016, Current: July 2017"
__maintainer__   = "R. Checa-Garcia"
__project__      = "SMURPHS and CMIP6 ozone database"
__email__        = "r.checa-garcia@reading.ac.uk"
__status__       = "Consolidating"
------------------------------------------------------------------------------
History:
    - Small refactory of the code in Nov-2017

"""

import numpy as np
import warnings
import logs
import cProfile
from netCDF4    import Dataset        # netcdf4
import resource
from netcdftime import utime       # UTIME operations
import sys
from netCDF4    import date2num
from netCDF4    import num2date
from datetime   import datetime
from datetime   import timedelta

def save_netcdf(new_var, all_time, val_plevs, val_lat, val_lon, model_name,
                var_ps='none', varname='vmro3',
                tim_units='months since 1850-01-01 00:00:00.0', calendar='standard'):
    """
    This function saves a netCDF file on pressure levels. It is used mainly for
    temporal netCDF files during processing.

    TESTED OK.
    """

    print('Main Memory use: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print('.... saving to : %s ' % model_name)


    nc_model = Dataset(model_name, mode='w', format='NETCDF4')

    medat = nc_model.createGroup('METADATA')
    medat.references   = 'http://www.met.reading.ac.uk/'
    medat.creator_name = "Ramiro Checa-Garcia (supervised by M.I. Hegglin)"
    medat.creator_mail = "r.checa-garcia@reading.ac.uk"
    medat.comment      = ('Created with pyMERGE_model done a University of Reading')
    medat.Conventions  = 'CF-1.0'
    medat.pyMerge_version = 'September 2016'

    # We create the main dimensions
    nc_model.createDimension('time', None)
    nc_model.createDimension('plev', len(val_plevs))
    nc_model.createDimension('lat', len(val_lat))
    nc_model.createDimension('lon', len(val_lon))

    time = nc_model.createVariable('time', 'f8', ('time',))
    levs = nc_model.createVariable('plev', 'f4', ('plev',))
    lats = nc_model.createVariable('lat', 'f4', ('lat',))
    lons = nc_model.createVariable('lon', 'f4', ('lon',))

    lons.units = 'degrees east'
    lons.standard_name = "longitude"
    lats.units = 'degrees north'
    lats.standard_name = "latitude"
    levs.units = 'Pa'
    time.units = tim_units
    time.calendar = calendar
    time[:] = all_time
    levs[:] = val_plevs
    lats[:] = val_lat
    lons[:] = val_lon

    if varname == 'vmro3':
        vmr = nc_model.createVariable('vmro3', 'f8',
                                    ('time', 'plev', 'lat', 'lon'),
                                     fill_value=-999.9)
        vmr.units = 'vmr'
        vmr.standard_name = 'Volume Mixing Ratio O3'

    if varname == 'vmrh2o':
        vmr = nc_model.createVariable('vmrh2o', 'f8',
                                    ('time', 'plev', 'lat', 'lon'),
                                    fill_value=-999.9)
        vmr.units = 'vmr'
        vmr.standard_name = 'Volume Mixing Ratio H2O'

    if var_ps != 'none':
        surf_press = nc_model.createVariable('ps', 'f8',
                                    ('time', 'lat', 'lon'),
                                     fill_value=-999.9)
        surf_press.units = 'Pa'
        surf_press.standard_name = 'Surface Pressure in Pa'
        surf_press[:] = var_ps

    vmr[:] = new_var

    nc_model.close()

    return

def concatenate_ordered(lvalue, ltimes):
    """
    First is estimate the interval with common times:

         From (list of arrays might not be

         (note:time value is on axis x)

         (1 array)        ------
         (2 array) -------
         (3 array)              -------
         We created a single array where first is 2 array then 1 array and
         then 3 array
         (new arr) --------------------

         For the arrays:
         ----------
                -----------
                         ------------
         We create:

         -------
                XXX
                   ------
                         XX
                           -----------

         X arrays are a mean (at this moment this is not implemented.)

    :param lvalue: list of arrays to concatenate
    :param ltimes: list of times of each array
    :return:
    """

    lminval = [np.min(a) for a in ltimes]
    lsorted = [i[0] for i in sorted(enumerate(lminval), key=lambda x: x[1])]

    l_ord_times = [ltimes[i] for i in lsorted]
    l_ord_value = [lvalue[i] for i in lsorted]

    new_value = np.concatenate(l_ord_value)
    new_time = np.concatenate(l_ord_times)

    return new_value, new_time


def do_cprofile(func):
    """
    With the decorator @do_cprofile the function is analyzed by cprofile module.

    :param func:
    :return:
    """
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func


def filter_values(myarray, value):
    """
    Assign nan to those index where the array has value.

    :param myarray:
    :param value:
    :return:
    """
    my_array = myarray.astype('float') # to ensure next statement will work
    my_array[my_array == value] = np.nan

    return my_array


def create_set_TS(lat_val, dim_plev, target_array,
                  pmax=100.1, pmin=99.9, latmin=-20., latmax=20.):
    """
        This function returns an flatten array whose index represents time
    The input is target_array(time, plev, lat) so an average over lat range
    is done and an specific plev is selected.

    :param lat_val:
    :param dim_plev:
    :param target_array:
    :param pmax:
    :param pmin:
    :param latmin:
    :param latmax:
    :param iloop:
    :return:
    """
    warnings.filterwarnings('ignore')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # lat_val = full_nc.variables['lat'][:]
        plev_inds = np.where((dim_plev >= pmin) & (dim_plev<= pmax))
        lat_inds  = np.where((lat_val >= latmin) & (lat_val<= latmax))

        plev_fact = 0.1
        plev_iter = 0
        pmin_new = pmin
        pmax_new = pmax
        while len(plev_inds[0]) == 0:
            plev_iter += 1
            pext = plev_iter*plev_fact
            plev_inds = np.where((dim_plev >= pmin-pext) & (dim_plev <= pmax+pext))
            logs.INFO.warning(' --- IN aux.create_set_TS --- plev limit increased')
            pmin_new = pmin-pext
            pmax_new = pmax+pext

        lat_fact = 0.5
        lat_iter = 0
        latmin_new = latmin
        latmax_new = latmax

        while len(lat_inds[0]) == 0:
            lat_iter += 1
            lext = lat_iter*lat_fact
            lat_inds = np.where((lat_val >= latmin-lext) & (lat_val <= latmax+lext))
            logs.INFO.warning(' --- IN aux.create_set_TS --- lat limit increased')
            latmin_new = latmin-lext
            latmax_new = latmax+lext

        if lat_iter > 0:
            str_log = ' IN aux.create_set_TS --- lat limit set from'
            str_log += '(%3.3f,%3.3f) to %3.3f,%3.3f) ' %(latmin, latmax, latmin_new, latmax_new)
            logs.DEBUG.warning(str_log)
        if plev_iter > 0:
            str_log = ' IN aux.create_set_TS --- plev limit set from'
            str_log += '(%3.3f,%3.3f) to %3.3f,%3.3f) ' %(pmin, pmax, pmin_new, pmax_new)
            logs.DEBUG.warning(str_log)

        target_test = target_array[:, plev_inds[0], lat_inds]
        target_weig = np.ones_like(target_test)
        sizes = target_weig.shape
        ts_out = np.zeros((sizes[0],sizes[1]))


        fact = 0.0
        for lat in lat_val[lat_inds[0]]:
            fact = fact + np.abs(np.cos(lat*np.pi/180.))

        for i_val in range(sizes[0]):
            for j_val in range(sizes[1]):

                for k_val, lat_id in zip(range(sizes[2]), lat_inds[0]):
                    norm =  np.abs(np.cos(lat_val[lat_id]*np.pi/180.))/fact
                    ts_out[i_val, j_val] = ts_out[i_val, j_val] + target_test[i_val, j_val, k_val]*norm

    return ts_out.flatten(), plev_inds[0][0]


def create_set_TS_2D(lat_val, dim_plev, target_array,
                  pmax=100000.1, pmin=0.01, latmin=-5., latmax=5.):
    """
        This function returns an flatten array whose index represents time
    The input is target_array(time, plev, lat) so an average over lat range
    is done and an specific plev is selected.

    :param lat_val:
    :param dim_plev:
    :param target_array:
    :param pmax:
    :param pmin:
    :param latmin:
    :param latmax:
    :param iloop:
    :return:
    """
    warnings.filterwarnings('ignore')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        plev_inds = np.where((dim_plev >= pmin) & (dim_plev<= pmax))
        lat_inds  = np.where((lat_val >= latmin) & (lat_val<= latmax))

        plev_fact = 0.1
        plev_iter = 0
        pmin_new = pmin
        pmax_new = pmax
        while len(plev_inds[0]) == 0:
            plev_iter += 1
            pext = plev_iter*plev_fact
            plev_inds = np.where((dim_plev >= pmin-pext) & (dim_plev <= pmax+pext))
            logs.INFO.warning(' --- IN aux.create_set_TS --- plev limit increased')
            pmin_new = pmin-pext
            pmax_new = pmax+pext

        lat_fact = 0.5
        lat_iter = 0
        latmin_new = latmin
        latmax_new = latmax

        while len(lat_inds[0]) == 0:
            lat_iter += 1
            lext = lat_iter*lat_fact
            lat_inds = np.where((lat_val >= latmin-lext) & (lat_val <= latmax+lext))
            logs.INFO.warning(' --- IN aux.create_set_TS --- lat limit increased')
            latmin_new = latmin-lext
            latmax_new = latmax+lext

        if lat_iter > 0:
            str_log = ' IN aux.create_set_TS --- lat limit set from'
            str_log += '(%3.3f,%3.3f) to %3.3f,%3.3f) ' %(latmin, latmax, latmin_new, latmax_new)
            logs.DEBUG.warning(str_log)
        if plev_iter > 0:
            str_log = ' IN aux.create_set_TS --- plev limit set from'
            str_log += '(%3.3f,%3.3f) to %3.3f,%3.3f) ' %(pmin, pmax, pmin_new, pmax_new)
            logs.DEBUG.warning(str_log)


        new_levs = plev_inds[0].tolist()
        target_testA = target_array[:, :, lat_inds[0]]
        target_test = target_testA[:, new_levs,:]
        target_weig = np.ones_like(target_test)
        sizes = target_weig.shape
        ts_out = np.zeros((sizes[0],sizes[1]))


        fact = 0.0

        for lat in lat_val[lat_inds[0]]:
            fact = fact + np.abs(np.cos(lat*np.pi/180.))

        for i_val in range(sizes[0]):
            for j_val in range(sizes[1]):

                for k_val, lat_id in zip(range(sizes[2]), lat_inds[0]):
                    norm =  np.abs(np.cos(lat_val[lat_id]*np.pi/180.))/fact
                    ts_out[i_val, j_val] = ts_out[i_val, j_val] + target_test[i_val, j_val, k_val]*norm

    return ts_out, new_levs


def up():
    # My terminal breaks if we don't flush after the escape-code
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()


def down():
    # I could use '\x1b[1B' here, but newline is faster and easier
    sys.stdout.write('\n')
    sys.stdout.flush()

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def parse_datetime(dataset_x):
    """
    This function just assign the day to 15. It might be useful for some data-analysis with other tools.

    :param dataset_x:
    :return: datetime_x
    """

    time_unit_model_x = utime(str(dataset_x.variables['time'].units).replace('"', ''))
    time_data_model_x = time_unit_model_x.num2date(dataset_x.variables['time'][:])

    dattime_x = [mytime.replace(day=15, hour=0, minute=0, second=0, microsecond=0)
                 for mytime in time_data_model_x]
    datetime_x = np.array(dattime_x)

    return datetime_x

def create_dataset_timeunits_other(file1, file2, newfile1):
    """
    This function take three arguments as file names of netcdf files
    file1: is a netcdf with time units to change
    file2: is a netcdf with the time units we want
    newfile1: is a name for a new netcdf file that will be the file1
              but with the file2 time units and calendar.
    """

    dataset_2 = Dataset(file2)
    time_units = dataset_2.variables['time'].units

    src = Dataset(file1,'r')
    dst = Dataset(newfile1, 'w')

    for namedimen in src.dimensions.keys():
        dimension = src.dimensions[namedimen]
        if not dimension.isunlimited():
            dst.createDimension(namedimen, size=len(dimension))
        else:
            dst.createDimension(namedimen, size=None)

    for varname in src.variables.keys():

        variable = src.variables[varname]

        if varname == 'some_variable':
            continue

        if varname == 'time':
                x = dst.createVariable(varname, variable.datatype,
                                       dimensions=variable.dimensions)
                x.units = time_units
                x.calendar = 'standard'

                time_x = num2date(src.variables['time'][:],
                                  src.variables['time'].units,
                                  calendar=src.variables['time'].calendar)

                dattime_x = time_x - timedelta(days=15)
                x[:] = date2num(dattime_x, time_units, calendar='standard')

        elif varname == 'time_bnds':
                x = dst.createVariable(varname, variable.datatype,
                                       dimensions=variable.dimensions)
                x.units = time_units
                time_bnd_0 = num2date(src.variables['time_bnds'][:][:,0],
                                      src.variables['time'].units,
                                      calendar=src.variables['time'].calendar)
                time_bnd_1 = num2date(src.variables['time_bnds'][:][:,1],
                                      src.variables['time'].units,
                                      calendar=src.variables['time'].calendar)

                x[:]=np.array([date2num(time_bnd_0, time_units, calendar='standard'),
                      date2num(time_bnd_1, time_units, calendar='standard')]).T

        else:
                x = dst.createVariable(varname,variable.datatype,
                                       dimensions=variable.dimensions)
                x[:] = src.variables[varname][:]

    src.close()
    dst.close()

    return


def create_dataset_timeunits_cmam(file1, newfile1):
    """
    This function take two arguments as file names of netcdf files
    file1: is a netcdf with time units to change
    newfile1: is a name for a new netcdf file that will be the file1
              but with the units 'days since 1850-01-01 00:00:00'
              and calendar standard.
    """

    new_time_units = 'days since 1850-01-01 00:00:00'

    src = Dataset(file1,'r')
    dst = Dataset(newfile1, 'w')

    cmam_time_units = src.variables['time'].units
    for namedimen in src.dimensions.keys():
        dimension = src.dimensions[namedimen]
        if not dimension.isunlimited():
            dst.createDimension(namedimen, size=len(dimension))
        else:
            dst.createDimension(namedimen, size=None)

    for varname in src.variables.keys():

        variable = src.variables[varname]

        if varname == 'some_variable':
            continue
        if varname=='time':
                x = dst.createVariable(varname, variable.datatype,
                                       dimensions=variable.dimensions)
                x.units = new_time_units
                x.calendar = variable.calendar
                time_obj = utime(str(cmam_time_units.replace('00:00','00:00:00')).replace('"', ''))
                time_x = time_obj.num2date(variable[:])
                new_time_obj = utime(new_time_units)
                new_cmam_tim = new_time_obj.date2num(time_x)
                x[:] = new_cmam_tim
        else:
                x = dst.createVariable(varname,variable.datatype,
                                       dimensions=variable.dimensions)
                x[:] = src.variables[varname][:]
    src.close()
    dst.close()

    return
