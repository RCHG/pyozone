# (C) R. Checa-Garcia, Department of Meteorology, University of Reading.
#     email: r.checa-garcia@reading.ac.uk
#
# This file is part of a set of Python Modules to process and
# perform data-analysis of the CMIP6 ozone dataset. Some of the
# methods included here, were developed for SMURPHS project.
#
# This software is licensed with GPLv3. Please see <http://www.gnu.org/licenses/>.
"""
Python Module calculate.py

Purpose ----------------------------------------------------------------------
   calculate.py :
      module created to perform calculations on the support of CMIP6 ozone
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

"""

# Load Libraries and Modules EXTERNAL
# -- note that not all these modules are actually needed by calculate.py

from optparse   import OptionParser     # Introduce options to code
from netCDF4    import Dataset          # netcdf4
from netCDF4    import date2num
from netcdftime import utime            # UTIME operations (local module)
from datetime   import datetime
from datetime   import timedelta

from scipy.interpolate import UnivariateSpline, interp1d
from tqdm import tnrange,  tqdm, tqdm_notebook
from os.path import isfile as exists_file

import numpy as np                    # numerical python
import os                             # Operating system operations
import glob                           # Find files
import pprint                         # Print arrays nice
import warnings                       # Manage warnings
import gc
import resource
import aux                            # This is a module of this software suite
import copy

# Functions ---------------------------------------------------------------------
#

def pure_weighted_mean(val, ome):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy arrays with the same shape.
    Also the function filters nan and gives as output the
    filtered arrays.

    :param val: for a given time, array mean values to merge
    :type val: numpy array with mean values
    :param ome: for a given time, array weights to merge
    :type ome: numpy array with weights for weigthing average
    :return mean, std, val_ok, ome_ok
    """
    val[np.argwhere(np.isnan(ome))] = np.nan
    ome[np.argwhere(np.isnan(val))] = np.nan

    val_ok = val[np.isfinite(val)]
    ome_ok = ome[np.isfinite(ome)]

    mean = np.sum(val_ok*ome_ok)/np.sum(ome_ok)
    std = 1.0/np.sqrt(np.sum(ome_ok))

    return mean, std, val_ok, ome_ok


def weigthed_mean(arr_ubi, arr_err):
    """
    Given the array of unbiased values and
    the array with the expected errors (std)
    calculates the merged values and merged
    error.

    NOTE: arr_ubi and arr_err are bi-dimensional arrays,
    the number of platforms to merge and the second is
    the values of each platform to each time:

    [[v(p1,t1),v(p2,t1),...,v(pk,t1)],
     [v(p1,t2),v(p2,t2),...,v(pk,t2)],...

    therefore the loops inside are a loop in the values of
    the different platforms for a given time.

    :param arr_ubi: list of arrays of unbiased time values
    :param arr_err: list of arrays of each platform error.
    :return:  merge_val, merge_err
    """

    omegas = 1.0/np.power(arr_err, 2.0)
    merge_val = []
    merge_err = []
    ii = 0
    for ave, omega in zip(arr_ubi, omegas):
        val, err, val_ok, ome_ok = pure_weighted_mean(ave, omega)
        merge_val.append(val)
        merge_err.append(err)
        ii += 1

    return merge_val, merge_err


def mad_based_outlier(points, thresh=3.5):
    """
    This function is testing an array of values
    contained in points and try to identify outliers
    given a thershold (by default 3.5).


    :param points:
    :param thresh:
    :return: bolean array with same dimensions than points.
    """

    diff = np.zeros_like(points)
    modified_z_score = np.zeros_like(points)
    median = np.nanmedian(points)

    for ival, val in enumerate(points):
        if np.isnan(val) is True:
            diff[ival] = np.nan
        else:
            diff[ival] = np.sqrt((val - median)**2)

    med_abs_deviation = np.nanmedian(diff)

    for ival, val in enumerate(points):
        if np.isnan(val) is True:
            modified_z_score[ival] = True
        else:
            modified_z_score[ival] = 0.6745 * diff[ival] / med_abs_deviation

    return modified_z_score > thresh


def smooth_tropopause(lat):
    """
    Defines a reasonable tropopause consistent with Hansen et al 2005, but
    full symmetrical NH vs SH. The units are in hPa. Provides the pressure
    height of the tropopause for a given value of the latitude. For more,
    information about this tropopause definition check the supplementary
    information of the paper "Historical tropospheric and stratospheric
    ozone radiative forcing using the CMIP6 database", Checa-Garcia et al.

    :param lat:
    :return: tropopause
    """


    center  = 45.0
    smooth  = 10.0
    voffset = 165.0
    vwidth  = 55.0

    tropopause = np.tanh((abs(lat)-center)/smooth)*vwidth+voffset

    return tropopause


def zonal_regrid(plev_0, lat_0, lat_1, zonal_map):
    """
    This function re-grid from lat_0 to lat_1 the original zonal_map.
    Zonal map is a map with dimensions [time, plev_0, lat_0] and the
    output will be the same array at [time, plev_0, lat_1]

    :param plev_0:
    :param lat_0:
    :param lat_1:
    :param zonal_map:
    :return: new_zonal_map
    """

    from scipy import interpolate

    len_time = zonal_map.shape[0]
    new_zonal_map = np.zeros((len_time, len(plev_0), len(lat_1)))

    for itime in range(len_time):
        for ipres in range(len(plev_0)):
            if False in np.isfinite(zonal_map[itime, ipres, :]):
                print('PROBLEM')
                exit()

            interfunc = interpolate.interp1d(lat_0, zonal_map[itime, ipres, :],
                                             fill_value='extrapolate', kind='nearest')
            # A potential problem that actually happens that than for some negative
            # latitudes the values can be negative. This is more likely to happen with
            # a linear interpolation so a nearest neighbour is included.

            negatives = np.argwhere(interfunc(lat_1) < 0)
            if len(negatives) > 0:
                print(itime, ipres, negatives)
            new_zonal = interfunc(lat_1)
            new_zonal[negatives] = 0.0
            new_zonal_map[itime, ipres, :] = new_zonal

    return new_zonal_map


def new_plevs_grid(field, plev_old, plev_new, keep=False, extend=True):
    """
    This function re-grid a field only in the vertical level which is
    supposed to be on the axis=1. This function is created to minimize
    the dependence with external libraries (only depends on scipy) but
    users with cf-python or iris python modules may be good alternatives.
    Other is the use of cdo/nco.

    :param field:
    :param plev_old:
    :param plev_new:
    :param extrapolate:
    :param keep:
    :return: new_field
    """

    sizes = field.shape

    new_sizes = (sizes[0], len(plev_new), sizes[2], sizes[3])
    new_field = np.zeros(new_sizes)
    for idim in tqdm(range(sizes[0]), desc='Lev DIM'):
        gc.collect()
        for jdim in range(sizes[2]):
            for kdim in range(sizes[3]):
                if extend==True:
                    extrapolator = interp1d(plev_old, field[idim, :, jdim, kdim],
                                            kind='linear', bounds_error=False,
                                            fill_value=(field[idim, 0, jdim, kdim],
                                                        field[idim, -1, jdim, kdim]))
                else:
                    extrapolator = interp1d(plev_old, field[idim, :, jdim, kdim],
                                            kind='linear', bounds_error=False,
                                            fill_value=(field[idim, 0, jdim, kdim], 0.0))

                new_field[idim, :, jdim, kdim] = extrapolator(plev_new)

    return new_field


def new_lons_grid(field, lons_old, lons_new, keep=False):
    """
    This function re-grid a field only in the longitude which is
    supposed to be on the axis=-1. This function is created to minimize
    the dependence with external libraries (only depends on scipy) but
    users with cf-python or iris python modules may be good alternatives.
    Other is the use of cdo/nco.
    
    This function kept commented the memory analysis calls in case the
    user find them useful.

    :param field:
    :param plev_old:
    :param plev_new:
    :param extrapolate:
    :param keep:
    :return: new_field
    """

    # print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    sizes = field.shape
    if len(sizes)>3:
        new_sizes = (sizes[0], sizes[1], sizes[2], len(lons_new))
        #print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        print('...Loops LONS:',new_sizes)
        #gc.collect()
        #print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


        new_field = np.zeros(new_sizes)
        for idim in tqdm(range(sizes[0]), desc='Lon DIM'):
            gc.collect()
            for jdim in range(sizes[1]):
                for kdim in range(sizes[2]):
                    #extrapolator = UnivariateSpline(lons_old, field[idim, jdim, kdim, :], k=extrapolate)
                    extrapolator = interp1d(lons_old, field[idim, jdim, kdim, :],
                                            kind='linear', fill_value='extrapolate')
                    new_field[idim, jdim, kdim, :] = extrapolator(lons_new)

        if keep == True:
            new_field[:, :, :, 0:len(lons_old)] = field[:,:,:,:]

    if len(sizes)==3:
        new_sizes = (sizes[0], sizes[1], len(lons_new))
        #print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        #print('...Loops:',new_sizes)
        #gc.collect()
       # print('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


        new_field = np.zeros(new_sizes)
        for idim in tqdm(range(sizes[0])):
            gc.collect()
            for jdim in range(sizes[1]):
                    #extrapolator = UnivariateSpline(lons_old, field[idim, jdim, kdim, :], k=extrapolate)
                    extrapolator = interp1d(lons_old, field[idim, jdim, :],
                                            kind='linear', fill_value='extrapolate')
                    new_field[idim, jdim, :] = extrapolator(lons_new)


    return new_field


def new_lats_grid(field, lats_old, lats_new, keep=False):
    """
    This function re-grid a field only in the latitude which is
    supposed to be on the axis=2. This function is created to minimize
    the dependence with external libraries (only depends on scipy) but
    users with cf-python or iris python modules may be good alternatives.
    Other is the use of cdo/nco.


    :param field:
    :param plev_old:
    :param plev_new:
    :param extrapolate:
    :param keep:
    :return: new_field
    """

    sizes = field.shape
    if len(sizes)>3:
        new_sizes = (sizes[0], sizes[1], len(lats_new), sizes[3])

        print('...Loops LATS:',new_sizes)
        new_field = np.zeros(new_sizes)
        for idim in tqdm(range(sizes[0]), desc='Lat DIM'):
            gc.collect()
            for jdim in range(sizes[1]):
                for kdim in range(sizes[3]):
                    extrapolator =  interp1d(lats_old, field[idim, jdim, :, kdim],
                                             kind='linear', fill_value='extrapolate')
                    new_field[idim, jdim, :, kdim] = extrapolator(lats_new)

    if len(sizes)==3:
        new_sizes = (sizes[0], len(lats_new), sizes[2])

        print('...Loops:',new_sizes)
        new_field = np.zeros(new_sizes)
        for idim in tqdm(range(sizes[0])):
            gc.collect()
            for kdim in range(sizes[2]):
                    extrapolator =  interp1d(lats_old, field[idim, :, kdim],
                                             kind='linear', fill_value='extrapolate')
                    new_field[idim, :, kdim] = extrapolator(lats_new)

    return new_field



def hybrid_to_uniform_press_plev_model(model_tim, model_plev, model_lon, model_lat,
                                      model_var, plevs, extend=True, keep=False):
    """
    This function transform a variable given with hybrid coordinates to a simple
    fixed pressure level vertical grid.
    
    :param model_tim:
    :param model_plev:
    :param model_lon:
    :param model_lat:
    :param model_var:
    :param plevs: 
    :param extend: 
    :param keep: 
    :return: new_field
    """
    
    print('Main Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    new_sizes = (len(model_tim), len(plevs), len(model_lat), len(model_lon))
    new_field = np.zeros(new_sizes)
    for ndim in tqdm(range(new_sizes[0]), ncols=80):
        for jdim in range(new_sizes[2]):
            for idim in range(new_sizes[3]):
                if extend==True:
                    fill_val = np.array([model_var[ndim, 0, jdim, idim], model_var[ndim, -1, jdim, idim]])
                    if False in np.isfinite(fill_val):
                        exit()

                    extrapolator = interp1d(model_plev[ndim, :, jdim, idim],
                                            model_var[ndim, :, jdim, idim],
                                            kind='linear', bounds_error=False,
                                            fill_value=(model_var[ndim, 0, jdim, idim],
                                            model_var[ndim, -1, jdim, idim]))
                else:
                    extrapolator = interp1d(model_plev[ndim, :, jdim, idim],
                                            model_var[ndim, :, jdim, idim],
                                            kind='linear', bounds_error=False,
                                            fill_value=(model_var[ndim, 0, jdim, idim], 0.0))

                new_field[ndim, :, jdim, idim] = extrapolator(plevs)

    return new_field


def input_hybrid_netcdf(file_name, varname):
    """
    Reads a netcdf file with hybrid coordinates.
    
    :param file_name: 
    :param varname: 
    :return: model_tim, model_plev, model_lon, model_lat,
             model_var, model_ps
    """
 

    nc_model = Dataset(file_name, mode='r')

    model_var = nc_model[varname][:]
    model_lon = nc_model['lon'][:]
    model_lat = nc_model['lat'][:]
    model_p0  = nc_model['P0'][:]
    model_ps  = nc_model['PS'][:]
    model_hyam= nc_model['hyam'][:]
    model_hybm= nc_model['hybm'][:]
    model_lev = nc_model['lev'][:]
    model_tim = nc_model['time'][:]

    model_plev = np.zeros_like(model_var)
    dim_model_plev = model_plev.shape
    for nval in tqdm(range(dim_model_plev[0]), ncols=80):
        for kval in range(dim_model_plev[1]):
            model_plev[nval,kval,:,:] =model_hyam[kval]*model_p0+model_hybm[kval]*model_ps[nval,:,:]

    return model_tim, model_plev, model_lon, model_lat, model_var, model_ps



def vertical_mean(vmr_A, vmr_B, plevs, goal='avg-fast'):
    """
    This subroutine estimate a kind of mean between vmr_A and vmr_B where wmr_A has a
    higher level of confidence in the trosphere. So the steps are:
    1. Identify the tropopause -> index in plevs -> i_tropo
    2. From surface to i_tropo we create a weighting function that gives 0.85 to the model
       in the surface and 0.50 in i_tropo. Between both a continuous monotonic decreasing
       function of the index is created and applied.
    3. From the tropopause ahead the weight is 0.5 for both models.

    This subroutine was created to test the vertical mean done to create the
    CMIP6 ozone dataset but also to test alternatives to be applied for the creation of
    SMURPHS datasets.
    :param vmr_A:
    :param vmr_B:
    :param plevs:
    :param goal:
    :return: new_mean
    """

    i_trop = np.argmin(np.abs(plevs - 15000.))

    w_A = np.array([0.70+i_plev*(-0.20/float(i_trop)) for i_plev in range(len(plevs))])
    w_A[i_trop::] = 0.5
    w_B = 1.0 - w_A
    new_size = vmr_A.shape
    new_mean = np.zeros_like(vmr_A)

    if goal=='avg':
        for nd in tqdm(range(new_size[0]), ncols=80, desc='1st DIM'):
            for jd in range(new_size[2]):
                for id in range(new_size[3]):
                    for kd in range(new_size[1]):
                        new_mean[nd, kd, jd, id] = w_A[kd]*vmr_A[nd, kd, jd, id] + \
                                                   w_B[kd]*vmr_B[nd, kd, jd, id]
    if goal=='avg-fast':
        for nd in tqdm(range(new_size[0]), ncols=80, desc='1st DIM'):
            for kd in range(new_size[1]):
                new_mean[nd, kd, :, :] = w_A[kd]*vmr_A[nd, kd, :, :] + \
                                           w_B[kd]*vmr_B[nd, kd, :, :]
    if goal=='err':
        for nd in tqdm(range(new_size[0]), ncols=80, desc='1st DIM'):
            for jd in range(new_size[2]):
                for ld in range(new_size[3]):
                    for kd in range(new_size[1]):
                        new_mean[nd, kd, jd, ld] = np.abs(w_A[kd]*vmr_A[nd, kd, jd, ld]-\
                                                   w_B[kd]*vmr_B[nd, kd, jd, ld])*0.5
    return new_mean


def vertical_mean_cmip6(vmr_A, vmr_B, plevs, goal='avg-fast'):
    """
    This subroutine estimate a kind of mean between vmr_A and vmr_B where wmr_A has a
    higher level of confidence in the troposphere. So the steps are:
    1. Identify the tropopause -> index in plevs -> i_tropo
    2. From surface to i_tropo we create a weighting function that gives 0.85 to the model
       in the surface and 0.50 in i_tropo. Between both a continuous monotonic decreasing
       function of the index is created and applied.
    3. From the tropopause ahead the weight is 0.5 for both models.

    This subroutine was created to test the vertical mean done to create the
    CMIP6 ozone dataset but also to test alternatives to be applied for the creation of
    SMURPHS datasets.

    :param vmr_A:
    :param vmr_B:
    :param plevs:
    :param goal:
    :return:
    """

    i_trop = np.argmin(np.abs(plevs - 15000.))

    i1 = 21

    w_A = np.array([0.40+i_plev*(0.10/float(i1)) for i_plev in range(len(plevs))])
    w_A[i1:i1+18] = 0.5
    w_A_alpha = (0.9-0.5)/float(len(plevs)-39)
    w_A_beta  = 0.5-w_A_alpha*float(21+18)
    w_A[i1+18::] = np.array([(w_A_beta+i_plev*w_A_alpha) for i_plev in np.arange(i1+18,len(plevs))])

    w_B = 1.0 - w_A


    new_size = vmr_A.shape
    new_mean = np.zeros_like(vmr_A)
    if goal=='avg':
        for nd in tqdm(range(new_size[0]), ncols=80, desc='1st DIM'):
            for jd in range(new_size[2]):
                for id in range(new_size[3]):
                    for kd in range(new_size[1]):
                        new_mean[nd, kd, jd, id] = w_A[kd]*vmr_A[nd, kd, jd, id] + \
                                                   w_B[kd]*vmr_B[nd, kd, jd, id]

    if goal=='avg-fast':
        for nd in tqdm(range(new_size[0]), ncols=80, desc='1st DIM'):
            for kd in range(new_size[1]):
                new_mean[nd, kd, :, :] = w_A[kd]*vmr_A[nd, kd, :, :] + \
                                         w_B[kd]*vmr_B[nd, kd, :, :]
    if goal=='err':
        for nd in tqdm(range(new_size[0]), ncols=80, desc='1st DIM'):
            for jd in range(new_size[2]):
                for ld in range(new_size[3]):
                    for kd in range(new_size[1]):
                        new_mean[nd, kd, jd, ld] = np.abs(w_A[kd]*vmr_A[nd, kd, jd, ld]-\
                                                   w_B[kd]*vmr_B[nd, kd, jd, ld])*0.5
    return new_mean


def create_seasonal_zonal_du(model_base, var_name='vmro3', add_du=True,
                             add_seasonal=False, add_zonal=True):
    """
    This function add a seasonal and zonal estimation of the variable.

    In general it relies on the assumption that the first month of the
    dataset is January and the netcdf has a monthly resolution.

    model_base is a file-name
    :param model_base:
    :param var_name:
    :param add_du:
    :param add_seasonal:
    :param add_zonal:
    :return:
    """


    nc_base = Dataset(model_base, mode='a')

    nc_lev = nc_base['plev'][:]
    nc_lon = nc_base['lon'][:]
    nc_lat = nc_base['lat'][:]
    nc_tim = nc_base['time'][:]

    dat_time = aux.parse_datetime(nc_base)

    nc_var = nc_base[var_name][:]
    pr_surf = nc_base['ps'][:]

    var_sizes = nc_var.shape

    seasonal = np.zeros((12,var_sizes[1],var_sizes[2],var_sizes[3]))

    if add_seasonal:
        print('Adding seasonal mean')
        for imonth in range(12):
            i_count = 0
            for itime in np.arange(imonth,var_sizes[0],12):
                seasonal[imonth,:,:,:] = seasonal[imonth,:,:,:]+nc_var[itime,:,:,:]
                i_count = i_count + 1
            seasonal[imonth,:,:,:] = seasonal[imonth,:,:,:]/i_count

        nc_base.createDimension('month', 12)
        months = nc_base.createVariable('month', 'i4', ('month',))
        months[:] = [1,2,3,4,5,6,7,8,9,10,11,12]
        vmr = nc_base.createVariable(var_name+'_seasonal', 'f8',
                                     ('month', 'plev', 'lat', 'lon'),
                                     fill_value=-999.9)
        vmr.units = 'vmr'
        vmr.standard_name = 'Volume Mixing Ratio'
        vmr[:] = seasonal
        vmr.warning = 'Seasonal field estimated for whole netcdf file'

    if add_zonal:
        print('Adding zonal mean')
        zonal_model = np.nanmean(nc_var, axis=3)  # zonal mean, mean in longitude
        vmr_zonal = nc_base.createVariable(var_name+'_zonal', 'f8',
                                           ('time', 'plev', 'lat'),
                                           fill_value=-999.9)
        vmr_zonal[:] = zonal_model
        vmr_zonal.warning = 'Zonal field estimated for whole netcdf file'

    #try:
    #    nc_err = nc_base[var_name+'_err'][:]
    #    zonal_error = np.nanmean(nc_err, axis=3)  # zonal mean, mean in longitude
    #    vmr_zonal_err = nc_base.createVariable(var_name+'_err_zonal', 'f4',
    #                                   ('time', 'plev', 'lat'),
    #                                   fill_value=-999.9)
    #    vmr_zonal_err[:] = zonal_error
    #except:
    #    print('Zonal uncer. not added, input file did not have uncertainty...')

    if add_du:
        print('Adding Partial Column Field DU')

        partial_DU = nc_base.createVariable('O3_partialcolumn', 'f8',
                                 ('time', 'plev', 'lat', 'lon'),
                                 fill_value=-999.9)
        partial_DU.units = 'DU'
        o3_DU = calculate_partial_column(nc_lat, nc_lon, nc_lev, nc_var,
                                         pr_surf, nc_tim, case='TOTAL')
        partial_DU.standard_name = 'Column concentration in DU'
        partial_DU[:] = o3_DU
        partial_DU.warning = 'Partial column from surface until plev'

    nc_base.close()

    return


def add_anomalies_field(model_base, var_name='vmro3'):
    """
    This function allows add anomalies to an specific seasonal dataset.

    :param model_base:
    :param var_name:
    :return:
    """

    nc_base = Dataset(model_base, mode='a')

    nc_lev = nc_base['plev'][:]
    nc_lon = nc_base['lon'][:]
    nc_lat = nc_base['lat'][:]
    nc_tim = nc_base['time'][:]
    nc_mon = nc_base['months'][:]

    dat_time = parse_datetime(nc_base)

    nc_var = nc_base[var_name][:]
    seasonal = nc_base[var_name+'_seasonal'][:]

    var_sizes = nc_var.shape
    anomaly=np.zeros_like(nc_var)
    for itime in np.arange(var_sizes[0]):
        # we suppose that first time is January
        imonth = (itime + 1)%12 - 1
        if imonth == -1:
            imonth == 11
        anomaly[itime,:,:,:] = nc_var[itime,:,:,:] - seasonal[imonth, :,:,:]

    vmr = nc_base.createVariable(var_name+'_anomaly', 'f8',
                                 ('time', 'plev', 'lat', 'lon'),
                                 fill_value=-999.9, zlib=True)
    vmr.units = 'vmr'
    vmr.standard_name = 'Volume Mixing Ratio'
    vmr[:] = anomaly
    vmr.warning = 'Differ. between monthly field and seasonal value on each month'

    zonal_model = np.nanmean(anomaly, axis=3)  # zonal mean, mean in longitude
    vmr_zonal = nc_base.createVariable(var_name+'_anomalyzonal', 'f8',
                                       ('time', 'plev', 'lat'),
                                       fill_value=-999.9, zlib=True)
    vmr_zonal[:] = zonal_model
    vmr_zonal.warning = 'Estimated as the zonal mean of the anomaly field'

    nc_base.close()

    return


def calculate_partial_column_nc(namedata, case='TROPOS'):
    """
    This function is designed to calculate the partial column of ozone in DU.
    The idea is transform the o3_vmr field to an partial column so

    o3_vmr(lat, lon, lev) => o3_DU(lat, lon, lev) and it is the partial column
                                   until level lev.

    :param namedata:
    :param case:
    :return:
    """

    dataset_o3 = Dataset(namedata)
    alats  = dataset_o3.variables['lat'][:]
    alons  = dataset_o3.variables['lon'][:]
    # Pressure level variable were renamed for RF calculations, the
    # original dataset has the name plev so it should be quite direct
    # use plev instead of pr_lay with appropiate changes.

    ps_surf = dataset_o3.variables['ps'][:]
    times   = dataset_o3.variables['time'][:]

    pr_lev = dataset_o3.variables['plev'][:] # begins at 1000 hPa until TOA
    ao3_mmr  = dataset_o3.variables['vmro3'][:][:,:,:]  #(plev, lat, lon)

    dataset_o3.close()

    ao3_DU = calculate_partial_column(alats, alons, pr_lev,
                                      ao3_mmr, ps_surf, times, case='TOTAL')

    return a3_DU


def calculate_partial_column(alats, alons, p_lev, ao3_mmr, ps_surf, times,
                             case='TOTAL', loopin=False):
    """
    This function is designed to calculate the partial column of ozone in DU.
    The idea is transform the o3_vmr field to an partial column so

    o3_vmr(lat, lon, lev) => o3_DU(lat, lon, lev) and it is the partial column
                                   until level lev.

    :param alats: 
    :param alons: 
    :param p_lev: 
    :param ao3_mmr: 
    :param ps_surf: 
    :param times: 
    :param case: 
    :param loopin: 
    :return: 
    """

    g0 = 9.80665
    T0 = 273.15
    p0 = 101325.
    R = 287.3

    factor = 10.0*R*T0*0.5/(g0*p0)

    kg_to_g      = 1.0e+03
    ppmv_to_ppv  = 1.0e-06
    mw_o3        = 47.9982
    mw_dryair    = 28.9648

    # 0.01 to calculate in hPa, 1e6*mw_dryair/mw_o3 to change mmr to ppmv
    f_mmr_to_vmr = 0.01*1.e6*mw_dryair/mw_o3

    f_units = 0.01*1.e6

    n_lev   = len(p_lev)

    ao3_DU = np.zeros_like(ao3_mmr)
    acc_DU = np.zeros_like(ao3_mmr[:,0,:,:])
    delta_pss = np.zeros_like(times)

    if loopin:
        for ilev in tqdm(range(n_lev-1), ncols=80, desc='2nd DIM'):
            for ilat in range(len(alats)):
                lati = alats[ilat]
                for ilon in range(len(alons)):
                    delta_mmr = ao3_mmr[:,ilev,ilat,ilon]+ao3_mmr[:,ilev+1,ilat,ilon]
                    delta_pss[:] = p_lev[ilev+1]-p_lev[ilev]


                    for itim,tm in enumerate(times):
                        if p_lev[ilev+1] < ps_surf[itim, ilat, ilon]:
                            delta_pss[itim] = 0.0
                        if p_lev[ilev] > ps_surf[itim,ilat, ilon] > p_lev[ilev+1]:
                            delta_pss[itim] = ps_surf[itim, ilat, ilon]-p_lev[ilev+1]

                    acc_DU[:,ilat,ilon] = acc_DU[:,ilat,ilon]+f_units*delta_mmr*delta_pss[:]
                    #if case=='TROPOS':
                    #    if (p_lev[ilev] >  100.0*smooth_tropopause(lati)):
                    #        acc_DU[:,ilat,ilon] = acc_DU[:,ilat,ilon]+f_mmr_to_vmr*delta_mmr*delta_pss[:]
                    #if case=='TOTAL':
                    #
                    #if case=='STRATO':
                    #    if (p_lev[ilev] <  100.0*smooth_tropopause(lati)):
                    #        acc_DU[:,ilat,ilon] = acc_DU[:,ilat,ilon]+f_mmr_to_vmr*delta_mmr*delta_pss[:]
                    #if case=='SURFACE':
                    #    if (p_lev[ilev] >  70000.0):
                    #        acc_DU[:,ilat,ilon] = acc_DU[:,ilat,ilon]+f_mmr_to_vmr*delta_mmr*delta_pss[:]
            ao3_DU[:,ilev+1,:,:] = factor*copy.deepcopy(acc_DU[:,:,:])

    else:
     for itim in tqdm(range(len(times)), ncols=80, desc='1st DIM'):
        for ilev in tqdm(range(n_lev-1), ncols=80, desc='2nd DIM'):
            for ilat in range(len(alats)):
                lati = alats[ilat]
                for ilon in range(len(alons)):
                    delta_mmr = ao3_mmr[itim,ilev,ilat,ilon]+ao3_mmr[itim,ilev+1,ilat,ilon]
                    delta_prs = p_lev[ilev]-p_lev[ilev+1]

                    if ilev<10:
                        if p_lev[ilev+1] < ps_surf[itim, ilat, ilon]:
                           delta_prs = 0.0
                        if p_lev[ilev] > ps_surf[itim,ilat, ilon] > p_lev[ilev+1]:
                           delta_prs = ps_surf[itim, ilat, ilon]-p_lev[ilev+1]

                    acc_DU[itim,ilat,ilon] = acc_DU[itim,ilat,ilon]+f_units*delta_mmr*delta_prs
            ao3_DU[itim,ilev+1,:,:] = factor*copy.deepcopy(acc_DU[itim,:,:])

    return ao3_DU


def mean_netcdf_cmam_waccm(file_cmam, file_waccm,
                           varname='vmro3', vertical_w=False, extend=True):
    """
    This function shows how to perform a mean of two model datasets given
    the functions present in this module.

    :param file_cmam:
    :param file_waccm:
    :param varname:
    :param vertical_w:
    :param extend:
    :return:
    """


    info_waccm = file_waccm.split('_')
    period = info_waccm[3]
    print('Merging to files: ', period)
    print('                  ', file_cmam)
    print('                  ', file_waccm)


    if extend==True:
        strextend = ''
    else:
        strextend = '_noextend'

    dir_out = '../OUTPUT/'
    if vertical_w == False:
        surname = 'vmro3_CMIP6_v1.0_py_'+period+'_monthly_standard_weights05.nc'
        model_mean = dir_out+surname
    if vertical_w==True:
        surname = 'vmro3_CMIP6_v1.0_py_'+period+'_monthly_standard_weightsA.nc'
        model_mean = dir_out+surname
    if vertical_w=='cmip6':
        surname = 'vmro3_CMIP6_v1.0_py_'+period+'_monthly_standard_weightsCMIP6.nc'
        model_mean = dir_out+surname

    nc_CMAM = Dataset(file_cmam, mode='r')
    nc_WACC = Dataset(file_waccm, mode='r')

    # Variables

    cmam_lev = nc_CMAM['plev'][:]
    wacc_lev = nc_WACC['plev'][:]

    cmam_lon = nc_CMAM['lon'][:]
    wacc_lon = nc_WACC['lon'][:]

    cmam_lat = nc_CMAM['lat'][:]
    wacc_lat = nc_WACC['lat'][:]

    cmam_tim = nc_CMAM['time'][:][:]
    wacc_tim = nc_WACC['time'][:][:]

    # These arrays should be identical for both files, we could test:

    if not np.allclose(cmam_lev, wacc_lev):
        print('Problem with levels')
        exit()
    if not np.allclose(cmam_lat, wacc_lat):
        print('Problem with lats')
        exit()
    if not np.allclose(cmam_lon, wacc_lon):
        print('Problem with lons')
        exit()
    wacc_ozo = nc_WACC[varname][:][:,:,:,:]
    cmam_ozo = nc_CMAM[varname][:][:,:,:,:]

    nc_mean = Dataset(model_mean, mode='w', format='NETCDF4')

    # We create the main dimensions
    nc_mean.createDimension('time', 0)
    nc_mean.createDimension('plev', len(cmam_lev))
    nc_mean.createDimension('lat',  len(cmam_lat))
    nc_mean.createDimension('lon', len(cmam_lon))
    nc_mean.createDimension('bnds', 2)

    time = nc_mean.createVariable('time',      'f8', ('time',))
    lats = nc_mean.createVariable('lat',  'f4', ('lat',))
    levs = nc_mean.createVariable('plev',      'f4', ('plev',))
    lons = nc_mean.createVariable('lon', 'f4', ('lon',))


    lons.units = 'degrees east'
    lons.standard_name = "longitude"
    lons.long_name = "longitude"

    lats.units = 'degrees north'
    lats.standard_name = "latitude"
    lats.long_name = "latitude"
    levs.units = 'Pa'

    time.units = 'days since 1850-01-01 00:00:00'
    time.calendar = 'standard'

    time[:] = wacc_tim
    levs[:] = cmam_lev
    lats[:] = cmam_lat
    lons[:] = cmam_lon

    vmr = nc_mean.createVariable('vmro3', 'f8',
                                 ('time', 'plev', 'lat', 'lon'),
                                 fill_value=-999.9)
    vmr.units = 'mole mole -1'
    vmr.standard_name = 'mole_fraction_of_ozone_in_air'

    pr_surf = nc_mean.createVariable('ps', 'f8',
                                 ('time', 'lat', 'lon'),
                                 fill_value=-999.9)
    pr_surf.units = 'Pa'
    pr_surf.standard_name = 'Surface Pressure'
    pr_surf.warning = 'Surface Pressure from cesm1-waccm'

    pr_surf = nc_WACC['ps'][:]

    print(    '=========== vertical mean calculation ===')
    if vertical_w== True:
        print('        --- own method                  >')

        vmr[:] = vertical_mean(cmam_ozo, wacc_ozo, cmam_lev)
        vmr.warning = 'merging method Ramiro'

    if vertical_w== False:
        print('        --- typical mean                >')
        vmr[:] =(cmam_ozo+wacc_ozo)*0.5
        vmr.warning = 'merging method 0.5 each'
        print('-----------------------------          ok')

    if vertical_w== 'cmip6':
        print('        --- cmip6 mean                  >')
        vmr[:] = vertical_mean_cmip6(cmam_ozo, wacc_ozo, cmam_lev)
        vmr.warning = 'merging method cmip6'


    nc_CMAM.close()
    nc_WACC.close()
    nc_mean.close()
    gc.collect()

    create_seasonal_zonal_du(model_mean, var_name='vmro3', add_du=True)
    gc.collect()

    return model_mean
