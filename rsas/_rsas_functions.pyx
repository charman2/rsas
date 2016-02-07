# cython: profile=True
# -*- coding: utf-8 -*-
"""
.. module:: _rsas_functions
   :platform: Unix, Windows
   :synopsis: Time-variable transport using storage selection (SAS) functions

.. moduleauthor:: Ciaran J. Harman
"""

from __future__ import division
import cython
import numpy as np
cimport numpy as np
dtype = np.float64
ctypedef np.float64_t dtype_t
ctypedef np.int_t inttype_t
ctypedef np.long_t longtype_t
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b
from scipy.special import gamma as gamma_function
from scipy.special import gammainc
from scipy.special import erfc
from scipy.interpolate import interp1d
from scipy.optimize import fmin, minimize_scalar, fsolve
import time
#import rsas._util

# for debugging
debug = True
def _verbose(statement):
    """Prints debuging messages if rsas_functions.debug==True

    """
    if debug:
        print statement

def create_function(rSAS_type, np.ndarray[dtype_t, ndim=2] params):
    """Initialize an rSAS function

    Args:
        rSAS_type : str
            A string indicating the requested rSAS functional form.
        params : n x k float64 ndarray
            Parameters for the rSAS function. The number of columns and
            their meaning depends on which rSAS type is chosen. For all the rSAS
            functions implemented so far, each row corresponds with a timestep.

    Returns:
        rSAS_fun : rSASFunctionClass
            An rSAS function of the chosen type

    The created function object will have methods that vary between types. All
    must have a constructor ("__init__") and two methods cdf_all and cdf_i. See
    the documentation for rSASFunctionClass for more information.

    Available choices for rSAS_type, and a description of parameter array, are below.
    These all take one parameter set (row) per timestep:

    'uniform' : Uniform distribution over the range [a, b].
            * ``Q_params[:, 0]`` = ST_min
            * ``Q_params[:, 1]`` = ST_max
    'kumaraswami': Kumaraswami distribution
            * ``Q_params[:, 0]`` = ST_min parameter
            * ``Q_params[:, 1]`` = ST_max parameter
            * ``Q_params[:, 2]`` = a parameter
            * ``Q_params[:, 3]`` = b parameter
    'gamma' : Gamma distribution, truncated at a maximum value (can be inf)
            * ``Q_params[:, 0]`` = ST_min parameter
            * ``Q_params[:, 1]`` = ST_max parameter (set as inf if undefined)
            * ``Q_params[:, 2]`` = scale parameter
            * ``Q_params[:, 3]`` = shape parameter
    'lookuptable' : Lookup table of values. First value of Omega must be 0 and last must be 1.
            * ``Q_params[:, 0]`` = S_T
            * ``Q_params[:, 1]`` = Omega(S_T)
    """
    function_dict = {'gamma':_gamma_rSAS,
                     'uniform':_uniform_rSAS,
                     'kumaraswami':_kumaraswami_rSAS,
                     'lookuptable':_lookup_rSAS}
                     #'from_steady_state_TTD':_from_steady_state_TTD_rSAS}
    return function_dict[rSAS_type](params)

class rSASFunctionClass:
    """Base class for constructing rSAS functions

    All rSAS functions must override the following methods:

    __init__(self, np.ndarray[dtype_t, ndim=2] params):
        Initializes the rSAS function for a timeseries of parameters params.
        Usually each row of params corresponds to a timestep and each column is
        a parameter, but this is not strictly required as long as cdf_all and
        cdf_i do what you want them to.
    rSAS_fun.cdf_all(ndarray ST)
        returns the cumulative distribution function for an array ST (which
        must be the same length as the params matrix used to create the
        function). Each value of ST is evaluated using the parameter values
        on the respective row of params
    rSAS_fun.cdf_i(ndarray ST, int i)
        returns the cumulative distribution function for an array ST (which
        can be of any size). Each value of ST is evaluated using the
        parameter values on row i.
    """
    def __init__(self, np.ndarray[dtype_t, ndim=2] params):
        raise NotImplementedError('__init__ not implemented in derived rSASFunctionClass')
    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        raise NotImplementedError('cdf_all not implemented in derived rSASFunctionClass')
    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        raise NotImplementedError('cdf_i not implemented in derived rSASFunctionClass')

class _kumaraswami_rSAS(rSASFunctionClass):
    def __init__(self, np.ndarray[dtype_t, ndim=2] params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.a = params[:,2]
        self.b = params[:,3]
    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        return np.where(self.ST_max>=self.ST_min,
                        np.where(ST > self.ST_min, np.where(ST < self.ST_max,
                                 1 - ( 1 - ((ST - self.ST_min)/(self.ST_max - self.ST_min))**self.a)**self.b, 1.), 0.),
                        np.where(ST > self.ST_min,
                                 1 - ( 1 - ((ST - self.ST_min)/(self.ST_max - self.ST_min))**self.a)**self.b, 0.))
    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        return np.where(self.ST_max[i]>=self.ST_min[i],
                        np.where(ST > self.ST_min[i], np.where(ST < self.ST_max[i],
                                 1 - ( 1 - ((ST - self.ST_min[i])/(self.ST_max[i] - self.ST_min[i]))**self.a[i])**self.b[i], 1.), 0.),
                        np.where(ST > self.ST_min[i],
                                 1 - ( 1 - ((ST - self.ST_min[i])/(self.ST_max[i] - self.ST_min[i]))**self.a[i])**self.b[i], 0.))

class _uniform_rSAS(rSASFunctionClass):
    def __init__(self, np.ndarray[dtype_t, ndim=2] params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.lam = 1.0/(self.ST_max-self.ST_min)
    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        return np.where(ST < self.ST_max, np.where(ST > self.ST_min,  self.lam * (ST - self.ST_min), 0.), 1.)
    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        return np.where(ST < self.ST_max[i], np.where(ST > self.ST_min[i], self.lam[i] * (ST - self.ST_min[i]), 0.), 1.)

class _gamma_rSAS(rSASFunctionClass):
    def __init__(self, np.ndarray[dtype_t, ndim=2] params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.scale = params[:,2]
        self.a = params[:,3]
        self.lam = 1.0/self.scale
        self.lam_on_gam = self.lam**self.a / gamma_function(self.a)
        self.rescale = np.where(np.isfinite(self.ST_max), 1/(gammainc(self.a, self.lam*(self.ST_max-self.ST_min))), 1.)
    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        return np.where(ST>self.ST_min, np.where(ST<self.ST_max,
                gammainc(self.a, self.lam*(ST-self.ST_min))*self.rescale, 1.), 0.)
    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        return np.where(ST>self.ST_min[i], np.where(ST<self.ST_max[i],
                gammainc(self.a[i], self.lam[i]*(ST-self.ST_min[i]))*self.rescale[i], 1.), 0.)

class _lookup_rSAS(rSASFunctionClass):
    def __init__(self, np.ndarray[dtype_t, ndim=2] params):
        params = params.copy()
        self.S_T = params[:,0]
        self.Omega = params[:,1]
        self.ST_min = self.S_T[0]
        self.ST_max = self.S_T[-1]
        if not (self.Omega[0]==0 and self.Omega[-1]==1):
            raise ValueError('The first and last value of S_T must correspond with probability 0 and 1 respectively')
        self.interp1d = interp1d(self.S_T, self.Omega, kind='linear', copy=False, bounds_error=True, assume_sorted=True)
    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        return self.interp1d(np.where(ST < self.ST_max, np.where(ST > self.ST_min,  ST, 0.), 1.))
    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        return self.interp1d(np.where(ST < self.ST_max, np.where(ST > self.ST_min, ST, 0.), 1.))
