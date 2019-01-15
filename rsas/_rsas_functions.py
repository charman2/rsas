# cython: profile=True
# -*- coding: utf-8 -*-
"""
.. module:: _rsas_functions
   :platform: Unix, Windows
   :synopsis: Time-variable transport using storage selection (SAS) functions

.. moduleauthor:: Ciaran J. Harman
"""

from __future__ import division
import numpy as np
dtype = np.float64
import scipy.stats
from scipy.special import gamma as gamma_function
from scipy.special import gammainc, gammaincinv
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

def create_function(rSAS_type, params):
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
                     'exponential':_exponential_rSAS,
                     'uniform':_uniform_rSAS,
                     'power':_power_rSAS,
                     'kumaraswami':_kumaraswami_rSAS,
                     'invgauss':_invgauss_rSAS,
                     'triangle':_triangle_rSAS,
                     'lookuptable':_lookup_rSAS}
    if rSAS_type in function_dict.keys():
        return function_dict[rSAS_type](params)
    elif hasattr(scipy.stats, rSAS_type):
        return _stats_rSAS(rSAS_type, params)
    else:
        raise ValueError('No such rSAS function type')


class rSASFunctionClass:
    """Base class for constructing rSAS functions

    All rSAS functions must override the following methods:

    __init__(self, params):
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
    def __init__(self, params):
        raise NotImplementedError('__init__ not implemented in derived rSASFunctionClass')
    def cdf_all(self, ST):
        raise NotImplementedError('cdf_all not implemented in derived rSASFunctionClass')
    def cdf_i(self, ST, i):
        raise NotImplementedError('cdf_i not implemented in derived rSASFunctionClass')

class _power_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.scale = params[:,2]
        self.bT = params[:,3]
        self.rescale = np.where(np.isfinite(self.ST_max), 1./(
             1 - ( 1 - ((self.ST_max - self.ST_min)/self.scale))**(1./(2-self.bT))), 
                                                          1.)
    def cdf_all(self, ST):
        return np.where(ST > self.ST_min, 
                    np.where(ST < self.ST_max,
                         1 - ( 1 - ((ST - self.ST_min)/(self.scale)))**(1./(2-self.bT)), 
                         1.), 
                    0.)
    def cdf_i(self, ST, i):
        return np.where(ST > self.ST_min[i], 
                    np.where(ST < self.ST_max[i],
                         1 - ( 1 - ((ST - self.ST_min[i])/(self.scale[i])))**(1./(2-self.bT[i])), 
                         1.), 
                    0.)
    def invcdf_i(self, P, i):
        return np.where(P>0, 
                    np.where(P<1, 
                         (1-(1-P)**(2-self.bT[i])),
                        np.inf), 
                    np.nan) * self.scale[i] + self.ST_min[i]

class _kumaraswami_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.a = params[:,2]
        self.b = params[:,3]
    def cdf_all(self, ST):
        return np.where(self.ST_max>=self.ST_min,
                        np.where(ST > self.ST_min, np.where(ST < self.ST_max,
                                 1 - ( 1 - ((ST - self.ST_min)/(self.ST_max - self.ST_min))**self.a)**self.b, 1.), 0.),
                        np.where(ST > self.ST_min,
                                 1 - ( 1 - ((ST - self.ST_min)/(self.ST_max - self.ST_min))**self.a)**self.b, 0.))
    def cdf_i(self, ST, i):
        return np.where(self.ST_max[i]>=self.ST_min[i],
                        np.where(ST > self.ST_min[i], np.where(ST < self.ST_max[i],
                                 1 - ( 1 - ((ST - self.ST_min[i])/(self.ST_max[i] - self.ST_min[i]))**self.a[i])**self.b[i], 1.), 0.),
                        np.where(ST > self.ST_min[i],
                                 1 - ( 1 - ((ST - self.ST_min[i])/(self.ST_max[i] - self.ST_min[i]))**self.a[i])**self.b[i], 0.))
    def invcdf_i(self, P, i):
        return np.where(self.ST_max[i]>=self.ST_min[i],
                        np.where(P > 0, np.where(P < 1,
                                 (1-(1-P)**(1/self.b[i]))**(1/self.a[i])*(self.ST_max[i] - self.ST_min[i]) + self.ST_min[i], self.ST_max[i]), self.ST_min[i]),
                        np.where(P > 0,
                                 (1-(1-P)**(1/self.b[i]))**(1/self.a[i])*(self.ST_max[i] - self.ST_min[i]) + self.ST_min[i], self.ST_min[i]))

class _uniform_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.lam = 1.0/(self.ST_max-self.ST_min)
    def cdf_all(self, ST):
        return np.where(ST < self.ST_max, np.where(ST > self.ST_min,  self.lam * (ST - self.ST_min), 0.), 1.)
    def cdf_i(self, ST, i):
        return np.where(ST < self.ST_max[i], np.where(ST > self.ST_min[i], self.lam[i] * (ST - self.ST_min[i]), 0.), 1.)
    def invcdf_i(self, P, i):
        return np.where(P < 1, np.where(P > 0, P, 0.), 1.)/self.lam[i] + self.ST_min[i]

class _triangle_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.ST_mode = params[:,2]
        self.ST_mode = np.maximum(self.ST_mode, self.ST_min)
        self.ST_mode = np.minimum(self.ST_mode, self.ST_max)
        self.lama = 1.0/(self.ST_max-self.ST_min)/(self.ST_mode-self.ST_min)
        self.lamb = 1.0/(self.ST_max-self.ST_min)/(self.ST_max-self.ST_mode)
        self.Pi = (self.ST_mode-self.ST_min)/(self.ST_max-self.ST_min)
    def cdf_all(self, ST):
        return np.where(ST < self.ST_max, 
                    np.where(ST < self.ST_mode,
                        np.where(ST < self.ST_min,
                            0.,
                            self.lama * (ST - self.ST_min)**2
                        ),
                        1-self.lamb * (self.ST_max - ST)**2
                    ),
                    1.
                )
    def cdf_i(self, ST, i):
        return np.where(ST < self.ST_max[i], 
                    np.where(ST < self.ST_mode[i],
                        np.where(ST < self.ST_min[i],
                            0.,
                            self.lama[i] * (ST - self.ST_min[i])**2
                        ),
                        1-self.lamb[i] * (self.ST_max[i] - ST)**2
                    ),
                    1.
                )
    def invcdf_i(self, P, i):
        return np.where(P < 1,
                    np.where(P < self.Pi[i],
                        np.where(P < 0,
                            self.ST_min[i],
                            self.ST_min[i] + np.sqrt(P/self.lama[i])
                        ),
                        self.ST_max[i] - np.sqrt((1-P)/self.lamb[i])
                    ),
                    self.ST_max[i]
                )

class _invgauss_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.loc = params[:,0]
        self.scale = params[:,1]
        self.mu = params[:,2:]
    def cdf_i(self, ST, i):
        x = (ST - self.loc[i]) / self.scale[i]
        return (erfc((-x + self.mu[i])/(np.sqrt(2*x)*self.mu[i]))
                + np.exp(2/self.mu[i])*erfc((x + self.mu[i])/(np.sqrt(2*x)*self.mu[i])))/2.

class _stats_rSAS(rSASFunctionClass):
    def __init__(self, rSAS_type, params):
        params = params.copy()
        self.dist_class = getattr(scipy.stats, rSAS_type)
        self.loc = params[:,0]
        self.scale = params[:,1]
        N = params.shape[0]
        if params.shape[1]>2:
            self.shape = params[:,2:]
            self.dist = [self.dist_class(*self.shape[i], loc=self.loc[i], scale=self.scale[i]) for i in range(N)]
        else:
            self.dist = [self.dist_class(loc=self.loc[i], scale=self.scale[i]) for i in range(N)]
    #def cdf_all(self, ST):
    #    return [dist.cdf(STi) for dist, STi in zip(self.dist, ST)]
    def cdf_i(self, ST, i):
        return [self.dist[i].cdf(STi) for STi in ST]

class _exponential_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.scale = params[:,2]
        self.lam = 1.0/self.scale
        self.rescale = np.where(np.isfinite(self.ST_max), 1/(1-np.exp(-self.lam * (self.ST_max-self.ST_min))), 1.)
    def cdf_all(self, ST):
        return np.where(ST>self.ST_min, np.where(ST<self.ST_max,
                (1-np.exp(-self.lam * (ST-self.ST_min)))*self.rescale, 1.), 0.)
    def cdf_i(self, ST, i):
        return np.where(ST>self.ST_min[i], np.where(ST<self.ST_max[i],
                (1-np.exp(-self.lam[i] * (ST-self.ST_min[i])))*self.rescale[i], 1.), 0.)
    def invcdf_i(self, P, i):
        return np.where(P>=0, 
                   np.where(P<1, 
                       - self.scale[i] * np.log(1-P/self.rescale[i]),
                       np.inf),
                   np.nan) + self.ST_min[i]

class _gamma_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.ST_min = params[:,0]
        self.ST_max = params[:,1]
        self.scale = params[:,2]
        self.a = params[:,3]
        self.lam = 1.0/self.scale
        self.lam_on_gam = self.lam**self.a / gamma_function(self.a)
        self.rescale = np.where(np.isfinite(self.ST_max), 1/(gammainc(self.a, self.lam*(self.ST_max-self.ST_min))), 1.)
    def cdf_all(self, ST):
        return np.where(ST>self.ST_min, np.where(ST<self.ST_max,
                gammainc(self.a, self.lam*(ST-self.ST_min))*self.rescale, 1.), 0.)
    def cdf_i(self, ST, i):
        return np.where(ST>self.ST_min[i], np.where(ST<self.ST_max[i],
                gammainc(self.a[i], self.lam[i]*(ST-self.ST_min[i]))*self.rescale[i], 1.), 0.)
    def invcdf_i(self, P, i):
        return np.where(P>=0, np.where(P<1, gammaincinv(self.a[i], P/self.rescale[i]),
                np.inf), np.nan)/self.lam[i] + self.ST_min[i]

class invariant:
    def __init__(self, value):
        self.value = value
    def __getitem__(self, anyindex):
        return self.value

class _lookup_rSAS(rSASFunctionClass):
    def __init__(self, params):
        params = params.copy()
        self.S_T = params[:,0]
        self.Omega = params[:,1]
        self.ST_min = invariant(self.S_T[0])
        self.ST_max = invariant(self.S_T[-1])
        if not (self.Omega[0]==0 and self.Omega[-1]==1):
            raise ValueError('The first and last value of S_T must correspond with probability 0 and 1 respectively')
        self.interp1d = interp1d(self.S_T, self.Omega, kind='linear', copy=False, bounds_error=True, assume_sorted=True)
        self.interp1d_inv = interp1d(self.Omega, self.S_T, kind='linear', copy=False, bounds_error=True, assume_sorted=True)
    def cdf_all(self, ST):
        return self.interp1d(np.where(ST < self.ST_max[0], np.where(ST > self.ST_min[0], ST, 0.), 1.))
    def cdf_i(self, ST, i):
        return self.interp1d(np.where(ST < self.ST_max[0], np.where(ST > self.ST_min[0], ST, 0.), 1.))
    def invcdf_i(self, P, i):
        return np.where(P>=self.Omega[0], np.where (P<=self.Omega[-1], self.interp1d_inv(P), self.ST_max[0]), self.ST_min[0])
