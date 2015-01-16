# cython: profile=True
# -*- coding: utf-8 -*-
"""
.. module:: rsas
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

# for debugging
debug = True
def _verbose(statement):
    """Prints debuging messages if rsas_functions.debug==True
    
    """
    if debug:
        print statement

def rSAS_setup(rSAS_type, np.ndarray[dtype_t, ndim=2] params):
    """Initialize an rSAS function

    Parameters
    ----------
    rSAS_type : str
        A string indicating the requested rSAS functional form.
    params : n x k float64 ndarray
        Parameters for the rSAS function. The number of columns and 
        their meaning depends on which rSAS type is chosen. For all the rSAS 
        functions implemented so far, each row corresponds with a timestep.

    Returns
    ----------
    rSAS_fun : rSASFunctionClass
        An rsas function of the chosen type
        
    The created function object will have methods that vary between types. All
    must have a constructor ("__init__") and two methods cdf_all and cdf_i.
    
    Implemented rSAS functions
    --------------------------
    Available choices for rSAS_type, and description of parameter array.
    These all take one parameter set (row) per timestep:
    
    'uniform'
        Uniform distribution over the range [a, b].
            Q_params[:, 0] : a
            Q_params[:, 1] : b
    'gamma'
        Gamma distribution
            Q_params[:, 0] : shift parameter
            Q_params[:, 1] : scale parameter
            Q_params[:, 2] : shape parameter
    'gamma_trunc'
        Gamma distribution, truncated at a maximum value
            Q_params[:, 0] : shift parameter
            Q_params[:, 1] : scale parameter
            Q_params[:, 2] : shape parameter
            Q_params[:, 3] : maximum value
    'SS_invgauss'
        Produces analytical solution to the advection-dispersion equation
        (inverse Gaussian distribution) under steady-state flow.
            Q_params[:, 0] : scale parameter
            Q_params[:, 1] : Peclet number
    'SS_mobileimmobile'
        Produces analytical solution to the advection-dispersion equation with
        linear mobile-immobile zone exchange under steady-state flow.
            Q_params[:, 0] : scale parameter
            Q_params[:, 1] : Peclet number
            Q_params[:, 2] : beta parameter
    """
    if rSAS_type == 'gamma':
        return gamma_rSAS(params)
    elif rSAS_type == 'gamma_trunc':
        return gamma_trunc_rSAS(params)
    elif rSAS_type == 'SS_invgauss':
        return SS_invgauss_rSAS(params)
    elif rSAS_type == 'SS_mobileimmobile':
        return SS_mobileimmobile_rSAS(params)
    elif rSAS_type == 'uniform':
        return uniform_rSAS(params)
        
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
        print "__init__ not constructed!"
    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        print "cdf_all not constructed!"
    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        print "cdf_i not constructed!"
    
class uniform_rSAS(rSASFunctionClass):
    def __init__(self, np.ndarray[dtype_t, ndim=2] params):
        self.a = params[:,0]
        self.b = params[:,1]
        self.lam = 1.0/(self.b-self.a)
    def pdf(self, np.ndarray[dtype_t, ndim=1] ST):
        return np.where(ST <= self.scale, self.lam, 0.)
    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        return np.where(np.logical_and(ST >= self.a, ST <= self.b), self.lam * (ST - self.a), 1.)
    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        return np.where(np.logical_and(ST >= self.a[i], ST <= self.b[i]), self.lam[i] * (ST - self.a[i]), 1.)

class gamma_rSAS(rSASFunctionClass): 
    def __init__(self, np.ndarray[dtype_t, ndim=2] params):
        self.shift = params[:,0]
        self.scale = params[:,1]
        self.a = params[:,2]
        self.lam = 1.0/self.scale
        self.lam_on_gam = self.lam**self.a / gamma_function(self.a)
    def pdf(self, np.ndarray[dtype_t, ndim=1] ST):
        return np.where(ST-self.shift > 0, (self.lam_on_gam * (ST-self.shift)**(self.a-1) * np.exp(-self.lam*(ST-self.shift))), 0.)
    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        return np.where(ST-self.shift > 0, gammainc(self.a, self.lam*(ST-self.shift)), 0.)
    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        return np.where(ST-self.shift[i] > 0, gammainc(self.a[i], self.lam[i]*(ST-self.shift[i])), 0.)
        
class gamma_trunc_rSAS(rSASFunctionClass): 
    def __init__(self, np.ndarray[dtype_t, ndim=2] params):
        self.shift = params[:,0]
        self.scale = params[:,1]
        self.a = params[:,2]
        self.max = params[:,3]
        self.lam = 1.0/self.scale
        self.lam_on_gam = self.lam**self.a / gamma_function(self.a)
        self.rescale = 1/(gammainc(self.a, self.lam*(self.max-self.shift)))
    def pdf(self, np.ndarray[dtype_t, ndim=1] ST):
        return np.where(ST-self.shift > 0, (self.lam_on_gam * (np.minimum(ST, self.max)-self.shift)**(self.a-1) * np.exp(-self.lam*(np.minimum(ST, self.max)-self.shift)))*self.rescale, 0.)
    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        return np.where(ST-self.shift > 0, gammainc(self.a, self.lam*(np.minimum(ST, self.max)-self.shift))*self.rescale, 0.)
    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        return np.where(ST-self.shift[i] > 0, gammainc(self.a[i], self.lam[i]*(np.minimum(ST, self.max[i])-self.shift[i]))*self.rescale[i], 0.)

class SS_invgauss_rSAS(rSASFunctionClass):
    def __init__(self, np.ndarray[dtype_t, ndim=2] params):
        self.scale = params[:,0]
        self.Pe = params[:,1]
        self.lastx = np.zeros(len(params))
        self.i = 0

    def F1(self, dtype_t x, dtype_t Pe):
        return erfc((1 - x) / np.sqrt(8 * x / Pe)) / 2.

    def F2(self, dtype_t x, dtype_t Pe):
        return erfc((1 + x) / np.sqrt(8 * x / Pe)) / 2. * np.exp(Pe / 2)

    def STx(self, dtype_t x):
        return (x + (1 - x) * self.F1(x, self.Pe[self.i]) - (1 + x) * self.F2(x, self.Pe[self.i]))

    def Omegax(self, dtype_t x):
        return self.F1(x, self.Pe[self.i]) + self.F2(x, self.Pe[self.i])

    def get_x(self, dtype_t ST):
        fun = lambda X: ST - self.STx(X)
        jac = lambda X: - self.dSTdx(X)
        x = fsolve(func=fun, fprime=jac, x0=[self.lastx[self.i]])
        self.lastx[self.i] = x[0]
        return x[0]

    def Omega(self, dtype_t ST_norm):
        x = self.get_x(ST_norm)
        return self.Omegax(x)

    def pdf(self, np.ndarray[dtype_t, ndim=1] ST):
        _verbose('SS_invgauss_rSAS > pdf not implemented')
        return None

    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        N = len(ST)
        return_cdf = np.zeros(N)
        for i in range(N):
            return_cdf[i] = self.Omega(ST[i]/self.scale[i])
        return return_cdf

    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        N = len(ST)
        return_cdf = np.zeros(N)
        for j in range(N):
            if ST[j]==0.:
                return_cdf[j] = 0.
            else:
                return_cdf[j] = self.Omega(ST[j]/self.scale[i])
        return return_cdf

class SS_mobileimmobile_rSAS(rSASFunctionClass):
    def __init__(self, np.ndarray[dtype_t, ndim=2] params):
        self.S_Mob = params[:,0]
        self.Pe = params[:,1]
        self.beta = params[:,2]
        self.rho = np.sqrt(1 - 8 / (self.Pe * self.beta))
        self.lastx = np.zeros(len(params))
        self.i = 0
        self.S_init_scale = self.S_Mob

    def F1(self, dtype_t x, dtype_t Pe):
        return erfc((1 - x) / np.sqrt(8 * x / Pe)) / 2.
        
    def F2(self, dtype_t x, dtype_t Pe):
        return erfc((1 + x) / np.sqrt(8 * x / Pe)) / 2. * np.exp(Pe/2)
                
    def Omegax(self, dtype_t x):
        ADE_term = self.F1(x, self.Pe[self.i]) + self.F2(x, self.Pe[self.i])
        Cross_term = self.F1(self.rho[self.i] * x, self.rho[self.i] * self.Pe[self.i]) + self.F2(self.rho[self.i] * x, self.rho[self.i] * self.Pe[self.i])
        Decay_term = np.exp(self.Pe[self.i] * (1 - self.rho[self.i])/4 - x / self.beta[self.i])
        return ADE_term - Cross_term * Decay_term

    def Omega_IM_inv(self, dtype_t q):
        return q * self.beta[self.i] * self.S_Mob[self.i]

    def STx(self, dtype_t x):
        q = self.Omegax(x)
        ST_IM = self.Omega_IM_inv(q)
        ST_ADE = self.S_Mob[self.i] * (x + (1 - x) * self.F1(x, self.Pe[self.i]) - (1 + x) * self.F2(x, self.Pe[self.i]))
        return ST_ADE + ST_IM

    def dSTdx(self, dtype_t x):
        return self.S_Mob[self.i] * (1 - self.Omegax(x))

    def get_x(self, dtype_t ST):
        fun = lambda X: ST - self.STx(X)
        jac = lambda X: np.array([- self.dSTdx(X)])
        x = fsolve(func=fun, fprime=jac, x0=[0.0])#ST/self.S_init_scale[self.i]])
        self.lastx[self.i] = x[0]
        return x[0]

    def Omega(self, dtype_t ST):
        x = self.get_x(ST)
        result = self.Omegax(x)
        if result==1.0:
            self.lastx[self.i] = 0.0
            x = self.get_x(ST)
            result = self.Omegax(x)
        return result

    def pdf(self, np.ndarray[dtype_t, ndim=1] ST):
        _verbose('SS_invgauss_rSAS > pdf not implemented')
        return None

    def cdf_all(self, np.ndarray[dtype_t, ndim=1] ST):
        N = len(ST)
        return_cdf = np.zeros(N)
        for i in range(N):
            self.i = i
            return_cdf[i] = self.Omega(ST[i])
        return return_cdf

    def cdf_i(self, np.ndarray[dtype_t, ndim=1] ST, int i):
        N = len(ST)
        return_cdf = np.zeros(N)
        self.i = i
        for j in range(N):
            if ST[j]==0.:
                return_cdf[j] = 0.
            else:
                return_cdf[j] = self.Omega(ST[j])
        return return_cdf