import cython
import numpy as np
cimport numpy as np
from warnings import warn
dtype = np.float64
ctypedef np.float64_t dtype_t
ctypedef np.int_t inttype_t
ctypedef np.long_t longtype_t
cdef inline np.float64_t float64_max(np.float64_t a, np.float64_t b): return a if a >= b else b
cdef inline np.float64_t float64_min(np.float64_t a, np.float64_t b): return a if a <= b else b
from _rsas_functions import rSASFunctionClass
from scipy.special import gamma as gamma_function
from scipy.special import gammainc
from scipy.special import erfc
from scipy.interpolate import interp1d
from scipy.optimize import fmin, minimize_scalar, fsolve
import time
import rsas._util



@cython.boundscheck(False)
@cython.wraparound(False)
def transport(np.ndarray[dtype_t, ndim=2] PQ, np.ndarray[dtype_t, ndim=1] C_in, float C_old):
    """Apply a time-varying transit time distribution to an input concentration timseries

    Args:
        PQ : numpy float64 2D array, size N x N
            The CDF of the backwards transit time distribution P_Q1(T,t)
        C_in : numpy float64 1D array, length N.
            Timestep-averaged inflow concentration.
        C_old : numpy float64 1D array, length N.
            Concentration to be assumed for portion of outflows older than initial
            timestep

    Returns:
        C_out : numpy float64 1D array, length N.
            Timestep-averaged outflow concentration.
        C_mod_raw : numpy float64 1D array, length N.
            Timestep-averaged outflow concentration, prior to correction with C_old.
        observed_fraction : numpy float64 1D array, length N.
            Fraction of outflow older than the first timestep
    """
    cdef int N, t, T, ti
    cdef np.ndarray[dtype_t, ndim=2] pQe
    cdef np.ndarray[dtype_t, ndim=1] C_mod_raw, observed_fraction
    N = len(C_in)
    C_mod_raw = np.zeros(N, dtype=np.float64)
    pQe = np.diff(PQ[:,1:],axis=0)
    for T in range(pQe.shape[0]):
        for t in range(pQe.shape[1]):
            ti = t - T
            C_mod_raw[t] += C_in[ti] * pQe[T,t]
    observed_fraction = np.diag(PQ[1:,1:]).copy()
    C_mod = (C_mod_raw + (1-observed_fraction) * C_old)
    return C_mod, C_mod_raw, observed_fraction
