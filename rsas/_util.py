import numpy as np
from scipy.integrate import cumtrapz
import scipy.weave

def steady_state_ST_from_TTD_1out(Q, TTD_CDF, dt=1):
    """Steady-state age-ranked storage
    
    At a given steady state flux rate Q, the age-ranked storage is given by:
    
    $$S_T = Q_0\left(T-\int_0^T\Pb{Q}(\tau)d\tau\right)$$
    
    Parameters:
    -----------
        TTD_CDF : ndarray
            Array of the cumulative distribution function, starting with T=0
        Q : float
            Flux rate
        dt : float (default=1)
            Timestep
        
    """
    T = np.arange(len(TTD_CDF)) * dt
    return Q * (T - cumtrapz(TTD_CDF, initial=0.) * dt)

def steady_state_ST_from_TTD_2out(TTD_CDF1, Q1, TTD_CDF2, Q2, dt=1):
    """Steady-state age-ranked storage with partitioning
    
    At a given steady state flux rate Q, the age-ranked storage is given by:
    
    $$S_T = Q_0\left(T-\int_0^T\Pb{Q}(\tau)d\tau\right)$$
    
    Parameters:
    -----------
        TTD_CDF1, TTD_CDF2 : ndarray
            Arrays of the cumulative distribution function, starting with T=0
            must be the same length
        Q1, Q2 : float
            Flux rates out. Assumed to sum to the influx rate.
        dt : float (default=1)
            Timestep
        
    """
    T = np.arange(len(TTD_CDF1)) * dt
    J = Q1 + Q2
    return J * T - Q1 * cumtrapz(TTD_CDF1, initial=0.) * dt - Q2 * cumtrapz(TTD_CDF1, initial=0.) * dt

def mean_TTD(PQ):
    """Unweighted mean TTD from PQ
    
    """
    N = PQ.size(axis=0)
    return np.cumsum(np.diff(PQ,axis=0).sum(axis=1)/(np.arange(N)[::-1]+1))
    
def transport(PQ, C_in, C_old):
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

    N = len(C_in)
    C_in=np.array(C_in)
    PQ = np.array(PQ)
    thetaQ = np.array(thetaQ)
    thetaS = np.array(thetaS)
    C_mod_raw = np.zeros(N, dtype=np.float64)
    pQe = np.where(thetaQ[1:,1:]>0, np.diff(PQ[:,1:],axis=0)/(thetaS[1:,1:] + thetaQ[1:,1:]), 0.)
    code = r"""
    int i, j, k;
    for(j=0; j<N; j++)
        for(i=0; i<j; i++)
            {
            k = j - i;
            C_mod_raw(j) += C_in(k) * pQe(i,j);
            }
    """
    scipy.weave.inline(code, arg_names=['C_mod_raw', 'C_in', 'pQe', 'N'], type_converters = scipy.weave.converters.blitz)
    observed_fraction = np.diag(PQ[1:,1:]).copy()
    C_mod = (C_mod_raw + (1-observed_fraction) * C_old)
    return C_mod, C_mod_raw, observed_fraction

    
def transport_with_evapoconcentration(PQ, thetaQ, thetaS, C_in, C_old):
    """Apply a time-varying transit time distribution to an input concentration timseries

    Args:
        PQ : numpy float64 2D array, size N+1 x N+1
            The CDF of the backwards transit time distribution P_Q1(T,t)
        thetaQ, thetaS : numpy float64 2D array, size N+1 x N+1
            Partial partition functions for discharge and storage
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

    N = len(C_in)
    C_in=np.array(C_in)
    PQ = np.array(PQ)
    thetaQ = np.array(thetaQ)
    thetaS = np.array(thetaS)
    C_mod_raw = np.zeros(N, dtype=np.float64)
    pQe = np.where(thetaQ[1:,1:]>0, np.diff(PQ[:,1:],axis=0)/(thetaS[1:,1:] + thetaQ[1:,1:]), 0.)
    code = r"""
    int i, j, k;
    for(j=0; j<N; j++)
        for(i=0; i<j; i++)
            {
            k = j - i;
            C_mod_raw(j) += C_in(k) * pQe(i,j);
            }
    """
    scipy.weave.inline(code, arg_names=['C_mod_raw', 'C_in', 'pQe', 'N'], type_converters = scipy.weave.converters.blitz)
    observed_fraction = np.diag(PQ[1:,1:]).copy()
    C_mod = (C_mod_raw + (1-observed_fraction) * C_old)
    return C_mod, C_mod_raw, observed_fraction