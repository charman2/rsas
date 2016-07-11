import numpy as np
dtype = np.float64
from f_convolve import f_convolve

def transport(PQ, C_J, C_old):
    """Apply a time-varying transit time distribution to an input concentration timseries

    Args:
        PQ : numpy float64 2D array, size N x N
            The CDF of the backwards transit time distribution P_Q1(T,t)
        C_J : numpy float64 1D array, length N.
            Timestep-averaged inflow concentration.
        C_old : float
            Concentration to be assumed for portion of outflows older than initial
            timestep

    Returns:
        C_J : numpy float64 1D array, length N.
            Timestep-averaged outflow concentration.
        C_Q_raw : numpy float64 1D array, length N.
            Timestep-averaged outflow concentration, prior to correction with C_old.
        observed_fraction : numpy float64 1D array, length N.
            Fraction of outflow older than the first timestep
    """
    max_age, timeseries_length = np.array(PQ.shape)-1
    return f_convolve(PQ, C_J, C_old, max_age, timeseries_length)
