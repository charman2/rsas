import numpy as np
dtype = np.float64

def transport(PQ, C_in, C_old):
    """Apply a time-varying transit time distribution to an input concentration timseries

    Args:
        PQ : numpy float64 2D array, size N x N
            The CDF of the backwards transit time distribution P_Q1(T,t)
        C_in : numpy float64 1D array, length N.
            Timestep-averaged inflow concentration.
        C_old : float
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
    #cdef int N, t, T
    #cdef np.ndarray[dtype_t, ndim=2] pQe
    #cdef np.ndarray[dtype_t, ndim=1] C_mod_raw, observed_fraction
    N = len(C_in)
    C_mod_raw = np.zeros(N, dtype=np.float64)
    pQe = np.diff(PQ[:,1:],axis=0)
    for t in range(pQe.shape[1]):
        for T in range(t+1):
            C_mod_raw[t] += C_in[t-T] * pQe[T,t]
    observed_fraction = np.diag(PQ[1:,1:]).copy()
    C_mod = (C_mod_raw + (1-observed_fraction) * C_old)
    return C_mod, C_mod_raw, observed_fraction
