import numpy as np
from scipy.integrate import cumtrapz
import scipy.weave

def steady_state_ST_from_TTD_1out(Q, TTD_CDF, dt=1):
    """Steady-state age-ranked storage

    At a given steady state flux rate Q, the age-ranked storage is given by:

    $$S_T = Q_0\left(T-\int_0^TP_{Q}(\tau)d\tau\right)$$

    Where $P_Q(\tau)$ is a steady-state transit time distribution at steady-state flux $Q_0$.

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

    $$S_T = (Q_1+Q_2)T-Q_1\int_0^TP_{Q_1}(\tau)-Q_2\int_0^TP_{Q_2}(\tau)d\tau$$

    Where $P_{Q_1}(\tau)$, $P_{Q_2}(\tau)$ are a steady-state transit time distribution at steady-state fluxs $Q_1$ and $Q_2$.

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
    C_mod_raw = np.zeros(N, dtype=np.float64)
    pQe = np.diff(PQ[:,1:],axis=0)
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


def transport_with_evapoconcentration(PQ, thetaQ, thetaS, C_in, C_old, observed_fraction_on_diagonal=True):
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
        observed_fraction_on_diagonal : Boolean
            Location of the observed fraction. 'True' (default) assumes observed
            fraction is on the diagonal of the PQ matrix. This is appropriate
            if no initial condition for ST is given. Otherwise use 'False' to
            use the oldest value of T for which PQ is given (i.e. the last row)

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
    if observed_fraction_on_diagonal:
        observed_fraction = np.diag(PQ[1:, 1:]).copy()
    else:
        observed_fraction = PQ[-1, 1:].copy()
    C_mod = (C_mod_raw + (1-observed_fraction) * C_old)
    return C_mod, C_mod_raw, observed_fraction

def transport_with_evapoconcentration_1st_order_reaction(PQ, thetaQ, thetaS, C_in,
                                                         C_old, k1=0.0, C_eq=0.0, C_sur=0.0):
    """Apply a time-varying transit time distribution to an input concentration timseries.
    Simulating a first order revisible reaction with reaction rate k1 towards
    the equilibrium concentration C_eq.  Allow for a surface concentration C_sur added to
    the input concentration at T=0.

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
        k1 : float (default=0.0)
            1st order rate of reaction (reversible) during transport
        C_eq : float (default=0.0)
            equilibrium concentration for 1st order reaction.
        C_sur : float (default=0.0)
            "instantaneous" additional concentration added at t=0.
            This may capture reactions occuring on timescales less than dt.



    Returns:
        C_out : numpy float64 1D array, length N.
            Timestep-averaged outflow concentration.
        C_mod_raw : numpy float64 1D array, length N.
            Timestep-averaged outflow concentration, prior to correction with C_old.
        observed_fraction : numpy float64 1D array, length N.
            Fraction of outflow older than the first timestep
    """

    N = len(C_in)
    ageT=np.arange(N)  #define age array
    C_in=np.array(C_in)
    C_in_sur=C_in+C_sur  #add surface concentration
    C_temp1=np.exp(-k1*ageT) #intermediate term #1
    C_temp2=C_eq*(1.-C_temp1)
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
            C_mod_raw(j) += C_in(k) * C_temp1(i) * pQe(i,j) + C_temp2(i) * pQe(i,j);
            }
    """
    scipy.weave.inline(code, arg_names=['C_mod_raw', 'C_in', 'pQe', 'N', 'C_temp1', 'C_temp2'], type_converters = scipy.weave.converters.blitz)
    observed_fraction = np.diag(PQ[1:,1:]).copy()
    C_mod = (C_mod_raw + (1-observed_fraction) * C_old)
    return C_mod, C_mod_raw, observed_fraction

def transport_with_decay(PQ, C_in, C_old, k1=0.0):
    """Apply a time-varying transit time distribution to an input concentration timseries.
    Simulating a first order decay reaction with decay rate k1

    Args:
        PQ : numpy float64 2D array, size N+1 x N+1
            The CDF of the backwards transit time distribution P_Q1(T,t)
        C_in : numpy float64 1D array, length N.
            Timestep-averaged inflow concentration.
        C_old : numpy float64 1D array, length N.
            Concentration to be assumed for portion of outflows older than initial
            timestep
        k1 : float (default=0.0)
            Decay rate. Units must be [1/timestep]

    Returns:
        C_out : numpy float64 1D array, length N.
            Timestep-averaged outflow concentration.
        C_mod_raw : numpy float64 1D array, length N.
            Timestep-averaged outflow concentration, prior to correction with C_old.
        observed_fraction : numpy float64 1D array, length N.
            Fraction of outflow older than the first timestep
    """

    N = len(C_in)
    ageT = np.arange(N)  #define age array
    C_in = np.array(C_in)
    decay = np.exp(-k1*ageT) #intermediate term #1
    PQ = np.array(PQ)
    C_mod_raw = np.zeros(N, dtype=np.float64)
    pQe = np.diff(PQ[:,1:],axis=0)
    code = r"""
    int i, j, k;
    for(j=0; j<N; j++)
        for(i=0; i<j; i++)
            {
            k = j - i;
            C_mod_raw(j) += C_in(k) * decay(i) * pQe(i,j);
            }
    """
    scipy.weave.inline(code, arg_names=['C_mod_raw', 'C_in', 'pQe', 'N', 'decay'], type_converters = scipy.weave.converters.blitz)
    observed_fraction = np.diag(PQ[1:,1:]).copy()
    C_mod = (C_mod_raw + (1-observed_fraction) * C_old)
    return C_mod, C_mod_raw, observed_fraction
