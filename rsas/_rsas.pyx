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
from _rsas_functions import rSAS_setup
from scipy.special import gamma as gamma_function
from scipy.special import gammainc
from scipy.special import erfc
from scipy.interpolate import interp1d
from scipy.optimize import fmin, minimize_scalar, fsolve
import time        

# for debugging
debug = True
def _verbose(statement):
    """Prints debuging messages if rsas.debug==True
    
    """
    if debug:
        print statement

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_all_by_age_2out(
        np.ndarray[dtype_t, ndim=1] J, 
        np.ndarray[dtype_t, ndim=1] Q1, 
        np.ndarray[dtype_t, ndim=2] rSAS1_params, 
        bytes rSAS1_type, 
        np.ndarray[dtype_t, ndim=1] Q2, 
        np.ndarray[dtype_t, ndim=2] rSAS2_params, 
        bytes rSAS2_type, 
        np.ndarray[dtype_t, ndim=1] ST_init = None, 
        dtype_t dt = 1, 
        int n_substeps = 1, 
        int n_iterations = 3):
    """rSAS model with 2 outflows, solved using the original age-based algorithm

    This is the original implementation used to generate the results in the paper.
    It solves for two outputs (Q1 and Q2, which might be discharge and ET) using
    an algorithm with an outer loop over all ages, and vectorized calculations over
    all times. It is slightly faster than the other implementations, but is more
    memory intensive. Unlike the others though, there is no option to calculate 
    output concentration timeseries inline. The calculated transit time
    distributions must be used to perform the convolutions after the code has 
    completed.

    Parameters
    ----------
    
    J : n x 1 float64 ndarray
         Timestep-averaged inflow timeseries
    Q1, Q2 : n x 1 float64 ndarray
         Timestep-averaged outflow timeseries. Must have same units and length as J
    rSAS1_params, rSAS2_params : n x k float64 ndarray
        Parameters for the Q1, Q2 rSAS function. The number of columns and 
        their meaning depends on which rSAS type is chosen. For all the rSAS 
        functions implemented so far, each row corresponds with a timestep, and
        so the first dimension must be the same as for J.
    rSAS1_type, rSAS2_type : string
        rSAS functional form. See below for available options.
    ST_init : m x 1 float64 ndarray (default None)
        Initial condition for the age-ranked storage. The length of ST_init
        determines the maximum age calculated. The first entry must be 0
        (corresponding to zero age). To calculate transit time dsitributions up
        to N timesteps in age, ST_init should have length m = M + 1. The default
        initial condition is ST_init=np.zeros(len(J) + 1).
    dt : float (default 1)
        Timestep, assuming same units as J, Q1, Q2
    n_substeps : int (default 1)
        if n_substeps>1, the timesteps are subdivided to allow a more accurate
        solution. Default is 1, which is also the value used in Harman (2015)
    n_iterations : int (default 3)
        Number of iterations to converge on a consistent solution. Convergence 
        in Harman (2015) was very fast, and n_iterations=3 was adequate (also 
        the default value here)

    Returns
    -------
    
    ST : numpy float64 2D array
        Array of age-ranked storage for all ages and times. 
    PQ1, PQ2 : numpy float64 2D array
        Time-varying cumulative transit time distributions.
    Q1out, Q2out : numpy float64 2D array
        Age-based outflow timeseries. Useful for visualization.
    theta1, theta2, thetaS : numpy float64 2D array
        Keeps track of the fraction of inputs that leave by each flux or remain
        in storage. This is needed to do transport with evapoconcentration.
    MassBalance : numpy float64 2D array
        Should always be within tolerances of zero, unless something is very wrong.

    For each of the above arrays, each row represents an age, and each
    column is a timestep. For N timesteps and M ages, ST will have dimensions
    (M+1) x (N+1), with the first row representing age T = 0 and the first
    column derived from the initial condition.
    
    Implemented rSAS functions
    --------------------------
    Available choices for rSAS1_type, rSAS2_type, and description of parameter array.
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
    # Initialization
    # Define some variables
    cdef int k, i, timeseries_length, num_inputs, max_age, N
    cdef np.float64_t start_time
    cdef np.ndarray[dtype_t, ndim=2] ST, PQ1, PQ2, Q1out, Q2out, theta1, theta2, thetaS, MassBalance
    cdef np.ndarray[dtype_t, ndim=1] STp, PQ1p, PQ2p, Q1outp, Q2outp
    cdef np.ndarray[dtype_t, ndim=1] STu, PQ1u, PQ2u, dPQ1u, dPQ2u, dQ1outu, dQ2outu, dSTp, dPQ1p, dPQ2p
    cdef np.ndarray[dtype_t, ndim=1] Q1r, Q2r, Jr
    cdef np.ndarray[dtype_t, ndim=2] Q1_paramsr, Q2_paramsr
    # Handle inputs
    if ST_init is None:
        ST_init=np.zeros(len(J) + 1)
    else:
        # This must be true
        ST_init[0] = 0
    # Some lengths
    timeseries_length = len(J)
    max_age = len(ST_init) - 1
    N = timeseries_length * n_substeps
    # Expand the inputs to accomodate the substep solution points
    Q1r = Q1.repeat(n_substeps)
    Q2r = Q2.repeat(n_substeps)
    Jr = J.repeat(n_substeps)
    Q1_paramsr = rSAS1_params.repeat(n_substeps, axis=0)
    Q2_paramsr = rSAS2_params.repeat(n_substeps, axis=0)
    dt = dt / n_substeps
    # Instantiate the rSAS functions
    rSAS_fun1 = rSAS_setup(rSAS1_type, rSAS1_params)
    rSAS_fun2 = rSAS_setup(rSAS2_type, rSAS2_params)
    # Create arrays to hold the state variables
    _verbose('...initializing arrays...')
    ST = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    PQ1 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    PQ2 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    Q1out = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    Q2out = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    theta1 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    theta2 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    thetaS = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    MassBalance = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    #Create arrays to hold intermediate solutions
    STp = np.zeros(N+1, dtype=np.float64)
    PQ1p = np.zeros(N+1, dtype=np.float64)
    PQ2p = np.zeros(N+1, dtype=np.float64)
    Q1outp = np.zeros(N+1, dtype=np.float64)
    Q2outp = np.zeros(N+1, dtype=np.float64)
    dSTp = np.zeros(N, dtype=np.float64)
    dPQ1p = np.zeros(N, dtype=np.float64)
    dPQ2p = np.zeros(N, dtype=np.float64)
    dSTu = np.zeros(N, dtype=np.float64)
    dPQ1u = np.zeros(N, dtype=np.float64)
    dPQ2u = np.zeros(N, dtype=np.float64)
    STu = np.zeros(N, dtype=np.float64)
    PQ1u = np.zeros(N, dtype=np.float64)
    PQ2u = np.zeros(N, dtype=np.float64)
    dQ1outu = np.zeros(N, dtype=np.float64)
    dQ2outu = np.zeros(N, dtype=np.float64)
    _verbose('done')
    # Now we solve the governing equation
    # Set up initial and boundary conditions
    dSTp[:] = Jr * dt
    dPQ1p[:] = np.where(dSTp>0., rSAS_fun1.cdf_all(dSTp), 0.)
    dPQ1p[:] = np.where(dSTp>0., rSAS_fun2.cdf_all(dSTp), 0.)
    ST[:, 0] = ST_init[:]
    PQ1_init = rSAS_fun1.cdf_i(ST_init, 0)
    PQ2_init = rSAS_fun2.cdf_i(ST_init, 0)
    PQ1[:, 0] = PQ1_init
    PQ2[:, 0] = PQ2_init
    start_time = time.clock()
    _verbose('...solving...')
    # Primary solution loop over ages T
    for i in range(max_age):
    # Loop over substeps
        for k in range(n_substeps):
            # dSTp is the increment of ST at the previous age and previous timestep.
            # It is therefore our first estimate of the increment of ST at this
            # age and timestep. 
            STu[:] = STp[1:] + dSTp
            # Use this estimate to get an initial estimate of the 
            # cumulative transit time distributions, PQ1 and PQ2
            PQ1u[:] = np.where(dSTp>0., rSAS_fun1.cdf_all(STu), PQ1p[1:])
            PQ2u[:] = np.where(dSTp>0., rSAS_fun2.cdf_all(STu), PQ2p[1:])
            # Iterate to refine these estimates
            for it in range(n_iterations):
                # Increments of the cumulative transit time distribution
                # approximate the values of the transit time PDF at this age
                dPQ1u[:] = (PQ1u - PQ1p[1:])
                dPQ2u[:] = (PQ2u - PQ2p[1:])
                # Estimate the outflow over the interval of ages dt with an age
                # T as the discharge over the timestep times the average of the
                # PDF values at the start and the end of the timestep
                dQ1outu[:] = Q1r * (dPQ1u + dPQ1p) / 2
                dQ2outu[:] = Q2r * (dPQ2u + dPQ2p) / 2
                # Update the estimate of dST, ST and the cumulative TTD to
                # account for these outflows
                dSTu[:] = np.maximum(dSTp - dt * dQ1outu - dt * dQ2outu, 0.)
                STu[:] = STp[1:] + dSTu
                PQ1u[:] = np.where(dSTp>0., rSAS_fun1.cdf_all(STu), PQ1p[1:])
                PQ2u[:] = np.where(dSTp>0., rSAS_fun2.cdf_all(STu), PQ2p[1:])
            # Update the 'previous solution' record in preparation of the
            # next solution timestep
            STp[1:] = STu[:]
            PQ1p[1:] = PQ1u[:]
            PQ2p[1:] = PQ2u[:]
            dSTp[1:]  = dSTu[:N-1]
            dPQ1p[1:] = dPQ1u[:N-1]
            dPQ2p[1:] = dPQ2u[:N-1]
            # Incorporate the boundary condition
            dSTp[0]  = (ST_init[i+1] - (ST_init[i])) / n_substeps
            dPQ1p[0] = (PQ1_init[i+1] - (PQ1_init[i])) / n_substeps
            dPQ2p[0] = (PQ2_init[i+1] - (PQ2_init[i])) / n_substeps
            # Keep a running tally of the outflows by age
            Q1outp[1:] = Q1outp[:N] + dQ1outu[:]
            Q2outp[1:] = Q2outp[:N] + dQ2outu[:]
            Q1out[i+1, 1:] += Q1outp[n_substeps::n_substeps]/n_substeps
            Q2out[i+1, 1:] += Q2outp[n_substeps::n_substeps]/n_substeps
            # If a full timestep is complete, store the result
            if k==n_substeps-1:
                ST[i+1, 1:] =   STp[n_substeps::n_substeps]
                PQ1[i+1, 1:] = PQ1p[n_substeps::n_substeps]
                PQ2[i+1, 1:] = PQ2p[n_substeps::n_substeps]
                theta1[i+1, i+1:] = np.where(J[:timeseries_length-i]>0, Q1out[i+1, i+1:] / J[:timeseries_length-i], 0.)
                theta2[i+1, i+1:] = np.where(J[:timeseries_length-i]>0, Q2out[i+1, i+1:] / J[:timeseries_length-i], 0.) 
                thetaS[i+1, i+1:] = np.where(J[:timeseries_length-i]>0, (ST[i+1, i+1:] - ST[i, i+1:]) / J[:timeseries_length-i], 0.)
                MassBalance[i+1, i+1:] = (J[:timeseries_length-i] 
                                        - Q1out[i+1, i+1:] - Q2out[i+1, i+1:] 
                                        - (ST[i+1, i+1:] - ST[i, i+1:])/dt)
        if np.mod(i+1,1000)==0:
            _verbose('...done ' + str(i+1) + ' of ' + str(max_age) + ' in ' + str(time.clock() - start_time) + ' seconds')
    return ST, PQ1, PQ2, Q1out, Q2out, theta1, theta2, thetaS, MassBalance

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_all_by_time_2out(np.ndarray[dtype_t, ndim=1] J, 
        np.ndarray[dtype_t, ndim=1] Q1, 
        np.ndarray[dtype_t, ndim=2] rSAS1_params, 
        bytes rSAS1_type, 
        np.ndarray[dtype_t, ndim=1] Q2, 
        np.ndarray[dtype_t, ndim=2] rSAS2_params, 
        bytes rSAS2_type, 
        np.ndarray[dtype_t, ndim=1] ST_init = None, 
        dtype_t dt = 1, 
        int n_iterations = 3,
        full_outputs=True, C_in=None, C_old=None, evapoconcentration=False):
    """rSAS model with 2 outflows, solved by looping over timesteps

    Solves for two outflows (Q1 and Q2, which might be discharge and ET).    
    Solution is found by looping over times, with all age calculations
    vectorized. Slower, but easier to understand and build on than 
    solve_all_by_age_2out. Includes option to determine output concentrations
    from a given input concentration progressively.

    Parameters
    ----------
    
    J : n x 1 float64 ndarray
         Timestep-averaged inflow timeseries
    Q1, Q2 : n x 1 float64 ndarray
         Timestep-averaged outflow timeseries. Must have same units and length as J
    rSAS1_params, rSAS2_params : n x k float64 ndarray
        Parameters for the Q1, Q2 rSAS function. The number of columns and 
        their meaning depends on which rSAS type is chosen. For all the rSAS 
        functions implemented so far, each row corresponds with a timestep, and
        so the first dimension must be the same as for J.
    rSAS1_type, rSAS2_type : string
        rSAS functional form. See below for available options.
    ST_init : m x 1 float64 ndarray
        Initial condition for the age-ranked storage. The length of ST_init
        determines the maximum age calculated. The first entry must be 0
        (corresponding to zero age). To calculate transit time dsitributions up
        to N timesteps in age, ST_init should have length m = M + 1. The default
        initial condition is ST_init=np.zeros(len(J) + 1).
    dt : float (default 1)
        Timestep, assuming same units as J, Q1, Q2
    n_substeps : int (default 1)
        if n_substeps>1, the timesteps are subdivided to allow a more accurate
        solution. Default is 1, which is also the value used in Harman (2015)
    n_iterations : int (default 3)
        Number of iterations to converge on a consistent solution. Convergence 
        in Harman (2015) was very fast, and n_iterations=3 was adequate (also 
        the default value here)
    full_outputs : bool (default True)
        Option to return the full state variables array ST the cumulative
        transit time distributions PQ1, PQ2, and other variables
    C_in : n x 1 float64 ndarray (default None)
        Optional timeseries of inflow concentrations to convolved progressively
        with the computed transit time distribution for flux Q1
    C_old : float (default None)
        Optional. Concentration of the 'unobserved fraction' of Q1 (from inflows 
        prior to the start of the simulation).
    evapoconcentration : bool (default False)
        Optional. If True, it will be assumed that species in C_in are not removed 
        by the second flux, Q2, and instead become increasingly concentrated in
        storage.

    Returns
    -------
    
    C_out : numpy float64 1D array
        If C_in is supplied, C_out is the timeseries of outflow concentration 
        in Q1.
    ST : numpy float64 2D array
        Array of age-ranked storage for all ages and times. 
    PQ1, PQ2 : numpy float64 2D array
        Time-varying cumulative transit time distributions.
    Q1out, Q2out : numpy float64 2D array
        Age-based outflow timeseries. Useful for visualization.
    theta1, theta2, thetaS : numpy float64 2D array
        Keeps track of the fraction of inputs that leave by each flux or remain
        in storage. This is needed to do transport with evapoconcentration.
    MassBalance : numpy float64 2D array
        Should always be within tolerances of zero, unless something is very wrong.

    For each of the arrays ST, PQ1, PQ2, Q1out, Q2out, theta1, theta2, thetaS,
    and MassBalance, each row represents an age, and each
    column is a timestep. For N timesteps and M ages, ST will have dimensions
    (M+1) x (N+1), with the first row representing age T = 0 and the first
    column derived from the initial condition.
    
    If full_outputs=True and C_in is supplied, the variables are returned as::
    
        C_out, ST, PQ1, PQ2, Q1out, Q2out, theta1, theta2, thetaS, MassBalance = rsas.solve_all_by_time_2out(...
    
    If C_in is not supplied, C_out is not returned::
    
        ST, PQ1, PQ2, Q1out, Q2out, theta1, theta2, thetaS, MassBalance = rsas.solve_all_by_time_2out(...

    Otherwise if full_outputs=False
    
        C_out = rsas.solve_all_by_time_2out(...
    
    Implemented rSAS functions
    --------------------------
    Available choices for rSAS1_type, rSAS2_type, and description of parameter array.
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
    # Initialization
    # Define some variables
    cdef int k, i, timeseries_length, num_inputs, max_age
    cdef np.float64_t start_time
    cdef np.ndarray[dtype_t, ndim=2] ST, PQ1, PQ2, Q1out, Q2out, theta1, theta2, thetaS, MassBalance
    cdef np.ndarray[dtype_t, ndim=1] STu, pQ1u, pQ2u, pQ1p, pQ2p, dQ1outu, dQ2outu, dSTu, dSTp, C_out, Q1out_total
    # Handle inputs
    if ST_init is None:
        ST_init=np.zeros(len(J) + 1)
    else:
        # This must be true
        ST_init[0] = 0
    # Some lengths
    timeseries_length = len(J)
    max_age = len(ST_init) - 1
    # Instantiate the rSAS functions
    rSAS_fun1 = rSAS_setup(rSAS1_type, rSAS1_params)
    rSAS_fun2 = rSAS_setup(rSAS2_type, rSAS2_params)
    # Create arrays to hold intermediate solutions
    _verbose('...initializing arrays...')
    pQ1p = np.zeros(max_age, dtype=np.float64)
    pQ2p = np.zeros(max_age, dtype=np.float64)
    STu = np.zeros(max_age+1, dtype=np.float64)
    dQ1outu = np.zeros(max_age, dtype=np.float64)
    dQ2outu = np.zeros(max_age, dtype=np.float64)
    pQ1u = np.zeros(max_age, dtype=np.float64)
    pQ2u = np.zeros(max_age, dtype=np.float64)
    dSTu = np.zeros(max_age, dtype=np.float64)
    dSTp = np.zeros(max_age, dtype=np.float64)
    if C_in is not None:
        if evapoconcentration:
            Q1out_total = np.zeros((max_age), dtype=np.float64)
        C_out = np.zeros(max_age, dtype=np.float64)
    # Create arrays to hold the state variables if they are to be outputted
    if full_outputs:
        ST = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        PQ1 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        PQ2 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        Q1out = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        Q2out = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        theta1 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        theta2 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        thetaS = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        MassBalance = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    _verbose('done')
    # Now we solve the governing equation
    # Set up initial and boundary conditions
    dSTp[0] = J[0] * dt
    dSTp[1:max_age] = np.diff(ST_init[:max_age])
    pQ1p[:] = np.diff(rSAS_fun1.cdf_i(ST_init, 0))
    pQ2p[:] = np.diff(rSAS_fun2.cdf_i(ST_init, 0))
    if full_outputs:
        ST[:,0] = ST_init[:]
        PQ1[:,0] = rSAS_fun1.cdf_i(ST_init, 0)
        PQ2[:,0] = rSAS_fun2.cdf_i(ST_init, 0)
    start_time = time.clock()
    _verbose('...solving...')
    # Primary solution loop over time t
    for i in range(timeseries_length):
        # dSTp is the increments of ST at the previous age and previous timestep.
        # It is therefore our first estimate of the increments of ST at this
        # age and timestep. Add up the increments to get an estimate of ST
        STu[0] = 0
        STu[1:max_age+1] = np.cumsum(dSTp)
        # Use this estimate to get an initial estimate of the 
        # transit time distribution PDFs, pQ1 and pQ2
        pQ1u[:max_age] = np.diff(rSAS_fun1.cdf_i(STu, i))
        pQ2u[:max_age] = np.diff(rSAS_fun2.cdf_i(STu, i))
        # Iterate to refine these estimates
        for it in range(n_iterations):
            # Estimate the outflow over the interval of time dt with an age
            # T as the discharge over the timestep times the average of the
            # PDF values at the start and the end of the timestep
            dQ1outu[0] = Q1[i] * pQ1u[0]
            dQ1outu[1:max_age] = Q1[i] * (pQ1u[1:max_age] + pQ1p[1:max_age])/2
            dQ2outu[0] = Q2[i] * pQ2u[0]
            dQ2outu[1:max_age] = Q2[i] * (pQ2u[1:max_age] + pQ2p[1:max_age])/2
            # Update the estimate of dST, ST and the TTD PDFs to
            # account for these outflows
            dSTu[:max_age] = np.maximum(dSTp - dt * dQ1outu - dt * dQ2outu, 0.)
            STu[1:max_age+1] = np.cumsum(dSTu)
            pQ1u[:max_age] = np.diff(rSAS_fun1.cdf_i(STu, i))
            pQ2u[:max_age] = np.diff(rSAS_fun2.cdf_i(STu, i))
        # Update the 'previous solution' record in preparation of the
        # next solution timestep
        if i<timeseries_length-1:
            dSTp[1:max_age] = dSTu[:max_age-1]
            # Incorporate the boundary condition
            dSTp[0] = J[i+1] * dt
            pQ1p[1:max_age] = pQ1u[:max_age-1]
            pQ2p[1:max_age] = pQ2u[:max_age-1]
            pQ1p[0] = 0
            pQ2p[0] = 0
        # Progressive evaluation of outflow concentration
        if C_in is not None:
            if evapoconcentration:
                # If evapoconcentration=True, keep a running tab of how much of
                # each timestep's inflow has become outflow
                Q1out_total[:i+1] = Q1out_total[:i+1] + dQ1outu[i::-1]
                # The enriched concentration in storge is the initial mass
                # divided by the volume that has not evaporated 
                # C_in * J / (Q1out_total + dSTu)
                # Get the current discharge concentration as the sum of previous
                # (weighted) inputs, accounting for evapoconcentration
                C_out[i] = np.sum(np.where(J[i::-1]>0, pQ1u[:i+1] * C_in[i::-1] * J[i::-1] / (Q1out_total[i::-1] + dSTu[:i+1]), 0.))
            else:
                # Get the current discharge concentration as the sum of previous
                # (weighted) inputs
                C_out[i] = np.sum(pQ1u[:i+1] * C_in[i::-1])
            if C_old:
                # Add the concentration of the 'unobserved fraction'
                C_out[i] += (1 - np.sum(pQ1u[:i+1])) * C_old
        # Store the result, if needed
        if full_outputs:
            ST[:max_age+1, i+1] =   STu[:max_age+1]
            PQ1[1:max_age+1, i+1] = np.cumsum(pQ1u)
            PQ2[1:max_age+1, i+1] = np.cumsum(pQ2u)
            Q1out[1:max_age+1, i+1] = Q1out[:max_age, i] + dQ1outu[:max_age]
            Q2out[1:max_age+1, i+1] = Q2out[:max_age, i] + dQ2outu[:max_age]
            theta1[1:i+2, i+1] = np.where(J[i::-1]>0, Q1out[1:i+2, i+1] / J[i::-1], 0.)
            theta2[1:i+2, i+1] = np.where(J[i::-1]>0, Q2out[1:i+2, i+1] / J[i::-1], 0.)
            thetaS[1:i+2, i+1] = np.where(J[i::-1]>0, (ST[1:i+2, i+1] - ST[:i+1, i+1]) / J[i::-1], 0.)
            MassBalance[1:i+2, i+1] = np.diff(ST[:i+2, i+1]) - dt * (J[i::-1] - Q1out[1:i+2, i+1] - Q2out[1:i+2, i+1])
            MassBalance[i+2:max_age+1, i+1] = np.diff(ST[i+1:max_age+1, i+1]) - dt * (np.diff(ST_init[:max_age-i]) - Q1out[i+2:max_age+1, i+1] - Q2out[i+2:max_age+1, i+1])
        if np.mod(i+1,1000)==0:
            _verbose('...done ' + str(i+1) + ' of ' + str(max_age) + ' in ' + str(time.clock() - start_time) + ' seconds')
    # Done. Return the outputs
    if full_outputs and C_in is not None:
        return C_out, ST, PQ1, PQ2, Q1out, Q2out, theta1, theta2, thetaS, MassBalance
    elif full_outputs and C_in is None:
        return ST, PQ1, PQ2, Q1out, Q2out, theta1, theta2, thetaS, MassBalance
    elif not full_outputs and C_in is not None:
        return C_out

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_all_by_time_1out(np.ndarray[dtype_t, ndim=1] J, 
        np.ndarray[dtype_t, ndim=1] Q1, 
        np.ndarray[dtype_t, ndim=2] rSAS1_params, 
        bytes rSAS1_type, 
        np.ndarray[dtype_t, ndim=1] ST_init = None, 
        dtype_t dt = 1, 
        int n_iterations = 3,
        full_outputs=True, C_in=None, C_old=None):
    """rSAS model with 1 flux, solved by looping over timesteps

    Same as solve_all_by_time_2out, but for only one flux out (Q1). 

    Parameters
    ----------
    
    J : n x 1 float64 ndarray
         Timestep-averaged inflow timeseries
    Q1 : n x 1 float64 ndarray
         Timestep-averaged outflow timeseries. Must have same units and length as J
    rSAS1_params : n x k float64 ndarray
        Parameters for the Q1 rSAS function. The number of columns and 
        their meaning depends on which rSAS type is chosen. For all the rSAS 
        functions implemented so far, each row corresponds with a timestep, and
        so the first dimension must be the same as for J.
    rSAS1_type : string
        rSAS functional form. See below for available options.
    ST_init : m x 1 float64 ndarray
        Initial condition for the age-ranked storage. The length of ST_init
        determines the maximum age calculated. The first entry must be 0
        (corresponding to zero age). To calculate transit time dsitributions up
        to N timesteps in age, ST_init should have length m = M + 1. The default
        initial condition is ST_init=np.zeros(len(J) + 1).
    dt : float (default 1)
        Timestep, assuming same units as J, Q1
    n_substeps : int (default 1)
        if n_substeps>1, the timesteps are subdivided to allow a more accurate
        solution. Default is 1, which is also the value used in Harman (2015)
    n_iterations : int (default 3)
        Number of iterations to converge on a consistent solution. Convergence 
        in Harman (2015) was very fast, and n_iterations=3 was adequate (also 
        the default value here)
    full_outputs : bool (default True)
        Option to return the full state variables array ST the cumulative
        transit time distributions PQ1, PQ2, and other variables
    C_in : n x 1 float64 ndarray (default None)
        Optional timeseries of inflow concentrations to convolved progressively
        with the computed transit time distribution for flux Q1
    C_old : float (default None)
        Optional. Concentration of the 'unobserved fraction' of Q1 (from inflows 
        prior to the start of the simulation).

    Returns
    -------
    
    C_out : numpy float64 1D array
        If C_in is supplied, C_out is the timeseries of outflow concentration 
        in Q1.
    ST : numpy float64 2D array
        Array of age-ranked storage for all ages and times. 
    PQ1 : numpy float64 2D array
        Time-varying cumulative transit time distributions.
    Q1out : numpy float64 2D array
        Age-based outflow timeseries. Useful for visualization.
    theta1, thetaS : numpy float64 2D array
        Keeps track of the fraction of inputs that leave by each flux or remain
        in storage.
    MassBalance : numpy float64 2D array
        Should always be within tolerances of zero, unless something is very wrong.

    For each of the arrays ST, PQ1, Q1out, theta1, thetaS,
    and MassBalance, each row represents an age, and each
    column is a timestep. For N timesteps and M ages, ST will have dimensions
    (M+1) x (N+1), with the first row representing age T = 0 and the first
    column derived from the initial condition.
    
    If full_outputs=True and C_in is supplied, the variables are returned as::
    
        C_out, ST, PQ1, PQ2, Q1out, theta1, thetaS, MassBalance = rsas.solve_all_by_time_2out(...
    
    If C_in is not supplied, C_out is not returned::
    
        ST, PQ1, PQ2, Q1out, theta1, thetaS, MassBalance = rsas.solve_all_by_time_2out(...

    Otherwise if full_outputs=False
    
        C_out = rsas.solve_all_by_time_2out(...
    
    Implemented rSAS functions
    --------------------------
    Available choices for rSAS1_type, and description of parameter array.
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
    # Initialization
    # Define some variables
    cdef int k, i, timeseries_length, num_inputs, max_age
    cdef np.float64_t start_time
    cdef np.ndarray[dtype_t, ndim=2] ST, PQ1, Q1out, theta1, thetaS, MassBalance
    cdef np.ndarray[dtype_t, ndim=1] STu, pQ1u, pQ1p, dQ1outu, dSTu, dSTp, C_out
    # Handle inputs
    if ST_init is None:
        ST_init=np.zeros(len(J) + 1)
    else:
        # This must be true
        ST_init[0] = 0
    # Some lengths
    timeseries_length = len(J)
    max_age = len(ST_init) - 1
    # Instantiate the rSAS function
    rSAS_fun1 = rSAS_setup(rSAS1_type, rSAS1_params)
    # Create arrays to hold intermediate solutions
    _verbose('...initializing arrays...')
    pQ1p = np.zeros(max_age, dtype=np.float64)
    pQ2p = np.zeros(max_age, dtype=np.float64)
    STu = np.zeros(max_age+1, dtype=np.float64)
    dQ1outu = np.zeros(max_age, dtype=np.float64)
    dQ2outu = np.zeros(max_age, dtype=np.float64)
    pQ1u = np.zeros(max_age, dtype=np.float64)
    pQ2u = np.zeros(max_age, dtype=np.float64)
    dSTu = np.zeros(max_age, dtype=np.float64)
    dSTp = np.zeros(max_age, dtype=np.float64)
    if C_in is not None:
        C_out = np.zeros(max_age, dtype=np.float64)
    # Create arrays to hold the state variables if they are to be outputted
    if full_outputs:
        ST = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        PQ1 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        PQ2 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        Q1out = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        Q2out = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        theta1 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        theta2 = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        thetaS = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
        MassBalance = np.zeros((max_age + 1, timeseries_length + 1), dtype=np.float64)
    _verbose('done')
    # Now we solve the governing equation
    # Set up initial and boundary conditions
    dSTp[0] = J[0] * dt
    dSTp[:] = np.diff(ST_init)
    pQ1p[:] = np.diff(rSAS_fun1.cdf_i(ST_init, 0))
    if full_outputs:
        ST[:,0] = ST_init[:]
        PQ1[:,0] = rSAS_fun1.cdf_i(ST_init, 0)
    start_time = time.clock()
    _verbose('...solving...')
    # Primary solution loop over time t
    for i in range(timeseries_length):
        # dSTp is the increments of ST at the previous age and previous timestep.
        # It is therefore our first estimate of the increments of ST at this
        # age and timestep. Use this estimate to get an initial estimate of the 
        # transit time distribution PDF, pQ1
        pQ1u[0] = rSAS_fun1.cdf_i(dSTp[:1], i)
        pQ1u[1:max_age] = np.diff(rSAS_fun1.cdf_i(np.cumsum(dSTp), i))
        # Iterate to refine the estimates
        for it in range(n_iterations):
            # Estimate the outflow over the interval of time dt with an age
            # T as the discharge over the timestep times the average of the
            # PDF values at the start and the end of the timestep
            dQ1outu[0] = Q1[i] * pQ1u[0]
            dQ1outu[1:max_age] = Q1[i] * (pQ1u[1:max_age] + pQ1p[1:max_age])/2
            # Update the estimate of dST and the TTD PDF to
            # account for the outflow
            dSTu[:max_age] = np.maximum(dSTp - dt * dQ1outu - dt * dQ2outu, 0.)
            pQ1u[0] = rSAS_fun1.cdf_i(dSTu[:1], i)
            pQ1u[1:max_age] = np.diff(rSAS_fun1.cdf_i(np.cumsum(dSTu), i))
        # Update the 'previous solution' record in preparation of the
        # next solution timestep
        if i<timeseries_length-1:
            dSTp[1:max_age] = dSTu[:max_age-1]
            pQ1p[1:max_age] = pQ1u[:max_age-1]
            # Incorporate the boundary condition
            dSTp[0] = J[i+1] * dt
            # This vale is never used
            pQ1p[0] = 0
        # Progressive evaluation of outflow concentration
        if C_in is not None:
            C_out[i] = np.sum(pQ1u[:i+1] * C_in[i::-1])
            if C_old:
                C_out[i] += (1 - np.sum(pQ1u[:i+1])) * C_old
        # Store the result, if needed
        if full_outputs:
            ST[:max_age+1, i+1] =   STu[:max_age+1]
            PQ1[1:max_age+1, i+1] = np.cumsum(pQ1u)
            Q1out[1:max_age+1, i+1] = Q1out[:max_age, i] + dQ1outu[:max_age]
            theta1[1:i+2, i+1] = np.where(J[i::-1]>0, Q1out[1:i+2, i+1] / J[i::-1], 0.)
            thetaS[1:i+2, i+1] = np.where(J[i::-1]>0, (ST[1:i+2, i+1] - ST[:i+1, i+1]) / J[i::-1], 0.)
            MassBalance[1:i+2, i+1] = np.diff(ST[:i+2, i+1]) - dt * (J[i::-1] - Q1out[1:i+2, i+1])
            MassBalance[i+2:max_age+1, i+1] = np.diff(ST[i+1:max_age+1, i+1]) - dt * (np.diff(ST_init[:max_age-i]) - Q1out[i+2:max_age+1, i+1])
        if np.mod(i+1,1000)==0:
            _verbose('...done ' + str(i+1) + ' of ' + str(max_age) + ' in ' + str(time.clock() - start_time) + ' seconds')
    # Done. Return the outputs
    if full_outputs and C_in is not None:
        return C_out, ST, PQ1, Q1out, theta1, thetaS, MassBalance
    elif full_outputs and C_in is None:
        return ST, PQ1, Q1out, theta1, thetaS, MassBalance
    elif not full_outputs and C_in is not None:
        return C_out

#def rSAS_setup(rSAS_type, np.ndarray[dtype_t, ndim=2] params):
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
    rSAS_fun : rsas function
        An rsas function of the chosen class
        
    The created function object will have methods that vary between types. All
    must have two methods cdf_all and cdf_i.
    
    rSAS_fun.cdf_all(ndarray ST)
        returns the cumulative distribution function for an array ST (which
        must be the same length as the params matrix used to create the 
        function). Each value of ST is evaluated using the parameter values
        on the respective row of params
    rSAS_fun.cdf_i(ndarray ST, int i)
        returns the cumulative distribution function for an array ST (which
        can be of any size). Each value of ST is evaluated using the
        parameter values on row i.

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
        
class uniform_rSAS:
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

class gamma_rSAS: 
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
        
class gamma_trunc_rSAS: 
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

class SS_invgauss_rSAS:
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

class SS_mobileimmobile_rSAS:
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
"""