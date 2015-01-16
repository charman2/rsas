# -*- coding: utf-8 -*-
"""Storage selection (SAS) functions: Lower Hafren example

Runs the rSAS model for the Lower Hafren stream to reproduce the results
presented in:

Harman, C. J. (2014), Time-variable transit time distributions and transport:
Theory and application to storage-dependent transport of chloride in a watershed,
Water Resour. Res., 51, doi:10.1002/2014WR015707.

The loaded dataset has already been gap-filled. The method to estimate the 
evapotranspiration timeseries and to account for occult deposition of chloride
are included here though.

This script is identical to lower_hafren_example_timestepping.py, except that
the numerical scheme originally used in the paper (solve_all_by_age_2out) is called,
rather than a new one (solve_all_by_time_2out).
"""
from __future__ import division
import rsas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.weave

def transport_with_evapoconcentration(PQ, thetaQ, thetaS, C_in, C_old):
    """Apply a time-varying transit time distribution to an input concentration timseries

    Parameters
    ----------
    pQ : numpy float64 2D array, size N x N
        The PDF of the backwards transit time distribution p'_Q1(T,t)
    C_in : numpy float64 1D array, length N.
        Timestep-averaged inflow concentration.

    Returns
    -------
    C_out : numpy float64 1D array, length N.
        Timestep-averaged outflow concentration.
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

# =====================================
# Load and process the input timeseries
# =====================================
# Load the data
data = pd.read_csv('lower_hafren_data.csv', index_col=0, parse_dates=True)
data = data[:1000]
N = len(data)
# Estimate ET from FAO reference evapotranspiration as ET = k_E * ETO, where
# k_E is chosen to ensurelong-term mass balance.
k_ET = (data['P water flux mm/day'].mean() - data['Q water flux mm/day'].mean())/data['ET0 mm/day'].mean()
data['ET water flux mm/day'] = data['ET0 mm/day'] * k_ET
# The fraction of CL deposition from occult deposition (mist droplets) was
# estimated from literature values and used to scale the timeseries values
occult_fraction = 0.27
data['P+occult Cl mg/l'] = data['P Cl mg/l'] / (1 - occult_fraction)
# =========================
# Parameters needed by rsas
# =========================
# The concentration of discharge older than the start of observations is
# assumed to be the long-term mean deposition rate divided by mean discharge
C_old = np.mean(data['P+occult Cl mg/l'] * data['P water flux mm/day']) / np.mean(data['Q water flux mm/day'])
# =========================
# Create the rsas functions
# =========================
# Parameters for the rSAS function (see Harman [2015])
ET_rSAS_fun_type = 'uniform'
S_ET = 398.
Q_rSAS_fun_type = 'gamma'
S_Q0 = 4922.
lambda_Q = -102.
Q_alpha = 0.6856
Q_dS_crit = 48.25
S_Q = np.maximum(1., lambda_Q * (data['S mm'] - Q_dS_crit))
ET_rSAS_fun_parameters = np.c_[np.zeros(N), np.ones(N) * S_ET ]
Q_rSAS_fun_parameters = np.c_[np.zeros(N), S_Q, np.ones(N) * Q_alpha]
# Initial condition
ST_init = np.zeros(N + 1)
# Run the model
results = rsas.solve_all_by_age_2out(
                                        data['P water flux mm/day'].values,
                                        data['Q water flux mm/day'].values,
                                        Q_rSAS_fun_parameters,
                                        Q_rSAS_fun_type,
                                        data['ET water flux mm/day'].values,
                                        ET_rSAS_fun_parameters,
                                        ET_rSAS_fun_type,
                                        ST_init,
                                        dt = 1.,
                                        n_substeps=1, 
                                        n_iterations = 4)
# Unroll the outputs
ST, PQ, PET, Qout, ETout, thetaQ, thetaET, thetaS, MassBalance = results
# Use the estimated transit time distribution and input timeseries to estimate
# the output timeseries
C_out, C_mod_raw, observed_fraction = transport_with_evapoconcentration(PQ, thetaQ, thetaS, data['P+occult Cl mg/l'], C_old)
data['Q Predicted Cl mg/l'] = C_out
# Plot the result
fig = plt.figure(1)
plt.clf()
plt.plot(data.index, data['Q Cl mg/l'], label='Observed')
plt.plot(data.index, data['Q Predicted Cl mg/l'], label='Predicted')