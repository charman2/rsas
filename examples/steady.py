# -*- coding: utf-8 -*-
"""Storage selection (SAS) functions: example with one flux out at steady state

Runs the rSAS model for a synthetic dataset with one flux in and out
and steady state flow

Theory is presented in:
Harman, C. J. (2014), Time-variable transit time distributions and transport:
Theory and application to storage-dependent transport of chloride in a watershed,
Water Resour. Res., 51, doi:10.1002/2014WR015707.
"""
from __future__ import division
import rsas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Initializes the random number generator so we always get the same result
np.random.seed(0)
# =====================================
# Generate the input timeseries
# =====================================
# length of the dataset
N = 100
S_0 = 5. # <-- volume of the uniformly sampled store
Q_0 = 1. # <-- steady-state flow rate
T_0 = S_0 / Q_0
# Note that the analytical solution for the cumulative TTD is
T = np.arange(N+1)
PQ_exact = 1 - np.exp(-T/T_0)
# Steady-state flow in and out for N timesteps
J = np.ones(N) * Q_0
Q1 = np.ones(N) * Q_0
# A random timeseries of concentrations
C_J = -np.log(np.random.rand(N))
# =========================
# Parameters needed by rsas
# =========================
# The concentration of water older than the start of observations
C_old = 0.
# =========================
# Create the rsas functions
# =========================
# Parameters for the rSAS function
# The uniform distribution extends between S_T=a and S_T=b.
Q_rSAS_fun_type = 'uniform'
a = np.ones(N) * 0.
b = np.ones(N) * S_0
Q_rSAS_fun_parameters = np.c_[a, b]
rSAS_fun_Q1 = rsas.create_function(Q_rSAS_fun_type, Q_rSAS_fun_parameters)
# =================
# Initial condition
# =================
# Unknown initial age distribution, so just set this to zeros
ST_init = np.zeros(N + 1)
# =============
# Run the model
# =============
# Run it
outputs = rsas.solve(J, [Q1], [rSAS_fun_Q1], ST_init=ST_init,
                     mode='time', dt = 1., n_substeps=10, C_in=C_J, C_old=C_old)
# Let's pull these out to make the outputs from rsas crystal clear
PQ1 = outputs['PQ'][0]
C_outi = outputs['C_out'][0]
ST = outputs['ST']
# ==================================
# Plot the transit time distribution
# ==================================
fig = plt.figure(1)
plt.clf()
plt.plot(PQ1[:,-1], 'b-', label='rsas model', lw=2)
plt.plot(PQ_exact, 'r-.', label='analytical solution', lw=2)
plt.ylim((0,1))
plt.xlim((0,4*T_0))
plt.legend(loc=0)
plt.ylabel('P_Q(T)')
plt.xlabel('age T')
plt.title('Cumulative transit time distribution')
#%%
# =====================================================================
# Convolve the transit time distributions with the input concentrations
# =====================================================================
# Use the estimated transit time distribution and input timeseries to estimate
# the output timeseries
C_outb, C_mod_raw, observed_fraction = rsas.transport(PQ1, C_J, C_old)
# Calculate the output concentration using the analytical TTD
T=np.arange(N*100.+1)/100
PQe = np.tile(1-np.exp(-T/T_0), (N*100.+1, 1)).T
C_oute, C_mod_raw, observed_fraction = rsas.transport(PQe, C_J.repeat(100), C_old)
# Plot the result
fig = plt.figure(2)
plt.clf()
plt.plot(np.arange(N)+1, C_outb, 'b-', label='rsas.transport', lw=2)
plt.plot(np.arange(N)+1, C_outi, 'g--', label='rsas internal', lw=2)
plt.plot(T[1:], C_oute, 'r-.', label='rsas internal', lw=2)
plt.legend(loc=0)
plt.ylabel('Concentration [-]')
plt.xlabel('time')
plt.title('Outflow concentration calculated three ways')
plt.show()
