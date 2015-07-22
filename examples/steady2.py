# -*- coding: utf-8 -*-
"""Storage selection (SAS) functions: example with multiple fluxes out at steady state

Runs the rSAS model for a synthetic dataset with one flux in and
multiple fluxes out and steady state flow

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
S_0 = 3. # <-- volume of the uniformly sampled store
Q1_0 = 1. # <-- steady-state flow rate
Q2_0 = 1. # <-- steady-state flow rate
T_0 = S_0 / (Q1_0 + Q2_0)
N = 10
n_substeps = 10
# Steady-state flow in and out for N timesteps
J = np.ones(N) * (Q1_0 + Q2_0)
Q = np.c_[[np.ones(N) * Q1_0, np.ones(N) * Q2_0]].T
# A random timeseries of concentrations
#C_J = np.ones((N,1))
C_J = -np.log(np.random.rand(N,1))
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
ST_min = np.ones(N) * 0.
ST_max = np.ones(N) * S_0
Q_rSAS_fun_parameters = np.c_[ST_min, ST_max]
rSAS_fun_Q1 = rsas.create_function(Q_rSAS_fun_type, Q_rSAS_fun_parameters)
Q_rSAS_fun_type = 'uniform'
ST_min = np.ones(N) * 0.
ST_max = np.ones(N) * S_0
Q_rSAS_fun_parameters = np.c_[ST_min, ST_max]
rSAS_fun_Q2 = rsas.create_function(Q_rSAS_fun_type, Q_rSAS_fun_parameters)
rSAS_fun = [rSAS_fun_Q1, rSAS_fun_Q2]
# =================
# Initial condition
# =================
# Unknown initial age distribution, so just set this to zeros
ST_init = np.zeros(N + 1)
# =============
# Run the model
# =============
# Run it
outputs = rsas.solve(J, Q, rSAS_fun, ST_init=ST_init,
                     mode='RK4', dt = 1., n_substeps=n_substeps, C_J=C_J, C_old=[C_old], verbose=True, debug=True)
# Let's pull these out to make the outputs from rsas crystal clear
# State variables: age-ranked storage of water and solutes
# ROWS of ST, MS are T - ages
# COLUMNS of ST, MS are t - times
# LAYERS of MS are s - solutes
ST = outputs['ST']
MS = outputs['MS'][:,:,0]
# Timestep-averaged backwards TTD
# ROWS of PQ are T - ages
# COLUMNS of PQ are t - times
# LAYERS of PQ are q - fluxes
PQ1m = outputs['PQ'][:,:,0]
PQ2m = outputs['PQ'][:,:,1]
# Timestep-averaged outflow concentration
# ROWS of C_Q are t - times
# COLUMNS of C_Q are q - fluxes
C_Q1m1 = outputs['C_Q'][:,0,0]
C_Q2m1 = outputs['C_Q'][:,1,0]
# Timestep averaged solute load out
# ROWS of MQ are T - ages
# COLUMNS of MQ are t - times
# LAYERS of MQ are q - fluxes
# Last dimension of MQ are s - solutes
MQ1m = outputs['MQ'][:,:,0,0]
MQ2m = outputs['MQ'][:,:,1,0]
# ==================================
# Plot the age-ranked storage
# ==================================
# The analytical solution for the age-ranked storage is
T = np.arange(N+1)
ST_exact = S_0 * (1 - np.exp(-T/T_0))
# plot this with the rsas estimate
fig = plt.figure(1)
plt.clf()
plt.plot(ST[:,-1], 'b-', label='rsas model', lw=2)
plt.plot(ST_exact, 'r-.', label='analytical solution', lw=2)
plt.ylim((0,S_0))
plt.legend(loc=0)
plt.ylabel('$S_T(T)$')
plt.xlabel('age $T$')
plt.title('Age-ranked storage')
#%%
# =====================================================================
# Outflow concentration estimated using several different TTD
# =====================================================================
# Lets get the instantaneous value of the TTD at the end of each timestep
PQ1i = np.zeros((N+1, N+1))
PQ1i[:,0]  = rSAS_fun_Q1.cdf_i(ST[:,0],0)
PQ1i[:,1:] = np.r_[[rSAS_fun_Q1.cdf_i(ST[:,i+1],i) for i in range(N)]].T
# Lets also get the exact TTD for the combined flux out
n=1000
T=np.arange(N*n+1.)/n
PQ1e = np.tile(1-np.exp(-T/T_0), (N*n+1., 1)).T
# Use the transit time distribution and input timeseries to estimate
# the output timeseries for the exact, instantaneous and timestep-averaged cases
C_Q1m2, C_mod_raw, observed_fraction = rsas.transport(PQ1m, C_J[:,0], C_old)
C_Q1i, C_mod_raw, observed_fraction = rsas.transport(PQ1i, C_J[:,0], C_old)
C_Q1ei, C_mod_raw, observed_fraction = rsas.transport(PQ1e, C_J[:,0].repeat(n), C_old)
# This calculates an exact timestep-averaged value
C_Q1em = np.reshape(C_Q1ei,(N,n)).mean(axis=1)
# Plot the results
fig = plt.figure(2)
plt.clf()
plt.step(np.arange(N), C_Q1em, 'r', ls='-', label='mean exact', lw=2, where='post')
plt.step(np.arange(N), C_Q1m1, 'g', ls='--', label='mean rsas internal', lw=2, where='post')
plt.step(np.arange(N), C_Q1m2, 'b', ls=':', label='mean rsas.transport', lw=2, where='post')
plt.plot((np.arange(N*n) + 1.)/n, C_Q1ei, 'r-', label='inst. exact', lw=1)
plt.plot(np.arange(N)+1, C_Q1i, 'b:o', label='inst. rsas.transport', lw=1)
plt.legend(loc=0)
plt.ylabel('Concentration [-]')
plt.xlabel('time')
plt.title('Outflow concentration')
plt.show()
