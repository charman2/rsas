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
S_0 = 4. # <-- volume of the uniformly sampled store
Q_0 = 1.0 # <-- steady-state flow rate
T_0 = S_0 / Q_0
N = 10
n_substeps = 10
# Steady-state flow in and out for N timesteps
J = np.ones(N) * Q_0
Q = np.ones((N,1)) * Q_0
# A timeseries of concentrations
C_J = np.ones((N,1))
#C_J = -np.log(np.random.rand(N,1))
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
# =================
# Initial condition
# =================
# Unknown initial age distribution, so just set this to zeros
ST_init = np.zeros(N + 1)
# =============
# Run the model - first method
# =============
# Run it
outputs = rsas.solve(J, Q, [rSAS_fun_Q1], ST_init=ST_init,
                     mode='RK4', dt = 1., n_substeps=n_substeps, C_J=C_J, C_old=[C_old], verbose=False, debug=False)
#%%
# Timestep-averaged outflow concentration
# ROWS of C_Q are t - times
# COLUMNS of PQ are q - fluxes
C_Qm1 = outputs['C_Q'][:,0,0]
# =============
# Run the model - second method
# =============
# Run it
outputs = rsas.solve(J, Q, [rSAS_fun_Q1], ST_init=ST_init,
                     mode='RK4', dt = 1., n_substeps=n_substeps, verbose=False, debug=False)
# Age-ranked storage
# ROWS of ST are T - ages
# COLUMNS of ST are t - times
# LAYERS of MS are s - solutes
ST = outputs['ST']
# Timestep-averaged backwards TTD
# ROWS of PQ are T - ages
# COLUMNS of PQ are t - times
# LAYERS of PQ are q - fluxes
PQm = outputs['PQ'][:,:,0]
# Timestep-averaged outflow concentration
# ROWS of C_Q are t - times
# COLUMNS of PQ are q - fluxes
# Use rsas.transport to convolve the input concentration with the TTD
C_Qm2, C_mod_raw, observed_fraction = rsas.transport(PQm, C_J[:,0], C_old)
# ==================================
# Plot the age-ranked storage
# ==================================
print 'Plotting ST at the last timestep'
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
print 'Getting the instantaneous TTD'
PQi = np.zeros((N+1, N+1))
PQi[:,0]  = rSAS_fun_Q1.cdf_i(ST[:,0],0)
PQi[:,1:] = np.r_[[rSAS_fun_Q1.cdf_i(ST[:,i+1],i) for i in range(N)]].T
# Lets also get the exact TTD
print 'Getting the exact solution'
n=100
T=np.arange(N*n+1.)/n
PQe = np.tile(1-np.exp(-T/T_0), (N*n+1., 1)).T
# Use the transit time distribution and input timeseries to estimate
# the output timeseries for the exact and instantaneous cases
print 'Getting the concentrations'
C_Qi, C_mod_raw, observed_fraction = rsas.transport(PQi, C_J[:,0], C_old)
C_Qei, C_mod_raw, observed_fraction = rsas.transport(PQe, C_J[:,0].repeat(n), C_old)
# This calculates an exact timestep-averaged value
C_Qem = np.reshape(C_Qei,(N,n)).mean(axis=1)
# Plot the results
print 'Plotting concentrations'
fig = plt.figure(2)
plt.clf()
plt.step(np.arange(N), C_Qem, 'r', ls='-', label='mean exact', lw=2, where='post')
plt.step(np.arange(N), C_Qm1, 'g', ls='--', label='mean rsas internal', lw=2, where='post')
plt.step(np.arange(N), C_Qm2, 'b', ls=':', label='mean rsas.transport', lw=2, where='post')
plt.plot((np.arange(N*n) + 1.)/n, C_Qei, 'r-', label='inst. exact', lw=1)
plt.plot(np.arange(N)+1, C_Qi, 'b:o', label='inst. rsas.transport', lw=1)
plt.legend(loc=0)
plt.ylim((0,1))
plt.ylabel('Concentration [-]')
plt.xlabel('time')
plt.title('Outflow concentration')
plt.show()
