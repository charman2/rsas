# -*- coding: utf-8 -*-
"""Storage selection (SAS) functions: example with two flux out at steady state

Runs the rSAS model for a synthetic dataset with two flux in and out
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
S_0 = 10. # <-- volume of the uniformly sampled store
Q_0 = 1. # <-- steady-state flow rate
T_0 = S_0 / (2 * Q_0)
# Note that the analytical solution for the cumulative TTD is
T = np.arange(N+1)
PQ_exact = 1 - np.exp(-T/T_0)
# Steady-state flow in and out for N timesteps
J = np.ones(N) * Q_0 * 2
Q = np.ones((N, 2)) * Q_0
# Three random timeseries of concentrations
C_J = np.tile(-np.log(np.random.rand(N,1)), (1,3))
# =========================
# Parameters needed by rsas
# =========================
# The concentration of water older than the start of observations
C_old = [0., 0., 0.]
alpha = np.ones((N,2,3))
alpha[:,0,1] = 0.5
alpha[:,0,2] = 0.
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
# =================
# Initial condition
# =================
# Unknown initial age distribution, so just set this to zeros
ST_init = np.zeros(N + 1)
MS_init = np.zeros((N + 1, 3))
# =============
# Run the model
# =============
# Run it
#TODO check PQ with n_substeps>1
outputs = rsas.solve(J, Q, [rSAS_fun_Q1, rSAS_fun_Q2], ST_init=ST_init, MS_init=MS_init,
                     mode='RK4', dt = 1., n_substeps=1, C_J=C_J, C_old=C_old, alpha=alpha, verbose=True, debug=True)
# Let's pull these out to make the outputs from rsas crystal clear
# State variables: age-ranked storage of water and solutes
# ROWS of ST, MS are T - ages
# COLUMNS of ST, MS are t - times
# LAYERS of MS are s - solutes
ST = outputs['ST']
MS1 = outputs['MS'][:,:,0]
MS2 = outputs['MS'][:,:,1]
MS3 = outputs['MS'][:,:,2]
# Timestep-averaged backwards TTD
# ROWS of PQ are T - ages
# COLUMNS of PQ are t - times
# LAYERS of PQ are q - fluxes
PQ1m = outputs['PQ'][:,:,0]
PQ2m = outputs['PQ'][:,:,1]
# Timestep-averaged outflow concentration
# ROWS of C_Q are t - times
# COLUMNS of C_Q are q - fluxes
# LAYERS of C_Q are s - solutes
C1_Q1m1 = outputs['C_Q'][:,0,0]
C1_Q2m1 = outputs['C_Q'][:,1,0]
C2_Q1m1 = outputs['C_Q'][:,0,1]
C2_Q2m1 = outputs['C_Q'][:,1,1]
C3_Q1m1 = outputs['C_Q'][:,0,2]
C3_Q2m1 = outputs['C_Q'][:,1,2]
# Timestep averaged solute load out
# ROWS of MQ are T - ages
# COLUMNS of MQ are t - times
# LAYERS of MQ are q - fluxes
# Last dimension of MS are s - solutes
M11m = outputs['MQ'][:,:,0,0]
M12m = outputs['MQ'][:,:,1,0]
M21m = outputs['MQ'][:,:,0,1]
M22m = outputs['MQ'][:,:,1,1]
M31m = outputs['MQ'][:,:,0,2]
M32m = outputs['MQ'][:,:,1,2]
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
n=20
T=np.arange(N*n+1.)/n
PQ1e = np.tile(1-np.exp(-T/T_0), (N*n+1., 1)).T
# Use the transit time distribution and input timeseries to estimate
# the output timeseries for the exact, instantaneous and timestep-averaged cases
C1_Q1m2, C_mod_raw, observed_fraction = rsas.transport(PQ1m, C_J[:,0], C_old[0])
C1_Q1i, C_mod_raw, observed_fraction = rsas.transport(PQ1i, C_J[:,0], C_old[0])
C1_Q1ei, C_mod_raw, observed_fraction = rsas.transport(PQ1e, C_J[:,0].repeat(n), C_old[0])
# This calculates an exact timestep-averaged value
C1_Q1em = np.reshape(C1_Q1ei,(N,n)).mean(axis=1)
# Plot the results
fig = plt.figure(2)
plt.clf()
plt.step(np.arange(N), C1_Q1em, 'r', ls='-', label='mean exact, C1', lw=2, where='post')
plt.step(np.arange(N), C1_Q1m2, 'b', ls=':', label='mean rsas.transport, C1', lw=2, where='post')
plt.plot((np.arange(N*n) + 1.)/n, C1_Q1ei, 'r-', label='inst. exact, C1', lw=1)
plt.plot(np.arange(N)+1, C1_Q1i, 'b:o', label='inst. rsas.transport, C1', lw=1)
plt.step(np.arange(N), C1_Q1m1, 'g', ls='-', label='mean rsas internal, C1', lw=2, where='post')
plt.step(np.arange(N), C2_Q1m1, 'g', ls='--', label='mean rsas internal, C2', lw=2, where='post')
plt.step(np.arange(N), C3_Q1m1, 'g', ls='-.', label='mean rsas internal, C3', lw=2, where='post')
plt.legend(loc=0)
plt.ylabel('Concentration [-]')
plt.xlabel('time')
plt.title('Outflow concentration')
plt.show()
