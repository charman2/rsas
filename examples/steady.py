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
N = 10
S_0 = 1. # <-- volume of the uniformly sampled store
Q_0 = 1. # <-- steady-state flow rate
T_0 = S_0 / Q_0
# Note that the analytical solution for the cumulative TTD is
T = np.arange(N+1)
ST_exact = S_0 * (1 - np.exp(-T/T_0))
# Steady-state flow in and out for N timesteps
J = np.ones(N) * Q_0
Q = np.ones((N,1)) * Q_0
# A random timeseries of concentrations
C_J = np.ones((N,1))#-np.log(np.random.rand(N,1))
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
outputs = rsas.solve(J, Q, [rSAS_fun_Q1], ST_init=ST_init,
                     mode='RK4', dt = 1., n_substeps=1, C_J=C_J, C_old=[C_old])
# Let's pull these out to make the outputs from rsas crystal clear
PQ1 = outputs['PQ'][:,:,0]
C_outi_m = outputs['C_Q'][:,0,0]
ST = outputs['ST']
MS = outputs['MS'][:,:,0]
MQ = outputs['MQ'][:,:,0,0]
PQ1_i = np.zeros((N+1, N+1))
PQ1_i[:,1:] = np.r_[[rSAS_fun_Q1.cdf_i(ST[:,i],i) for i in range(N)]].T
# ROWS of ST, PQ1 are T - ages
# COLUMNS of ST, PQ1 are t - times
# ==================================
# Plot the transit time distribution
# ==================================
fig = plt.figure(1)
plt.clf()
plt.plot(ST[:,-1], 'b-', label='rsas model', lw=2)
plt.plot(ST_exact, 'r-.', label='analytical solution', lw=2)
plt.ylim((0,S_0))
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
C_outb_m, C_mod_raw, observed_fraction = rsas.transport(PQ1, C_J[:,0], C_old)
C_outb_i, C_mod_raw, observed_fraction = rsas.transport(PQ1_i, C_J[:,0], C_old)
# Calculate the output concentration using the analytical TTD
T=np.arange(N*100.+1)/100
PQe = np.tile(1-np.exp(-T/T_0), (N*100.+1, 1)).T
C_oute_i, C_mod_raw, observed_fraction = rsas.transport(PQe, C_J[:,0].repeat(100), C_old)
C_oute_m = np.reshape(C_oute_i,(N,100)).mean(axis=1)
# Plot the result
fig = plt.figure(2)
plt.clf()
plt.plot(np.arange(N)+1, C_outb_m, 'b-', label='rsas.transport', lw=2)
plt.plot(np.arange(N)+1, C_outb_i, 'b-', label='rsas.transport', lw=1)
plt.plot(np.arange(N)+1, C_outi_m, 'g--', label='rsas internal', lw=2)
plt.plot(np.arange(N)+1, C_oute_m, 'r-.', label='exact', lw=2)
plt.plot((np.arange(N*100.) + 1)/100, C_oute_i, 'r-.', label='exact', lw=1)
plt.legend(loc=0)
plt.ylabel('Concentration [-]')
plt.xlabel('time')
plt.title('Outflow concentration calculated three ways')
plt.show()
