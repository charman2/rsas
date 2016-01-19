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
# Load the input data
# =====================================
data = pd.read_csv('Q1.csv', index_col=0, parse_dates=[1])
# length of the dataset
N = len(data)
# The individual timeseries can be pulled out of the dataframe
J = data['J'].values
Q = data['Q1'].values
C_J = data['C_J'].values-2
C_Q1 = data['C_Q1'].values
ST_min = data['ST_min'].values
ST_max = data['ST_max'].values
# =========================
# Parameters needed by rsas
# =========================
# The concentration of water older than the start of observations
C_old = ((J*C_J)[J>0]).sum()/((J)[J>0]).sum()
# =========================
# Create the rsas functions
# =========================
S_dead = 10.
#lam = 0.
# Uniform
# Parameters for the rSAS function
Q_rSAS_fun_type = 'uniform'
ST_min = np.zeros(N)
ST_max = S + S_dead
Q_rSAS_fun_parameters = np.c_[ST_min, ST_max]
rSAS_fun_Q1 = rsas.create_function(Q_rSAS_fun_type, Q_rSAS_fun_parameters)
rSAS_fun = [rSAS_fun_Q1]
# Kumaraswami
## Parameters for the rSAS function
#Q_rSAS_fun_type = 'kumaraswami'
#ST_min = np.ones(N) * 0.
#ST_max = S + S_dead
#a = np.maximum(0.01, 2. +  lam * (S - S.mean())/S.std())
#b = np.ones(N) * 5.
#Q_rSAS_fun_parameters = np.c_[a, b, ST_min, ST_max]
#rSAS_fun_Q1 = rsas.create_function(Q_rSAS_fun_type, Q_rSAS_fun_parameters)
#rSAS_fun = [rSAS_fun_Q1]
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
                     mode='RK4', dt = 1., n_substeps=3, C_J=C_J, C_old=[C_old], verbose=False, debug=False)
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
# Timestep-averaged outflow concentration
# ROWS of C_Q are t - times
# COLUMNS of PQ are q - fluxes
C_Q1m1 = outputs['C_Q'][:,0,0]
# Timestep averaged solute load out
# ROWS of MQ are T - ages
# COLUMNS of MQ are t - times
# LAYERS of MQ are q - fluxes
# Last dimension of MS are s - solutes
MQ1m = outputs['MQ'][:,:,0,0]
#%%
# ==================================
# Plot the rSAS function
# ==================================
STx = np.linspace(0,S.max()+S_dead,100)
Omega = np.r_[[rSAS_fun_Q1.cdf_i(STx,i) for i in range(N)]].T
import matplotlib.cm as cm
fig = plt.figure(0)
plt.clf()
for i in range(N):
    plt.plot(STx, Omega[:,i], lw=1, color=cm.jet((S[i]-S.min())/S.ptp()))
plt.ylim((0,1))
plt.ylabel('$\Omega_Q(T)$')
plt.xlabel('age-ranked storage $S_T$')
plt.title('Cumulative rSAS function')
#%%
# ==================================
# Plot the transit time distribution
# ==================================
fig = plt.figure(1)
plt.clf()
plt.plot(PQ1m, lw=1)
plt.ylim((0,1))
plt.ylabel('$P_Q(T)$')
plt.xlabel('age $T$')
plt.title('Cumulative transit time distribution')
#%%
# =====================================================================
# Outflow concentration estimated using several different TTD
# =====================================================================
# Lets get the instantaneous value of the TTD at the end of each timestep
PQ1i = np.zeros((N+1, N+1))
PQ1i[:,0]  = rSAS_fun_Q1.cdf_i(ST[:,0],0)
PQ1i[:,1:] = np.r_[[rSAS_fun_Q1.cdf_i(ST[:,i+1],i) for i in range(N)]].T
# Use the transit time distribution and input timeseries to estimate
# the output timeseries for the instantaneous and timestep-averaged cases
C_Q1i, C_Q1i_raw, Q1i_observed_fraction = rsas.transport(PQ1i, C_J, C_old)
C_Q1m2, C_Q1m2_raw, Q1m2_observed_fraction = rsas.transport(PQ1m, C_J, C_old)
# Plot the results
fig = plt.figure(2)
plt.clf()
plt.step(data['datetime'], C_Q1m1, 'g', ls='--', label='mean rsas internal', lw=2, where='post')
plt.step(data['datetime'], C_Q1m2, 'b', ls=':', label='mean rsas.transport', lw=2, where='post')
plt.step(data['datetime'], C_Q1m2_raw, '0.5', ls=':', label='mean rsas.transport (obs part)', lw=2, where='post')
plt.plot(data['datetime'], C_Q1i, 'b:o', label='inst. rsas.transport', lw=1)
#plt.plot(data['datetime'], data['C_Q1'], 'r.', label='observed', lw=2)
plt.ylim((-2, 0))
plt.legend(loc=0)
plt.ylabel('Concentration [-]')
plt.xlabel('time')
plt.title('Outflow concentration')
plt.show()
