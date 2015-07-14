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
# Load the input data
# =====================================
data = pd.read_csv('Q1.csv', index_col=0, parse_dates=[1])
# length of the dataset
N = len(data)
# The individual timeseries can be pulled out of the dataframe
J = data['J'].values
Q1 = data['Q1'].values
C_J = data['C_J'].values
C_Q1 = data['C_Q1'].values
S = data['S'].values
# =================
# Initial condition
# =================
# Unknown initial age distribution, so just set this to zeros
ST_init = np.zeros(N + 1)
# =========================
# Parameters needed by rsas
# =========================
# The concentration of water older than the start of observations
C_old = np.mean(C_J)
# =========================
# Uniform case
# =========================
# Parameters for the rSAS function
S_0 = 20.
# The uniform distribution extends between S_T=a and S_T=b.
Q_rSAS_fun_type = 'uniform'
a = np.ones(N) * 0.
b = np.ones(N) * S_0
Q_rSAS_fun_parameters = np.c_[a, b]
rSAS_fun_Q1 = rsas.create_function(Q_rSAS_fun_type, Q_rSAS_fun_parameters)
# =============
# Run the model
# =============
# Run it
outputs = rsas.solve(J, [Q1], [rSAS_fun_Q1], ST_init=ST_init,
                     mode='time', dt = 1., n_substeps=5)
# Let's pull these out to make the outputs from rsas crystal clear
PQ1 = outputs['PQ'][0]
ST = outputs['ST']
# ==================================
# Plot the transit time distribution
# ==================================
fig = plt.figure(1)
plt.clf()
plt.plot(PQ1, lw=1)
plt.ylim((0,1))
plt.ylabel('P_Q(T)')
plt.xlabel('age T')
plt.title('Cumulative transit time distribution')
#%%
# =====================================================================
# Convolve the transit time distributions with the input concentrations
# =====================================================================
# Use the estimated transit time distribution and input timeseries to estimate
# the output timeseries
C_out, C_mod_raw, observed_fraction = rsas.transport(PQ1, C_J, C_old)
# Plot the result
fig = plt.figure(2)
#plt.clf()
plt.plot(data['datetime'], C_out, 'b-', label='rsas predictions', lw=2)
plt.plot(data['datetime'], data['C_Q1'], 'r.', label='observed', lw=2)
plt.legend(loc=0)
plt.ylabel('Concentration [-]')
plt.xlabel('time')
plt.title('Outflow concentration')
plt.show()
