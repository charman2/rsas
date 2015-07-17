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
N = 4
J = np.array([1, 1, 1, 2], dtype = np.float64)
Q1 = np.array([1, 1, 1, 2], dtype = np.float64)
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
a = np.ones(N) * 2.5
b = np.ones(N) * 2.5
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
                     mode='RK4', dt = 1., n_substeps=100, C_in=C_J, C_old=C_old)
# Let's pull these out to make the outputs from rsas crystal clear
PQ1 = outputs['PQ'][0]
C_outi = outputs['C_out'][0]
ST = outputs['ST']
# ROWS of ST, PQ1 are T - ages
# COLUMNS of ST, PQ1 are t - times
# ==================================
# Plot the transit time distribution
# ==================================
fig = plt.figure(1)
plt.clf()
plt.plot(PQ1[:,-1], 'b-', label='rsas model', lw=2)
#plt.plot(PQ_exact, 'r-.', label='analytical solution', lw=2)
plt.ylim((0,1))
#plt.xlim((0,4*T_0))
plt.legend(loc=0)
plt.ylabel('P_Q(T)')
plt.xlabel('age T')
plt.title('Cumulative transit time distribution')
#%%
## =====================================================================
## Convolve the transit time distributions with the input concentrations
## =====================================================================
## Use the estimated transit time distribution and input timeseries to estimate
## the output timeseries
#C_outb, C_mod_raw, observed_fraction = rsas.transport(PQ1, C_J, C_old)
## Calculate the output concentration using the analytical TTD
#T=np.arange(N*100.+1)/100
#PQe = np.tile(1-np.exp(-T/T_0), (N*100.+1, 1)).T
#C_oute, C_mod_raw, observed_fraction = rsas.transport(PQe, C_J.repeat(100), C_old)
## Plot the result
#fig = plt.figure(2)
#plt.clf()
#plt.plot(np.arange(N)+1, C_outb, 'b-', label='rsas.transport', lw=2)
#plt.plot(np.arange(N)+1, C_outi, 'g--', label='rsas internal', lw=2)
#plt.plot(T[1:], C_oute, 'r-.', label='exact', lw=2)
#plt.legend(loc=0)
#plt.ylabel('Concentration [-]')
#plt.xlabel('time')
#plt.title('Outflow concentration calculated three ways')
#plt.show()
#
