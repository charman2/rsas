# -*- coding: utf-8 -*-
"""Storage selection (SAS) functions: example with parmeter estimation

Estimate parameters for the rSAS model for a synthetic dataset
with one flux in and one flux out

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
C_J = data['C_J'].values
C_Q1 = data['C_Q1'].values
S = data['S'].values
# =========================
# Parameters needed by rsas
# =========================
# The concentration of water older than the start of observations
C_old = ((J*C_J)[J>0]).sum()/(J[J>0]).sum()
# =================
# Initial condition
# =================
# Unknown initial age distribution, so just set this to zeros
ST_init = np.zeros(N + 1)
# =================
# Parameter optimization
# =================
# Make a function that runs the model given set of parameters
def run(params):
    # Unroll the parameters
    S_dead, lam = params
    # Define the rSAS function
    Q_rSAS_fun_type = 'kumaraswami'
    ST_min = np.ones(N) * 0.
    ST_max = S + S_dead
    a = np.maximum(0.01, 2. +  lam * (S - S.mean())/S.std())
    b = np.ones(N) * 5.
    Q_rSAS_fun_parameters = np.c_[ST_min, ST_max, a, b]
    rSAS_fun_Q1 = rsas.create_function(Q_rSAS_fun_type, Q_rSAS_fun_parameters)
    rSAS_fun = [rSAS_fun_Q1]
    # Run the model
    outputs = rsas.solve(J, Q, rSAS_fun, ST_init=ST_init,
                         mode='RK4', dt = 1., n_substeps=1, C_J=C_J, C_old=[C_old], verbose=False, debug=False)
    # Return the results
    return outputs, rSAS_fun
# Make a function that returns the RMS error associated with a
# particular set of parameters
def err(params):
    outputs, _ = run(params)
    C_mod = outputs['C_Q'][:,0,0]
    isobs = np.isfinite(C_Q1)
    err = np.sqrt(((C_Q1[isobs]-C_mod[isobs])**2).mean())
    print params, err
    return err
# Start with a good initial guess of the parameters
# Notice above that the first line of the run function unrolls
# this list by assuming the first entry is S_dead and the second
# is lam. You can include as many parameters here as you like, but
# you must also change the line where they are unrolled.
params0 = [10., -0.2]
# run the optimzer
from scipy.optimize import fmin
params_opt = fmin(err, params0)
S_dead, lam = params_opt
print "Optimum parameter set = ", params_opt
# run the model for these parameters
outputs, rSAS_fun = run(params_opt)
# extract the outputs
ST = outputs['ST']
PQ1m = outputs['PQ'][:,:,0]
C_Q1m1 = outputs['C_Q'][:,0,0]
#%%
# ==================================
# Plot the rSAS function
# ==================================
STx = np.linspace(0,S.max()+S_dead,100)
Omega = np.r_[[rSAS_fun[0].cdf_i(STx,i) for i in range(N)]].T
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
PQ1i[:,0]  = rSAS_fun[0].cdf_i(ST[:,0],0)
PQ1i[:,1:] = np.r_[[rSAS_fun[0].cdf_i(ST[:,i+1],i) for i in range(N)]].T
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
plt.plot(data['datetime'], data['C_Q1'], 'r.', label='observed', lw=2)
plt.legend(loc=0)
plt.ylabel('Concentration [-]')
plt.xlabel('time')
plt.title('Outflow concentration')
plt.show()
