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
the numerical scheme originally used in the paper (mode='age') is called,
rather than a new one (mode='time').
"""
from __future__ import division
import rsas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# =====================================
# Load and process the input timeseries
# =====================================
# Load the data
data = pd.read_csv('lower_hafren_data.csv', index_col=0, parse_dates=True)
# Uncomment the following line if you want to just run the first 2000 days
# data = data[:2000]
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
# - Discharge
Q_rSAS_fun_type = 'gamma'
lambda_Q = -102.
Q_alpha = 0.6856
Q_dS_crit = 48.25
S_Q = np.maximum(1., lambda_Q * (data['S mm'] - Q_dS_crit))
Q_rSAS_fun_parameters = np.c_[np.zeros(N), S_Q, np.ones(N) * Q_alpha]
rSAS_fun_Q = rsas.create_function(Q_rSAS_fun_type, Q_rSAS_fun_parameters)
# - ET
ET_rSAS_fun_type = 'uniform'
S_ET = 398.
ET_rSAS_fun_parameters = np.c_[np.zeros(N), np.ones(N) * S_ET ]
rSAS_fun_ET = rsas.create_function(ET_rSAS_fun_type, ET_rSAS_fun_parameters)
# =================
# Initial condition
# =================
# Unknown total storage
ST_init = np.zeros(N + 1)
# =============
# Run the model
# =============
# Let's pull these out to make the inputs to rsas crystal clear
J = data['P water flux mm/day'].values
Q = data['Q water flux mm/day'].values
ET = data['ET water flux mm/day'].values
C_in = data['P+occult Cl mg/l'].values
# Run it
outputs = rsas.solve(J, [Q, ET], [rSAS_fun_Q, rSAS_fun_ET], ST_init=ST_init, 
                     mode='age', dt = 1., n_substeps=1, n_iterations=4)
# Let's pull these out to make the outputs from rsas crystal clear
PQ = outputs['PQ'][0]
thetaQ = outputs['thetaQ'][0]
thetaS = outputs['thetaS']
# =====================================================================
# Convolve the transit time distributions with the input concentrations
# =====================================================================
# Use the estimated transit time distribution and input timeseries to estimate
# the output timeseries
C_out, C_mod_raw, observed_fraction = rsas.transport_with_evapoconcentration(PQ, thetaQ, thetaS, data['P+occult Cl mg/l'], C_old)
data['Q Predicted Cl mg/l'] = C_out
# Plot the result
fig = plt.figure(1)
plt.clf()
plt.plot(data.index, data['Q Cl mg/l'], label='Observed')
plt.plot(data.index, data['Q Predicted Cl mg/l'], label='Predicted')