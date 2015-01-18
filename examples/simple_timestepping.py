# -*- coding: utf-8 -*-
"""Storage selection (SAS) functions: simple example

Runs the rSAS model for an example dataset with one outflow. A file containing 
a steady-state transit time distribution is loaded, and used to construct an 
equivalent storage seletion function.

This script uses the time-stepping algorithm (mode='time'), and 
calculates discharge concentrations progressively. This saves memory, but is not
advised if you want to run the model with the same rSAS parameters but different
concentration input timeseries.

The example dataset is the same as in lower_hafren_example.py, but the results here are not
intended to be realistic.
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
# Calculate a (bogus) effective rainfall rate
data['Peff water flux mm/day'] = data['Q water flux mm/day'].mean() * data['P water flux mm/day'] / data['P water flux mm/day'].mean()
# Calculate a (bogus) adjusted concentration
data['Peff Cl mg/l'] = data['Q Cl mg/l'].mean() * data['P Cl mg/l'] / data['P Cl mg/l'].mean()
# =========================
# Parameters needed by rsas
# =========================
# The concentration of discharge older than the start of observations is
# assumed to be the long-term mean deposition rate divided by mean discharge
C_old = np.mean(data['Peff Cl mg/l'] * data['P Cl mg/l'])/data['Peff Cl mg/l'].mean()
# =========================
# Create the rsas functions
# =========================
Q_rSAS_fun_type = 'from_steady_state_TTD'
TT_CDF = pd.read_csv('example_steady_state_TTD.csv', names='P')
Q_rSAS_fun_parameters = np.zeros((len(TT_CDF),1))
Q_rSAS_fun_parameters[0,0] = 5.72 # steady-state flow rate
Q_rSAS_fun_parameters[1:,0] = TT_CDF.values[1:,0]
rSAS_fun_Q = rsas.create_function(Q_rSAS_fun_type, Q_rSAS_fun_parameters)
# =================
# Initial condition
# =================
# Unknown total storage
ST_init = np.zeros(N + 1)
# =============
# Run the model
# =============
# Let's pull these out to make the inputs to rsas crystal clear
Peff = data['Peff water flux mm/day'].values
Q = data['Q water flux mm/day'].values
C_in = data['Peff Cl mg/l'].values
# Run it
output = rsas.solve(Peff, Q, rSAS_fun_Q, mode='time', ST_init=ST_init, C_in=C_in, C_old=C_old)
# Plot the result
fig = plt.figure(1)
plt.clf()
plt.fill_between(data.index, data['P Cl mg/l'], color='0.7', label='Load in')
plt.plot(data.index, data['Q Cl mg/l'], color='c', label='Obs load out')
plt.plot(data.index, output['C_out'][0], 'b', label='Pred load out')