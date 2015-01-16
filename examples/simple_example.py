# -*- coding: utf-8 -*-
"""Storage selection (SAS) functions: simple example

Runs the rSAS model for an example dataset with one outflow. The example dataset
is the same as in lower_hafren_example.py, but the results here are not
intended to be realistic.

This script uses the time-stepping algorithm solve_all_by_age_1out, and 
calculates discharge concentrations progressively.
"""
from __future__ import division
import rsas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Load thedata
data = pd.read_csv('lower_hafren_data.csv', index_col=0, parse_dates=True)
data=data[:1000]
N = len(data)
# The concentration of discharge older than the start of observations
C_old = np.mean(data['P Cl mg/l'])
# Parameters for the rSAS function (see Harman [2015])
Q_rSAS_fun_type = 'gamma'
S_Q0 = 4922.
lambda_Q = -102.
Q_alpha = 0.6856
Q_dS_crit = 48.25
S_Q = np.maximum(1., lambda_Q * (data['S mm'] - Q_dS_crit))
Q_rSAS_fun_parameters = np.c_[np.zeros(N), S_Q, np.ones(N) * Q_alpha]
# Run the model
results = rsas.solve_all_by_time_1out(
                                    data['P water flux mm/day'].values,
                                    data['Q water flux mm/day'].values,
                                    Q_rSAS_fun_parameters,
                                    Q_rSAS_fun_type,
                                    C_in=data['P Cl mg/l'],
                                    C_old=C_old)
# Unroll the outputs
C_out, ST, PQ, Qout, thetaQ, thetaS, MassBalance = results
#%%
# Plot the result
fig = plt.figure(1)
plt.clf()
plt.fill_between(data.index, data['P Cl mg/l'], color='0.7', label='Load in')
plt.plot(data.index, C_out, 'b', label='Load out')