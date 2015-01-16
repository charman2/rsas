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

This script is identical to lower_hafren_example.py, except that the numerical scheme 
solve_all_by_time_2out is used instead of the one used in the paper, solve_all_by_age_2out
"""
from __future__ import division
import rsas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Load thedata
data = pd.read_csv('lower_hafren_data.csv', index_col=0, parse_dates=True)
N = len(data)
# Estimate ET from FAO reference evapotranspiration as ET = k_E * ETO, where
# k_E is chosen to ensurelong-term mass balance.
k_ET = (data['P water flux mm/day'].mean() - data['Q water flux mm/day'].mean())/data['ET0 mm/day'].mean()
data['ET water flux mm/day'] = data['ET0 mm/day'] * k_ET
# The fraction of CL deposition from occult deposition (mist droplets) was
# estimated from literature values and used to scale the timeseries values
occult_fraction = 0.27
data['P+occult Cl mg/l'] = data['P Cl mg/l'] / (1 - occult_fraction)
# The concentration of discharge older than the start of observations was
# assumed to be the long-term mean deposition rate divided by mean discharge
C_old = np.mean(data['P+occult Cl mg/l'] * data['P water flux mm/day']) / np.mean(data['Q water flux mm/day'])
# Parameters for the rSAS function (see Harman [2015])
ET_rSAS_fun_type = 'uniform'
S_ET = 398.
Q_rSAS_fun_type = 'gamma'
S_Q0 = 4922.
lambda_Q = -102.
Q_alpha = 0.6856
Q_dS_crit = 48.25
S_Q = np.maximum(1., lambda_Q * (data['S mm'] - Q_dS_crit))
ET_rSAS_fun_parameters = np.c_[np.zeros(N), np.ones(N) * S_ET ]
Q_rSAS_fun_parameters = np.c_[np.zeros(N), S_Q, np.ones(N) * Q_alpha]
# Initial condition
ST_init = np.zeros(N + 1)
# Run the model
results = rsas.solve_all_by_time_2out(
                                        data['P water flux mm/day'].values,
                                        data['Q water flux mm/day'].values,
                                        Q_rSAS_fun_parameters,
                                        Q_rSAS_fun_type,
                                        data['ET water flux mm/day'].values,
                                        ET_rSAS_fun_parameters,
                                        ET_rSAS_fun_type,
                                        ST_init,
                                        dt = 1.,
                                        n_iterations = 4,
                                        full_outputs=True,
                                        C_in=data['P+occult Cl mg/l'],
                                        C_old=C_old,
                                        evapoconcentration=True)
# Unroll the outputs
C_out, ST, PQ, PET, Qout, ETout, thetaQ, thetaET, thetaS, MassBalance = results
data['Q Predicted Cl mg/l'] = C_out
# Plot the result
fig = plt.figure(1)
plt.clf()
plt.plot(data.index, data['Q Cl mg/l'], label='Observed')
plt.plot(data.index, data['Q Predicted Cl mg/l'], label='Predicted')