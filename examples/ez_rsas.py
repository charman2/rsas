# -*- coding: utf-8 -*-
"""Storage selection (SAS) functions: easy-run script

Runs the rSAS model for a synthetic dataset with one flux in and
multiple fluxes out

Theory is presented in:
Harman, C. J. (2014), Time-variable transit time distributions and transport:
Theory and application to storage-dependent transport of chloride in a watershed,
Water Resour. Res., 51, doi:10.1002/2014WR015707.
"""
# =====================================
# load necessary libraries
# =====================================
from __future__ import division
import rsas
import numpy as np
import pandas as pd
import argparse

def run(inputfile, function_list, outputfile=None, alpha=None, C_old=0., n_substeps=1, save_arrays=False):
    # Initializes the random number generator so we always get the same result
    np.random.seed(0)
    # =====================================
    # Load the input data
    # =====================================
    data = pd.read_csv(inputfile, index_col=0, parse_dates=[1])
    inputfile.close()
    # length of the dataset
    N = len(data)
    # The individual timeseries can be pulled out of the dataframe
    J = data['J'].values.astype(np.float)
    C_J = data['C_J'].values.astype(np.float)
    # =========================
    # Create the rsas functions
    # =========================
    rSAS_fun = list()
    Q = list()
    for i, function_type in enumerate(function_list):
        if i>0:
            istr = '_' + str(i+1)
        else:
            istr = ''
        # Parameters for the rSAS function
        rSAS_fun_param_list = ['ST_min', 'ST_max']
        if function_type=='kumaraswami':
            rSAS_fun_param_list.append('a')
            rSAS_fun_param_list.append('b')
        elif function_type=='gamma':
            rSAS_fun_param_list.append('scale')
            rSAS_fun_param_list.append('shape')
        try:
            rSAS_fun_parameters = data[[param + istr for param in rSAS_fun_param_list]].values.astype(float)
        except:
            print("Something went wrong when I tried to read columns {} from {}".format(', '.join(rSAS_fun_param_list), inputfile))
            raise
        rSAS_fun.append(rsas.create_function(function_type, rSAS_fun_parameters))
        Q.append(data['Q'+istr].values.astype(np.float))
    if alpha is None:
        alpha = [1.0] * len(function_list)
    alpha = np.array([alpha]).T
    # =================
    # Initial condition
    # =================
    # Unknown initial age distribution, so just set this to zeros
    ST_init = np.zeros(N + 1)
    # =============
    # Run the model
    # =============
    # Run it
    outputs = rsas.solve(J, Q, rSAS_fun, ST_init=ST_init, alpha=alpha,
                        dt = 1., n_substeps=n_substeps, C_J=C_J, C_old=np.array([C_old]), verbose=True, debug=False)
    # Note that n_substeps may need to be increased to get an optimal result.
    for i in range(len(function_list)):
        data['C_Qm_'+str(i+1)] = outputs['C_Q'][:,i,0]
    if outputfile is not None:
        data.to_csv(outputfile)
    if save_arrays:
        for i in range(len(function_list)):
            np.savetxt(outputfile[:-4]+'_PQ_'+str(i+1)+'.csv', outputs['PQ'][:,:,i], delimiter=",")
            np.savetxt(outputfile[:-4]+'_ST'+'.csv', outputs['ST'], delimiter=",")
    return data, outputs

if __name__ == '__main__':
    # =====================================
    # Parse the arguments
    # =====================================
    parser = argparse.ArgumentParser(description='Easy-run script for the rsas solute transport code. Only handles one output flux, with alpha=1.')
    parser.add_argument('inputfile', type=argparse.FileType('r'), help="input file name. Must be a .csv file with columns 'timestep', 'datetime', 'J', 'Q1', 'C_J', 'ST_min', and 'ST_max'. Other parameter columns are required depending on the function_type.")
    parser.add_argument('function_list', nargs='*', type=str, choices=['uniform', 'kumaraswami', 'gamma'], help="type of rsas function, one for each outflow. For Kumaraswami, the inputfile must have columns 'a', and 'b'. For the 'gamma' type, there must be 'scale', and 'shape' columns")
    parser.add_argument('-o', '--outputfile', type=str, help="output file name")
    # The concentration of water older than the start of observations
    parser.add_argument('-C', '--C_old', type=float, default=0., help="concentration assumed for water older than timestep zero")
    # alpha parameter
    parser.add_argument('-a', '--alpha', nargs='*', type=float, help="alpha parameter controling the partitioning of the solute to each outflow. Must be the same number of alphas as rsas functions. Defaults to 1 for each outflow")
    # number of substeps
    parser.add_argument('-n', '--n_substeps', type=int, default=1, help="number of substeps required to ensure numerical accuracy. Try increasing this number to get a more accurate result")
    parser.add_argument('-s', '--save_arrays', action='store_true', default=False, help="if set to true, save the PQ and ST matricies")
    args = parser.parse_args()
    if args.outputfile is None:
        args.outputfile = args.inputfile.name
    run(**vars(args))
