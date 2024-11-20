#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:55:30 2021

@author: mwinter
"""

import os
import json
import numpy as np

# Check whether this is running on my laptop
current_dir = os.getcwd()
if current_dir[:14] == '/Users/mwinter':
    base_path = '/Users/mwinter/Documents/python_code/NN_measurements/'
else:
    base_path = '/home/phys/20214003/python_code/NN_measurements/'

# Networks for phase diagram
w_for_printing = []
L2_for_printing = []
rep_for_printing = []

L2s = [0]
L2_strings = ['0']
for L2 in np.arange(0.01, 0.053, 0.003):
    L2s.append(L2)
    L2_str = '{:.3f}'.format(L2).replace('.', '_')
    L2_strings.append(L2_str)

for r in range(10):
    for L2_count, L2 in enumerate(L2s):
        L2_str = L2_strings[L2_count]
        for width in range(30, 82, 3):

            depth = 6
            dataset = 'binary_PCA_MNIST'
            loss_function = 'quadratic_hinge'
            hidden_layers = depth*[width]

            ic_path = '{}models/binary_pca_mnist_w{}_ic_{}'.format(
                base_path, width, r)

            model_params = {}
            model_params['batch_size'] = 100
            model_params['dataset'] = dataset
            model_params['loss_function'] = loss_function
            model_params['learning_rate'] = 1e-3
            model_params['L2_penalty'] = L2
            model_params['N_inputs'] = 10
            model_params['N_outputs'] = 1
            model_params['h_layer_widths'] = hidden_layers
            model_params['initial_condition_path'] = ic_path

            model_name = 'binary_pca_mnist_ic{}_w{}_b100_reg{}'.format(
                r, width, L2_str)

            outpath = base_path + 'models/{}/{}_model_params.json'.format(model_name,
                                                                          model_name)

            output_dir = '{}models/{}/'.format(base_path, model_name)
            dir_exists = os.path.isdir(output_dir)
            if not dir_exists:
                os.mkdir(output_dir)

            with open(outpath, 'w') as outfile:
                json.dump(model_params, outfile)

            w_for_printing.append(width)
            L2_for_printing.append(L2_str)
            rep_for_printing.append(r)


w_string = '('
for w in w_for_printing:
    w_str = str(w)
    w_string += w_str
    w_string += ' '

w_string = w_string[:-1]
w_string += ')'

L2_string = '('
for L2 in L2_for_printing:
    L2_str = str(L2)
    L2_string += L2_str
    L2_string += ' '

L2_string = L2_string[:-1]
L2_string += ')'

rep_string = '('
for rep in rep_for_printing:
    rep_str = str(rep)
    rep_string += rep_str
    rep_string += ' '

rep_string = rep_string[:-1]
rep_string += ')'


# Relaxation runs
kT_for_printing = []
rep_for_printing = []

kTs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
kT_strs = ['-4', '5-4', '-3', '5-3', '-2', '5-2', '-1', '5-1', '1']
width = 30
depth = 6
L2 = 1e-2
L2_str = '-2'
ics = np.arange(10)

for ic in ics:
    for kT_count, kT in enumerate(kTs):
        kT_str = kT_strs[kT_count]
        ic_path = '{}models/binary_pca_mnist_ic{}_w30_b100_reg0_010'.format(
            base_path, ic)

        hidden_layers = depth*[width]

        model_params = {}
        model_params['batch_size'] = -1
        model_params['dataset'] = 'binary_PCA_MNIST'
        model_params['loss_function'] = 'quadratic_hinge'
        model_params['learning_rate'] = 1e-3
        model_params['L2_penalty'] = L2
        model_params['N_inputs'] = 10
        model_params['N_outputs'] = 1
        model_params['h_layer_widths'] = hidden_layers
        model_params['kT'] = kT
        model_params['initial_condition_path'] = ic_path

        model_name = 'binary_pca_mnist_b100_Langevin_relaxation_ic{}_w{}_kT{}_reg{}'.format(
            ic, width, kT_str, L2_str)

        outpath = base_path + 'models/{}/{}_model_params.json'.format(model_name,
                                                                      model_name)

        output_dir = '{}models/{}/'.format(base_path, model_name)
        dir_exists = os.path.isdir(output_dir)
        if not dir_exists:
            os.mkdir(output_dir)

        with open(outpath, 'w') as outfile:
            json.dump(model_params, outfile)

        kT_for_printing.append(kT_str)
        rep_for_printing.append(ic)

kT_string = '('
for kT in kT_for_printing:
    kT_str = str(kT)
    kT_string += kT_str
    kT_string += ' '

kT_string = kT_string[:-1]
kT_string += ')'

rep_string = '('
for rep in rep_for_printing:
    rep_str = str(rep)
    rep_string += rep_str
    rep_string += ' '

rep_string = rep_string[:-1]
rep_string += ')'
