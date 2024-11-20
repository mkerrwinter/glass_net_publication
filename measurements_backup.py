#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:43:58 2024

@author: mwinter
"""

import torch
from torch import nn
from torch.autograd.functional import hessian
# from torchvision import datasets
# from torchvision.transforms import ToTensor, Lambda, Compose
from torch.linalg import eigvals
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import numpy as np
import numpy.linalg
from NN import NeuralNetwork
from train_a_model import (train, load_dataset, make_quadratic_hinge_loss, 
                           read_in_std_data_from_teacher_dataset, 
                           load_final_model, load_models, load_model_params, evaluate_loss)
import json
import os
import sys
import pickle
from collections import defaultdict
import copy
from scipy.special import erf
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import CubicSpline
from hessian_eigenthings import compute_hessian_eigenthings

plt.close('all')
start = time.time()

debug = False
if debug:
    print('RUNNING IN DEBUG MODE')

# Set number of threads to 1 for cpu jobs on cluster
if not torch.cuda.is_available():
    current_dir = os.getcwd()
    if not current_dir[:14] == '/Users/mwinter':
        torch.set_num_threads(1)

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]
    Taken from https://medium.com/@BrendanArtley/matplotlib-color-gradients-21374910584b """
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors. Taken from https://medium.com/@BrendanArtley/matplotlib-color-gradients-21374910584b
    """
    if n==1:
        return [c1]
        
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def convert_arrays_for_json(my_dict):
    """
    Run through a dictionary and convert any lists of np.arrays to lists of 
    lists so they can be json serialised.
    """
    
    for key in my_dict.keys():
        val = my_dict[key]
        
        if isinstance(val, np.float64):
            new_val = float(val)
            my_dict[key] = new_val
            
        elif isinstance(val, np.ndarray):
            new_val = list(val)
            my_dict[key] = new_val
        
        else:
            try:
                val0 = val[0]
                                
                if isinstance(val0, np.ndarray):
                    new_val = []
                    for elem in val:
                        new_elem = list(elem)
                        new_val.append(new_elem)
                
                    my_dict[key] = new_val
                                
            except (TypeError, IndexError):
                pass

    return my_dict

def variable_length_mean(arrays, times, debug_path=None):
    if type(arrays)==pd.core.frame.DataFrame:
        data = arrays
    else:
        series_list = []
        for i in range(len(arrays)):
            array = arrays[i]
            time = times[i]
            temp_series = pd.Series(array, index=time)
            series_list.append(temp_series)
        
        data = pd.concat(series_list, axis=1)
    if debug_path is not None:
        data.to_csv(debug_path)
        
    # Remove timepoints where we have less than a critical number of curves
    N_crit = data.shape[1]-5
    mask = data.isnull().sum(axis=1)<N_crit
    clean_data = data.loc[mask, :]
    
    mean = clean_data.mean(axis=1)
    
    return np.array(mean), np.array(mean.index).astype(float)

def check_for_NaN_network(model):
    NaN_network = False
    
    for param in model.parameters():
        param_array = param.detach().numpy()
        bool_array = np.isnan(param_array)
        bool_sum = bool_array.sum()
        
        if bool_sum>0:
            NaN_network = True
            break
    
    return NaN_network

# Fit linear func to log of exponential data, including NaNs
def find_grad(y_array, x_array):
    nan_mask = ~np.isnan(y_array)
    inf_mask = ~np.isinf(y_array)
    zero_mask = ~(y_array==0)
    mask = nan_mask & inf_mask & zero_mask
    
    y_array = y_array[mask]
    x_array = x_array[mask]
    
    ln_y = np.log(y_array)
    ln_x = np.log(x_array)
    
    try:
        m, c = np.polyfit(ln_x, ln_y, 1)
    except (np.linalg.LinAlgError, TypeError):
        m = np.nan
        c = np.nan
    
    x_vals = np.unique(x_array)
    
    return m, c, x_vals

def calc_alpha2(models):
    w4s = []
    w2s = []
    
    model_0 = models[0]
    
    for model in models:
        w4 = 0.0
        w2 = 0.0
        N = 0
        
        for w0, w in zip(model_0.state_dict().values(), model.state_dict().values()):
            w4 += ((w-w0)**4).sum()
            w2 += ((w-w0)**2).sum()
            N += w.numel()
        
        w4s.append(w4/N)
        w2s.append(w2/N)
    
    w4s = np.array(w4s)
    w2s = np.array(w2s)
    
    alpha2 = (1.0/3.0)*((w4s-3.0*w2s**2)/w2s**2)
    
    return alpha2

def calc_weight_displacements(model_0, model_1):
    params_0 = []
    for param in model_0.parameters():
        temp = list(param.detach().flatten())
        params_0 += temp
    
    params_0 = np.array(params_0)
    
    params_1 = []
    for param in model_1.parameters():
        temp = list(param.detach().flatten())
        params_1 += temp
    
    params_1 = np.array(params_1)
    
    displacements = np.abs(params_0-params_1)
    
    return displacements

# Estimate discretisation by which to bin 
def calc_epsilon_for_van_Hove_calc(model, width_factor=100):
    params = []
    
    deltas = range(0, len(model.net)-1, 2)
    
    for delta in deltas:
        layer_tensor = model.net[delta].weight
        layer = layer_tensor.detach().numpy()
        params += list(layer.flatten())
    
    std = np.std(params)
    
    print('Weight std = ', std)
    
    epsilon = std/width_factor # Factor of 100 chosen somewhat arbitrarily.
    
    # Check this is resolvable 
    if epsilon<np.finfo(np.float32).eps:
        # The resolution of float32 at 1.0.
        raise ValueError('epsilon is below float32 precision.')
    
    return epsilon

# Calculate two point t0 self correlation function
def mean_self_van_Hove_correlation(model_0, epoch_0, model, epoch, epsilon):
    deltas = range(0, len(model.net)-1, 2)
    
    N = 0
    overlap = 0
    
    for delta in deltas:
        layer_1_tensor = model_0.net[delta].weight
        layer_2_tensor = model.net[delta].weight

        layer_1 = layer_1_tensor.detach().numpy()
        layer_2 = layer_2_tensor.detach().numpy()
        
        difference = np.abs(layer_1-layer_2)
        overlap_array = difference<=epsilon
        overlap_sum = overlap_array.sum()
        
        overlap += overlap_sum
        
        N += layer_1.size

    correlation = float(overlap)/N

    return correlation

def calc_analytic_van_Hove(epsilon, kT, mu, t):
    arg = epsilon/(np.sqrt(4.0*kT*mu*t))
    func = erf(arg)
    
    return func

def mean_GD_loss_phase_diag_stage1(device, base_path, rep):
    L2s = [0]
    L2_strings = ['0']
    for L2 in np.arange(0.01, 0.053, 0.003):
        L2s.append(L2)
        L2_str = '{:.3f}'.format(L2).replace('.', '_')
        L2_strings.append(L2_str)
        
    widths = np.arange(30, 82, 3)
    b = 100
    
    loss_array = np.zeros((len(L2s), len(widths)))
        
    for w_count, width in enumerate(widths):
        for L2_count, L2 in enumerate(L2s):
            L2_str = L2_strings[L2_count]
            
            model_name = 'binary_pca_mnist_ic{}_w{}_b{}_reg{}'.format(rep, 
                          width, b, L2_str)
            input_dir = '{}models/{}/'.format(base_path, model_name)
            params = load_model_params('{}/{}_model_params.json'.format(
                                       input_dir, model_name))
            model, epoch = load_final_model(input_dir, model_name, params, 
                                          device)
                        
            batch_size = -1
            dataset = params['dataset']
            
            (train_dataloader, test_dataloader, training_data, test_data, 
                N_inputs, N_outputs) = load_dataset(dataset, batch_size, 
                                                    base_path)
                
            # Define loss function 
            loss_function = params['loss_function']
            lr_original = params['learning_rate']
            if loss_function == 'CrossEntropy':
                loss_fn = nn.BCELoss()
                w_decay = params['L2_penalty']
                
            elif loss_function == 'MSELoss':
                loss_fn = nn.MSELoss()
                w_decay = params['L2_penalty']
                
            elif loss_function == 'Hinge':
                loss_fn = nn.HingeEmbeddingLoss()
                w_decay = params['L2_penalty']
                
            elif loss_function == 'quadratic_hinge':
                loss_fn = make_quadratic_hinge_loss()
                w_decay = params['L2_penalty']
                  
            else:
                print('PROVIDE A LOSS FUNCTION')
                        
            epochs = 50
            learning_rates = [lr_original, lr_original/10]
            
            GD_losses = []
            l = evaluate_loss(train_dataloader, model, loss_fn, device, w_decay)
            GD_losses.append(l)
            
            for learning_rate in learning_rates:
                optimizer = torch.optim.SGD(model.parameters(), 
                                            lr=learning_rate, 
                                            weight_decay=w_decay)
    
                for t in range(epochs): 
                    print('Training time step {}'.format(t))
                    train(train_dataloader, model, loss_fn, optimizer, device)
                    l = evaluate_loss(train_dataloader, model, loss_fn, device, w_decay)
                    GD_losses.append(l)
            
            # Plot gradient descent to check it works
            plot_output_dir = '{}plots/{}/'.format(base_path, model_name)
            dir_exists = os.path.isdir(plot_output_dir)
            if not dir_exists:
                os.mkdir(plot_output_dir)
                
            fig, ax = plt.subplots()
            plt.plot(GD_losses)
            plt.title('Gradient descent')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.yscale('log')
            plt.savefig('{}{}_GD_training_loss.pdf'.format(plot_output_dir, 
                        model_name), bbox_inches='tight')
            
            l = evaluate_loss(train_dataloader, model, loss_fn, device, w_decay)
            loss_array[L2_count, w_count] = l
                        
    # Save loss array for averaging in stage 2
    data_output_dir = '{}plots/L2_NP_phase_diagrams/'.format(base_path)
    dir_exists = os.path.isdir(data_output_dir)
    if not dir_exists:
        os.mkdir(data_output_dir)  

    data_outpath = data_output_dir + 'data_array_rep{}.txt'.format(rep)
    
    np.savetxt(data_outpath, loss_array)
    
    list_widths = []
    for width in widths:
        list_widths.append(float(width))
    
    list_L2s = []
    for L2 in L2s:
        list_L2s.append(float(L2))
    
    meta_data = {'b':b,
                 'widths':list_widths,
                 'L2s':list_L2s}
    
    json_outpath = data_output_dir+'averaging_meta_data.json'
        
    with open(json_outpath, 'w') as outfile:
        json.dump(meta_data, outfile)
        print('Json data saved')

def mean_GD_loss_phase_diag_stage2(device, base_path):
    data_input_dir = '{}plots/L2_NP_phase_diagrams/'.format(base_path)
    
    json_inpath = data_input_dir+'averaging_meta_data.json'
    with open(json_inpath) as infile:
        meta_data = json.load(infile)
    
    b = meta_data['b']
    f_widths = meta_data['widths']
    L2s = meta_data['L2s']
    widths = np.array([int(w) for w in f_widths])
    N_array = 10*widths + 5*widths**2 + widths
    N_over_P = N_array/60000
    
    files = os.listdir(data_input_dir)
    
    data_arrays = []
    for file in files:
        if file[:14] == 'data_array_rep':
            data_array = np.loadtxt(data_input_dir+file)
            data_arrays.append(data_array)
        
    # Deal with zeros
    delta = 1e-20
    log_data = np.log(data_arrays)
    mask = log_data<np.log(delta)
    log_data[mask] = np.log(delta)
    
    geom_loss_array = np.exp(np.nanmean(log_data, axis=0))
    
    # Plot
    plot_output_dir = '{}plots/L2_NP_phase_diagrams/'.format(base_path)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)  
    
    nodes = [0.0, 7.2/8, 7.8/8, 1.0]
    my_colors = ['black', 'blue', 'red', 'yellow']
    mycmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, my_colors)))
    
    stride = 4
    orig_x_labels = ['{:.2f}'.format(NP) for NP in N_over_P]
    x_labels = np.array(orig_x_labels[::stride])[[0, 2, 4]]
    orig_y_labels = ['{:.3f}'.format(L2) for L2 in L2s]
    
    # Manual y labels
    y0_label = '0.00'; pos0 = 0
    y1_label = '0.03'; pos1 = 8
    y2_label = '0.05'; pos2 = 14
    y_labels = [y0_label, y1_label, y2_label]
    
    fig, ax = plt.subplots()
    plt.imshow(geom_loss_array, 
               extent=[-0.5, len(orig_x_labels)-0.5, -0.5, len(orig_y_labels)-0.5],
               norm=colors.LogNorm(), cmap=mycmap, origin='lower')
    plt.title('Final geom mean loss, batch size = {}'.format(b))
    plt.colorbar(orientation='vertical')
    ax.set_xticks(np.arange(len(orig_x_labels))[::stride][[0, 2, 4]])
    ax.set_yticks(np.arange(len(orig_y_labels))[[pos0, pos1, pos2]])
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    plt.xlabel(r'N/P')
    plt.ylabel(r'$\lambda$')
    plt.savefig('{}/geom_mean_GD_phase_diag_SGD_b{}_log.pdf'.format(plot_output_dir, 
                b), bbox_inches='tight')

def plot_loss_vs_N_with_ics(device, base_path, default_model_name=None):    
    widths = np.arange(30, 82, 3)
    b = 100
    L2_str = '0'
    ics = np.arange(10)
    stub = ''
        
    recalculate_data = True
    datapath = base_path + 'measured_data/{}train_loss_after_GD_ics.pkl'.format(stub)
    data_exists = os.path.isfile(datapath)
    
    if data_exists and not recalculate_data:
        # Load data
        print('Loading loss data from {}'.format(datapath))
        with open(datapath, 'rb') as infile:
            losses_for_saving = pickle.load(infile)
    
    else:
        # Load dataset with batch=60k
        model_name = '{}binary_pca_mnist_ic0_w30_b{}_reg{}'.format(stub, b, L2_str)
        input_dir = '{}models/{}/'.format(base_path, model_name)
        params = load_model_params('{}{}_model_params.json'.format(
                                   input_dir, model_name))
        batch_size = -1
        dataset = params['dataset']
        loss_function = params['loss_function']
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path, 
                                                loss_fn=loss_function)
        
        train_set_size = len(training_data)
        
        losses_for_saving = defaultdict(list)
        
        for ic in ics:
            for w in widths:
                model_name = '{}binary_pca_mnist_ic{}_w{}_b{}_reg{}'.format(stub, ic, 
                                                                         w, b, 
                                                                         L2_str)
                    
                input_dir = '{}models/{}/'.format(base_path, model_name)
        
                params = load_model_params('{}{}_model_params.json'.format(
                                           input_dir, model_name))
            
                models, epochs = load_models(input_dir, model_name, params, 
                                             device)
                
                if len(epochs) == 0 or len(models) == 0:
                    print('No models for ', model_name)
                    continue
                
                # Extract the model at timestep 10^7 or the final timestep
                batch_size = params['batch_size']
                timesteps = np.array(epochs)*len(training_data)//batch_size
                if timesteps[-1]<=10**7:
                    model = models[-1]
                else:
                    idx = np.where(timesteps>10**7)[0][0]
                    model = models[idx]
                            
                # Do deterministic gradient descent
                N_total_params = sum(p.numel() for p in model.parameters() 
                                           if p.requires_grad)
                print('Total number of parameters = ', N_total_params)
                print('Size of training set = ', len(training_data)) 
                
                # Define loss function
                loss_function = params['loss_function']
                lr_original = params['learning_rate']
                    
                if loss_function == 'CrossEntropy':
                    loss_fn = nn.BCELoss()
                    w_decay = params['L2_penalty']
                
                elif loss_function == 'MSELoss':
                    loss_fn = nn.MSELoss()
                    w_decay = params['L2_penalty']
                
                elif loss_function == 'Hinge':
                    loss_fn = nn.HingeEmbeddingLoss()
                    w_decay = params['L2_penalty']
                        
                elif loss_function == 'quadratic_hinge':
                    loss_fn = make_quadratic_hinge_loss()
                    w_decay = params['L2_penalty']
                else:
                    print('PROVIDE A LOSS FUNCTION')
                
                # Define the optimizer. weight_decay>0 adds L2 regularisation to weights
                epochs = 50
                learning_rates = [lr_original, lr_original/10]
                
                GD_losses = []
                
                for learning_rate in learning_rates:
                    optimizer = torch.optim.SGD(model.parameters(), 
                                                lr=learning_rate, 
                                                weight_decay=w_decay)
        
                    for t in range(epochs): 
                        print('Training time step {}'.format(t))
                        train(train_dataloader, model, loss_fn, optimizer, device)
                        l = evaluate_loss(train_dataloader, model, loss_fn, device, w_decay)
                        GD_losses.append(l)
                 
                loss = evaluate_loss(train_dataloader, model, loss_fn, device, w_decay)
                losses_for_saving[w].append(loss)
                
                # Print GD loss curve
                GD_plot_output_dir = '{}plots/{}/'.format(base_path, model_name)
                fig, ax = plt.subplots()
                plt.plot(GD_losses)
                plt.yscale('log')
                plt.savefig('{}GD_train_loss_log.pdf'.format(GD_plot_output_dir), bbox_inches='tight')
                
    # Save calculated losses
    with open(datapath, 'wb') as outfile:
        pickle.dump(losses_for_saving, outfile)
                    
    # Plot
    plot_output_dir = '{}plots/training_loss_vs_N/'.format(base_path)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)  
        
    fig, ax = plt.subplots()
    keys = list(losses_for_saving.keys())
    keys.sort()
    means = []
    x_vals_for_means = []
        
    for count, width in enumerate(keys):
        N = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x_vals = [N/train_set_size]*len(losses_for_saving[width])
        
        x_vals_for_means.append(x_vals[0])
        mean = np.exp(np.nanmean(np.log(losses_for_saving[width])))
        means.append(mean)
        
        plt.scatter(x_vals, losses_for_saving[width], marker='o', s=15.0, color='k')
    
    plt.plot(x_vals_for_means, means)

    plt.title('Final loss vs network size')
    plt.xlabel(r'Number of parameters/Number of data')
    plt.ylabel('Training loss')
    plt.yscale('log')
    plt.savefig('{}{}training_loss_vs_N_with_ics_log.pdf'.format(plot_output_dir, stub), 
                bbox_inches='tight')
    
    fig, ax = plt.subplots()
    for count, width in enumerate(keys):
        N = sum(p.numel() for p in model.parameters() if p.requires_grad)
        x_vals = [N/train_set_size]*len(losses_for_saving[width])
        plt.scatter(x_vals, losses_for_saving[width], marker='o', s=15.0, color='k')
    plt.title('Final loss vs network size')
    plt.xlabel(r'Number of parameters/Number of data')
    plt.ylabel('Training loss')
    plt.yscale('log')
    plt.savefig('{}{}training_loss_vs_N_with_ics_nomean_log.pdf'.format(plot_output_dir, stub), 
                bbox_inches='tight')

def plot_loss_vs_L2_with_alphas(device, base_path):
    width = 30
    N_inputs = 10
    N_outputs = 1
        
    L2s = [1e-3, 5e-3, 1e-2, 1.2e-2, 1.4e-2, 1.6e-2, 1.8e-2, 
                        2e-2, 2.2e-2, 2.4e-2, 2.6e-2, 2.8e-2,
                        3e-2, 3.2e-2, 3.4e-2, 3.6e-2, 3.8e-2, 4e-2, 8e-2, 1.5e-1]
    
    L2_strs = ['-3', '5-3', '-2', '1_2-2', '1_4-2', '1_6-2', '1_8-2', 
                            '2-2', '2_2-2', '2_4-2', '2_6-2', '2_8-2',
                            '3-2', '3_2-2', '3_4-2', '3_6-2', '3_8-2', '4-2', 
                            '8-2', '1_5-1']
    
    # Load data if it exists
    data_output_dir = '{}measured_data/L2_transition/'.format(base_path)
    data_outpath = '{}losses_by_ic.pkl'.format(data_output_dir)
    
    dir_exists = os.path.isdir(data_output_dir)
    if not dir_exists:
        os.mkdir(data_output_dir)  
    
    data_exists = os.path.isfile(data_outpath)
    if data_exists:
        with open(data_outpath, 'wb') as infile:
            losses_by_ic = pickle.load(infile)
        
    else:
        losses_by_ic = []
            
        for rep in range(21, 41):
            # Load one model to get correct dataset
            model_name = 'binary_pca_mnist_ic{}_w30_b100_reg-2'.format(rep)
                
            input_dir = '{}models/{}/'.format(base_path, model_name)
            params = load_model_params('{}{}_model_params.json'.format(
                                       input_dir, model_name))
            batch_size = params['batch_size']
            dataset = params['dataset']
            
            (train_dataloader, test_dataloader, training_data, test_data, 
                N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path)
            
            losses = []
            
            for L2_count, L2 in enumerate(L2s):
                L2_str = L2_strs[L2_count]
    
                model_name = 'binary_pca_mnist_ic{}_w{}_b100_reg{}'.format(rep, 
                                                                width, L2_str)
                    
                input_dir = '{}models/{}/'.format(base_path, model_name)
                params = load_model_params('{}/{}_model_params.json'.format(
                                           input_dir, model_name))
        
                models, epochs = load_models(input_dir, model_name, params, 
                                             device)
                
                try:
                    model = models[-1]
                except IndexError:
                    losses.append(np.nan)
                    print('No models loaded for ', model_name)
                    continue
                
                # Define loss function
                loss_function = params['loss_function']
                lr_original = params['learning_rate']
                w_decay = params['L2_penalty']
                    
                if loss_function == 'CrossEntropy':
                    loss_fn = nn.BCELoss()
                
                elif loss_function == 'MSELoss':
                    loss_fn = nn.MSELoss()
                
                elif loss_function == 'Hinge':
                    loss_fn = nn.HingeEmbeddingLoss()
                        
                elif loss_function == 'quadratic_hinge':
                    loss_fn = make_quadratic_hinge_loss()
    
                else:
                    print('PROVIDE A LOSS FUNCTION')
                
                # Define the optimizer. weight_decay>0 adds L2 regularisation to weights
                epochs = 50
                learning_rates = [lr_original, lr_original/10]
                
                GD_losses = []
                
                for learning_rate in learning_rates:
                    optimizer = torch.optim.SGD(model.parameters(), 
                                                lr=learning_rate, 
                                                weight_decay=w_decay)
        
                    for t in range(epochs): 
                        print('Training time step {}'.format(t))
                        train(train_dataloader, model, loss_fn, optimizer, device)
                        l = evaluate_loss(train_dataloader, model, loss_fn, device, w_decay)
                        GD_losses.append(l)
                             
                # Print GD loss curve
                GD_plot_output_dir = '{}plots/{}/'.format(base_path, model_name)
                fig, ax = plt.subplots()
                plt.plot(GD_losses)
                plt.yscale('log')
                plt.savefig('{}GD_train_loss_log.pdf'.format(GD_plot_output_dir), bbox_inches='tight')
                
                # Save loss data
                GD_losses_array = np.array(GD_losses)
                np.savetxt('{}GD_train_loss_log.txt'.format(GD_plot_output_dir), GD_losses_array)
                
                loss = GD_losses[-1]
                if loss != loss:
                    print('NaN at w{}, L2{}'.format(width, L2_str))
                losses.append(loss)
            
            losses_by_ic.append(losses)
            
        # Save data
        with open(data_outpath, 'wb') as outfile:
            pickle.dump(losses_by_ic, outfile)
    
    # Average over i.c's
    average_losses = np.array(losses_by_ic[0])
    for loss in losses_by_ic[1:]:
        average_losses += np.array(losses)
        
    average_losses /= len(losses_by_ic)

    # Plot
    plot_output_dir = '{}plots/L2_transition/'.format(base_path)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)  
        
    fig, ax = plt.subplots()        
    for loss_by_ic in losses_by_ic:
        plt.plot(L2s, loss_by_ic, alpha=0.3, linewidth=1.0, c='k')
        
    plt.plot(L2s, average_losses, linewidth=3.0)
    
    plt.title(r'Average final loss vs L2')
    plt.xlabel(r'L2')
    plt.ylabel(r'$\langle L \rangle$')
    plt.xscale('log')
    plt.yticks([0.1, 0.3, 0.5])
    plt.savefig('{}/Average_loss_vs_L2_b100_log_w{}_alpha.pdf'.format(
                plot_output_dir, width), bbox_inches='tight')
    
    fig, ax = plt.subplots()        
    for loss_by_ic in losses_by_ic:
        plt.plot(L2s, loss_by_ic, alpha=0.3, linewidth=1.0, c='k')
        
    plt.plot(L2s, average_losses, linewidth=3.0)
    
    plt.title(r'Average final loss vs L2')
    plt.xlabel(r'L2')
    plt.ylabel(r'$\langle L \rangle$')
    plt.xticks([0, 0.02, 0.04])
    plt.yticks([0.1, 0.3, 0.5])
    plt.savefig('{}/Average_loss_vs_L2_b100_w{}_alpha.pdf'.format(
                plot_output_dir, width), bbox_inches='tight')

def plot_two_T_tau90_fit(device, base_path):    
    kT_strings = ['-4', '-6']
    kTs = [1e-4, 1e-6]
    L2_str = '-2'
    width = 30
    eps_factor = 400

    t_0 = 5000 # timestep at which measurements start

    # Load dataset with batch size = dataset size
    model_name = 'binary_pca_mnist_b100_Langevin_relaxation_ic0_w30_kT-1_reg-2'
    
    input_dir = '{}models/{}/'.format(base_path, model_name)
    params = load_model_params('{}{}_model_params.json'.format(
                               input_dir, model_name))    
    batch_size = -1
    dataset = params['dataset']
    
    (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = load_dataset(dataset, batch_size, 
                                                base_path)
                                                
    P = len(training_data)
    
    ic = 0
    correlation_curves = []
    analytic_curves= []
    times_list = []
    
    for kT_count, kT in enumerate(kTs):
        kT_str = kT_strings[kT_count]
        model_name = ('binary_pca_mnist_b100_Langevin_relaxation_ic{}_w{}_'.format(ic, width)+
                              'kT{}_reg{}'.format(kT_str, L2_str))
        
        print('Modal name = ', model_name)

        input_dir = '{}models/{}/'.format(base_path, model_name)

        params = load_model_params('{}{}_model_params.json'.format(input_dir, 
                                                                   model_name))
    
        models, epochs = load_models(input_dir, model_name, params, device)
        
        if len(epochs)>0:
            print('max(epochs) = ', max(epochs))
            print('len(epochs) = ', len(epochs))
        
        # Extract the model at timestep t_0
        timesteps = np.array(epochs)
        if timesteps[-1]<=t_0:
            print('WARNING: ')
            print('kT {} DOES NOT REACH {} TIMESTEPS'.format(kT, t_0))
            continue
        else:
            idx = np.where(timesteps>=t_0)[0][0]
            model = models[idx]
        
        epsilon = calc_epsilon_for_van_Hove_calc(model, eps_factor)
        epsilon_str = str(eps_factor)    
        
        epoch_seg = epochs[idx:]
        model_seg = models[idx:]
        
        epoch_0 = epoch_seg[0]
        model_0 = model_seg[0]
                
        self_correlations = []
        epochs_for_plotting = []
        
        for i in range(len(epoch_seg)):
            epoch = epoch_seg[i]
            model = model_seg[i]
            
            epochs_for_plotting.append(epoch)
            
            self_correlation = mean_self_van_Hove_correlation(model_0, 
                                                              epoch_0,
                                                              model, 
                                                              epoch,
                                                              epsilon)
            
            self_correlations.append(self_correlation)

        self_correlations = np.array(self_correlations)
        
        # Rescale time to be in time steps not epochs
        timestep_array = np.array(epochs_for_plotting)
        
        times_for_fitting = timestep_array-timestep_array[0]
        VH_curve = calc_analytic_van_Hove(epsilon, kT/P, 
                                          params['learning_rate'], 
                                          times_for_fitting)
        
        correlation_curves.append(self_correlations)
        analytic_curves.append(VH_curve)
        times_list.append(times_for_fitting)
        
    # Plot decorrelation curve
    plot_output_dir = '{}plots/analytic_comparison/'.format(base_path)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)

    my_colors = ['red', 'deepskyblue']
    my_labels = ['Noise dom.', 'Landscape dom.']

    print('Data:')
    print('times_list = ', times_list)
    print('correlation_curves = ', correlation_curves)
    print('analytic_curves = ', analytic_curves)

    fig, ax = plt.subplots()
    for i in range(len(times_list)):
        times = times_list[i]
        corr_curve = correlation_curves[i]
        analytic = analytic_curves[i]
        
        plt.plot(times, corr_curve, color=my_colors[i], label=my_labels[i])
        plt.plot(times, analytic, linestyle='dashed', color='k')
        
    plt.legend(loc='lower left')
    # plt.title('Overlap and analytic fit')
    # plt.ylabel(r'Overlap')
    # plt.xlabel('Timesteps')
    plt.xscale('log')
    plt.xticks([1e3, 1e5])
    plt.yticks([0, 0.5, 1.0])
    plt.axhline(y=0, color='k', alpha=0.5)
    plt.axhline(y=0.9, color='k', linestyle='dashed', linewidth=1.0)
    plt.savefig('{}Overlap_comparison_{}_lnt_epsilon{}.pdf'.format(
    plot_output_dir, t_0, epsilon_str), bbox_inches='tight')

def plot_overlap_with_repeats(device, base_path):    
    dataset_name = 'binary_pca_mnist'
    
    if dataset_name == 'binary_pca_mnist':
        orig_kT_strings = ['-4', '5-4', '-3', '5-3', '-2', '5-2', '-1', '5-1', '1']
        orig_kTs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
        L2s = [0.0, 1e-2, 0.0]
        L2_strings = ['0', '-2', '0']
        widths = [30, 30, 72]
        eps_factors = [500, 300, 6000]
        
    elif dataset_name == 'cifar10':
        orig_kT_strings = ['e-10', 'e-8', 'e-6', 'e-5', 'e-4', 'e-3', 'e-2', 'e-1', 'e0']
        orig_kTs = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        L2s = [1e-3, 0.0, 0.0]
        L2_strings = ['e-3', '0', '0']
        widths = [30, 80, 30]
        eps_factors = [150, 600, 150]
    
    else:
        print('Select a dataset.')
    
    
    for count in range(len(L2s)):
        width = widths[count]
        L2 = L2s[count]
        L2_str = L2_strings[count]
        eps_factor = eps_factors[count]
        epsilon_str = str(eps_factor)
        
        kT_strings = orig_kT_strings
        kTs = orig_kTs
   
        t_0 = 20000 # timestep at which measurements start
    
        if dataset_name=='binary_pca_mnist':
            if width==30 and L2==0:
                new_kT_strings = ['-6', '-5', '0']
                new_kTs = [1e-6, 1e-5, 0]
                
                kT_strings = kT_strings + new_kT_strings
                kTs = kTs + new_kTs
            
            elif width==30 and L2==1e-2:
                new_kT_strings = ['0', '-10', '-8', '-6', '-5']
                new_kTs = [0, 1e-10, 1e-8, 1e-6, 1e-5]
                
                kT_strings = kT_strings + new_kT_strings
                kTs = kTs + new_kTs
                
            else:
                new_kT_strings = ['0', '-10', '-8', '-6', '-5']
                new_kTs = [0, 1e-10, 1e-8, 1e-6, 1e-5]
                
                kT_strings = kT_strings + new_kT_strings
                kTs = kTs + new_kTs
        
        if dataset_name=='cifar10':
            new_kT_strings = ['0']
            new_kTs = [0]
            
            kT_strings = new_kT_strings + kT_strings 
            kTs = new_kTs + kTs
        
        ordered_kTs = [x for x,_ in sorted(zip(kTs,kT_strings))]
        kT_strings = [x for _,x in sorted(zip(kTs,kT_strings))]
        kTs = ordered_kTs
        
        # Load data if it exists
        plot_output_dir = '{}plots/mean_overlap_curves/'.format(base_path)
        dir_exists = os.path.isdir(plot_output_dir)
        if not dir_exists:
            os.mkdir(plot_output_dir)
        
        mean_curve_outpath = '{}mean_curves_df_w{}_L2{}_epsfac{}.csv'.format(plot_output_dir, width, L2_str, eps_factor)
        
        mean_data_exists = os.path.isfile(mean_curve_outpath)
        
        if mean_data_exists:
            mean_curves_df = pd.read_csv(mean_curve_outpath, index_col=0)

            mean_curves_by_kT = []
            times_by_kT = []
            
            for i in range(mean_curves_df.shape[0]):
                temp_series = mean_curves_df.loc[i, :]
                temp_series.dropna(inplace=True)
                times_by_kT.append(np.array(temp_series.index.astype(float)))
                mean_curves_by_kT.append(np.array(temp_series))
            
            print('Loaded mean_curves_df from text file.')
        
        else:
            print('Calculating data')
            # Load dataset with batch size = dataset size
            if dataset_name=='binary_pca_mnist':
                model_name = 'binary_pca_mnist_b100_Langevin_relaxation_ic0_w30_kT-1_reg-2'
        
            elif dataset_name == 'cifar10':
                model_name = 'cifar10_b100_Langevin_relaxation_ic1_w30_kTe-1_reg0'
                
                
            input_dir = '{}models/{}/'.format(base_path, model_name)
            params = load_model_params('{}{}_model_params.json'.format(
                                       input_dir, model_name))    
            batch_size = -1
            dataset = params['dataset']
            
            (train_dataloader, test_dataloader, training_data, test_data, 
                    N_inputs, N_outputs) = load_dataset(dataset, batch_size, 
                                                        base_path)
                                                        
            P = len(training_data)
            
            if dataset_name == 'cifar10':
                if width==30:
                    ics = np.arange(1, 10) # These are the runs that did not diverge
                else:
                    ics = np.array([2, 3, 5])
            else:
                if width==72:
                    ics = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 
                                    15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 
                                    28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 
                                    42, 43, 44, 45, 46, 47, 48, 49])
                elif width==30 and L2==0:
                    ics = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 19, 
                                    25, 26, 27, 29, 30, 31, 36, 37, 44, 45, 46, 48])
                else:
                    ics = np.arange(50)
            
            # Calculate data  
            mean_curves_by_kT = []
            times_by_kT = []
            
            for kT_count, kT in enumerate(kTs):
                kT_str = kT_strings[kT_count]
                curves_by_ic = []
                times_by_ic = []
                
                for ic in ics:
                    # Correct change in naming convention
                    L2_str = L2_strings[count]
                    if L2_str=='-2' and ic>9:
                        L2_str = '0_010'
                    
                    kT_str = kT_strings[kT_count]
                    if dataset_name=='binary_pca_mnist':
                        model_name = ('binary_pca_mnist_b100_Langevin_relaxation_ic{}_w{}_'.format(ic, width)+
                                          'kT{}_reg{}'.format(kT_str, L2_str)) 
                    elif dataset_name=='cifar10':
                        model_name = 'cifar10_b100_Langevin_relaxation_ic{}_w{}_kT{}_reg{}'.format(ic, width, kT_str, L2_str)
                                        
                    input_dir = '{}models/{}/'.format(base_path, model_name)
            
                    params = load_model_params('{}{}_model_params.json'.format(
                                               input_dir, model_name))
                
                    models, epochs = load_models(input_dir, model_name, params, 
                                                 device)

                    # Extract the model at timestep t_0
                    timesteps = np.array(epochs)
                    if timesteps[-1]<=t_0:
                        curves_by_ic.append([])
                        print('WARNING: ')
                        print('kT {} DOES NOT REACH {} TIMESTEPS'.format(kT, t_0))
                        continue
                    else:
                        idx = np.where(timesteps>=t_0)[0][0]
                        model = models[idx]
                    
                    epsilon = calc_epsilon_for_van_Hove_calc(model, eps_factor)
                    
                    epoch_seg = epochs[idx:]
                    model_seg = models[idx:]
                    
                    epoch_0 = epoch_seg[0]
                    model_0 = model_seg[0]
                                        
                    self_correlations = []
                    epochs_for_plotting = []
                    
                    for i in range(len(epoch_seg)):
                        epoch = epoch_seg[i]
                        model = model_seg[i]
                        
                        epochs_for_plotting.append(epoch)
                        
                        self_correlation = mean_self_van_Hove_correlation(model_0, 
                                                                          epoch_0,
                                                                          model, 
                                                                          epoch,
                                                                          epsilon)
                        
                        self_correlations.append(self_correlation)
                
                    self_correlations = np.array(self_correlations)
                    self_correlations /= self_correlations[0]
                    
                    # Rescale time to be in time steps not epochs
                    timestep_array = np.array(epochs_for_plotting)
                    
                    times_for_fitting = timestep_array-timestep_array[0]
                    VH_curve = calc_analytic_van_Hove(epsilon, kT/P, 
                                                      params['learning_rate'], 
                                                      times_for_fitting)
                    
                    # Plot decorrelation curve
                    plot_output_dir2 = '{}plots/{}/'.format(base_path, model_name)
                    dir_exists = os.path.isdir(plot_output_dir)
                    if not dir_exists:
                        os.mkdir(plot_output_dir)
                    
                    fig, ax = plt.subplots()
                    plt.plot(times_for_fitting, self_correlations, label='Measured')
                    plt.plot(times_for_fitting, VH_curve, linestyle='dashed', label='Analytic')
                    plt.legend()
                    plt.title('self van Hove correlation')
                    plt.ylabel(r'van Hove correlation')
                    plt.xlabel('Timesteps')
                    plt.xscale('log')
                    plt.savefig('{}{}_van_Hove_self_correlation_{}_lnt_epsilon{}.pdf'.format(
                    plot_output_dir2, model_name, t_0, epsilon_str), bbox_inches='tight')
                    
                    curves_by_ic.append(self_correlations)
                    times_by_ic.append(times_for_fitting)
                
                # Average over ics
                list_of_series = []
                for i in range(len(curves_by_ic)):
                    curve = curves_by_ic[i]
                    time = times_by_ic[i]
                    series = pd.Series(curve, index=time)
                    list_of_series.append(series)
                
                corr_df = pd.DataFrame(list_of_series).transpose()
                correlation_df_no_nans = corr_df[~corr_df.isnull().any(axis=1)]
                mean_corr_curve = correlation_df_no_nans.mean(axis=1)
                
                time_array = np.array(mean_corr_curve.index)
                mean_corr_curve = np.array(mean_corr_curve)
                
                mean_curves_by_kT.append(mean_corr_curve)
                times_by_kT.append(time_array)
                
            # Save curves as a text file
            list_of_series = []
            for i in range(len(mean_curves_by_kT)):
                curve_series = pd.Series(mean_curves_by_kT[i], index=times_by_kT[i])
                list_of_series.append(curve_series)
            
            mean_curves_df = pd.DataFrame(list_of_series)
            mean_curves_df.to_csv(mean_curve_outpath)


        # Plot mean curves
        colour1 = '#0096FF' #blue
        colour2 = '#FF0000' #red
        colours = get_color_gradient(colour1, colour2, len(mean_curves_by_kT))

        fig, ax = plt.subplots()
        for i in range(len(mean_curves_by_kT)):
            times = times_by_kT[i]
            overlap = mean_curves_by_kT[i]
            kT_str=kT_strings[i]
            my_c=colours[i]
            if times[0]==0:
                plt.plot(times[1:], overlap[1:], label='kT={}'.format(kT_str), c=my_c)
            else:
                plt.plot(times, overlap, label='kT={}'.format(kT_str), c=my_c)

        plt.legend()
        plt.title('Average overlap functions')
        plt.ylabel(r'$Q(t)$')
        plt.xlabel('t')
        plt.xscale('log')
        plt.savefig('{}Overlap_w{}_L2{}_t0_{}_log_log_epsilon{}_legend.pdf'.format(
        plot_output_dir, width, L2_str, t_0, epsilon_str), bbox_inches='tight')

        fig, ax = plt.subplots()
        for i in range(len(mean_curves_by_kT)):
            times = times_by_kT[i]
            overlap = mean_curves_by_kT[i]
            kT_str=kT_strings[i]
            my_c=colours[i]
            if times[0]==0:
                plt.plot(times[1:], overlap[1:], c=my_c)
            else:
                plt.plot(times, overlap, c=my_c)
        
        plt.axhline(0.9, label='T=0', linestyle='dotted')
        plt.yticks([0, 0.5, 1.0])

        plt.xscale('log')
        plt.savefig('{}Overlap_w{}_L2{}_t0_{}_log_log_epsilon{}.pdf'.format(
        plot_output_dir, width, L2_str, t_0, epsilon_str), bbox_inches='tight')

def plot_tau90_vs_kT_with_repeats(device, base_path):    
    dataset_name = 'binary_pca_mnist'
    # dataset_name = 'cifar10'
    
    if dataset_name == 'binary_pca_mnist':
        orig_kT_strings = ['-4', '5-4', '-3', '5-3', '-2', '5-2', '-1', '5-1', '1']
        orig_kTs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
        L2s = [0.0, 1e-2, 0.0]
        L2_strings = ['0', '-2', '0']
        widths = [30, 30, 72]
        eps_factors = [500, 300, 6000]
        name_prefix = ''
        dataset_pathstub = ''
        
    elif dataset_name == 'cifar10':
        orig_kT_strings = ['e-10', 'e-8', 'e-6', 'e-5', 'e-4', 'e-3', 'e-2', 'e-1', 'e0']
        orig_kTs = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        L2s = [1e-3, 0.0, 0.0]
        L2_strings = ['e-3', '0', '0']
        widths = [30, 80, 30]
        name_prefix = 'cifar10'
        eps_factors = [150, 600, 150]
    
    else:
        print('Select a dataset.')
    
    for count in range(len(L2s)):
        width = widths[count]
        L2 = L2s[count]
        L2_str = L2_strings[count]
        eps_factor = eps_factors[count]
        epsilon_str = str(eps_factor)  
        
        kT_strings = orig_kT_strings
        kTs = orig_kTs
   
        t_0 = 20000 # timestep at which measurements start
    
        if dataset_name=='binary_pca_mnist':
            if width==30 and L2==0:
                new_kT_strings = ['-6', '-5', '0']
                new_kTs = [1e-6, 1e-5, 0]
                
                kT_strings = kT_strings + new_kT_strings
                kTs = kTs + new_kTs
            
            elif width==30 and L2==1e-2:
                new_kT_strings = ['0', '-10', '-8', '-6', '-5']
                new_kTs = [0, 1e-10, 1e-8, 1e-6, 1e-5]
                
                kT_strings = kT_strings + new_kT_strings
                kTs = kTs + new_kTs
                
            else:
                new_kT_strings = ['0', '-10', '-8', '-6', '-5']
                new_kTs = [0, 1e-10, 1e-8, 1e-6, 1e-5]
                
                kT_strings = kT_strings + new_kT_strings
                kTs = kTs + new_kTs
        
        if dataset_name=='cifar10':
            new_kT_strings = ['0']
            new_kTs = [0]
            
            kT_strings = new_kT_strings + kT_strings 
            kTs = new_kTs + kTs
        
        ordered_kTs = [x for x,_ in sorted(zip(kTs,kT_strings))]
        kT_strings = [x for _,x in sorted(zip(kTs,kT_strings))]
        kTs = ordered_kTs

        # Load dataset with batch size = dataset size
        if dataset_name=='binary_pca_mnist':
            model_name = 'binary_pca_mnist_b100_Langevin_relaxation_ic0_w30_kT-1_reg-2'
            
        elif dataset_name == 'mini_binary_PCA_MNIST':
            model_name = 'tiny_net_b100_Langevin_relaxation_ic0_w10_kT-4_reg0'
            
        elif dataset_name == 'cifar10':
            model_name = 'cifar10_b100_Langevin_relaxation_ic1_w30_kTe-1_reg0'
            
            
        input_dir = '{}models/{}/'.format(base_path, model_name)
        params = load_model_params('{}{}_model_params.json'.format(
                                   input_dir, model_name))    
        batch_size = -1
        dataset = params['dataset']
        
        (train_dataloader, test_dataloader, training_data, test_data, 
                N_inputs, N_outputs) = load_dataset(dataset, batch_size, 
                                                    base_path)
                                                    
        P = len(training_data)

        # See if data already exists
        data_output_dir = '{}measured_data/tau_vs_kT/'.format(base_path)
        dir_exists = os.path.isdir(data_output_dir)
        if not dir_exists:
            os.mkdir(data_output_dir)  
        
        tau90_outpath = '{}{}tau90s_by_ic_w{}_reg{}_t0_{}_eps{}.pkl'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0, eps_factor)
        taue_outpath = '{}{}taues_by_ic_w{}_reg{}_t0_{}_eps{}.pkl'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0, eps_factor)
        eps_outpath = '{}{}epsilon_w{}_reg{}_t0_{}_eps{}.pkl'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0, eps_factor)
        tau90_data_exists = os.path.isfile(tau90_outpath)
        taue_data_exists = os.path.isfile(taue_outpath)
        eps_data_exists = os.path.isfile(eps_outpath)
        data_exists = tau90_data_exists and taue_data_exists and eps_data_exists
            
        if data_exists:
            with open(tau90_outpath, 'rb') as infile:
                tau_90s_by_ic = pickle.load(infile)
            
            with open(taue_outpath, 'rb') as infile:
                tau_es_by_ic = pickle.load(infile)
            
            with open(eps_outpath, 'rb') as infile:
                epsilon = pickle.load(infile)
                
            print('Loaded taus from text file.')
    
        else:
            # Calculate data  
            tau_90s_by_ic = []
            tau_es_by_ic = []
            
            if dataset_name == 'cifar10':
                if width==30:
                    ics = np.arange(1, 10) # The runs that didn't diverge
                else:
                    ics = np.array([2, 3, 5])
            else:
                if width==72:
                    ics = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
                           15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 
                           29, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 
                           44, 45, 46, 47, 48, 49]
                elif L2==0:
                    ics = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 19, 25, 26, 
                           27, 29, 30, 31, 36, 37, 44, 45, 46, 48]
                else:
                    ics = np.arange(50)
            
            for ic in ics:
                # Correct change in naming convention
                if L2_str=='-2' and ic>9:
                    L2_str = '0_010'
                
                tau_90s = []
                tau_es = []
                for kT_count, kT in enumerate(kTs):
                    kT_str = kT_strings[kT_count]
                    if dataset_name=='binary_pca_mnist':
                        model_name = ('binary_pca_mnist_b100_Langevin_relaxation_ic{}_w{}_'.format(ic, width)+
                                          'kT{}_reg{}'.format(kT_str, L2_str))
                        
                    elif dataset_name=='mini_binary_PCA_MNIST':
                        model_name = ('tiny_net_b100_Langevin_relaxation_ic{}_w{}_'.format(ic, width)+
                                         'kT{}_reg{}'.format(kT_str, L2_str)) 
                    elif dataset_name=='cifar10':
                        model_name = 'cifar10_b100_Langevin_relaxation_ic{}_w{}_kT{}_reg{}'.format(ic, width, kT_str, L2_str)
            
                    input_dir = '{}models/{}/'.format(base_path, model_name)
            
                    params = load_model_params('{}{}_model_params.json'.format(
                                               input_dir, model_name))
                
                    models, epochs = load_models(input_dir, model_name, params, 
                                                 device)

                    # Extract the model at timestep t_0
                    timesteps = np.array(epochs)
                    if timesteps[-1]<=t_0:
                        tau_90s.append(np.nan)
                        tau_es.append(np.nan)
                        print('WARNING: ')
                        print('kT {} DOES NOT REACH {} TIMESTEPS'.format(kT, t_0))
                        continue
                    else:
                        idx = np.where(timesteps>=t_0)[0][0]
                        model = models[idx]
                    
                    epsilon = calc_epsilon_for_van_Hove_calc(model, eps_factor)
                    
                    epoch_seg = epochs[idx:]
                    model_seg = models[idx:]
                    
                    epoch_0 = epoch_seg[0]
                    model_0 = model_seg[0]
                                        
                    self_correlations = []
                    epochs_for_plotting = []
                    
                    for i in range(len(epoch_seg)):
                        epoch = epoch_seg[i]
                        model = model_seg[i]
                        
                        epochs_for_plotting.append(epoch)
                        
                        self_correlation = mean_self_van_Hove_correlation(model_0, 
                                                                          epoch_0,
                                                                          model, 
                                                                          epoch,
                                                                          epsilon)
                        
                        self_correlations.append(self_correlation)
                
                    tau_relax = np.nan
                    tau_relax_90 = np.nan
            
                    self_correlations = np.array(self_correlations)
                    
                    # Rescale time to be in time steps not epochs
                    timestep_array = np.array(epochs_for_plotting)
                    
                    # Calculate e-fold relaxation times
                    ic_value = self_correlations[0]
                    try:
                        idx = np.where(self_correlations<ic_value/np.exp(1))[0][0]
                        if idx<=1:
                            tau_relax = np.nan
                        else:
                            tau_relax = timestep_array[idx]-timestep_array[0]
                    except IndexError:
                        tau_relax = np.nan
                        
                    try:
                        idx = np.where(self_correlations<ic_value*0.9)[0][0]
                        if idx<=1:
                            tau_relax_90 = np.nan
                        else:
                            tau_relax_90 = timestep_array[idx]-timestep_array[0]
                    except IndexError:
                        tau_relax_90 = np.nan
                    
                    tau_90s.append(tau_relax_90)
                    tau_es.append(tau_relax)
                    
                    times_for_fitting = timestep_array-timestep_array[0]
                    VH_curve = calc_analytic_van_Hove(epsilon, kT/P, 
                                                      params['learning_rate'], 
                                                      times_for_fitting)
                    
                    # Save decorrelation times
                    data_output_dir = '{}plots/{}/'.format(base_path, model_name)
                    dir_exists = os.path.isdir(data_output_dir)
                    if not dir_exists:
                        os.mkdir(data_output_dir)
                    
                    data_outpath = '{}tau90_eps{}.txt'.format(data_output_dir, epsilon_str)
                    np.savetxt(data_outpath, np.array([tau_relax_90]))
                    
                    data_outpath = '{}taue_eps{}.txt'.format(data_output_dir, epsilon_str)
                    np.savetxt(data_outpath, np.array([tau_relax]))
                    
                    # Plot decorrelation curve
                    plot_output_dir = '{}plots/{}/'.format(base_path, model_name)
                    dir_exists = os.path.isdir(plot_output_dir)
                    if not dir_exists:
                        os.mkdir(plot_output_dir)
                    
                    fig, ax = plt.subplots()
                    plt.plot(timestep_array-timestep_array[0], self_correlations, label='Measured')
                    plt.plot(times_for_fitting, VH_curve, linestyle='dashed', label='Analytic')
                    plt.legend()
                    plt.title('self van Hove correlation')
                    plt.ylabel(r'van Hove correlation')
                    plt.xlabel('Timesteps')
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.savefig('{}{}_van_Hove_self_correlation_{}_log_log_epsilon{}.pdf'.format(
                    plot_output_dir, model_name, t_0, epsilon_str), bbox_inches='tight')
                    
                tau_90s_by_ic.append(tau_90s)
                tau_es_by_ic.append(tau_es)
                
            # Save data
            dir_exists = os.path.isdir(data_output_dir)
            if not dir_exists:
                os.mkdir(data_output_dir) 
            
            with open(tau90_outpath, 'wb') as outfile:
                pickle.dump(tau_90s_by_ic, outfile)
            
            with open(taue_outpath, 'wb') as outfile:
                pickle.dump(tau_es_by_ic, outfile)
            
            with open(eps_outpath, 'wb') as outfile:
                pickle.dump(epsilon, outfile)

        # Plot averages
        plot_output_dir = '{}plots/decorrelation_vs_kT/'.format(base_path)
        dir_exists = os.path.isdir(plot_output_dir)
        if not dir_exists:
            os.mkdir(plot_output_dir)  
        
        # Remove any tau90s and taues shorter than 4 as resolution is bad
        clean_tau_90s_by_ic = []
        clean_tau_es_by_ic = []
        clean_90s_kTs_by_ic = []
        clean_es_kTs_by_ic = []

        for tau_90s in tau_90s_by_ic:
            idx = np.where(np.array(tau_90s)>=4)[0]
            clean_tau_90s = np.array(tau_90s)[idx]
            clean_kTs = np.array(kTs)[idx]
            
            clean_tau_90s_by_ic.append(clean_tau_90s)
            clean_90s_kTs_by_ic.append(clean_kTs)
        
        for tau_es in tau_es_by_ic:
            idx = np.where(np.array(tau_es)>=4)[0]
            clean_tau_es = np.array(tau_es)[idx]
            clean_e_kTs = np.array(kTs)[idx]
            
            clean_tau_es_by_ic.append(clean_tau_es)
            clean_es_kTs_by_ic.append(clean_e_kTs)
        
        tau_90s_by_ic = clean_tau_90s_by_ic
        tau_es_by_ic = clean_tau_es_by_ic
        
        # Get points for line fitting
        if width == 30:
            short_tau_90s = []
            short_kTs = []
            for count, tau_90s in enumerate(tau_90s_by_ic):
                temp_kTs = clean_90s_kTs_by_ic[count]
                
                if dataset_name=='cifar10':
                    if width==30:
                        short_tau_90s.append(tau_90s[-3:])
                        short_kTs.append(temp_kTs[-3:])
                    else:
                        short_tau_90s.append(tau_90s[-6:])
                        short_kTs.append(temp_kTs[-6:])
                else:
                    if width==72:
                        idx = np.where(np.array(temp_kTs)>=1e-8)[0]
                        short_tau_90s.append(np.array(tau_90s)[idx])
                        short_kTs.append(np.array(temp_kTs)[idx])
                    elif L2==0:
                        idx = np.where(np.array(temp_kTs)>1e-3)[0]
                        short_tau_90s.append(np.array(tau_90s)[idx])
                        short_kTs.append(np.array(temp_kTs)[idx])
                    else:
                        idx = np.where(np.array(temp_kTs)>1e-5)[0]
                        short_tau_90s.append(np.array(tau_90s)[idx])
                        short_kTs.append(np.array(temp_kTs)[idx])
            
            short_tau_es = []
            short_e_kTs = []
            for count, tau_es in enumerate(tau_es_by_ic):
                temp_kTs = clean_es_kTs_by_ic[count]
                
                if dataset_name=='cifar10':
                    short_tau_es.append(tau_es[-2:])
                    short_e_kTs.append(temp_kTs[-2:])
                else:
                    if L2==0 and width==30:
                        idx = np.where(np.array(temp_kTs)>=1e-4)[0]                        
                        short_tau_es.append(np.array(tau_es)[idx])
                        short_e_kTs.append(np.array(temp_kTs)[idx])
                    elif L2==0 and width==72:
                        idx = np.where(np.array(temp_kTs)>=1e-6)[0]                        
                        short_tau_es.append(np.array(tau_es)[idx])
                        short_e_kTs.append(np.array(temp_kTs)[idx])
                    else:
                        idx = np.where(np.array(temp_kTs)>=1e-6)[0]                        
                        short_tau_es.append(np.array(tau_es)[idx])
                        short_e_kTs.append(np.array(temp_kTs)[idx])
            
            tau_90s_array = np.concatenate(short_tau_90s, axis=0)
            tau_es_array = np.concatenate(short_tau_es, axis=0)
            kTs_90s_array = np.concatenate(short_kTs, axis=0)
            kTs_es_array = np.concatenate(short_e_kTs, axis=0)
            
            m_90, c_90, line_kTs = find_grad(tau_90s_array, kTs_90s_array)
            m_e, c_e, line_e_kTs = find_grad(tau_es_array, kTs_es_array)
            
            tau_90s_line = np.exp(c_90)*line_kTs**m_90        
            tau_es_line = np.exp(c_e)*line_e_kTs**m_e
        else:
            tau_90s_array = np.concatenate(tau_90s_by_ic, axis=0)
            tau_es_array = np.concatenate(tau_es_by_ic, axis=0)
            kTs_90s_array = np.concatenate(clean_90s_kTs_by_ic, axis=0)
            kTs_es_array = np.concatenate(clean_es_kTs_by_ic, axis=0)
        
            m_90, c_90, line_kTs = find_grad(tau_90s_array, kTs_90s_array)
            m_e, c_e, line_e_kTs = find_grad(tau_es_array, kTs_es_array)
        
            tau_90s_line = np.exp(c_90)*line_kTs**m_90        
            tau_es_line = np.exp(c_e)*line_e_kTs**m_e
        
        # Get T=0 values
        tau_90_zero_val = np.nan
        tau_e_zero_val = np.nan
        try:
            tau_90_zero_vals = []
            for count, tau_90s in enumerate(tau_90s_by_ic):
                clean_kTs = clean_90s_kTs_by_ic[count]
                zero_idx = np.where(np.array(clean_kTs)==0)[0]
                if not np.isnan(tau_90s[zero_idx].sum()):
                    tau_90_zero_vals.append(tau_90s[zero_idx])
            
            tau_90_zero_val = np.nanmean(tau_90_zero_vals)
                
            
            tau_e_zero_vals = []
            for count, tau_es in enumerate(tau_es_by_ic):
                clean_kTs = clean_es_kTs_by_ic[count]
                zero_idx = np.where(np.array(clean_kTs)==0)[0]
                if not np.isnan(tau_es[zero_idx].sum()):
                    tau_e_zero_vals.append(tau_es[zero_idx])
            
            tau_e_zero_val = np.nanmean(tau_e_zero_vals)
            
        except IndexError:
            pass

        my_dict = {'tau_90_zero_val':tau_90_zero_val, 'tau_90s_by_ic':tau_90s_by_ic, 
           'clean_90s_kTs_by_ic':clean_90s_kTs_by_ic, 'line_kTs':line_kTs, 
           'tau_90s_line':tau_90s_line, 'm_90':m_90}

        my_dict = convert_arrays_for_json(my_dict)
        tau90_plotting_data_outpath = '{}{}tau90_plotting_w{}_reg{}_t0_{}_eps{}.json'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0, eps_factor)
        with open(tau90_plotting_data_outpath, 'w') as outfile:
            json.dump(my_dict, outfile)

        # Average over ics
        list_of_series = []
        for count, tau_90s in enumerate(tau_90s_by_ic):
            clean_kTs = clean_90s_kTs_by_ic[count]
            temp_series = pd.Series(tau_90s, index=clean_kTs)
            list_of_series.append(temp_series)
        
        corr_df = pd.DataFrame(list_of_series).transpose()
        correlation_df_no_nans = corr_df[~corr_df.isnull().any(axis=1)]
        mean_tau_90s_df = correlation_df_no_nans.mean(axis=1)
        std_tau_90s_df = correlation_df_no_nans.std(axis=1)
        
        clean_kT_90_array = np.array(mean_tau_90s_df.index)
        mean_tau_90s = np.array(mean_tau_90s_df)
        std_tau_90s = np.array(std_tau_90s_df)
        
        list_of_series = []
        for count, tau_es in enumerate(tau_es_by_ic):
            clean_kTs = clean_es_kTs_by_ic[count]
            temp_series = pd.Series(tau_es, index=clean_kTs)
            list_of_series.append(temp_series)
        
        corr_df = pd.DataFrame(list_of_series).transpose()
        correlation_df_no_nans = corr_df[~corr_df.isnull().any(axis=1)]
        mean_tau_es_df = correlation_df_no_nans.mean(axis=1)
        std_tau_es_df = correlation_df_no_nans.std(axis=1)
        
        clean_kT_e_array = np.array(mean_tau_es_df.index)
        mean_tau_es = np.array(mean_tau_es_df)
        std_tau_es = np.array(std_tau_es_df)
        
        # Plot on log scale
        fig, ax = plt.subplots()
        if width==30:
            plt.axhline(tau_90_zero_val, label='T=0', linewidth=1.0, color='k')
        for count, tau_90s in enumerate(tau_90s_by_ic):
            clean_kTs = clean_90s_kTs_by_ic[count]
            plt.scatter(clean_kTs, tau_90s)
        left, right = plt.xlim()
        if t_0==20000:
            plt.plot(line_kTs, tau_90s_line, linewidth=1.0, label='grad={:.2f}'.format(m_90), color='b')
        # plt.xlabel('kT')
        # plt.ylabel(r'$\tau_{90}$')
        plt.xscale('log')
        plt.yscale('log')
        if L2==0 and width==30 and dataset_name=='binary_pca_mnist':
            plt.xlim(5e-7, 2)
            plt.xticks([1e-6, 1e-3, 1e0])
        if L2==1e-2 and width==30 and dataset_name=='binary_pca_mnist':
            plt.xticks([1e-9, 1e-5, 1e-1])
        if width==72:
            plt.xticks([1e-8, 1e-6, 1e-4])
        
        plt.yticks([1e2, 1e4])
        
        if dataset_name=='cifar10':
            if width==80 and L2==0:
                plt.xlim(5e-6, 2)
                plt.ylim(top=1e6)
                plt.xticks([1e-5, 1e-3, 1e-1])
                plt.yticks([1e0, 1e3, 1e6])
            else:
                plt.xlim(left=5e-11)
                plt.xticks([1e-9, 1e-5, 1e-1])
                plt.yticks([1e2, 1e4])
        plt.legend()
        # plt.title('90% decorrelation time vs kT at t={}'.format(t_0))
        plt.savefig('{}{}tau90_vs_kT_repeats_b100_w{}_reg{}_log_epsilon{}_t0{}.pdf'.format(plot_output_dir, name_prefix,
                    width, L2_str, epsilon_str, t_0), bbox_inches='tight')
        
        
        fig, ax = plt.subplots()
        if width==30:
            plt.axhline(tau_90_zero_val, label='T=0', linewidth=1.0, color='k')
        
        plt.errorbar(clean_kT_90_array, mean_tau_90s, yerr=std_tau_90s, capsize=3.0)
        left, right = plt.xlim()
        if t_0==20000:
            plt.plot(line_kTs, tau_90s_line, linewidth=1.0, label='grad={:.2f}'.format(m_90), color='r')
        plt.xscale('log')
        plt.yscale('log')
        if L2==0 and width==30 and dataset_name=='binary_pca_mnist':
            plt.xlim(5e-7, 2)
            plt.xticks([1e-6, 1e-3, 1e0])
        if L2==1e-2 and width==30 and dataset_name=='binary_pca_mnist':
            plt.xlim(5e-11, 5e-2)
            plt.xticks([1e-9, 1e-6, 1e-3])
        if width==72:
            plt.xticks([1e-8, 1e-6, 1e-4])
        
        plt.yticks([1e2, 1e4])
        
        if dataset_name=='cifar10':
            if width==80 and L2==0:
                plt.xlim(5e-6, 2)
                plt.ylim(top=1e6)
                plt.xticks([1e-5, 1e-3, 1e-1])
                plt.yticks([1e0, 1e3, 1e6])
            else:
                plt.xlim(left=5e-11)
                plt.xticks([1e-9, 1e-5, 1e-1])
                plt.yticks([1e2, 1e4])
        plt.legend()
        plt.savefig('{}{}tau90_vs_kT_repeats_b100_w{}_reg{}_log_epsilon{}_t0{}_mean_std.pdf'.format(plot_output_dir, name_prefix,
                    width, L2_str, epsilon_str, t_0), bbox_inches='tight')
        
        my_dict = {'tau_e_zero_val':tau_e_zero_val, 'tau_es_by_ic':tau_es_by_ic, 
           'clean_es_kTs_by_ic':clean_es_kTs_by_ic, 'line_e_kTs':line_e_kTs, 
           'tau_es_line':tau_es_line, 'm_e':m_e}

        my_dict = convert_arrays_for_json(my_dict)
        taue_plotting_data_outpath = '{}{}taue_plotting_w{}_reg{}_t0_{}_eps{}.json'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0, eps_factor)
        with open(taue_plotting_data_outpath, 'w') as outfile:
            json.dump(my_dict, outfile)
        
        fig, ax = plt.subplots()
        if width==30:
            plt.axhline(tau_e_zero_val, label='T=0', linewidth=1.0, color='k')
        for count, tau_es in enumerate(tau_es_by_ic):
            clean_kTs = clean_es_kTs_by_ic[count]
            plt.scatter(clean_kTs, tau_es)
        left, right = plt.xlim()
        plt.plot(line_e_kTs, tau_es_line, linewidth=1.0, label='grad={:.2f}'.format(m_e), color='r')
        # plt.xlabel('ln(kT)')
        # plt.ylabel(r'ln($\tau_{e}$)')
        plt.xscale('log')
        plt.yscale('log')
        # if width==30 and L2==0 and dataset_name=='binary_pca_mnist':
        #     plt.xlim(5e-7, 2)
        plt.legend()
        if width==30 and L2_str=='-2':
            plt.xticks([1e-9, 1e-6, 1e-3])
            plt.yticks([1e1, 1e3, 1e5])
        elif width==30 and L2_str=='0':
            plt.xticks([1e-6, 1e-3, 1e0])
            plt.yticks([1e1, 1e2, 1e3, 1e4])
        else:
            plt.xticks([1e-6, 1e-4, 1e-2])
            plt.yticks([1e1, 1e3, 1e5])
        # plt.title('e-fold decorrelation time vs kT at t={}'.format(t_0))
        plt.savefig('{}{}taue_vs_kT_repeats_b100_w{}_reg{}_log_epsilon{}_t0{}.pdf'.format(plot_output_dir, name_prefix,
                    width, L2_str, epsilon_str, t_0), bbox_inches='tight')
        
        
        fig, ax = plt.subplots()
        if width==30:
            plt.axhline(tau_e_zero_val, label='T=0', linewidth=1.0, color='k')
        plt.errorbar(clean_kT_e_array, mean_tau_es, yerr=std_tau_es, capsize=3.0)
        left, right = plt.xlim()
        plt.plot(line_e_kTs, tau_es_line, linewidth=1.0, label='grad={:.2f}'.format(m_e), color='r')
        plt.xscale('log')
        plt.yscale('log')
        if width==30 and L2==0 and dataset_name=='binary_pca_mnist':
            plt.xlim(5e-7, 2)
        plt.legend()
        if width==30 and L2_str=='-2':
            plt.xticks([1e-9, 1e-6, 1e-3])
            plt.yticks([1e1, 1e3, 1e5])
        elif width==30 and L2_str=='0':
            plt.xticks([1e-6, 1e-3, 1e0])
            plt.yticks([1e1, 1e3, 1e5])
        else:
            plt.xticks([1e-6, 1e-4, 1e-2])
            plt.yticks([1e1, 1e3, 1e5])
        plt.savefig('{}{}taue_vs_kT_repeats_b100_w{}_reg{}_log_epsilon{}_t0{}_mean_std.pdf'.format(plot_output_dir, name_prefix,
                    width, L2_str, epsilon_str, t_0), bbox_inches='tight')
        
def plot_collapsed_overlap_curves(device, base_path): 
    # dataset_name = 'binary_pca_mnist'
    dataset_name = 'cifar10'
    
    if dataset_name == 'binary_pca_mnist':
        kT_strings_list = [['0','-10','-8','-6','-5','-4','5-4','-3','5-3','-2','5-2','-1','5-1','1'],
                            ['0', '-6', '-5', '-4', '5-4', '-3', '5-3', '-2', '5-2', '-1', '5-1', '1'],
                            ['0','-10','-8','-6','-5','-4','5-4','-3','5-3','-2','5-2','-1','5-1','1']]
        
        kTs_list = [[0,1e-10,1e-08,1e-06,1e-05,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1],
                    [0, 1e-06, 1e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
                    [0, 1e-10, 1e-08, 1e-06, 1e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]]
        
        widths = [30, 30, 72]
        L2_strs = ['-2', '0', '0']
        t_fs = [300000, 300000, 200000]
        eps_factor_list = [[500], [1000], [500]]
        name_stub = ''
    
    else:
        kT_strings_list = [['0', 'e-10', 'e-8', 'e-6', 'e-5', 'e-4', 'e-3', 'e-2', 'e-1', 'e0'],
                            ['0', 'e-10', 'e-8', 'e-6', 'e-5', 'e-4', 'e-3', 'e-2', 'e-1', 'e0'],
                            ['0', 'e-10', 'e-8', 'e-6', 'e-5', 'e-4', 'e-3', 'e-2', 'e-1', 'e0']]
        
        kTs_list = [[0, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                    [0, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                    [0, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]]
        
        L2_strs = ['e-3', '0', '0']
        widths = [30, 80, 30]
        name_stub = 'cifar10_'
        eps_factor_list = [[250], [700], [250]]
        t_fs = [300000, 80000, 300000]
    
    for count, width in enumerate(widths):
        L2_str = L2_strs[count]
        t_f = t_fs[count]
        kT_strings = kT_strings_list[count]
        kTs = kTs_list[count]
        
        eps_factors = eps_factor_list[count]
        t_0 = 20000
        
        for eps_factor in eps_factors:
            # Calculate data  
            vH_curves = []
            time_arrays = []
            tau_es = []
            tau_90s = []
        
            for kT_count, kT in enumerate(kTs):
                correlation_curves = []
                times_for_curves = []
                
                if dataset_name == 'binary_pca_mnist':
                    if width==30:
                        ics = np.arange(10)
                    else:
                        ics = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9])
                
                else:
                    if width==30:
                        ics = np.arange(1, 10)
                    else:
                        ics = np.array([2, 3, 5])
                
                for ic in ics:
                    kT_str = kT_strings[kT_count]
                    
                    stub = '{}_b100_Langevin_relaxation_'.format(dataset_name)
                    model_name = '{}ic{}_w{}_kT{}_reg{}'.format(stub, ic, width, 
                                                                kT_str, L2_str)
                                              
                    input_dir = '{}models/{}/'.format(base_path, model_name)
            
                    params = load_model_params('{}{}_model_params.json'.format(
                                               input_dir, model_name))
                
                    models, epochs = load_models(input_dir, model_name, params, 
                                                 device)
                    
                    # Calculate a reasonable epsilon
                    epsilon = calc_epsilon_for_van_Hove_calc(models[-1], eps_factor)
                    epsilon_str = str(eps_factor)
                    
                    # Extract the model at timestep t_0
                    timesteps = np.array(epochs)
                    if timesteps[-1]<=t_0:
                        print('WARNING: ')
                        print('kT {} DOES NOT REACH {} TIMESTEPS'.format(kT, t_0))
                    else:
                        idx = np.where(timesteps>t_0)[0][0]
                        
                    epoch_seg = epochs[idx:]
                    model_seg = models[idx:]
                    
                    epoch_0 = epoch_seg[0]
                    model_0 = model_seg[0]
                            
                    self_correlations = []
                    epochs_for_plotting = []
                    
                    for i in range(len(epoch_seg)):
                        epoch = epoch_seg[i]
                        model = model_seg[i]
                        
                        epochs_for_plotting.append(epoch)
                        
                        self_correlation = mean_self_van_Hove_correlation(model_0, 
                                                                          epoch_0,
                                                                          model, 
                                                                          epoch,
                                                                          epsilon)
                        
                        self_correlations.append(self_correlation)
                    
                    correlation_curves.append(np.array(self_correlations))
                    times_for_curves.append(np.array(epochs_for_plotting))
                
                # Average over repetitions, accounting for different time grids
                list_of_series = []
                for i in range(len(correlation_curves)):
                    curve = correlation_curves[i]
                    time = times_for_curves[i]
                    series = pd.Series(curve, index=time)
                    list_of_series.append(series)
                
                corr_df = pd.DataFrame(list_of_series).transpose()
                correlation_df_no_nans = corr_df[~corr_df.isnull().any(axis=1)]
                mean_corr_curve = correlation_df_no_nans.mean(axis=1)
                
                time_array = np.array(mean_corr_curve.index)
                mean_corr_curve = np.array(mean_corr_curve)
                
                e_point = mean_corr_curve[0]/np.exp(1)
                tau90_point = 0.9*mean_corr_curve[0]
                
                if mean_corr_curve[-1]>e_point:
                    tau_e = np.nan
                    print("{} doesn't reach e-fold point.".format(model_name))
                else:
                    tau_e_idx = np.where(mean_corr_curve<=e_point)[0][0]
                    if tau_e_idx < 2:
                        tau_e = np.nan
                        print("{} tau_e_idx is less than 2".format(model_name))
                    else:
                        tau_e = time_array[tau_e_idx] - time_array[0]
                        print("{} tau_e = {}".format(model_name, tau_e))
                
                if mean_corr_curve[-1]>tau90_point:
                    tau_90 = np.nan
                    print("{} doesn't reach 0.9 point.".format(model_name))
                else:
                    tau_90_idx = np.where(mean_corr_curve<=tau90_point)[0][0]
                    if tau_90_idx < 2:
                        tau_90 = np.nan
                        print("{} tau_90_idx is less than 2".format(model_name))
                    else:
                        tau_90 = time_array[tau_90_idx] - time_array[0]
                        print("{} tau_90 = {}".format(model_name, tau_90))
                
                vH_curves.append(mean_corr_curve)
                time_arrays.append(time_array)
                tau_es.append(tau_e)
                tau_90s.append(tau_90)
             
            # Fit power law to master curve
            high_T_corr_curve = vH_curves[-1]
            high_T_time_array = time_arrays[-1]
            tau_e = tau_es[-1]
            
            modified_high_T_time_array = (high_T_time_array-high_T_time_array[0])/tau_e
            start_idx = np.where(modified_high_T_time_array>=1)[0][0]
            end_idx = np.where(modified_high_T_time_array>1e3)[0][0]
            modified_high_T_time_array = modified_high_T_time_array[start_idx:end_idx]
            high_T_corr_curve = high_T_corr_curve[start_idx:end_idx]
            
            m, c, line_times = find_grad(high_T_corr_curve, modified_high_T_time_array)
            
            line = np.exp(c*2)*line_times**m     
             
            # Plot decorrelation curve
            plot_output_dir = '{}plots/Overlap_collapse_curves/'.format(base_path)
            dir_exists = os.path.isdir(plot_output_dir)
            if not dir_exists:
                os.mkdir(plot_output_dir) 
            
            finite_taues = np.where(np.isfinite(tau_es)==True)[0]
            
            fig, ax = plt.subplots()
            for count in finite_taues:
                self_correlations = vH_curves[count]
                timestep_array = time_arrays[count]
                kT = kTs[count]
                tau_e = tau_es[count]
                modified_t = (timestep_array-timestep_array[0])/tau_e
                mask = modified_t>0
                modified_t = modified_t[mask]
                self_correlations = self_correlations[mask]
            
                plt.plot(modified_t, self_correlations, 
                         label='kT={}'.format(kT), 
                         c=plt.cm.Spectral(1.0-count/len(finite_taues)))
                
            plt.axhline(y=0, color='k', linestyle='dashed', linewidth=1.0)
            plt.title('Mean Overlap Correlation')
            plt.ylabel(r'Overlap')
            plt.xlabel(r'$t/\tau_e$')
            plt.xscale('log')
            plt.legend()
            plt.savefig('{}{}collapsed_overlap_t0{}_eps_fac{}_width{}_L2{}_log_small_text.pdf'.format(
            plot_output_dir, name_stub, t_0, epsilon_str, width, L2_str), bbox_inches='tight')
            
            
            fig, ax = plt.subplots()
            for i, count in enumerate(finite_taues):
                self_correlations = vH_curves[count]
                timestep_array = time_arrays[count]
                try:
                    t_f_idx = np.where(timestep_array>=t_f)[0][0]
                    t_f_val = self_correlations[t_f_idx]
                except IndexError:
                    t_f_val = np.nan
                kT = kTs[count]
                tau_e = tau_es[count]
                modified_t = (timestep_array-timestep_array[0])/tau_e
                mask = modified_t>0
                modified_t = modified_t[mask]
                self_correlations = self_correlations[mask]
            
                plt.plot(modified_t, self_correlations, 
                         c=plt.cm.Spectral(1.0-i/len(finite_taues)))
                
            plt.plot(line_times, line, color='k', linestyle='dashed', label='grad={:.2f}'.format(m))
            plt.title('Mean Overlap Correlation')
            plt.ylabel(r'Overlap')
            plt.xlabel(r'$t/\tau_e$')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
            if dataset_name == 'binary_pca_mnist' or width==80:
                plt.ylim(bottom=3e-4)
            plt.savefig('{}{}collapsed_overlap_t0{}_eps_fac{}_width{}_L2{}_log_log_small_text.pdf'.format(
            plot_output_dir, name_stub, t_0, epsilon_str, width, L2_str), bbox_inches='tight')
            
            
            finite_tau90s = np.where(np.isfinite(tau_90s)==True)[0]
            
            fig, ax = plt.subplots()
            for i, count in enumerate(finite_tau90s):
                self_correlations = vH_curves[count]
                timestep_array = time_arrays[count]
                kT = kTs[count]
                tau_90 = tau_90s[count]
                modified_t = (timestep_array-timestep_array[0])/tau_90
                mask = modified_t>0
                modified_t = modified_t[mask]
                self_correlations = self_correlations[mask]
            
                plt.plot(modified_t, self_correlations, label=kT,
                         c=plt.cm.Spectral(1.0-i/len(finite_tau90s)))
                
            plt.title('Mean Overlap Correlation')
            plt.ylabel(r'Overlap')
            plt.xlabel(r'$t/\tau_{90}$')
            plt.legend()
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('{}{}collapsed_overlap_tau90_t0{}_eps_fac{}_width{}_L2{}_log_log_small_text.pdf'.format(
            plot_output_dir, name_stub, t_0, epsilon_str, width, L2_str), bbox_inches='tight')   
            
# Measure mean square displacement
def calc_mean_square_displacement(models, epochs):
    # Get network architecture
    model = models[0]
    layers = []
    for layer in model.net:
        # Deal with ReLU layers
        try:
            layer_shape = layer.weight.shape
            layer_with_time = np.zeros((len(models),layer_shape[0],layer_shape[1]))
            layers.append(layer_with_time)
        except AttributeError:
            layers.append(None)
    
    # Put weights in this structure
    for t, model in enumerate(models):
        for l, layer in enumerate(model.net):
            try:
                weights = layer.weight.detach().numpy()
                layers[l][t, :, :] = weights
            
            except AttributeError:
                pass
    
    # Measure mean square displacement
    MSDs = []
    times = []
    
    for t in range(layers[0].shape[0]):
        my_sum = 0.0
        my_count = 0.0
        for layer_with_time in layers:
            if layer_with_time is None:
                continue
            
            # Square displacement from t_0 to t
            d2 = (layer_with_time[t, :, :] - layer_with_time[0, :, :])**2
            my_sum += d2.sum()
            my_count += d2.size                

        
        # Add safety check for very large jumps
        if len(MSDs)>0:
            if abs(my_sum/my_count - MSDs[-1])>1e15:
                break
        
        MSDs.append(my_sum/my_count)
        times.append(epochs[t]-epochs[0])
    
    return MSDs, times
            
def plot_MSD_vs_kT(device, base_path):
    dataset_name = 'binary_pca_mnist'
    # dataset_name = 'cifar10'
    
    if dataset_name == 'binary_pca_mnist':
        orig_kT_strings = ['-3', '-2', '-1']
        orig_kTs = [1e-3, 1e-2, 1e-1]
        L2s = [0.0, 1e-2, 0.0]
        L2_strings = ['0', '-2', '0']
        widths = [30, 30, 72]
        dataset_pathstub = ''

    elif dataset_name == 'cifar10':
        orig_kT_strings = ['e-10', 'e-8', 'e-6', 'e-5', 'e-4', 'e-3', 'e-2', 'e-1', 'e0']
        orig_kTs = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        L2s = [1e-3, 0.0, 0.0]
        L2_strings = ['e-3', '0', '0']
        widths = [30, 80, 30]
        
        dataset_pathstub = 'cifar10_'

    else:
        print('Select a dataset.')

    
    for count in range(len(L2s)):
        width = widths[count]
        L2 = L2s[count]
        L2_str = L2_strings[count]
   
        t_0 = 20000
        
        kT_strings = orig_kT_strings
        kTs = orig_kTs
        
        if dataset_name=='binary_pca_mnist':
            if width==30 and L2==0:
                new_kT_strings = ['0', '-6', '-5']
                new_kTs = [0, 1e-6, 1e-5]
                
                kT_strings = new_kT_strings + kT_strings
                kTs = new_kTs + kTs
            
            elif width==30 and L2==1e-2:
                new_kT_strings = ['0', '-10', '-8', '-6', '-5']
                new_kTs = [0, 1e-10, 1e-8, 1e-6, 1e-5]
                
                kT_strings = new_kT_strings + kT_strings
                kTs = new_kTs + kTs
                
            else:
                new_kT_strings = ['-10', '-8', '-6', '-5']
                new_kTs = [1e-10, 1e-8, 1e-6, 1e-5]
                
                kT_strings = new_kT_strings + kT_strings
                kTs = new_kTs + kTs
    
        if dataset_name=='cifar10':
            new_kT_strings = ['0']
            new_kTs = [0]
            
            kT_strings = new_kT_strings + kT_strings 
            kTs = new_kTs + kTs
        
        # Calculate data 
        MSDs_by_ic = []
        times_by_ic = []
        
        if dataset_name == 'cifar10':
            if width==30:
                ics = np.arange(1, 10)
            else:
                ics = np.array([2, 3, 5])
        elif dataset_name == 'binary_pca_mnist_LR-4':
            ics = np.arange(10)
        else:
            if width==72:
                ics = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 
                                15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 
                                28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 
                                42, 43, 44, 45, 46, 47, 48, 49])
            elif width==30 and L2==0:
                ics = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 19, 
                                25, 26, 27, 29, 30, 31, 36, 37, 44, 45, 46, 48])
            else:
                ics = np.arange(50)
            
        plot_output_dir = '{}plots/MSD_vs_kT/'.format(base_path)
        dir_exists = os.path.isdir(plot_output_dir)
        if not dir_exists:
            os.mkdir(plot_output_dir)  
        
        # See if data already exists
        data_output_dir = '{}measured_data/MSD_vs_kT/'.format(base_path)
        dir_exists = os.path.isdir(data_output_dir)
        if not dir_exists:
            os.mkdir(data_output_dir)  
        
        MSD_outpath = '{}{}MSDs_by_ic_w{}_reg{}_t0_{}.pkl'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0)
        t_outpath = '{}{}times_by_ic_w{}_reg{}_t0_{}.pkl'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0)
        eps_outpath = '{}{}epsilon_w{}_reg{}_t0_{}.pkl'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0)
        MSD_data_exists = os.path.isfile(MSD_outpath)
        t_data_exists = os.path.isfile(t_outpath)
        eps_data_exists = os.path.isfile(eps_outpath)
        data_exists = MSD_data_exists and t_data_exists and eps_data_exists
            
        if data_exists:
            with open(MSD_outpath, 'rb') as infile:
                MSDs_by_ic = pickle.load(infile)
            
            with open(t_outpath, 'rb') as infile:
                times_by_ic = pickle.load(infile)
            
            with open(eps_outpath, 'rb') as infile:
                epsilon = pickle.load(infile)
                
            print('Loaded MSDs from text file.')
    
        else:
            print('Calculating MSDs')
            for ic in ics: 
                # Correct change in naming convention
                L2_str = L2_strings[count]
                if L2_str=='-2' and ic>9:
                    L2_str = '0_010'
                    
                MSDs = []
                times = []
                
                for kT_count, kT in enumerate(kTs):
                    kT_str = kT_strings[kT_count]
                    if dataset_name == 'binary_pca_mnist_LR-4':
                        model_name = ('binary_pca_mnist_b100_Langevin_relaxation_ic{}_w{}_'.format(ic, width)+
                                      'kT{}_reg{}_LR-4'.format(kT_str, L2_str))
                    else:
                        model_name = ('{}_b100_Langevin_relaxation_ic{}_w{}_'.format(dataset_name, ic, width)+
                                          'kT{}_reg{}'.format(kT_str, L2_str))
                    print('model_name = ', model_name)
                    input_dir = '{}models/{}/'.format(base_path, model_name)
            
                    params = load_model_params('{}{}_model_params.json'.format(
                                               input_dir, model_name))
                
                    models, epochs = load_models(input_dir, model_name, params, 
                                                 device)
                    
                    if len(epochs)==0:
                        print('No epochs!')
                        print('Model: ', model_name)
                        MSDs.append(np.array([np.nan]))
                        times.append(np.array([np.nan]))
                        continue
                    
                    # Extract the model at timestep t_0
                    timesteps = np.array(epochs)
                    if timesteps[-1]<t_0:
                        idx = len(models)-1
                        print('WARNING: ')
                        print('kT {} DOES NOT REACH {} TIMESTEPS'.format(kT, t_0))
                        MSDs.append(np.array([np.nan]))
                        times.append(np.array([np.nan]))
                        continue
                    else:
                        idx = np.where(timesteps>=t_0)[0][0]
                        
                    epoch_seg = epochs[idx:]
                    model_seg = models[idx:]
                    
                    if dataset_name == 'binary_pca_mnist':
                        # Get the epsilon used for decorrelation plot at kT=1e-3. eps doesn't vary much with T.
                        if kT==1e-3:
                            if width==30:
                                epsilon_factor = 500
                            else:
                                epsilon_factor = 6000
                                
                            final_model = model_seg[-1]
                            if check_for_NaN_network(final_model):
                                final_model = model_seg[-2]
                                
                            params = []
                            deltas = range(0, len(final_model.net)-1, 2)
                            
                            for delta in deltas:
                                layer_tensor = final_model.net[delta].weight
                                layer = layer_tensor.detach().numpy()
                                params += list(layer.flatten())
                            
                            std = np.std(params)
                            epsilon = std/epsilon_factor
                    
                    else:
                        epsilon = np.nan
                    
                    # Calculate MSD
                    MSD, epochs = calc_mean_square_displacement(model_seg, epoch_seg)
                    timesteps_for_plotting = np.array(epochs)
                     
                    MSDs.append(MSD)
                    times.append(timesteps_for_plotting)
                
                # Plot
                fig, ax = plt.subplots()
                for i in range(len(MSDs)):
                    plt.plot(times[i], MSDs[i], label=kTs[i])
                plt.xlabel('timesteps')
                plt.ylabel(r'MSD')
                plt.xscale('log')
                plt.yscale('log')
                plt.legend()
                plt.title('MSD for different kT from t={}'.format(t_0))
                plt.savefig('{}{}MSD_vs_kT_w{}_reg{}_ic{}_t0_{}.pdf'.format(plot_output_dir, dataset_pathstub, width, 
                                                                 L2_str, ic, t_0), 
                                                                 bbox_inches='tight')
                
                MSDs_by_ic.append(MSDs)
                times_by_ic.append(times)
            
            # Save data
            dir_exists = os.path.isdir(data_output_dir)
            if not dir_exists:
                os.mkdir(data_output_dir) 
            
            with open(MSD_outpath, 'wb') as outfile:
                pickle.dump(MSDs_by_ic, outfile)
            
            with open(t_outpath, 'wb') as outfile:
                pickle.dump(times_by_ic, outfile)
            
            with open(eps_outpath, 'wb') as outfile:
                pickle.dump(epsilon, outfile)
            
        # Rearrange to get MSDs_by_kT
        MSDs_by_kT = []
        times_by_kT = []
        for i in range(len(kTs)):
            MSD_temp = []
            time_temp = []
            for j in range(len(MSDs_by_ic)):
                MSD_temp.append(MSDs_by_ic[j][i])
                time_temp.append(times_by_ic[j][i])
            
            MSDs_by_kT.append(MSD_temp)
            times_by_kT.append(time_temp)
        
        # Average MSD over many networks
        mean_MSDs = []
        mean_times = []
        longest_time = times_by_kT[0][0]
        for i, MSD_by_kT in enumerate(MSDs_by_kT):
            time_by_kT = times_by_kT[i]
            
            for time_list in time_by_kT:
                if time_list[-1]>longest_time[-1]:
                    longest_time = time_list
                    
            mean_MSD, mean_time = variable_length_mean(MSD_by_kT, time_by_kT)
            mean_MSDs.append(mean_MSD)
            mean_times.append(mean_time)
        
        # Fit line to T=0 line at long t
        T0_times = np.array(mean_times[0])
        T0_MSD = np.array(mean_MSDs[0])
        if dataset_name == 'binary_pca_mnist_LR-4':
            m_T0_line = np.nan
            c_T0_line = np.nan
        else:
            late_t_idx = np.where(T0_times>=1e5)[0][0]
            data_segment = T0_MSD[late_t_idx:]
            time_segment = T0_times[late_t_idx:]
            m_T0_line, c_T0_line, T0_line_ts = find_grad(data_segment, time_segment)
        
        if width==30 and L2_str=='-2':
            c_for_plotting = c_T0_line*0.95
        elif width==30 and L2_str=='0':
            c_for_plotting = c_T0_line*1.1
        else:
            c_for_plotting = c_T0_line*0.95
            
        plotting_t = np.array([])
        T0_line_vals = np.array([])
        
        # Draw grad=1 line
        line_times = np.array(mean_times[-1])
        line_times = line_times[line_times>0.9]
        line_vals = line_times*(10**(-8))
        
        colour1 = '#0096FF' #blue
        colour2 = '#FF0000' #red
        if len(mean_MSDs)<=1:
            colours = ['b']
        else:
            colours = get_color_gradient(colour1, colour2, len(mean_MSDs))
        
        fig, ax = plt.subplots()
        plt.plot(line_times, line_vals, label='grad=1', linestyle='dotted', color='k')
        if width==30:
            plt.axhline(y=epsilon**2, color='k', linewidth=1.0)
        for i in range(len(mean_MSDs)):
            my_c = colours[i]
            plt.plot(mean_times[i], mean_MSDs[i], color=my_c)
        # plt.xlabel('timesteps')
        # plt.ylabel(r'MSD')
        if dataset_name == 'binary_pca_mnist':
            plt.ylim([1e-18, 1e-2])   
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        if dataset_name == 'binary_pca_mnist':
            plt.xticks([1e0, 1e3, 1e6])
            plt.yticks([1e-16, 1e-10, 1e-4])
        # else:
        #     plt.locator_params(nbins=3)
        # plt.title('Mean MSD for different kT from t={}'.format(t_0))
        plt.savefig('{}{}MSD_vs_kT_mean_w{}_reg{}_t0_{}.pdf'.format(plot_output_dir, dataset_pathstub,
                                                              width, L2_str, t_0), 
                                                         bbox_inches='tight')
        
def plot_alpha2(device, base_path):
    dataset_name = 'binary_pca_mnist'
    # dataset_name = 'cifar10'
    
    if dataset_name == 'binary_pca_mnist':
        orig_kT_strings = ['-3', '-2', '-1']
        orig_kTs = [1e-3, 1e-2, 1e-1]
        L2s = [0.0, 1e-2, 0.0]
        L2_strings = ['0', '-2', '0']
        widths = [30, 30, 72]
        dataset_pathstub = ''

    elif dataset_name == 'cifar10':
        orig_kT_strings = ['e-10', 'e-8', 'e-6', 'e-5', 'e-4', 'e-3', 'e-2', 'e-1', 'e0']
        orig_kTs = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        L2s = [1e-3, 0.0, 0.0]
        L2_strings = ['e-3', '0', '0']
        widths = [30, 80, 30]
        dataset_pathstub = 'cifar10_'
    
    else:
        print('Select a dataset.')

    for count in range(len(L2s)):
        width = widths[count]
        L2 = L2s[count]
        L2_str = L2_strings[count]
   
        t_0 = 20000
        
        kT_strings = orig_kT_strings
        kTs = orig_kTs
        
        if dataset_name=='binary_pca_mnist':
            if width==30 and L2==0:
                new_kT_strings = ['0', '-6', '-5']
                new_kTs = [0, 1e-6, 1e-5]
                
                kT_strings = new_kT_strings + kT_strings
                kTs = new_kTs + kTs
            
            elif width==30 and L2==1e-2:
                new_kT_strings = ['0', '-10', '-8', '-6', '-5']
                new_kTs = [0, 1e-10, 1e-8, 1e-6, 1e-5]
                
                kT_strings = new_kT_strings + kT_strings
                kTs = new_kTs + kTs
                
            else:

                new_kT_strings = ['-10', '-8', '-6', '-5']
                new_kTs = [1e-10, 1e-8, 1e-6, 1e-5]
                
                kT_strings = new_kT_strings + kT_strings
                kTs = new_kTs + kTs
    
        if dataset_name=='cifar10':
            if width<80:
                new_kT_strings = ['0']
                new_kTs = [0]
                
                kT_strings = new_kT_strings + kT_strings 
                kTs = new_kTs + kTs
        
        # Remove T=0.1 from cifar10 overparam
        if dataset_name=='cifar10' and width==80:
            kTs.remove(1e-1)
            kT_strings.remove('e-1')
        
        # Calculate data 
        alpha2_by_ic = []
        times_by_ic = []
        
        if dataset_name == 'cifar10':
            if width==30:
                ics = np.arange(1, 10)
            else:
                ics = np.array([2, 3, 5])
        else:
            if width==72:
                ics = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 
                                15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 
                                28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 
                                42, 43, 44, 45, 46, 47, 48, 49])
            elif width==30 and L2==0:
                ics = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 19, 
                                25, 26, 27, 29, 30, 31, 36, 37, 44, 45, 46, 48])
            else:
                ics = np.arange(50)
            
        plot_output_dir = '{}plots/alpha2/'.format(base_path)
        dir_exists = os.path.isdir(plot_output_dir)
        if not dir_exists:
            os.mkdir(plot_output_dir)  
        
        # See if data already exists
        data_output_dir = '{}measured_data/alpha2/'.format(base_path)
        alpha2_outpath = '{}{}alpha2_by_ic_w{}_reg{}_t0_{}.pkl'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0)
        t_outpath = '{}{}times_by_ic_w{}_reg{}_t0_{}.pkl'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0)
        alpha2_data_exists = os.path.isfile(alpha2_outpath)
        t_data_exists = os.path.isfile(t_outpath)
        data_exists = alpha2_data_exists and t_data_exists
            
        if data_exists:
            with open(alpha2_outpath, 'rb') as infile:
                alpha2_by_ic = pickle.load(infile)
            
            with open(t_outpath, 'rb') as infile:
                times_by_ic = pickle.load(infile)
            
            print('Loaded alpha2s from text file.')
    
        else:
            print('Calculating alpha2s')
            for ic in ics:   
                alpha2 = []
                times = []
                
                for kT_count, kT in enumerate(kTs):
                    kT_str = kT_strings[kT_count]
                    model_name = ('{}_b100_Langevin_relaxation_ic{}_w{}_'.format(dataset_name, ic, width)+
                                      'kT{}_reg{}'.format(kT_str, L2_str))
                    input_dir = '{}models/{}/'.format(base_path, model_name)
            
                    params = load_model_params('{}{}_model_params.json'.format(
                                               input_dir, model_name))
                
                    models, epochs = load_models(input_dir, model_name, params, 
                                                 device)
                    
                    if len(epochs)==0:
                        print('No epochs!')
                        print('Model: ', model_name)
                        alpha2.append(np.array([np.nan]))
                        times.append(np.array([np.nan]))
                        continue
                    
                    # Extract the model at timestep t_0
                    timesteps = np.array(epochs)
                    if timesteps[-1]<t_0:
                        idx = len(models)-1
                        print('WARNING: ')
                        print('kT {} DOES NOT REACH {} TIMESTEPS'.format(kT, t_0))
                        alpha2.append(np.array([np.nan]))
                        times.append(np.array([np.nan]))
                        continue
                    else:
                        idx = np.where(timesteps>=t_0)[0][0]
                        
                    epoch_seg = epochs[idx:]
                    model_seg = models[idx:]
                    
                    # Calculate alpha2
                    a2 = calc_alpha2(model_seg)
                    timesteps_for_plotting = np.array(epoch_seg)-epoch_seg[0]
                     
                    alpha2.append(a2)
                    times.append(timesteps_for_plotting)
                
                # Plot
                fig, ax = plt.subplots()
                for i in range(len(alpha2)):
                    plt.plot(times[i], alpha2[i], label=kTs[i])
                plt.xlabel('timesteps')
                plt.ylabel(r'$\alpha_2$')
                plt.xscale('log')
                plt.yscale('log')
                plt.legend()
                plt.title('alpha2 for different kT from t={}'.format(t_0))
                plt.savefig('{}{}alpha2_vs_kT_w{}_reg{}_ic{}_t0_{}.pdf'.format(plot_output_dir, dataset_pathstub, width, 
                                                                 L2_str, ic, t_0), 
                                                                 bbox_inches='tight')
                
                alpha2_by_ic.append(alpha2)
                times_by_ic.append(times)
            
            # Save data
            dir_exists = os.path.isdir(data_output_dir)
            if not dir_exists:
                os.mkdir(data_output_dir) 
            
            with open(alpha2_outpath, 'wb') as outfile:
                pickle.dump(alpha2_by_ic, outfile)
            
            with open(t_outpath, 'wb') as outfile:
                pickle.dump(times_by_ic, outfile)
                
        # Rearrange to get 
        alpha2_by_kT = []
        times_by_kT = []
        for i in range(len(kTs)):
            alpha2_temp = []
            time_temp = []
            for j in range(len(alpha2_by_ic)):
                alpha2_temp.append(alpha2_by_ic[j][i])
                time_temp.append(times_by_ic[j][i])
            
            alpha2_by_kT.append(alpha2_temp)
            times_by_kT.append(time_temp)
        
        fig, ax = plt.subplots()
        for i in range(len(alpha2_by_kT)):
            alpha2s = alpha2_by_kT[i]
            times = times_by_kT[i]
            my_c=plt.cm.Spectral(1.0-i/len(alpha2_by_kT))
            
            for j in range(len(alpha2s)):
                if j==0:
                    plt.plot(times[j], alpha2s[j], label=kTs[i], c=my_c)
                else:
                    plt.plot(times[j], alpha2s[j], c=my_c)
        
        fig, ax = plt.subplots()
        for i in range(len(alpha2_by_kT)):
            alpha2s = alpha2_by_kT[i]
            times = times_by_kT[i]
            my_c=plt.cm.Spectral(1.0-i/len(alpha2_by_kT))
            
            for j in range(len(alpha2s)):
                if j==0:
                    plt.plot(times[j], alpha2s[j], label=kTs[i], c=my_c)
                else:
                    plt.plot(times[j], alpha2s[j], c=my_c)
                    
        plt.xlabel('timesteps')
        plt.ylabel(r'$\alpha_2$')
        plt.legend()
        plt.xscale('log')
        plt.title('alpha2 for different kT from t={}'.format(t_0))
        plt.savefig('{}{}alpha2_w{}_reg{}_t0_{}_linear.pdf'.format(plot_output_dir, dataset_pathstub,
                                                              width, L2_str, t_0), 
                                                         bbox_inches='tight')
        
        # Average over ics
        mean_alpha2s_by_kT = []
        mean_times_by_kT = []
        
        for i in range(len(alpha2_by_kT)):
            alpha2s = alpha2_by_kT[i]
            times = times_by_kT[i]
            
            list_of_series = []
            for j in range(len(alpha2s)):
                curve = alpha2s[j]
                t = times[j]
                series = pd.Series(curve, index=t)
                list_of_series.append(series)
            
            alpha2_df = pd.DataFrame(list_of_series).transpose()
            mean_alpha2_curve = alpha2_df.mean(axis=1)
            
            time_array = np.array(mean_alpha2_curve.index)
            mean_alpha2_curve = np.array(mean_alpha2_curve)
            
            mean_alpha2s_by_kT.append(mean_alpha2_curve)
            mean_times_by_kT.append(time_array)
        
        # Plot average curves
        colour1 = '#0096FF' #blue
        colour2 = '#FF0000' #red
        colours = get_color_gradient(colour1, colour2, len(mean_alpha2s_by_kT))
        
        fig, ax = plt.subplots()
        for i in range(len(mean_alpha2s_by_kT)):
            if width==72 and kTs[i]==1e-10:
                continue
            alpha2 = mean_alpha2s_by_kT[i]
            times = mean_times_by_kT[i]
            my_c=colours[i]
            plt.plot(times, alpha2, label=kTs[i], c=my_c)
        
        # plt.xlabel('timesteps')
        # plt.ylabel(r'$\alpha_2$')
        # plt.legend()
        if dataset_name == 'cifar10':
            plt.yticks([0, 5, 10])
        else:
            if width==30 and L2_str=='0':
                plt.yticks([0, 2, 4])
            if width==30 and L2_str=='-2':
                plt.yticks([0, 5, 10])
            if width==72 and L2_str=='0':
                plt.yticks([0, 0.1, 0.2])
        plt.xscale('log')
        # plt.title('alpha2 for different kT from t={}'.format(t_0))
        plt.savefig('{}{}mean_alpha2_w{}_reg{}_t0_{}_linear.pdf'.format(plot_output_dir, dataset_pathstub,
                                                              width, L2_str, t_0), 
                                                         bbox_inches='tight')

def plot_disp_dist_multiple_temps(device, base_path):
    dataset_name = 'binary_pca_mnist'
    # dataset_name = 'cifar10'
    
    if dataset_name == 'binary_pca_mnist':
        kT_strings_list = [['-2', '-3', '-5', '-6'],
                           ['-3', '-5', '-6', '-8'],
                           ['-1', '-3', '-6', '-8']]
        kTs_list = [[1e-2, 1e-3, 1e-5, 1e-6],
                    [1e-3, 1e-5, 1e-6, 1e-8],
                    [1e-1, 1e35, 1e-6, 1e-8]]
        L2s = [0.0, 1e-2, 0.0]
        L2_strings = ['0', '-2', '0']
        widths = [30, 30, 72]
        
        delta_t = 200000
        
        dataset_pathstub = ''
    
    elif dataset_name == 'cifar10':
        kT_strings_list = [['e-3', 'e-5', 'e-6', 'e-8'],
                           ['e-3', 'e-5', 'e-6', 'e-8'],
                           ['e-3', 'e-5', 'e-6', 'e-8']]
        
        kTs_list = [[1e-3, 1e-5, 1e-6, 1e-8],
                    [1e-3, 1e-5, 1e-6, 1e-8],
                    [1e-3, 1e-5, 1e-6, 1e-8]]
        
        L2s = [1e-3, 0.0, 0.0]
        L2_strings = ['e-3', '0', '0']
        widths = [30, 80, 30]
        
        delta_t = 200000
        
        dataset_pathstub = 'cifar10_'
        
    else:
        print('Select a dataset.')

    
    for count in range(len(L2s)):
        width = widths[count]
        L2_str = L2_strings[count]
        kT_strings = kT_strings_list[count]
        kTs = kTs_list[count]
   
        t_0 = 20000
       
        if dataset_name == 'binary_pca_mnist':
            if width==72:
                ics = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
                       15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 
                       29, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 
                       44, 45, 46, 47, 48, 49]
            elif L2_str=='0':
                ics = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 19, 25, 26, 
                       27, 29, 30, 31, 36, 37, 44, 45, 46, 48]
            else:
                ics = np.arange(50)
        else:
            if width==30:
                ics = np.arange(1, 10)
            else:
                ics = np.array([2, 3, 5])
            
        plot_output_dir = '{}plots/disp_multi_temp/'.format(base_path)
        dir_exists = os.path.isdir(plot_output_dir)
        if not dir_exists:
            os.mkdir(plot_output_dir)  
        
        displacements_by_kT = []
        
        for kT_count, kT in enumerate(kTs):
            kT_str = kT_strings[kT_count]
            # See if data already exists
            data_output_dir = '{}measured_data/disp_multi_temp/'.format(base_path)
            disp_outpath = '{}{}displacements_w{}_reg{}_t0_{}_kT{}.pkl'.format(data_output_dir, dataset_pathstub, width, L2_str, t_0, kT_str)
            data_exists = os.path.isfile(disp_outpath)
                
            if data_exists:
                with open(disp_outpath, 'rb') as infile:
                    displacements = pickle.load(infile)
                
                print('Loaded displacements from text file.')
        
            else:
                print('Calculating displacements')
                displacements = []
                
                for ic in ics: 
                    # Correct change in naming convention
                    L2_str = L2_strings[count]
                    if L2_str=='-2' and ic>9:
                        L2_str = '0_010'      
                        
                    model_name = ('{}_b100_Langevin_relaxation_ic{}_w{}_'.format(dataset_name, ic, width)+
                                      'kT{}_reg{}'.format(kT_str, L2_str))
                    input_dir = '{}models/{}/'.format(base_path, model_name)
            
                    params = load_model_params('{}{}_model_params.json'.format(
                                               input_dir, model_name))
                
                    models, epochs = load_models(input_dir, model_name, params, 
                                                 device)
                    
                    if len(epochs)==0:
                        print('No epochs!')
                        print('Model: ', model_name)
                        continue
                    
                    # Extract the model at timestep t_0
                    timesteps = np.array(epochs)
                    if timesteps[-1]<t_0:
                        idx = len(models)-1
                        print('WARNING: ')
                        print('kT {} DOES NOT REACH {} TIMESTEPS'.format(kT, t_0))
                        continue
                    else:
                        idx = np.where(timesteps>=t_0)[0][0]
                    
                    # Extract the model at timestep t_0 + delta_t
                    if timesteps[-1]<t_0+delta_t:
                        idx2 = len(models)-1
                        print('WARNING: ')
                        print('kT {} DOES NOT REACH {} TIMESTEPS'.format(kT, t_0+delta_t))
                        continue
                    else:
                        idx2 = np.where(timesteps>=t_0+delta_t)[0][0]
                        
                    model_0 = models[idx]
                    model_1 = models[idx2]
                    
                    # Calculate displacements
                    disp = calc_weight_displacements(model_0, model_1)
                    
                    displacements += list(disp)
                
                # Save data
                dir_exists = os.path.isdir(data_output_dir)
                if not dir_exists:
                    os.mkdir(data_output_dir) 
                
                with open(disp_outpath, 'wb') as outfile:
                    pickle.dump(displacements, outfile)
                
            
            displacements_by_kT.append(displacements)
            
        # Save debug data in text file
        disp_by_kT_path = '{}{}disp_by_kT_data.txt'.format(plot_output_dir, dataset_pathstub)
        np.savetxt(disp_by_kT_path, np.array(displacements_by_kT))                  
        
        # Plot data
        colour1 = '#FF0000' #red
        colour2 = '#0096FF' #blue
        colours = get_color_gradient(colour1, colour2, len(displacements_by_kT))

        fig, ax = plt.subplots()
        for count, displacements in enumerate(displacements_by_kT):
            
            displacements = np.array(displacements)
            positive_disp = displacements[displacements>0]
            my_c=colours[count]
            
            try:
                plt.hist(np.log10(positive_disp), bins=100, histtype='step', 
                         ec=my_c)
            except ValueError:
                print('ValueError, kT_str={}'.format(kT_str))
                pass # In case of NaN displacements
                    
        # plt.xlabel(r'log10($|w(t+\Delta t)-w(t)|$)')
        # plt.ylabel(r'Frequency')
        plt.yscale('log')
        if dataset_name == 'binary_pca_mnist':
            plt.yticks([1e0, 1e2, 1e4])
            plt.xticks(ticks=[-8, -5, -2], labels=[r'$10^{-8}$', r'$10^{-5}$', r'$10^{-2}$'])
        # plt.legend(loc='lower center')
        # plt.title('Displacement distributions from t={}'.format(t_0))
        plt.savefig('{}{}displacements_prelogged_w{}_reg{}_t0_{}_kT{}.pdf'.format(plot_output_dir, dataset_pathstub,
                                                              width, L2_str, t_0, kT_str), 
                                                         bbox_inches='tight')
        
def plot_mean_van_Hove_Langevin_curves_for_aging(device, base_path):  
    dataset = 'binary_pca_mnist'
    # dataset = 'cifar10'

    if dataset == 'binary_pca_mnist':      
        kT_strs = ['-6', '-5', '-5']
        kTs = [1e-6, 1e-5, 1e-5]
        widths = [30, 30, 72]
        L2_strs = ['-2', '0', '0']
        eps_factors = [300, 500, 6000]
        dataset_pathstub = ''
    
    if dataset == 'cifar10':
        kT_strs = ['e-5', 'e-4', 'e-4']
        kTs = [1e-5, 1e-4, 1e-4]
        widths = [30, 30, 80]
        L2_strs = ['e-3', '0', '0']
        dataset_pathstub = 'cifar10_'
    
    t_0s = [5000, 10000, 20000, 40000, 80000, 160000]
    
    # Make folder for saved data
    data_outdir = '{}/measured_data/aging_curves/'.format(base_path)
    dir_exists = os.path.isdir(data_outdir)
    if not dir_exists:
        os.mkdir(data_outdir)

    for count, kT in enumerate(kTs):
        kT_str = kT_strs[count]
        width = widths[count]
        eps_factor = eps_factors[count]
        L2_str = L2_strs[count]
        
        # Check if data exists
        data_outpath = '{}vH_curves_w{}_reg{}_kT{}_eps{}.pkl'.format(data_outdir, width, L2_str, kT_str, eps_factor)
        time_outpath = '{}vH_times_w{}_reg{}_kT{}_eps{}.pkl'.format(data_outdir, width, L2_str, kT_str, eps_factor)
        
        curve_data_exists = os.path.isfile(data_outpath)
        time_data_exists = os.path.isfile(time_outpath)
        data_exists = curve_data_exists and time_data_exists
        
        if data_exists:
            with open(data_outpath, 'rb') as infile:
                vH_curves = pickle.load(infile)
            
            with open(time_outpath, 'rb') as infile:
                time_arrays = pickle.load(infile)
            
            print('Loaded data from file.')
        
        else:
            print('Calculating data')
            vH_curves = []
            time_arrays = []
            
            for t_0 in t_0s:
                correlation_curves = []
                times_for_curves = []
                
                if dataset == 'binary_pca_mnist':
                    if width==72:
                        ics1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9])
                        ics2 = np.arange(10, 50)
                        ics = np.concatenate((ics1, ics2))
                    else:
                        ics = np.arange(50)
                else:
                    if width==30:
                        ics = np.arange(1, 10)
                    else:
                        ics = np.array([2, 3, 5])
                    
                for rep in ics:
                    # Correct change in naming convention
                    L2_str = L2_strs[count]
                    if L2_str=='-2' and rep>9:
                        L2_str = '0_010'
                    
                    model_name = '{}_b100_Langevin_relaxation_ic{}_w{}_kT{}_reg{}'.format(dataset, rep, width, kT_str, L2_str)            
                    input_dir = '{}models/{}/'.format(base_path, model_name)
                    params = load_model_params('{}{}_model_params.json'.format(
                                               input_dir, model_name))
                    models, epochs = load_models(input_dir, model_name, params, 
                                                 device)
                    
                    # Calculate a reasonable epsilon
                    epsilon = calc_epsilon_for_van_Hove_calc(models[-1], eps_factor)
                    epsilon_str = str(eps_factor)
                    
                    # Extract the model at timestep t_0
                    timesteps = np.array(epochs)
                    if timesteps[-1]<=t_0:
                        print('WARNING: ')
                        print('kT {} DOES NOT REACH {} TIMESTEPS'.format(kT, t_0))
                    else:
                        idx = np.where(timesteps>t_0)[0][0]
                        
                    epoch_seg = epochs[idx:]
                    model_seg = models[idx:]
                    
                    epoch_0 = epoch_seg[0]
                    model_0 = model_seg[0]
                            
                    self_correlations = []
                    epochs_for_plotting = []
                    
                    for i in range(1, len(epoch_seg)):
                        epoch = epoch_seg[i]
                        model = model_seg[i]
                        
                        epochs_for_plotting.append(epoch)
                        
                        self_correlation = mean_self_van_Hove_correlation(model_0, 
                                                                          epoch_0,
                                                                          model, 
                                                                          epoch,
                                                                          epsilon)
                        
                        self_correlations.append(self_correlation)
                    
                    correlation_curves.append(np.array(self_correlations))
                    times_for_curves.append(np.array(epochs_for_plotting))
                
                # Average over repetitions, accounting for different time grids
                list_of_series = []
                for i in range(len(correlation_curves)):
                    curve = correlation_curves[i]
                    time = times_for_curves[i]
                    series = pd.Series(curve, index=time)
                    list_of_series.append(series)
                
                corr_df = pd.DataFrame(list_of_series).transpose()
                correlation_df_no_nans = corr_df[~corr_df.isnull().any(axis=1)]
                mean_corr_curve = correlation_df_no_nans.mean(axis=1)
                
                vH_curves.append(np.array(mean_corr_curve))
                time_arrays.append(np.array(mean_corr_curve.index))
            
            # Save data
            with open(data_outpath, 'wb') as outfile:
                pickle.dump(vH_curves, outfile)
                
            with open(time_outpath, 'wb') as outfile:
                pickle.dump(time_arrays, outfile)
                
        # Plot decorrelation curve
        plot_output_dir = '{}plots/van_Hove_curves/'.format(base_path)
        dir_exists = os.path.isdir(plot_output_dir)
        if not dir_exists:
            os.mkdir(plot_output_dir) 

        fig, ax = plt.subplots()
        for i in range(len(vH_curves)):
            self_correlations = vH_curves[i]
            timestep_array = time_arrays[i]
            t_0 = t_0s[i]
            plt.plot(timestep_array-timestep_array[0], self_correlations, 
                     label=r'$t_0$={}'.format(t_0), 
                     c=plt.cm.Spectral(1.0-i/len(vH_curves)))
            
        # plt.axhline(y=0, color='k', linestyle='dashed', linewidth=1.0)
        # plt.title('Mean self van Hove correlation')
        # plt.ylabel(r'Mean van Hove correlation')
        # plt.xlabel('Timesteps')
        plt.xscale('log')
        # plt.legend(loc='lower left')
        # if dataset == 'binary_pca_mnist':
        #     if width==30 and L2_str=='0':
        #         plt.xlim(left=5e3)
        #         plt.ylim(bottom=0.4)
        #     if width==30 and L2_str=='-2':
        #         plt.xlim(left=1e4)
        #         plt.ylim(bottom=0.8)
        #     else:
        #         plt.xlim(left=5e3)
        #         plt.ylim(bottom=0.2)
        
        plt.savefig('{}{}aging_mean_van_Hove_kT{}_w{}_reg{}_lnt_epsilon{}.pdf'.format(
        plot_output_dir, dataset_pathstub, kT_str, width, L2_str, epsilon_str), bbox_inches='tight')

def calc_weight_and_loss_distributions(device, base_path):
    L2s = [0.1, 0.4]
    L2_strings = ['0_010', '0_040']
    
    width = 30
    b = 100
    
    model_input_dir = '{}models/inherent_structures/'.format(base_path)
      
    w_dists = []
    L_dists = []
    
    for L2_count, L2 in enumerate(L2s):
        L2_str = L2_strings[L2_count]
        
        w_dist = []
        L_dist = []
        
        for rep in range(10):
            name_stub = 'binary_pca_mnist_ic'
            model_name = '{}{}_w{}_b{}_reg{}'.format(name_stub, rep, width, 
                                                      b, L2_str)
            input_dir = '{}models/{}/'.format(base_path, model_name)
            params = load_model_params('{}/{}_model_params.json'.format(
                                       input_dir, model_name))
            
            model, _ = load_final_model(model_input_dir, model_name, params, device)
            
            for param in model.parameters():
                param_array = param.detach().numpy()
                param_array[np.abs(param_array)<1e-20] = 0.0
                param_list = list(param_array.flatten())
                w_dist += param_list
            
            loss_input_path = '{}measured_data/{}/train_loss.txt'.format(base_path, model_name)
            losses = np.loadtxt(loss_input_path)
            L_dist.append(losses[-1, 1])
        
        w_dists.append(w_dist)
        L_dists.append(L_dist)
    
    plot_output_dir = '{}plots/inherent_structure_clusters/'.format(base_path)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)
    
    N_big = 100
    
    fig, ax = plt.subplots()
    plt.hist(w_dists[1], bins=N_big, label=r'$L_2$=0.04', align='left', histtype='step')
    plt.hist(w_dists[0], bins=N_big, label=r'$L_2$=0.01', histtype='step')
    plt.xlim(xmin=-0.25, xmax=0.25)
    plt.xticks([-0.2, 0, 0.2])
    plt.yticks([0, 25000, 50000])
    plt.legend()
    plt.savefig('{}weight_distribution.pdf'.format(plot_output_dir), bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.hist(w_dists[0], bins=N_big, label=r'$L_2$=0.01')
    plt.savefig('{}weight_distribution_L2_0_01.pdf'.format(plot_output_dir), bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.hist(w_dists[1], bins=N_big, label=r'$L_2$=0.04')
    plt.savefig('{}weight_distribution_L2_0_04.pdf'.format(plot_output_dir), bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.scatter([0.01]*len(w_dists[0]), w_dists[0], c='b')
    plt.scatter([0.04]*len(w_dists[1]), w_dists[1], c='b')
    plt.savefig('{}weight_distribution_L2_scatter.pdf'.format(plot_output_dir), bbox_inches='tight')
    
    # Loss stuff    
    fig, ax = plt.subplots()
    plt.hist(L_dists[0], bins=10, label=r'$L_2$=0.01')
    plt.hist(L_dists[1], bins=10, label=r'$L_2$=0.04')
    plt.legend()
    plt.savefig('{}loss_distribution.pdf'.format(plot_output_dir), bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.hist(L_dists[0], bins=10, histtype='step', align='left', label=r'$L_2$=0.01')
    plt.legend()
    plt.xticks([0.08, 0.081, 0.082])
    plt.xlim([0.0799, 0.0824])
    plt.yticks([0, 1.5, 3])
    plt.savefig('{}loss_distribution_L2_0_01.pdf'.format(plot_output_dir), bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.hist(L_dists[1], bins=10, histtype='step', align='left', label=r'$L_2$=0.04')
    plt.legend()
    plt.xticks([0, 0.5, 1])
    plt.xlim([0, 1])
    plt.yticks([0, 5, 10])
    plt.savefig('{}loss_distribution_L2_0_04.pdf'.format(plot_output_dir), bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.scatter([0.01]*len(L_dists[0]), L_dists[0], c='b')
    plt.scatter([0.04]*len(L_dists[1]), L_dists[1], c='b')
    plt.savefig('{}loss_distribution_L2_scatter.pdf'.format(plot_output_dir), bbox_inches='tight')

def find_inherent_structures(device, base_path):
    L2s = [0.1, 0.4]
    L2_strings = ['0_010', '0_040']
    
    width = 30
    b = 100
    
    model_output_dir = '{}models/inherent_structures/'.format(base_path)
    dir_exists = os.path.isdir(model_output_dir)
    if not dir_exists:
        os.mkdir(model_output_dir)  
    
    for rep in range(10):       
        for L2_count, L2 in enumerate(L2s):
            L2_str = L2_strings[L2_count]
            
            model_name = 'binary_pca_mnist_ic{}_w{}_b{}_reg{}'.format(rep, 
                          width, b, L2_str)
            input_dir = '{}models/{}/'.format(base_path, model_name)
            params = load_model_params('{}/{}_model_params.json'.format(
                                       input_dir, model_name))
            model, epoch = load_final_model(input_dir, model_name, params, 
                                          device)
                    
            batch_size = -1
            dataset = params['dataset']
            
            (train_dataloader, test_dataloader, training_data, test_data, 
                N_inputs, N_outputs) = load_dataset(dataset, batch_size, 
                                                    base_path)
                
            # Define loss function 
            loss_function = params['loss_function']
            learning_rate = params['learning_rate']
            if loss_function == 'CrossEntropy':
                loss_fn = nn.BCELoss()
                w_decay = params['L2_penalty']
                
            elif loss_function == 'MSELoss':
                loss_fn = nn.MSELoss()
                w_decay = params['L2_penalty']
                
            elif loss_function == 'Hinge':
                loss_fn = nn.HingeEmbeddingLoss()
                w_decay = params['L2_penalty']
                
            elif loss_function == 'quadratic_hinge':
                loss_fn = make_quadratic_hinge_loss()
                w_decay = params['L2_penalty']
                  
            else:
                print('PROVIDE A LOSS FUNCTION')
            
            epochs = 100
            GD_losses = []
            l = evaluate_loss(train_dataloader, model, loss_fn, device, w_decay)
            GD_losses.append(l)
            
            optimizer = torch.optim.SGD(model.parameters(), 
                                        lr=learning_rate, 
                                        weight_decay=w_decay)
    
            for t in range(epochs): 
                print('Training time step {}'.format(t))
                train(train_dataloader, model, loss_fn, optimizer, device)
                l = evaluate_loss(train_dataloader, model, loss_fn, device, w_decay)
                GD_losses.append(l)
            
            
            # Plot gradient descent to check it works
            plot_output_dir = '{}plots/{}/'.format(base_path, model_name)
            dir_exists = os.path.isdir(plot_output_dir)
            if not dir_exists:
                os.mkdir(plot_output_dir)
                
            fig, ax = plt.subplots()
            plt.plot(GD_losses)
            plt.title('Gradient descent')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.yscale('log')
            plt.savefig('{}{}_GD_training_loss.pdf'.format(plot_output_dir, 
                        model_name), bbox_inches='tight')
                
            # Save model
            torch.save(model.state_dict(), 
                       '{}{}_model_epoch_{}.pth'.format(model_output_dir, 
                                                        model_name, t))

def calc_net_dist(model, model2):
    dist = 0.0
    
    for x, y in zip(model.state_dict().values(), model2.state_dict().values()):
        dist += ((x - y)**2).sum()
    
    dist = float(torch.sqrt(dist))
    
    return dist

def calc_inherent_structure_dist_mats(device, base_path):
    L2s = [0.1, 0.4]
    L2_strings = ['0_010', '0_040']
    
    width = 30
    b = 100
    
    model_input_dir = '{}models/inherent_structures/'.format(base_path)
      
    for L2_count, L2 in enumerate(L2s):
        L2_str = L2_strings[L2_count]
        
        model_names = []
        for rep in range(10):
            name_stub = 'binary_pca_mnist_ic'
            model_name = '{}{}_w{}_b{}_reg{}'.format(name_stub, rep, width, 
                                                      b, L2_str)
            model_names.append(model_name)
        
        N = len(model_names)
        dist_mat = np.zeros((N, N))
        
        
        for i in range(len(model_names)):
            model_name = model_names[i]
            input_dir = '{}models/{}/'.format(base_path, model_name)
            params = load_model_params('{}/{}_model_params.json'.format(
                                       input_dir, model_name))
            
            model, _ = load_final_model(model_input_dir, model_name, params, device)
            
            for j in range(i):
                model_name = model_names[j]
                input_dir = '{}models/{}/'.format(base_path, model_name)
                params = load_model_params('{}/{}_model_params.json'.format(
                                           input_dir, model_name))
                
                model2, _ = load_final_model(model_input_dir, model_name, params, device)
                
                dist = calc_net_dist(model, model2)
                dist_mat[i, j] = dist
        
        # Make distance matrix symmetric
        dist_mat = np.maximum(dist_mat, dist_mat.transpose())
        
        # Save distance matrix
        output_dir = '{}measured_data/inherent_structures/'.format(base_path)
        dir_exists = os.path.isdir(output_dir)
        if not dir_exists:
            os.mkdir(output_dir) 
                
        dist_outpath = '{}dist_matrix_L2{}.txt'.format(output_dir, L2_str)
        np.savetxt(dist_outpath, dist_mat)

def network_distance_distribution(device, base_path): 
    L2s = [0.01, 0.04]
    L2_strings = ['0_010', '0_040']
  
    for L2_count, L2 in enumerate(L2s):
        L2_str = L2_strings[L2_count]
    
        dist_outpath = '{}measured_data/inherent_structures/dist_matrix_L2{}.txt'.format(base_path, L2_str)
        try:
            dist_mat = np.loadtxt(dist_outpath)
            print('Loading ', dist_outpath.split('/')[-1])
        except OSError:
            print('File not found: ', dist_outpath.split('/')[-1])
            continue
        
        # Plot distance distributions
        dist_list = []
        for i in range(dist_mat.shape[0]):
            for j in range(i):
                dist_list.append(dist_mat[i, j])
        
        # Plot on linear scale
        plot_output_dir = '{}plots/inherent_structure_clusters/'.format(base_path)
        dir_exists = os.path.isdir(plot_output_dir)
        if not dir_exists:
            os.mkdir(plot_output_dir)  
    
        fig, ax = plt.subplots()
        plt.hist(dist_list, bins=100, histtype='step', label=r'$L_2$={}'.format(L2))
        plt.legend()
        if L2_count == 0:
            plt.xticks([5.5, 5.75, 6])
            plt.yticks([0, 2, 4])
        else:
            plt.xticks([-0.5, 0, 0.5])
            plt.yticks([0, 20, 40])
        # plt.xlabel('Inter-network distance')
        # plt.ylabel('Frequency')
        # plt.title('Distance distribution, L2{}'.format(L2_str))
        plt.savefig('{}distance_distribution_L2_{}.pdf'.format(plot_output_dir, L2_str), bbox_inches='tight')


##################
#####  Main  #####
##################   

if __name__=='__main__':
    base_path = './'
    
    if torch.cuda.is_available():
        device = "cuda"
        print('Using GPU.')
    else:
        device = "cpu"
        print('Using CPU.')
    
    if len(sys.argv)>1:
        actions = []
        if len(sys.argv)==3:
            print('Running with 3 arguments.')
            for x in sys.argv:
                print(x)
            try:
                actions.append(sys.argv[1])
                rep = int(sys.argv[2])
            except ValueError:
                for arg in sys.argv[1:]:
                    actions.append(arg)
        else:
            for arg in sys.argv[1:]:
                actions.append(arg)
                
        print('Running with user provided arguments')
    else:
        actions = ['plot_two_T_tau90_fit']
        print('Running with default arguments')    
    
    if 'mean_GD_loss_phase_diag_stage1' in actions:
        mean_GD_loss_phase_diag_stage1(device, base_path, rep)
        
    if 'mean_GD_loss_phase_diag_stage2' in actions:
        mean_GD_loss_phase_diag_stage2(device, base_path)
        
    if 'plot_loss_vs_N_with_ics' in actions:
        plot_loss_vs_N_with_ics(device, base_path)
    
    if 'plot_loss_vs_L2_with_alphas' in actions:
        plot_loss_vs_L2_with_alphas(device, base_path)
    
    if 'plot_two_T_tau90_fit' in actions:
        plot_two_T_tau90_fit(device, base_path)
    
    if 'plot_overlap_with_repeats' in actions:
        plot_overlap_with_repeats(device, base_path)
        
    if 'plot_tau90_vs_kT_with_repeats' in actions:
        plot_tau90_vs_kT_with_repeats(device, base_path)
    
    if 'plot_MSD_vs_kT' in actions:
        plot_MSD_vs_kT(device, base_path)
    
    if 'plot_alpha2' in actions:
        plot_alpha2(device, base_path)
    
    if 'plot_disp_dist_multiple_temps' in actions:
        plot_disp_dist_multiple_temps(device, base_path)
    
    if 'plot_mean_van_Hove_Langevin_curves_for_aging' in actions:
        plot_mean_van_Hove_Langevin_curves_for_aging(device, base_path)
    
    if 'calc_weight_and_loss_distributions' in actions:
        calc_weight_and_loss_distributions(device, base_path)
        
    if 'find_inherent_structures' in actions:
        find_inherent_structures(device, base_path)
        
    if 'calc_inherent_structure_dist_mats' in actions:
        calc_inherent_structure_dist_mats(device, base_path)
    
    if 'network_distance_distribution' in actions:
        network_distance_distribution(device, base_path)
        
        
        