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