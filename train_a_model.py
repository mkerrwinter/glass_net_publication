#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:55:30 2021

@author: mwinter
"""

# Train and save a model

import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from NN import NeuralNetwork
import json
import os
import sys
import numpy as np

start = time.time()

# Load the params json that contains the widths of each layer
def load_model_params(filepath):
    with open(filepath) as infile:
        params = json.load(infile)

    return params

# Load existing models
def load_models(input_dir, model_name, params, device):
    files = os.listdir(input_dir)
    
    # order files
    epochs = []
    epoch_strs = []
    for file in files:
        file2 = file
        try:
            start, end = file2.split('_epoch_')
        except ValueError:
            continue
        
        if start != '{}_model'.format(model_name):
            continue
        
        epoch_str = end[:-4]
        epoch_strs.append(epoch_str)
        epoch = float(epoch_str)
        epochs.append(epoch)
    
    epochs, epoch_strs = (list(t) for t in zip(*sorted(zip(epochs, epoch_strs))))
    if len(epoch_strs)>1 and epoch_strs[1] == '0.0':
        epochs = epochs[1:]
        epoch_strs = epoch_strs[1:]
    
    # load models
    N_inputs = params['N_inputs']
    N_outputs = params['N_outputs']
    h_layer_widths = params['h_layer_widths']
    loss_function = params['loss_function']
    
    sigmoid_out = loss_function == 'CrossEntropy'
    
    models = []
    for epoch_str in epoch_strs:
        filename = '{}{}_model_epoch_{}.pth'.format(input_dir, model_name, 
                                                     epoch_str)
        model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs, 
                              h_layer_widths=h_layer_widths, 
                              sigmoid_output=sigmoid_out).to(device)
        model.load_state_dict(torch.load(filename, 
                                         map_location=torch.device(device)))
        
        # Check device
        if epoch_str == epoch_strs[0]:
            print('Current device = ', device)
            for l in range(0, len(model.net), 2):
                l_weight = model.net[l].weight
                print('Layer {} device '.format(l), l_weight.device)
        
        models.append(model)

    return models, epochs

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

def load_final_model(input_dir, model_name, params, device):
    files = os.listdir(input_dir)
    
    # order files
    epochs = []
    for file in files:
        file2 = file
        try:
            start, end = file2.split('_epoch_')
        except ValueError:
            continue
        
        if start != '{}_model'.format(model_name):
            continue
        
        epoch = int(end[:-4])
        epochs.append(epoch)
    
    epochs.sort()
    
    N_inputs = params['N_inputs']
    N_outputs = params['N_outputs']
    h_layer_widths = params['h_layer_widths']
    loss_function = params['loss_function']
    
    sigmoid_out = loss_function == 'CrossEntropy'
    
    # search backwards for first non-NaN network    
    final_idx = len(epochs)-1
    new_idx = final_idx
    init_idx = -final_idx
    print('Loading epoch ', epochs[final_idx])
    epoch = epochs[final_idx]
    filename = '{}{}_model_epoch_{}.pth'.format(input_dir, model_name, 
                                                 epoch)
    model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs, 
                          h_layer_widths=h_layer_widths, 
                          sigmoid_output=sigmoid_out).to(device)
    model.load_state_dict(torch.load(filename, 
                                     map_location=torch.device(device)))
    
    NaN_network = check_for_NaN_network(model)
    if not NaN_network:
        init_idx = final_idx
    
    while abs(final_idx-init_idx)>1:
        new_idx = int((final_idx+init_idx)/2.0)
        
        epoch = epochs[new_idx]
        filename = '{}/{}_model_epoch_{}.pth'.format(input_dir, model_name, 
                                                     epoch)
        model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs, 
                              h_layer_widths=h_layer_widths, 
                              sigmoid_output=sigmoid_out).to(device)
        model.load_state_dict(torch.load(filename, 
                                         map_location=torch.device(device)))
        
        NaN_network = check_for_NaN_network(model)
        
        # Check for epoch 0 NaN
        if NaN_network and new_idx==0:
            raise ValueError('ERROR: First epoch is a NaN network.')
        
        if NaN_network:
            final_idx = new_idx
        else:
            init_idx = new_idx
    
    epoch = epochs[init_idx]
    filename = '{}/{}_model_epoch_{}.pth'.format(input_dir, model_name, 
                                                     epoch)
    model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs, 
                          h_layer_widths=h_layer_widths, 
                          sigmoid_output=sigmoid_out).to(device)
    model.load_state_dict(torch.load(filename, 
                                     map_location=torch.device(device)))
                    
    return model, epoch

# Load a dataset
def load_dataset(dataset, batch_size, base_path, model_name=None, debug=False, 
                 loss_fn=None):
    # Download training and testing data from open datasets 
    if dataset=='downsampled_MNIST':
        data_output_dir = '{}data/downsampled_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'training_data.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'test_data.pt'
        test_data = torch.load(fname)
        
    elif dataset=='binary_MNIST':
        data_output_dir = '{}data/binary_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'binary_MNIST_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'binary_MNIST_test.pt'
        test_data = torch.load(fname)
        
    elif dataset=='binary_PCA_MNIST':
        data_output_dir = '{}data/binary_PCA_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'binary_PCA_MNIST_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'binary_PCA_MNIST_test.pt'
        test_data = torch.load(fname)
    
    elif dataset=='mini_binary_PCA_MNIST':
        data_output_dir = '{}data/binary_PCA_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'mini_binary_PCA_MNIST_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'mini_binary_PCA_MNIST_test.pt'
        test_data = torch.load(fname)
    
    elif dataset=='mini_binary_PCA_MNIST_2':
        data_output_dir = '{}data/binary_PCA_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'mini_binary_PCA_MNIST_train_2.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'mini_binary_PCA_MNIST_test_2.pt'
        test_data = torch.load(fname)
    
    elif dataset=='mini_binary_PCA_MNIST_2k':
        data_output_dir = '{}data/binary_PCA_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'mini_binary_PCA_MNIST_train_2k.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'mini_binary_PCA_MNIST_test_2k.pt'
        test_data = torch.load(fname)
    
    elif dataset=='binary_PCA_CIFAR_10':
        data_output_dir = '{}data/binary_PCA_CIFAR_10/'.format(base_path)
        
        fname = data_output_dir + 'binary_PCA_CIFAR_10_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'binary_PCA_CIFAR_10_test.pt'
        test_data = torch.load(fname)
    
    elif dataset=='binary_random':
        # Same as binary_PCA_MNIST but with random labels
        data_output_dir = '{}data/binary_random/'.format(base_path)
        
        fname = data_output_dir + 'binary_random_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'binary_random_test.pt'
        test_data = torch.load(fname)
    
    elif dataset=='random_binary':
        data_output_dir = '{}data/random_data/'.format(base_path)
        
        fname = data_output_dir + 'random_binary_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'random_binary_test.pt'
        test_data = torch.load(fname)
        
    elif dataset=='mini_pca_mnist':
        data_output_dir = '{}data/PCA_MNIST/'.format(base_path)
        
        fname = data_output_dir + 'mini_pca_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'mini_pca_test.pt'
        test_data = torch.load(fname)
        
    elif dataset=='F_to_M_noisy_-2':
        stem_1 = '/Users/mwinter/Documents/python_code/Learning_M/'
        stem_2 = 'data/F_to_M_data/noise_-2/'
        data_output_dir = stem_1 + stem_2
        
        fname = data_output_dir + 'MCT_train.pt'
        training_data = torch.load(fname)
        
        fname = data_output_dir + 'MCT_test.pt'
        test_data = torch.load(fname)
        
    else:
        print('PROVIDE A DATASET')
    
    # Get img size. data element is (tensor, lbl), image is tensor with 
    # shape either [1, rows, cols], or [rows, cols]. PCA vector is tensor with 
    # shape [1, rows]
    
    if dataset in ['mini_pca_mnist', 'teacher', 'F_to_M_noisy_-2']:
        sample_vec = training_data[0][0][0, :]
        N_inputs = sample_vec.shape[0]
        
    elif dataset in ['binary_MNIST', 'binary_PCA_MNIST', 
                     'mini_binary_PCA_MNIST', 'mini_binary_PCA_MNIST_2', 
                     'mini_binary_PCA_MNIST_2k', 'random_binary', 
                     'binary_random', 'binary_PCA_CIFAR_10']:
        sample_img = training_data[0][0]
        N_inputs = sample_img.shape[0]*sample_img.shape[1]
        
    else:
        sample_img = training_data[0][0][0, :, :]
        N_inputs = sample_img.shape[0]*sample_img.shape[1]
    
    if dataset in ['teacher', 'binary_MNIST', 'binary_PCA_MNIST', 
                   'mini_binary_PCA_MNIST', 'mini_binary_PCA_MNIST_2', 
                   'mini_binary_PCA_MNIST_2k', 'random_binary', 
                     'binary_random', 'binary_PCA_CIFAR_10']:
        N_outputs = len(training_data[0][1])
    
    elif dataset in ['F_to_M_noisy_-2']:
        N_outputs = training_data[0][1].shape[0]
    else:
        unique_labels = []
        for t in training_data:
            if len(t[1])==1:
                try:
                    label = t[1].item()
                except AttributeError:
                    label = t[1]
            else:
                label = np.argmax(t[1]).item()
                
            if label not in unique_labels:
                unique_labels.append(label)
        
        
        N_outputs = len(unique_labels)
    
    # For cross entropy shift labels to be 0, 1
    if loss_fn == 'CrossEntropy':
        training_tensor_list = []
        training_label_tensor = torch.zeros(len(training_data), 1)
        for idx in range(len(training_data)):
            a, b = training_data[idx]
            training_tensor_list.append(a)
            training_label_tensor[idx, 0] = max(0, b)
        
        training_tensor = torch.cat(training_tensor_list)
        training_data = torch.utils.data.TensorDataset(training_tensor, 
                                                       training_label_tensor)
        
        testing_tensor_list = []
        testing_label_tensor = torch.zeros(len(test_data), 1)
        for idx in range(len(test_data)):
            a, b = test_data[idx]
            testing_tensor_list.append(a)
            testing_label_tensor[idx, 0] = max(0, b)
        
        testing_tensor = torch.cat(testing_tensor_list)
        test_data = torch.utils.data.TensorDataset(testing_tensor, 
                                                       testing_label_tensor)
        
    # Deal with batch size = dataset size case
    set_b_for_test = False
    if batch_size==-1:
        batch_size = len(training_data)
        set_b_for_test = True
    elif batch_size>len(test_data):
        set_b_for_test = True
    
    if debug:
        shuffle_data = False
        print('Shuffle data turned off for debug mode.')
    else:
        shuffle_data = True
    
    train_dataloader = DataLoader(training_data, batch_size=batch_size, 
                                  shuffle=shuffle_data, drop_last=True)
    
    # Deal with batch size = dataset size case
    if set_b_for_test:
        batch_size = len(test_data)
        
    test_dataloader = DataLoader(test_data, batch_size=batch_size, 
                                 shuffle=shuffle_data, drop_last=True)
    
    return (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs)

# Define training process
def train(dataloader, model, loss_fn, optimizer, device):
        
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y) 

        for param in model.parameters():
            param.grad = None  

        loss.backward() 
        optimizer.step() 

# Define training process with Gaussian noise
def train_Langevin(dataloader, model, loss_fn, optimizer, device, kT, lr, 
                   N_examples):
        
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y) 

        for param in model.parameters():
            param.grad = None  

        loss.backward() 
        optimizer.step() 

        # Add noise given by temperature T
        strength = np.sqrt(2.0*kT*lr/N_examples)
        with torch.no_grad():
            for param in model.parameters():
                noise_tensor = torch.randn(param.size()) * strength
                noise_tensor = noise_tensor.to(device)
                param.add_(noise_tensor)

# Define testing process
def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): 
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

# Calculate value of loss function
def evaluate_loss(dataloader, model, loss_fn, device, L2):
    num_batches = len(dataloader)
    model.eval()
    loss = 0
    with torch.no_grad(): 
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()

    
    if num_batches>0:
        loss /= num_batches
    else:
        loss = np.inf
    
    # Add L2 regularisation
    L2_loss = 0.0
    if L2>0:
        for param in model.parameters():
            param_sq = param**2
            param_sq_sum = param_sq.sum()
            L2_loss += L2*param_sq_sum/2.0
    
        L2_loss = L2_loss.detach().numpy()
    
    loss += L2_loss
        
    return loss

# Save json of model params
def save_model_params(params, outpath):
    with open(outpath, 'w') as outfile:
        json.dump(params, outfile)

def split_path(ic_path):    
    chunks = ic_path.split('/')
    ic_path += '/'

    return ic_path, chunks[-1]

def make_quadratic_hinge_loss():
    
    def quadratic_hinge(output, target):
        Delta = 1.0-target*output
        zeros = torch.zeros_like(Delta)
        
        max_Delta = torch.max(zeros, Delta)
        
        sq_max_Delta = max_Delta*max_Delta
        
        return 0.5*torch.mean(sq_max_Delta)
    
    return quadratic_hinge

if __name__ == "__main__":
    base_path = './'
    Langevin_dynamics = False
    
    # Get cpu or gpu device for training. Note: my MacBook doesn't have CUDA gpu    
    if torch.cuda.is_available():
        device = "cuda"
        print('Training with CUDA GPU.')
    else:
        device = "cpu"
        print('Training with CPU.')
    
    # Load any command line arguments (sys.argv has len 1 if no args provided)
    if len(sys.argv)==7:
        model_stub = sys.argv[6]
        
        if model_stub in ['binary_pca_mnist', 
                          'binary_pca_mnist_relaxation', 
                          'binary_pca_mnist_b100_relaxation']:
            w = sys.argv[1]
            b = sys.argv[2]
            L2 = sys.argv[3]
            
            if sys.argv[4]=='True':
                start_from_existing_model = True
            else:
                start_from_existing_model = False
            
            if sys.argv[5]=='True':
                start_from_param_file = True
            else:
                start_from_param_file = False
                        
            model_name = model_stub + '_w{}_b{}_reg{}'.format(w, b, L2)
        
        elif model_stub in ['binary_pca_mnist_ic']:
            r = sys.argv[1]
            w = sys.argv[2]
            L2 = sys.argv[3]
            
            if sys.argv[4]=='True':
                start_from_existing_model = True
            else:
                start_from_existing_model = False
            
            if sys.argv[5]=='True':
                start_from_param_file = True
            else:
                start_from_param_file = False
                        
            model_name = model_stub + '{}_w{}_b100_reg{}'.format(r, w, L2)
   
        else:
            print('Unrecognised input arguments')
        
        print('Running with user provided arguments')
        print(sys.argv)
    
    elif len(sys.argv) == 8:  
        model_stub = sys.argv[7]
        
        if model_stub in ['binary_pca_mnist_ic',
                          'binary_pca_mnist_b100_relaxation_ic',
                          'cifar10_b100_relaxation_ic',
                          'cifar10_ic']:
            r = sys.argv[1]
            w = sys.argv[2]
            b = sys.argv[3]
            L2 = sys.argv[4]
            
            if sys.argv[5]=='True':
                start_from_existing_model = True
            else:
                start_from_existing_model = False
            
            if sys.argv[6]=='True':
                start_from_param_file = True
            else:
                start_from_param_file = False
                        
            model_name = model_stub + '{}_w{}_b{}_reg{}'.format(r, w, b, L2)
        
        elif model_stub in ['binary_pca_mnist_b100_Langevin_relaxation_ic',
                            'binary_pca_mnist_Langevin_ic']:
            rep = sys.argv[1]
            w = sys.argv[2]
            kT = sys.argv[3]
            L2 = sys.argv[4]
            
            if sys.argv[5]=='True':
                start_from_existing_model = True
            else:
                start_from_existing_model = False
            
            if sys.argv[6]=='True':
                start_from_param_file = True
            else:
                start_from_param_file = False
            
            model_name = model_stub + '{}_w{}_kT{}_reg{}'.format(rep, w, kT, L2)
            Langevin_dynamics = True
            
        else:
            r = sys.argv[1]
            w = sys.argv[2]
            b = sys.argv[3]
            L2 = sys.argv[4]
            
            if sys.argv[5]=='True':
                start_from_existing_model = True
            else:
                start_from_existing_model = False
            
            if sys.argv[6]=='True':
                start_from_param_file = True
            else:
                start_from_param_file = False
            
            model_stub = sys.argv[7]
                        
            model_name = model_stub + '{}_w{}_b{}_reg{}'.format(r, w, b, L2)
                
    else:
        params = {}
        w = 100
        d = 6
        b = 100
        start_from_existing_model = False
        start_from_param_file = True
        model_name = 'default_model'

        params['batch_size'] = b
        hidden_layers = d*[w]
        params['h_layer_widths'] = hidden_layers
        
        print('Running with default arguments')

    ### Load or create model ###
    start_epoch = -1
    
    model_output_dir = '{}models/{}/'.format(base_path, model_name)
    
    if start_from_existing_model:
        print('Loading existing model {}'.format(model_name))
        input_dir = '{}models/{}/'.format(base_path, model_name)
        params = load_model_params('{}{}_model_params.json'.format(input_dir, 
                                                                model_name))
        
        hidden_layers = params['h_layer_widths']
        loss_function = params['loss_function']
        learning_rate = params['learning_rate']
        L2_penalty = params['L2_penalty']
        batch_size = params['batch_size']
        dataset = params['dataset']
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path, 
                                                loss_fn=loss_function)
        
        model, start_epoch = load_final_model(input_dir, model_name, params, 
                                              device)
        
        print('Starting from epoch {}'.format(start_epoch+1))
    
    elif start_from_param_file:
        print('Creating model from parameter file')
        input_dir = '{}models/{}/'.format(base_path, model_name)
        params = load_model_params('{}{}_model_params.json'.format(input_dir, 
                                                                model_name))
        hidden_layers = params['h_layer_widths']
        
        loss_function = params['loss_function']
        learning_rate = params['learning_rate']
        L2_penalty = params['L2_penalty']
        batch_size = params['batch_size']
        dataset = params['dataset']
    
        sigmoid_out = loss_function == 'CrossEntropy'
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path,
                                                model_name, loss_fn=loss_function)
                
        if 'initial_condition_path' in params:
            ic_path = params['initial_condition_path']
            ic_input_dir, ic_model_name = split_path(ic_path)
            print('input_dir = ', ic_input_dir)
            print('model_name = ', ic_model_name)
            model, _ = load_final_model(ic_input_dir, ic_model_name, params, 
                                              device)
            print('Loading from initial condition:')
            print(ic_path)
        else:
            model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs,
                              h_layer_widths=hidden_layers, 
                              sigmoid_output=sigmoid_out).to(device)
        
        # Save initial state
        dir_exists = os.path.isdir(model_output_dir)
        if not dir_exists:
            os.mkdir(model_output_dir)
            
        torch.save(model.state_dict(),'{}{}_model_epoch_0.pth'.format(
            model_output_dir, model_name))
        print('Saved PyTorch Model State to '+
              '{}{}_model_epoch_0.pth'.format(model_output_dir, model_name))
        
    else:
        print('Creating new model.')
                
        loss_function = 'CrossEntropy'
        learning_rate = 1e-3
        batch_size = b
        width = w
        depth = d
        dataset = 'binary_PCA_MNIST'
        params['L2_penalty'] = 0.0
        
        sigmoid_out = loss_function == 'CrossEntropy'
        
        (train_dataloader, test_dataloader, training_data, test_data, 
            N_inputs, N_outputs) = load_dataset(dataset, batch_size, base_path, 
                                                loss_fn=loss_function)
        
        hidden_layers = [width]*depth
        model = NeuralNetwork(n_in=N_inputs, n_out=N_outputs,
                              h_layer_widths=hidden_layers, 
                              sigmoid_output=sigmoid_out).to(device)
        
        # Save initial state
        model_output_dir = '{}models/{}/'.format(base_path, model_name)
        dir_exists = os.path.isdir(model_output_dir)
        if not dir_exists:
            os.mkdir(model_output_dir)
            
        torch.save(model.state_dict(),'{}{}_model_epoch_0.pth'.format(
            model_output_dir, model_name))
        print('Saved PyTorch Model State to '+
              '{}{}_model_epoch_0.pth'.format(model_output_dir, model_name))
        
    # Check data_output_dir exists
    data_output_dir = '{}measured_data/{}/'.format(base_path, model_name)
    dir_exists = os.path.isdir(data_output_dir)
    if not dir_exists:
        os.mkdir(data_output_dir)
    
    # Check how many parameters the network has
    N_total_params = sum(p.numel() for p in model.parameters() 
                               if p.requires_grad)       
    
    # Define loss function
    if 'intensive_L2' in params:
        if params['intensive_L2']:
            int_factor = N_total_params
        else:
            int_factor = 1
    else:
        int_factor = 1
        
    if loss_function == 'CrossEntropy':
        loss_fn = nn.BCELoss()
        w_decay = params['L2_penalty']/int_factor
    
    elif loss_function == 'MSELoss':
        loss_fn = nn.MSELoss()
        w_decay = params['L2_penalty']/int_factor
    
    elif loss_function == 'Hinge':
        loss_fn = nn.HingeEmbeddingLoss()
        w_decay = params['L2_penalty']/int_factor
            
    elif loss_function == 'quadratic_hinge':
        loss_fn = make_quadratic_hinge_loss()
        w_decay = params['L2_penalty']/int_factor
    
    else:
        print('PROVIDE A LOSS FUNCTION')
    
    # Define the optimizer. weight_decay>0 adds L2 regularisation to weights
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
                                weight_decay=w_decay)
        
    # Save parameters
    params['dataset'] = dataset
    params['loss_function'] = loss_function
    params['N_inputs'] = N_inputs
    params['N_outputs'] = N_outputs
    
    save_model_params(params,'{}{}_model_params.json'.format(
            model_output_dir, model_name))
    
    # Calculate number of parameters and datapoints
    N_total_params = sum(p.numel() for p in model.parameters() 
                               if p.requires_grad)
    N_examples = len(training_data)
        
    ### Train for some number of epochs ###
    epochs = 500
    test_loss = []
    train_loss = []
    epoch_list = []
    timechecks_list = []
    for t in range(epochs):
        t += start_epoch + 1
        if Langevin_dynamics:
            train_Langevin(train_dataloader, model, loss_fn, optimizer, device, 
                           kT, learning_rate, N_examples)
        else:
            train(train_dataloader, model, loss_fn, optimizer, device)
        
        # Save every Nth epoch
        time_check = t - start_epoch
        save = False
        if time_check<100:
            save = True
        if (time_check)>=100 and (time_check)<1000 and (time_check)%10==0:
            save = True
        if (time_check)>=1000 and (time_check)%100==0:
            save = True
        
        if save:
            l = evaluate_loss(train_dataloader, model, loss_fn, 
                                            device, w_decay)
            train_loss.append(l)
            
            torch.save(model.state_dict(), 
                       '{}{}_model_epoch_{}.pth'.format(model_output_dir, 
                                                               model_name, t))
            
            train_loss_outpath = data_output_dir + 'train_loss.txt'
            out_string = '{} {}\n'.format(t, train_loss[-1])
            try:
                with open(train_loss_outpath, 'a') as f:
                        f.write(out_string)
            except FileNotFoundError:
                with open(train_loss_outpath, 'w') as f:
                        f.write(out_string)
        
            test_loss.append(evaluate_loss(test_dataloader, model, loss_fn, 
                                            device, 0.0))
            test_loss_outpath = data_output_dir + 'test_loss.txt'
            out_string = '{} {}\n'.format(t, test_loss[-1])
            try:
                with open(test_loss_outpath, 'a') as f:
                        f.write(out_string)
            except FileNotFoundError:
                with open(test_loss_outpath, 'w') as f:
                        f.write(out_string)
                        
            epoch_list.append(t)
            
            print(f"Saved data at epoch {t+1}\n-----------------------------")
            
            if np.isnan(l):
                print('WARNING: Got a NaN loss. Ending training.')
                break
    
    print("Done!")
    
    # Rescale time to be in timesteps not epochs
    epoch_array = np.array(epoch_list)
    N_steps = len(train_dataloader)
    timestep_array = epoch_array*N_steps
    
    # Plot losses
    plot_output_dir = '{}plots/{}/'.format(base_path, model_name)
    dir_exists = os.path.isdir(plot_output_dir)
    if not dir_exists:
        os.mkdir(plot_output_dir)

    fig, ax = plt.subplots()
    plt.plot(timestep_array, test_loss)
    plt.title('Loss on test set')
    plt.xlabel('Time/steps')
    plt.yscale('log')
    plt.savefig('{}{}_testing_loss.pdf'.format(plot_output_dir, model_name), 
                bbox_inches='tight')
    
    fig, ax = plt.subplots()
    plt.plot(timestep_array, train_loss)
    plt.title('Loss on training set')
    plt.xlabel('Time/steps')
    plt.yscale('log')
    plt.savefig('{}{}_training_loss.pdf'.format(plot_output_dir, model_name), 
                bbox_inches='tight')
    
    
    print('Running time = ', time.time()-start)







