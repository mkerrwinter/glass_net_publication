#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:22:04 2022

@author: mwinter
"""

# A script to produce a PCA version of MNIST for binary classification, where
# odd and even numbers make up the two classes. The original data, in the 
# files mnist_train.csv and mnist_test.csv are from this webpage: 
# https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_test.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle

def get_PCA_components(F_df, order):
    
    pca = decomposition.PCA()
    pca.n_components = F_df.shape[1]
    pca_data = pca.fit_transform(F_df)

    exp_var = pca.explained_variance_ratio_
    exp_var_short = exp_var[:40]
    
    # Plot explained variance
    fig, ax = plt.subplots()
    plt.bar(range(0, len(exp_var_short)), exp_var_short, align='center')
    plt.ylabel('Explained variance')
    plt.xlabel('PCA index')
    plt.title('PCA explaine variance')
    plt.yscale('log')
    
    return pca_data[:, :order], pca


base_path = './'
data_name = 'MNIST'
# data_name = 'CIFAR_10'
    

input_dir = base_path + 'data/{}/'.format(data_name)
   
d0 = pd.read_csv(input_dir + '{}_train.csv'.format(data_name))

# save the labels into a variable labels.
labels = d0['label']

# Drop the label feature and store the pixel data in data.
data = d0.drop("label", axis=1)

# Remove mean and scale to unit variance
standardized_data = StandardScaler().fit_transform(data)

# Do PCA on training data
order = 10 
pca_data, pca_obj = get_PCA_components(standardized_data, order)

# Pickle PCA object
pkl_path = input_dir + 'pca_object.pkl'
with open(pkl_path, 'wb') as outfile:
    pickle.dump(pca_obj, outfile)

labels = np.expand_dims(labels, axis=1)
output_data = np.concatenate((labels, pca_data), axis=1)
cols = np.arange(output_data.shape[1])

output_train_df = pd.DataFrame(output_data, columns=cols)
    
# Do the same to the test set  
d0 = pd.read_csv(input_dir + '{}_test.csv'.format(data_name))

# save the labels into a variable labels.
labels = d0['label']

# Drop the label feature and store the pixel data in data.
data = d0.drop("label", axis=1)

# Remove mean and scale to unit variance
standardized_data = StandardScaler().fit_transform(data)

# PCA transform testing data
pca_data = pca_obj.transform(standardized_data)
pca_data = pca_data[:, :order]

labels = np.expand_dims(labels, axis=1)
output_data = np.concatenate((labels, pca_data), axis=1)
cols = np.arange(output_data.shape[1])

output_test_df = pd.DataFrame(output_data, columns=cols)

# Make low dimensional train and test sets with first 10 PCA components
# Note pandas .loc slicing includes start AND END points, but cols[:11] does not
mini_train = output_train_df.loc[:, cols[:11]]
mini_test = output_test_df.loc[:, cols[:11]]

mini_test.index = range(mini_test.shape[0])

# Also convert these into a pytorch friendly form
pt_train = []
pt_train_l = torch.zeros(mini_train.shape[0], 1)
cols = mini_train.columns

for row in range(mini_train.shape[0]):
    label = mini_train.loc[row, cols[0]]
    data_row = mini_train.loc[row, cols[1:]]
    data_row = np.expand_dims(data_row, axis=0)
    data_row = np.expand_dims(data_row, axis=0)
    if label%2==0:
        pt_train_l[row, 0] = 1
    else:
        pt_train_l[row, 0] = -1

    pt_train.append(torch.Tensor(data_row))

pt_train_tensor = torch.cat(pt_train)
pt_train_data = TensorDataset(pt_train_tensor, pt_train_l)

pt_test = []
pt_test_l = torch.zeros(mini_test.shape[0], 1)
cols = mini_test.columns

for row in range(mini_test.shape[0]):
    label = mini_test.loc[row, cols[0]]
    data_row = mini_test.loc[row, cols[1:]]
    data_row = np.expand_dims(data_row, axis=0)
    data_row = np.expand_dims(data_row, axis=0)
    if label%2==0:
        pt_test_l[row, 0] = 1
    else:
        pt_test_l[row, 0] = -1
        
    pt_test.append(torch.Tensor(data_row))

pt_test_tensor = torch.cat(pt_test)
pt_test_data = TensorDataset(pt_test_tensor, pt_test_l)

# Save pytorch data
data_output_dir = '{}data/binary_PCA_{}/'.format(base_path, data_name)
dir_exists = os.path.isdir(data_output_dir)
if not dir_exists:
    os.mkdir(data_output_dir)

fname = data_output_dir + 'binary_PCA_{}_train.pt'.format(data_name)
torch.save(pt_train_data, fname)

fname = data_output_dir + 'binary_PCA_{}_test.pt'.format(data_name)
torch.save(pt_test_data, fname)










