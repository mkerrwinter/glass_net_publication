#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:56:02 2021

@author: mwinter
"""

import torch
from torch import nn

# Define model class
class NeuralNetwork(nn.Module):
    def __init__(self, n_in=28*28, n_out=10, h_layer_widths=[512], bias=False,
                 gaussian_prior=False, sigmoid_output=False):
        super(NeuralNetwork, self).__init__()

        self.flatten = nn.Flatten()

        layer_widths = [n_in] + h_layer_widths

        layers = []
        for count in range(len(layer_widths)-1):
            layers.append(
                nn.Linear(int(layer_widths[count]), int(layer_widths[count+1]),
                          bias=bias)
                         )

            layers.append(nn.ReLU())

        layers.append(nn.Linear(int(layer_widths[-1]), n_out, bias=bias))
        
        if sigmoid_output:
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        
        if gaussian_prior:
            self.net.apply(self.init_weights)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.25)
