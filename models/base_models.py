
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Base classes for feedforward, convolutional and recurrent 
# neural network (DNN, CNN, RNN) models in pytorch
# ---------------------------------------------------------

# -------------------------------------
# |  TO DO:                           | 
# |  ------                           | 
# |  Loss functions file              |
# |  ADD EPOCHS                       |
# |  argument explanation for the DNN |
# |  Exception handling               |
# |  Multiple architectures in RNN    | 
# |  cmd arguments                    |
# |  logger, misc and config files    |
# -------------------------------------

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import torch
from torch.autograd import Variable 
import torch.nn.functional as nnf
from torch.utils.data import random_split
from torch.optim import SGD 
from torch.distributions import constraints
import torchvision as torchv
import torchvision.transforms as torchvt
from torch import nn
import torchvision.transforms as transforms
from torch.autograd import grad
import scipy.stats as st

from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import time

from utils.parameters import *

torch.manual_seed(1) 


class DNN(nn.Module):
    
    def __init__(self, 
                 n_dim=1, 
                 dropout_prob=0.0,
                 dropout_active=False,  
                 num_layers=2, 
                 num_hidden=200,
                 output_size=1,
                 activation="Tanh", 
                 mode="Regression"
                ):
        
        super(DNN, self).__init__()
        
        self.n_dim          = n_dim
        self.num_layers     = num_layers
        self.num_hidden     = num_hidden
        self.mode           = mode
        self.activation     = activation
        self.device         = torch.device('cpu') # Make this an option
        self.output_size    = output_size
        self.dropout_prob   = dropout_prob
        self.dropout_active = dropout_active  
        self.model          = build_architecture(self)


    def fit(self, X, y, learning_rate=1e-3, loss_type="MSE", batch_size=100, num_iter=500, verbosity=False):
        
        self.X           = torch.tensor(X.reshape((-1, self.n_dim))).float()
        self.y           = torch.tensor(y).float()
        
        loss_dict        = {"MSE": torch.nn.MSELoss}
        
        self.loss_fn     = loss_dict[loss_type](reduction='mean')
        self.loss_trace  = []     
        
        batch_size       = np.min((batch_size, X.shape[0]))
        
        optimizer        = torch.optim.Adam(self.parameters(), lr=learning_rate) 
        
        for _ in range(num_iter):

            batch_idx = np.random.choice(list(range(X.shape[0])), batch_size )
            
            y_pred    = self.model(self.X[batch_idx, :])
    
            self.loss = self.loss_fn(y_pred.reshape((batch_size, self.n_dim)), self.y[batch_idx].reshape((batch_size, self.n_dim)))
        
            self.loss_trace.append(self.loss.detach().numpy())
            
            if verbosity:
    
                print("--- Iteration: %d \t--- Loss: %.3f" % (_, self.loss.item()))
  
            self.model.zero_grad()
            
            optimizer.zero_grad()   # clear gradients for this training step
            self.loss.backward()    # backpropagation, compute gradients
            optimizer.step()

    
    
    def predict(self, X, numpy_output=True):
        
        X = torch.tensor(X.reshape((-1, self.n_dim))).float()

        if numpy_output:

            prediction = self.model(X).detach().numpy()

        else:

            prediction = self.model(X)    


        return prediction
    


