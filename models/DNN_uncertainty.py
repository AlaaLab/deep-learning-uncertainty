
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Base classes for feedforward, convolutional and recurrent 
# neural network (DNN, CNN, RNN) models in pytorch
# ---------------------------------------------------------

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
from torch.autograd import grad
import torch.nn.functional as F
import scipy.stats as st

from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import time

from models.base_models import DNN

torch.manual_seed(1) 


class MCDP_DNN(DNN):
    
    def __init__(self, 
                 dropout_prob=0.5,
                 dropout_active=True,                  
                 n_dim=1, 
                 num_layers=2, 
                 num_hidden=200,
                 output_size=1,
                 activation="ReLU", 
                 mode="Regression"):
        
        super(MCDP_DNN, self).__init__()
        
        self.dropout_prob   = dropout_prob 
        self.dropout        = nn.Dropout(p=dropout_prob)
        self.dropout_active = True


    def forward(self, X):
        
        _out= self.dropout(self.model(X))  
        
        return _out

    
    def predict(self, X, alpha=0.1, MC_samples=100):
        
        z_c         = st.norm.ppf(1-alpha/2)
        X           = torch.tensor(X.reshape((-1, self.n_dim))).float()
        samples_    = [self.forward(X).detach().numpy() for u in range(MC_samples)]
        pred_sample = np.concatenate(samples_, axis=1)
        pred_mean   = np.mean(pred_sample, axis=1)  
        pred_std    = z_c * np.std(pred_sample, axis=1)         

        return pred_mean, pred_std     
