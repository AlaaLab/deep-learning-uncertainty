
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Helper functions and utilities for deep learning models
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
import torchvision.transforms as transforms
from torch.autograd import grad
import scipy.stats as st

from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import time

torch.manual_seed(1) 


ACTIVATION_DICT = {"ReLU": torch.nn.ReLU(), "Hardtanh": torch.nn.Hardtanh(),
                   "ReLU6": torch.nn.ReLU6(), "Sigmoid": torch.nn.Sigmoid(),
                   "Tanh": torch.nn.Tanh(), "ELU": torch.nn.ELU(),
                   "CELU": torch.nn.CELU(), "SELU": torch.nn.SELU(), 
                   "GLU": torch.nn.GLU(), "LeakyReLU": torch.nn.LeakyReLU(),
                   "LogSigmoid": torch.nn.LogSigmoid(), "Softplus": torch.nn.Softplus()}


def build_architecture(base_model):

    modules          = []

    if base_model.dropout_active:

        modules.append(torch.nn.Dropout(p=base_model.dropout_prob))

    modules.append(torch.nn.Linear(base_model.n_dim, base_model.num_hidden))
    modules.append(ACTIVATION_DICT[base_model.activation])

    for u in range(base_model.num_layers - 1):

        if base_model.dropout_active:

            modules.append(torch.nn.Dropout(p=base_model.dropout_prob))

        modules.append(torch.nn.Linear(base_model.num_hidden, base_model.num_hidden))
        modules.append(ACTIVATION_DICT[base_model.activation])

    modules.append(torch.nn.Linear(base_model.num_hidden, base_model.output_size))

    _architecture    = nn.Sequential(*modules)

    return _architecture


