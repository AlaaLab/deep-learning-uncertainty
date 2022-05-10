# Copyright (c) 2022, Clinical ML lab
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Synthetic data generation models
# ---------------------------------------------------------

from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.special import erfinv
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# dictionary of synthetic models

function_forms   = dict({"cos": np.cos, "sin": np.sin, 
                         "abs": np.abs, "sqrt": np.sqrt})   

feature_samplers = dict({"uniform": np.random.uniform,
                         "fixed": np.linspace,
                         "gaussian": np.random.normal}) 


# outcome distribution P_Y|X

def outcome_model(x, 
                  T=10, 
                  C_1=4.5, 
                  C_2=0, 
                  alpha=0.05,
                  form="cos"):
  
  """
  Synthetic model for generating samples from a prespecified
  model for the conditional distribution Y|X=x

  This function returns a sample and the true model quantiles at inputs x
  
  """

  # outcomes for |x| <= C_1 and |x| > C_2

  y_expected = function_forms[form](np.pi * x / T)                     # conditional mean E[Y|X=x] 
  y_sample_1 = y_expected * np.random.normal()                         # samples from the model Y|X=x
  q_true_1   = y_expected * np.sqrt(2) * erfinv(2 * (1-(alpha/2)) - 1) # true quantile function

  # outcomes for |x| > C_1 and |x| <= C_2

  y_sample_2 = 2 * np.random.normal()                         
  q_true_2   = 2 * np.sqrt(2) * erfinv(2 * (1-(alpha/2)) - 1) 

  # combine the piecewise functions

  indicators = (np.abs(x) <= C_1) * (np.abs(x) > C_2) * 1
  y_sample   = indicators * y_sample_1 + (1 - indicators) * y_sample_2
  q_true     = indicators * q_true_1 + (1 - indicators) * q_true_2

  return y_sample, q_true     


# feature distribution P_X

def feature_distribution(a=-5, 
                         b=5, 
                         n=100, 
                         dist="uniform"):
  

  X = feature_samplers[dist](a, b, n)

  return X  


# data generation process

def sample_data(n=100, 
                a=-5, 
                b=5,
                T=10, 
                C_1=4.5, 
                C_2=0, 
                alpha=0.05,
                form="cos", 
                feature_dist="uniform",
                **kwargs):

  outcome_params = dict({"T":T, "C_1": C_1, "C_2": C_2, "alpha": alpha, "form": form})
  feature_params = dict({"a": a, "b": b, "n": n})

  if feature_dist=="gaussian":

    X            = 2 * feature_distribution(dist=feature_dist, **dict({"a": 0, "b": 1, "n": n}))

  else:
  
    X            = feature_distribution(dist=feature_dist, **feature_params)  


  YQ             = [outcome_model(X[_], **outcome_params) for _ in range(n)]
  Y, Q           = [YQ[_][0] for _ in range(n)], [YQ[_][1] for _ in range(n)]

  return X, Y, Q

