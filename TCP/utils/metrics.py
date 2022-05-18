# Copyright (c) 2022, Clinical ML lab
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Conformal prediction tools
# ---------------------------------------------------------

from __future__ import absolute_import, division, print_function
import numpy as np
from matplotlib import pyplot as plt
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def compute_subgroup_coverage(subgroup_idxs, y_true, y_lower, y_upper): 
  subgroup_coverages = []
  for sg in subgroup_idxs: 
      y_true_sg = y_true[sg]
      y_lower_sg = y_lower[sg]
      y_upper_sg = y_upper[sg]
      c = compute_coverage(y_true_sg, y_lower_sg, y_upper_sg)[0]
      subgroup_coverages.append(c)
  return subgroup_coverages
  
def compute_coverage(y_true, y_lower, y_upper, w=None):
  if w is None:
    coverage_     = 1 - np.mean((y_true < y_lower) | (y_true > y_upper))
  else:
    weighted_cov  = (w * np.repeat(((y_true < y_lower) | (y_true > y_upper)).reshape((1, -1)), w.shape[0], axis=0))
    coverage_     = 1 - np.sum(weighted_cov, axis=1)

  return coverage_, np.mean(np.abs(y_upper - y_lower))  
