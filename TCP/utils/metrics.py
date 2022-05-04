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


def compute_coverage(y_true, y_lower, y_upper, w=None):

  if w is None:

    coverage_     = 1 - np.mean((y_true < y_lower) | (y_true > y_upper))

  else:

    weighted_cov  = (w * np.repeat(((y_true < y_lower) | (y_true > y_upper)).reshape((1, -1)), w.shape[0], axis=0))
    coverage_     = 1 - np.sum(weighted_cov, axis=1)

  return coverage_, np.mean(np.abs(y_upper - y_lower))  
