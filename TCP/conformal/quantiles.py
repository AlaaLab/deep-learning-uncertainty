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


def empirical_quantile(residuals, alpha=.05):

  return np.quantile(residuals, 1-(alpha/2))
