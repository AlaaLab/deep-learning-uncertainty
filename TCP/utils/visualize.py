# Copyright (c) 2022, Clinical ML lab
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------

from __future__ import absolute_import, division, print_function
import numpy as np
from matplotlib import pyplot as plt
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


plotting_modes   = dict({"scatter": plt.scatter, 
                         "plot": plt.plot, 
                         "axhline": plt.axhline})


def prepare_plot(X=None, Y=None, type="plot", args=None):

  plot_params         = dict({"type": type, 
                              "X": X, 
                              "Y": Y, 
                              "args": args})
  
  return plot_params 


def plotting(plot_dict, 
             legend=False, 
             xlabel=None,
             ylabel=None,
             legend_loc="upper right",
             xlim=None,
             ylim=None,
             save=False,
             filename=None): # save function

  fig, ax     = plt.subplots()

  right_side  = ax.spines["right"]
  right_side.set_visible(False)

  upper_side  = ax.spines["top"]
  upper_side.set_visible(False)

  _keys       = list(plot_dict.keys())

  for _ in range(len(list(plot_dict.keys()))):

    plot_func = plotting_modes[plot_dict[_keys[_]]["type"]]

    plot_func(plot_dict[_keys[_]]["X"], 
              plot_dict[_keys[_]]["Y"], 
              **plot_dict[_keys[_]]["args"])

  if legend is not None:

    plt.legend(loc=legend_loc)

  if xlabel is not None:  

    plt.xlabel(xlabel)

  if ylabel is not None:  

    plt.ylabel(ylabel)

  if xlim is not None:

    plt.xlim(xlim[0], xlim[1])

  if ylim is not None:

    plt.ylim(ylim[0], ylim[1])

  if save:

    plt.savefig(filename, transparent=True, dpi=1200)

    if 'google.colab' in sys.modules:

      from google.colab import files
      
      files.download(filename)  
