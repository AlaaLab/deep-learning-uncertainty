# Copyright (c) 2022, Clinical ML lab
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Transparent conformal prediction (TCP)
# ---------------------------------------------------------

from __future__ import absolute_import, division, print_function
import numpy as np
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsRegressor

# unconditional quantile regression (UQR)
class UQR:
    
    def __init__(self, 
                 alpha=0.1,
                 residual_density="kde",
                 knn_size=50):

        self.alpha             = alpha
        self.residual_density  = residual_density
        self.knn_size          = knn_size    

    def fit(self, X, residuals, kernel_bandwidth=.1):

        q_alpha            = np.quantile(residuals, 1-self.alpha) 
        self.q_alpha       = q_alpha       
        kde                = KernelDensity(kernel='gaussian', bandwidth=kernel_bandwidth).fit(np.array(residuals).reshape((-1, 1))) 

        self.fr_density    = np.exp(kde.score_samples(np.array([q_alpha]).reshape((-1, 1))))[0]
        self.RIF           = self.q_alpha + (((1-self.alpha) - (residuals < self.q_alpha))/ self.fr_density)
        self.RIF_model     = KNeighborsRegressor(n_neighbors=self.knn_size) #GradientBoostingRegressor(n_estimators=100) #KernelRegression(gamma=10) #

        self.RIF_model.fit(X, self.RIF)
    
    def predict(self, X):

        RIF_pred           = self.RIF_model.predict(X) 

        return RIF_pred

def get_relevance_group_size(delta, n_calib):
  return int(delta * n_calib * (1 - np.sqrt(2 * np.log(n_calib) / (delta * n_calib))))

def get_achieved_coverage(knnresiduals, q_UQRs, alpha):

  acheved_cov = np.array([np.mean((knnresiduals > q_UQRs[u]) | (knnresiduals < -1 * q_UQRs[u])) for u in range(q_UQRs.shape[0])])

  return q_UQRs[np.argmin(np.abs(acheved_cov - alpha))]


# Transparent conoformal with RIF nested sequences

class TCP_RIF:
    
    def __init__(self, 
                 alphas=list(np.linspace(0.005, 0.975, 100)),
                 alpha=0.1,
                 delta=0.05,
                 residual_density="kde"):
      
      self.UQR_models        = []
      self.q_UQRs            = []
      self.alphas            = alphas
      self.residual_density  = residual_density
      self.delta             = delta
      self.alpha             = alpha  

      # Initialize UQR models

      for k in range(len(self.alphas)):
        self.UQR_models.append(UQR(alpha=self.alphas[k]))
        
    def fit(self, X, Y):

      self.q_UQRs      = []
      X, Y_            = np.array(X), np.array(Y)
      self.n_neighbors = get_relevance_group_size(self.delta, n_calib=X.shape[0])

      if len(X.shape)==1:
        X_ = X.reshape((-1, 1))
      elif X.shape[1]==1 or X.shape[0]==1:
        X_ = X.reshape((-1, 1))
      else:
        X_ = X    

      for k in range(len(self.alphas)):
        self.UQR_models[k].knn_size = self.n_neighbors
        self.UQR_models[k].fit(X_, Y_)

      self.X_calib, self.Y_calib = X, Y_   
        
    def predict(self, X): 

      X           = np.array(X)  
      self.q_UQRs = []

      if len(X.shape)==1:
        X_ = X.reshape((-1, 1))
      elif X.shape[1]==1 or X.shape[0]==1:
        X_ = X.reshape((-1, 1))
      else:
        X_ = X   

      for k in range(len(self.alphas)):
        self.q_UQRs.append(self.UQR_models[k].predict(X_))

      self.q_UQRs = np.array(self.q_UQRs)

      # conformalization

      q_interval   = []
  
      for k in range(len(X)):
        knnresiduals = self.Y_calib[np.argsort(np.abs(self.X_calib - X[k]))[:self.n_neighbors]] # replace this for high dimensional
        interval_    = get_achieved_coverage(knnresiduals, self.q_UQRs[:, k], self.alpha)
        q_interval.append(interval_)
      q_int_arr = np.array(q_interval)

      return [-q_int_arr, q_int_arr]
      # report subgroup radiuss        