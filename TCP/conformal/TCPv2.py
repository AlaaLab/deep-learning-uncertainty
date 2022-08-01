# Copyright (c) 2022, Clinical ML lab
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Transparent conformal prediction (TCP)
# ---------------------------------------------------------

from __future__ import absolute_import, division, print_function
from cgi import test
import numpy as np
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsRegressor

def euclidean_distance(X_single_pt, X_calib):
  if len(X_single_pt.shape) == 0:
    X_dists = sum((x - y) ** 2 for (x, y) in zip(np.array(X_calib).reshape((1, -1)), \
            np.array([X_single_pt]).reshape((1, -1)))) ** 0.5
  else:
    X_dists = np.linalg.norm(X_calib - X_single_pt[None,:], ord=2, axis=1)

  return  X_dists   

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
        # self.RIF_model     = KNeighborsRegressor(n_neighbors=self.knn_size) #GradientBoostingRegressor(n_estimators=100) #KernelRegression(gamma=10) #
        self.RIF_model     = GradientBoostingRegressor(n_estimators=100) #KernelRegression(gamma=10) #
        self.RIF_model.fit(X, self.RIF)
    
    def predict(self, X):
        RIF_pred           = self.RIF_model.predict(X) 
        return RIF_pred

def get_relevance_group_size(delta, n_calib):
  return int(delta * n_calib * (1 - np.sqrt(2 * np.log(n_calib) / (delta * n_calib))))

def get_achieved_coverage(knnresiduals, q_UQRs, alpha):
  achieved_cov = np.array([np.mean((knnresiduals > q_UQRs[u]) | (knnresiduals < -1 * q_UQRs[u])) for u in range(q_UQRs.shape[0])])
  # print(1-achieved_cov[np.argmin(np.abs(achieved_cov - alpha))])
  return q_UQRs[np.argmin(np.abs(achieved_cov - alpha))]

# Transparent conformal with RIF nested sequences
class TCP_quantile: 
    def __init__(self, 
              alpha=0.1, 
              delta=10,
              subgroup_model=None): 
      self.alpha = alpha 
      self.delta = delta 
      assert subgroup_model is not None, 'need to pass in a fit clustering model, e.g. kmeans on proper training'
      self.subgroup_model = subgroup_model
    
    def fit(self, X, Y, seed=42, test_size=0.5): 
      X, Y_            = np.array(X), np.array(Y)

      if len(X.shape)==1:
        X_ = X.reshape((-1, 1))
      elif X.shape[1]==1 or X.shape[0]==1:
        X_ = X.reshape((-1, 1))
      else:
        X_ = X    
      
      self.X_calib, self.Y_calib = X_, Y_

    def predict(self, X): 
      X           = np.array(X)  
      if len(X.shape)==1:
        X_ = X.reshape((-1, 1))
      elif X.shape[1]==1 or X.shape[0]==1:
        X_ = X.reshape((-1, 1))
      else:
        X_ = X   

      # conformalization
      q_interval   = []
      
      # store all the idxs of the cluster 
      self.test_clusters = []
      calib_clusters = self.subgroup_model.predict(self.X_calib)
      intervals = []

      # pre-compute a fixed interval for each cluster based on empirical quantile
      for c in range(self.delta): 
        C = self.subgroup_model.cluster_centers_
        cluster_dists = np.linalg.norm(C - C[c][None,:], ord=2, axis=1)
        nearest_clusters = np.argsort(cluster_dists)
        assert nearest_clusters[0] == c 

        # possible that there are no calibration samples in a subgroup, so just assign interval of nearest subgroup.
        kmeans_residuals = self.Y_calib[np.where(calib_clusters == c)[0]]
        cluster_idx = 1
        while len(kmeans_residuals) == 0: 
          new_c = nearest_clusters[cluster_idx]
          kmeans_residuals = self.Y_calib[np.where(calib_clusters == new_c)[0]]
          cluster_idx += 1

        intervals.append(np.quantile(kmeans_residuals, 1-(self.alpha)))
      
      # for each test point, find which cluster it's in, and then assign the interval 
      # associated with that cluster
      for k in range(len(X)): 
        test_point_cluster = self.subgroup_model.predict(X_[k][None,:])[0]
        q_interval.append(intervals[test_point_cluster])
        self.test_clusters.append(test_point_cluster)

      return np.array(q_interval), self.test_clusters

class TCP_RIF:
    
    def __init__(self, 
                 alphas=list(np.linspace(0.025, 0.975, 20)),
                 alpha=0.1,
                 delta=10,
                 residual_density='kde',
                 subgroup_model=None):
      
      self.UQR_models        = []
      self.q_UQRs            = []
      self.alphas            = alphas
      self.residual_density  = residual_density
      self.delta             = delta
      self.alpha             = alpha  
      self.test_subgroup_idxs     = None
      assert subgroup_model is not None, 'need to pass in a fit clustering model, e.g. kmeans on proper training'
      self.subgroup_model = subgroup_model

      # Initialize UQR models
      for k in range(len(self.alphas)):
        self.UQR_models.append(UQR(alpha=self.alphas[k]))
        
    def fit(self, X, Y, seed=42, test_size=0.5):

      self.q_UQRs      = []
      X, Y_            = np.array(X), np.array(Y)

      if len(X.shape)==1:
        X_ = X.reshape((-1, 1))
      elif X.shape[1]==1 or X.shape[0]==1:
        X_ = X.reshape((-1, 1))
      else:
        X_ = X    
      # self.n_neighbors = get_relevance_group_size(self.delta, n_calib=X.shape[0])
      X_fit, X_calib, Y_fit, Y_calib = train_test_split(
        X_, Y_, test_size=test_size, random_state=seed)
      
      for k in range(len(self.alphas)):
        if k % 10 == 0: 
          print(f'fitting UQR number {k}')
        self.UQR_models[k].fit(X_fit, Y_fit)

      self.X_calib, self.Y_calib = X_calib, Y_calib
        
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
        if k % 10 == 0: 
          print(f'predicting with UQR number {k}')
        self.q_UQRs.append(self.UQR_models[k].predict(X_))
      self.q_UQRs = np.array(self.q_UQRs)

      # conformalization
      q_interval   = []
      self.internal_residuals = []
      # store all the idxs of the cluster 
      self.test_clusters = []
      calib_clusters = self.subgroup_model.predict(self.X_calib)
      for k in range(len(X)):
        test_point_cluster = self.subgroup_model.predict(X[k][None,:])[0]
        kmeans_residuals   = self.Y_calib[np.where(calib_clusters == test_point_cluster)[0]]
        self.internal_residuals.append(kmeans_residuals)

        interval_ = get_achieved_coverage(kmeans_residuals.reshape((-1,1)), self.q_UQRs[:,k], self.alpha)
        q_interval.append(interval_)
        self.test_clusters.append(test_point_cluster)

      return np.array(q_interval), self.test_clusters
    
    def get_subgroup_idxs(self): 
      assert self.test_clusters is not None 
      return self.test_clusters

             





