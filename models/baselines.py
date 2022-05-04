import numpy as np 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from numpy.random import default_rng
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsRegressor, KDTree 
from sklearn.kernel_ridge import KernelRidge

# cqr imports
from cqr.cqr import helper
from cqr.nonconformist.nc import RegressorNc
from cqr.nonconformist.cp import IcpRegressor
from cqr.nonconformist.nc import QuantileRegErrFunc

class ConformalBase: 
    '''
        Implementation inspired from: 
            https://github.com/yromano/cqr/blob/master/cqr_synthetic_data_example_1.ipynb
    '''
    def _get_model_dict(self): 
        return { 
            'boost': GradientBoostingRegressor()
        }


    def __init__(self, alpha=0.1):
        model_dict = self._get_model_dict
        self.alpha = alpha 

    def fit(self, x_train, y_train):
        raise NotImplementedError()

    def calibrate(self, x_calibrate, y_calibrate): 
        raise NotImplementedError()

    def predict(self, x_test): 
        raise NotImplementedError()
    

class CQR(ConformalBase): 

    def __init__(self, alpha=0.1): 
        super().__init__(alpha)        
        n_estimators = 100 
        min_samples_leaf = 40 
        max_features = 1 
        random_state = 0
        quantiles = [alpha*10/2, 100-(alpha*10/2)]         

        # define dictionary for quantile estimator
        params_qforest = dict()
        params_qforest['n_estimators'] = n_estimators
        params_qforest['min_samples_leaf'] = min_samples_leaf
        params_qforest['max_features'] = max_features
        params_qforest['CV'] = True
        params_qforest['coverage_factor'] = 0.9
        params_qforest['test_ratio'] = 0.1
        params_qforest['random_state'] = random_state
        params_qforest['range_vals'] = 10
        params_qforest['num_vals'] = 4

        quantile_estimator = helper.QuantileForestRegressorAdapter(model=None,
                                                           fit_params=None,
                                                           quantiles=quantiles,
                                                           params=params_qforest)
        nc  = RegressorNc(quantile_estimator, QuantileRegErrFunc())
        self.model = IcpRegressor(nc)

    def fit(self, x_train, y_train): 
        self.model.fit(x_train, y_train)

    def calibrate(self, x_calibrate, y_calibrate): 
        self.model.calibrate(x_calibrate, y_calibrate)

    def predict(self, x_test): 
        return self.model.predict(x_test, significance=self.alpha)