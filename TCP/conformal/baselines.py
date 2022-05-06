import numpy as np 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from numpy.random import default_rng
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsRegressor, KDTree 
from sklearn.kernel_ridge import KernelRidge

# cqr imports
import sys 
sys.path.append('./cqr/cqr')
from conformal.cqr.cqr import helper
from conformal.cqr.nonconformist.nc import RegressorNc
from conformal.cqr.nonconformist.cp import IcpRegressor
from conformal.cqr.nonconformist.nc import QuantileRegErrFunc

# chr imports 
from conformal.chr.chr.black_boxes import QNet, QRF
from conformal.chr.chr.black_boxes_r import QBART
from conformal.chr.chr.methods import CHR

# locally adaptive conformal prediction imports
from conformal.cqr.nonconformist.nc import AbsErrorErrFunc
from conformal.cqr.nonconformist.nc import RegressorNormalizer

class ConformalBase: 
    '''
        Implementation inspired from: 
            https://github.com/yromano/cqr/blob/master/cqr_synthetic_data_example_1.ipynb
    '''
    def __init__(self, alpha=0.1):
        self.alpha = alpha 

    def fit(self, x_calibrate, y_calibrate, frac=0.7, random_state=10): 
        '''
            * split data into train and calibrate
            * y_calibrate contains residuals 
            step 1: fit model on training data + training residuals 
            step 2: call calibrate  
        '''
        raise NotImplementedError()

    def predict(self, x_test): 
        '''
            We return both predictions and interval for each prediction
        '''
        raise NotImplementedError()
    
class QR(ConformalBase): 

    def __init__(self, alpha=0.1): 
        super().__init__(alpha)
    
    def fit(self, x_calibrate, y_calibrate):
        y_calibrate = np.array(y_calibrate) 
        self.all_models    = {}
        common_params = dict(
            learning_rate=0.05,
            n_estimators=200,
            max_depth=2,
            min_samples_leaf=9,
            min_samples_split=9,
        )
        for alpha_ in [self.alpha/2, 1-(self.alpha/2)]:
            gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha_, **common_params)
            self.all_models["q %1.2f" % alpha_] = \
                gbr.fit(x_calibrate.reshape((-1, 1)), np.array(y_calibrate).reshape((-1, 1)))

    def predict(self, x_test): 
        Quant_lo = self.all_models['q 0.05'].predict(x_test.reshape((-1, 1)))
        Quant_up = self.all_models['q 0.95'].predict(x_test.reshape((-1, 1)))
        return [Quant_lo, Quant_up] 

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
        self.icp = IcpRegressor(nc)

    def fit(self, x_calibrate, y_calibrate, frac=0.7, random_state=10):
        '''
            * split data into train and calibrate
            * y_calibrate contains residuals 
            step 1: fit model on training data + training residuals 
            step 2: call calibrate  
        '''
        if len(x_calibrate.shape)==1:
            x_calibrate_ = x_calibrate.reshape((-1, 1))
        elif x_calibrate.shape[1]==1 or x_calibrate.shape[0]==1:
            x_calibrate_ = x_calibrate.reshape((-1, 1))
        else:
            x_calibrate_ = x_calibrate  
        y_calibrate = np.array(y_calibrate) 
        x_train, x_calib, y_train, y_calib = \
            train_test_split(x_calibrate_, y_calibrate, test_size=1-frac, random_state=random_state)
        
        self.icp.fit(x_train, y_train)
        self.icp.calibrate(x_calib, y_calib)

    def predict(self, x_test): 
        '''
            We return both predictions and interval for each prediction
        '''
        if len(x_test.shape)==1:
            x_test_ = x_test.reshape((-1, 1))
        elif x_test.shape[1]==1 or x_test.shape[0]==1:
            x_test_ = x_test.reshape((-1, 1))
        else:
            x_test_ = x_test   
        return self.icp.predict(x_test_, significance=self.alpha)


class CondHist(ConformalBase): 

    def __init__(self, alpha=0.1): 
        super().__init__(alpha)        
        grid_quantiles = np.arange(0.01,1.0,0.01)
        self.bbox = QNet(grid_quantiles, 1, no_crossing=True, batch_size=1000, dropout=0.1,
            num_epochs=10000, learning_rate=0.0005, num_hidden=256, calibrate=0)

    def fit(self, x_calibrate, y_calibrate, frac=0.7, random_state=10): 
        '''
            * split data into train and calibrate
            * y_calibrate contains residuals 
            step 1: fit model on training data + training residuals 
            step 2: call calibrate  
        '''
        if len(x_calibrate.shape)==1:
            x_calibrate_ = x_calibrate.reshape((-1, 1))
        elif x_calibrate.shape[1]==1 or x_calibrate.shape[0]==1:
            x_calibrate_ = x_calibrate.reshape((-1, 1))
        else:
            x_calibrate_ = x_calibrate  
        y_calibrate = np.array(y_calibrate) 
        x_train, x_calib, y_train, y_calib = \
            train_test_split(x_calibrate_, y_calibrate, test_size=1-frac, random_state=random_state) 
        self.bbox.fit(x_train, y_train)
        # Initialize and calibrate the new method
        self.chr = CHR(self.bbox, ymin=-3, ymax=20, y_steps=200, delta_alpha=0.001, randomize=True)
        self.chr.calibrate(x_calib, y_calib, self.alpha)

    def predict(self, x_test): 
        if len(x_test.shape)==1:
            x_test_ = x_test.reshape((-1, 1))
        elif x_test.shape[1]==1 or x_test.shape[0]==1:
            x_test_ = x_test.reshape((-1, 1))
        else:
            x_test_ = x_test   
        return self.chr.predict(x_test_)

class LACP(ConformalBase): 

    def __init__(self, alpha=0.1): 
        super().__init__(alpha)
        n_estimators = 100 
        min_samples_leaf = 40 
        max_features = 1 
        random_state = 0
        # define the conditonal mean estimator as random forests (used to predict the labels)
        mean_estimator = RandomForestRegressor(n_estimators=n_estimators,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            random_state=random_state)

        # define the MAD estimator as random forests (used to scale the absolute residuals)
        mad_estimator = RandomForestRegressor(n_estimators=n_estimators,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            random_state=random_state)

        # define a conformal normalizer object that uses the two regression functions.
        # The nonconformity score is absolute residual error
        normalizer = RegressorNormalizer(mean_estimator,
                                        mad_estimator,
                                        AbsErrorErrFunc())

        # define the final local conformal object 
        nc = RegressorNc(mean_estimator, AbsErrorErrFunc(), normalizer)

        # build the split local conformal object
        self.icp = IcpRegressor(nc)
    
    def fit(self, x_calibrate, y_calibrate, frac=0.7, random_state=10):
        if len(x_calibrate.shape)==1:
            x_calibrate_ = x_calibrate.reshape((-1, 1))
        elif x_calibrate.shape[1]==1 or x_calibrate.shape[0]==1:
            x_calibrate_ = x_calibrate.reshape((-1, 1))
        else:
            x_calibrate_ = x_calibrate   
        y_calibrate = np.array(y_calibrate)
        x_train, x_calib, y_train, y_calib = \
            train_test_split(x_calibrate_, y_calibrate, test_size=1-frac, random_state=random_state)
        self.icp.fit(x_train, y_train)
        self.icp.calibrate(x_calib, y_calib)

    def predict(self, x_test): 
        if len(x_test.shape)==1:
            x_test_ = x_test.reshape((-1, 1))
        elif x_test.shape[1]==1 or x_test.shape[0]==1:
            x_test_ = x_test.reshape((-1, 1))
        else:
            x_test_ = x_test   
        return self.icp.predict(x_test_, significance=self.alpha)

