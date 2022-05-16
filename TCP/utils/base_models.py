import torch
import torch.nn as nn
import numpy as np 
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from skorch import NeuralNetRegressor
from torch import optim
from sklearn.model_selection import train_test_split
from itertools import product


class Model: 
    
    def __init__(self, model_name='LogisticRegression', hp={}): 
        self.model_name = model_name
        if model_name == 'MLP': 
            if torch.cuda.is_available():
                self.device = torch.device('cuda:1')
            else:
                self.device  = torch.device('cpu')
        self.model = self._init_model(model_name, hp)
    
    def _init_model(self, name, params): 
        if name == 'LogisticRegression':
            C = params.get('C',1.)
            return Pipeline([('var_threshold', VarianceThreshold()), \
                             ('LR', LogisticRegression(C=C, max_iter=1000))])
        elif name == 'RandomForestRegressor': 
            nt = params.get('n_estimators',50)
            md = params.get('max_depth', 5)
            mn = params.get('min_samples_split', 10)
            mf = params.get('max_features', 'all')
            return RandomForestRegressor(n_estimators=nt, max_depth=md, \
                                          min_samples_split=mn, max_features=mf)         
        elif name == 'Lasso': 
            alpha = params.get('alpha',1.)
            return Lasso(alpha=alpha, max_iter=1000)
        elif name == 'LinearRegression': 
            return LinearRegression()
        elif name == 'MLP': 
            hidden_layer_sizes = params.get('hidden_layer_sizes', (50,50))
            activation = params.get('activation', 'relu')
            solver = params.get('solver', 'adam')
            alpha = params.get('alpha', .001)
            learning_rate = params.get('learning_rate', 'adaptive')
            learning_rate_init = params.get('learning_rate_init', 1e-3)
            max_iter = params.get('max_iter', 200)
            input_dim = params.get('input_dim',-1)
            
            if solver == 'adam': 
                o = optim.Adam
            elif solver == 'sgd': 
                o = optim.SGD

            return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation,
                                solver=solver,
                                alpha=alpha,
                                learning_rate=learning_rate,
                                learning_rate_init=learning_rate_init,
                                max_iter=max_iter)
    
    def fit(self, X, y): 
        self.model.fit(X,y)
    
    def predict(self, X): 
        return self.model.predict(X).squeeze()
    
    def predict_proba(self, X): 
        return self.model.predict_proba(X)
    
    def compute_metric(self, y_true, y_predict): 
        return -mean_squared_error(y_true, y_predict)


def hp_selection(data, 
                test_size=0.2, 
                seed=42, 
                model_name='LogisticRegression', 
                hp={}): 
        
    X   = data['X']; y = data['y']        
    tr_idxs, val_idxs = train_test_split(np.arange(X.shape[0]),\
                        test_size=test_size,random_state=seed)
    
    X_train, X_val = X[tr_idxs], X[val_idxs]
    y_train, y_val = y[tr_idxs], y[val_idxs]
    best_hp = None # to store best hps 

    param_names = hp.keys()
    param_lists = [hp[k] for k in param_names]
    for elem in product(*param_lists): 
        print(f'[trying hp {elem} for {model_name}]')
        params = {k:elem[i] for i,k in enumerate(param_names)}
        params['input_dim'] = X_train.shape[1]
        
        model = Model(model_name, hp=params)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_val)
        metric = model.compute_metric(y_val, y_predict)
        
        if best_hp is None or metric > best_hp[0]: 
            best_hp = (metric, params)

    print(f'best hp: {best_hp[1]}')
    return best_hp[1]