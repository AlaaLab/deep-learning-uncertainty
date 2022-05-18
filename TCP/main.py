from __future__ import absolute_import, division, print_function
from distutils.util import strtobool
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import numpy as np
import sys

from data.synthetic import *
from data.real_data import *
from conformal.quantiles import *
from conformal.TCP import *
from utils.visualize import *
from utils.metrics import *
from utils.base_models import *
from sklearn.ensemble import GradientBoostingRegressor
import sys 
import six
sys.path.append('./conformal/cqr/')
sys.path.append('./conformal/chr/')
sys.modules['sklearn.externals.six'] = six
from conformal.baselines import *

def to_numpy(T): 
    return T.cpu().detach().numpy()

def run_experiment(model='TCP', data_params={}, delta=.05, alpha=0.1): 
    dataset_name   = data_params['name']
    dataset_base_path = data_params['base_path']
    params = data_params['params']
    data_out = get_scaled_dataset(dataset_name, dataset_base_path, params=params)
    X_train, y_train = to_numpy(data_out.X_tr), to_numpy(data_out.y_tr).squeeze()
    X_calib, y_calib = to_numpy(data_out.X_ca)[1000:2000], to_numpy(data_out.y_ca).squeeze()[1000:2000]
    X_test, y_test   = to_numpy(data_out.X_te), to_numpy(data_out.y_te).squeeze()

    # fit model to proper training set
    '''
    
    data_prop = {'X': X_train, 'y': y_train}
    hp = {'hidden_layer_sizes': [(100,100)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [.0001],
        'learning_rate': ['adaptive'],
        'learning_rate_init': [1e-3],
        'max_iter': [300]}
    print(f'1] Computing best hyperparameters for MLP.')
    best_hp_prop = hp_selection(data_prop, 
                test_size=0.5, 
                seed=42, 
                model_name='MLP', 
                hp=hp)
    f = Model('MLP', hp=best_hp_prop)
    print(f'2] Fitting MLP on proper training set.')
    f.fit(X_train, y_train)
    '''
    print(f'2] Fitting GradientBoostingRegressor on proper training set.')
    f = GradientBoostingRegressor()
    f.fit(X_train, y_train)

    # compute residuals on calibration set
    print(f'3] Computing residuals on calibration set.')
    y_pred = f.predict(X_calib)
    y_resid_calib = np.abs(y_calib - y_pred)
    y_pred_test  = f.predict(X_test)
    y_resid_test = np.abs(y_test - y_pred_test)

    # conformal method
    print(f'4] Running {model}...')
    if model == 'TCP':      
        TCP_model    = TCP_RIF(delta=delta)
        TCP_model.fit(X_calib, y_resid_calib, seed=params['seed'])
        q_TCP_RIF_test, r_TCP_RIF_test = TCP_model.predict(X_test)
        subgroup_idxs = TCP_model.get_subgroup_idxs()

        ''' 
            Metric 1
            external_residuals = []
            for sg in subgroup_idxs: 
                external_residuals.append(y_resid_calib[sg])
            internal_residuals = TCP_model.internal_residuals        
            print(np.mean(((internal_residuals[0] < q_upper[0]) & (internal_residuals[0] > q_lower[0]))))
        '''
        q_lower      = -1 * q_TCP_RIF_test
        q_upper      = q_TCP_RIF_test
        coverage_subgroups = compute_subgroup_coverage(subgroup_idxs, y_resid_test, q_lower, q_upper)
        print(f'coverage in subgroup (metric 2): {np.mean(coverage_subgroups)}')
    elif model == 'CP': 
        q_conformal = empirical_quantile(y_resid_calib, alpha=alpha)
        q_lower     = -1 * q_conformal * np.ones(X_test.shape[0])
        q_upper     = q_conformal * np.ones(X_test.shape[0])
    elif model == 'CQR': 
        cqr = CQR(alpha=alpha)
        cqr.fit(X_calib, y_resid_calib, frac=0.5)
        q_intervals = cqr.predict(X_test)
        q_lower = q_intervals[:,0]; q_upper = q_intervals[:,1]
    elif model == 'CondHist': 
        if len(X_calib.shape) == 1: 
            n_features = 1
        else: 
            n_features = X_calib.shape[1]
        ch = CondHist(alpha=alpha, n_features=n_features)
        ch.fit(X_calib, y_resid_calib, frac=0.5)
        q_intervals = ch.predict(X_test)
        q_lower = q_intervals[:,0]; q_upper = q_intervals[:,1]
    else: 
        raise ValueError('invalid method specified. must be one of ["CP", "TCP", "CQR", or "CondHist"]')

    return compute_coverage(y_resid_test, q_lower, q_upper)

if __name__ == '__main__': 
    parser = ArgumentParser()
    parser.add_argument('--grand_seed', default=42, type=int, help='meta level seed')
    parser.add_argument('-n', '--n_experiments', default=2, type=int, help='# of experiments to run')
    parser.add_argument('--alpha', default=0.1, type=float, help='level of confidence intervals produced')
    parser.add_argument('--base_path', type=str, default='./data/real_data/')
    parser.add_argument('--save', type=strtobool, default=True)
    parser.add_argument('-d','--datasets', nargs='+', help='list of datasets', required=True) #['meps_19', 'meps_20', 'meps_21'] 
    parser.add_argument('-m','--methods', nargs='+', help='list of methods', required=True) #['TCP', 'CP', 'CQR', 'CondHist']   

    args = parser.parse_args()
    
    meps_19 = dict({'name': 'meps_19', 'base_path': args.base_path, 'params': None})
    meps_20 = dict({'name': 'meps_20', 'base_path': args.base_path, 'params': None})
    meps_21 = dict({'name': 'meps_21', 'base_path': args.base_path, 'params': None})
    facebook_1 = dict({'name': 'facebook_1', 'base_path': args.base_path, 'params': None})
    facebook_2 = dict({'name': 'facebook_2', 'base_path': args.base_path, 'params': None})
    bio = dict({'name': 'bio', 'base_path': args.base_path, 'params': None})
    blog_data = dict({'name': 'blog_data', 'base_path': args.base_path, 'params': None})
    kin8nm = dict({'name': 'kin8nm', 'base_path': args.base_path, 'params': None})
    naval  = dict({'name': 'naval', 'base_path': args.base_path, 'params': None})
    real_world_datasets = dict({'meps_19': meps_19, 
                                'meps_20': meps_20, 
                                'meps_21': meps_21, 
                                'facebook_1': facebook_1, 
                                'facebook_2': facebook_2, 
                                'bio': bio, 
                                'blog_data': blog_data,
                                'kin8nm': kin8nm,
                                'naval': naval})

    grand_seed = args.grand_seed
    n_experiments = args.n_experiments
    np.random.seed(grand_seed)
    seeds = np.random.randint(0,100,size=n_experiments)
    datasets = args.datasets 
    methods  = args.methods
    exp_results = []

    for i in range(n_experiments): 
        for dataset in datasets:
            params = {'seed': seeds[i], 'test_size': 0.15}
            real_world_datasets[dataset]['params'] = params

            for method in methods: 
                print(f'Running Experiment Configuration - [experiment {i+1}, {dataset}, {method}]')
                marginal_coverage, average_length = run_experiment(model=method, 
                                                data_params=real_world_datasets[dataset],
                                                alpha=args.alpha)
                result = {'exp_num': i, 
                          'dataset': dataset, 
                          'model': method, 
                          'marginal_coverage': marginal_coverage, 
                          'average_length': average_length}
                exp_results.append(result)
    
    R = pd.DataFrame(exp_results)
    print(R)
    if args.save: 
        R.to_csv('./results/real_world_results_working_5runs.csv', index=False)