from __future__ import absolute_import, division, print_function
from distutils.util import strtobool
from matplotlib import pyplot as plt
from argparse import ArgumentParser
import numpy as np
import sys

from data.synthetic import *
from data.real_data import *
from conformal.quantiles import *
from conformal.TCPv2 import *
from utils.visualize import *
from utils.metrics import *
from utils.base_models import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
import sys 
import six
sys.path.append('./conformal/cqr/')
sys.path.append('./conformal/chr/')
sys.modules['sklearn.externals.six'] = six
from conformal.baselines import *

def to_numpy(T): 
    return T.cpu().detach().numpy()

def run_experiment(model='TCP', 
                    data_params={}, 
                    K=10, 
                    alpha=0.1, 
                    num_alphas=20,
                    save_subgroup_ids=False): 
    dataset_name   = data_params['name']
    dataset_base_path = data_params['base_path']
    params = data_params['params']

    data_out = get_scaled_dataset(dataset_name, dataset_base_path, params=params)
    X_train, y_train = to_numpy(data_out.X_tr), to_numpy(data_out.y_tr)
    X_calib, y_calib = to_numpy(data_out.X_ca), to_numpy(data_out.y_ca)
    X_test, y_test   = to_numpy(data_out.X_te), to_numpy(data_out.y_te)

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
    print(f'1] Fitting GradientBoostingRegressor on proper training set.')
    f = GradientBoostingRegressor()
    f.fit(X_train, y_train)

    print(f'2] K-means clustering of training set.')
    kmeans_seed = params['kmeans_seed']
    km = KMeans(n_clusters=K, random_state=kmeans_seed)
    km.fit(X_train)

    # compute residuals on calibration set
    print(f'3] Computing residuals on calibration set.')
    y_pred = f.predict(X_calib)[:,None]
    y_resid_calib = np.abs(y_calib - y_pred)
    y_pred_test  = f.predict(X_test)[:,None]
    y_resid_test = np.abs(y_test - y_pred_test)

    # conformal method
    print(f'4] Running {model}...')
    if model == 'TCP':      
        alphas = list(np.linspace(0.025, 0.975, num_alphas))
        TCP_model    = TCP_RIF(alphas=alphas, 
                                alpha=alpha, 
                                delta=K, 
                                subgroup_model=km)
        TCP_model.fit(X_calib, y_resid_calib, seed=params['calib_seed']) # CHANGE THIS BACK TO seed WHEN DONE!!!!
        q_TCP_RIF_test, r_TCP_RIF_test = TCP_model.predict(X_test)

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
    elif model == 'TCP-quantile': 
        TCP_quant = TCP_quantile(alpha=alpha, 
                                delta=K, 
                                subgroup_model=km)
        TCP_quant.fit(X_calib, y_resid_calib, seed=params['calib_seed'])
        q_TCP_quant_test, _ = TCP_quant.predict(X_test)
        q_lower = -1 * q_TCP_quant_test
        q_upper = q_TCP_quant_test
    elif model == 'CP': 
        q_conformal = empirical_quantile(y_resid_calib.squeeze(), alpha=alpha)
        q_lower     = (-1 * q_conformal * np.ones(X_test.shape[0])).reshape((-1,1))
        q_upper     = (q_conformal * np.ones(X_test.shape[0])).reshape((-1,1))
    elif model == 'CQR': 
        cqr = CQR(alpha=alpha)
        cqr.fit(X_calib, y_resid_calib.squeeze(), frac=0.5, random_state=params['calib_seed'])
        q_intervals = cqr.predict(X_test)
        q_lower = q_intervals[:,[0]]; q_upper = q_intervals[:,[1]]
    elif model == 'CondHist': 
        if len(X_calib.shape) == 1: 
            n_features = 1
        else: 
            n_features = X_calib.shape[1]
        ch = CondHist(alpha=alpha, n_features=n_features)
        ch.fit(X_calib, y_resid_calib.squeeze(), frac=0.5, random_state=params['calib_seed'])
        q_intervals = ch.predict(X_test)
        q_lower = q_intervals[:,[0]]; q_upper = q_intervals[:,[1]]
    elif model == 'LACP': 
        lacp = LACP(alpha=alpha) 
        lacp.fit(X_calib, y_resid_calib.squeeze(), frac=0.5, random_state=params['calib_seed'])
        q_intervals = lacp.predict(X_test)
        q_lower = q_intervals[:,[0]]; q_upper = q_intervals[:,[1]]
    elif model == 'QR-RF': 
        qr_rf = QR_RF(alpha=alpha)
        qr_rf.fit(X_calib, y_resid_calib.squeeze())
        q_lower, q_upper = qr_rf.predict(X_test)
        q_lower = q_lower.reshape((-1,1)); q_upper = q_upper.reshape((-1,1)) 
    elif model == 'QR-NN': 
        qr_nn = QR_NN(alpha=alpha, in_shape=X_calib.shape[1])
        qr_nn.fit(X_calib, y_resid_calib.squeeze())
        q_lower, q_upper = qr_nn.predict(X_test)
        q_lower = q_lower.reshape((-1,1)); q_upper = q_upper.reshape((-1,1))  
    elif model == 'PASS': 
        pass
    else: 
        raise ValueError('invalid method specified. must be one of ["CP", "TCP", "CQR", "CondHist", "LACP", "QR-RF", or "QR-NN"]')
    
    x_query = X_test
    test_clusters = np.array([km.predict(x_query[k][None,:])[0] for k in range(len(x_query))])
    test_subgroup_idxs = [np.where(test_clusters == cluster_num)[0] for cluster_num in range(K)]

    coverage_subgroups = compute_subgroup_coverage(test_subgroup_idxs, y_resid_test, q_lower, q_upper)

    print(f'{y_resid_test.shape}, {q_lower.shape}, {q_upper.shape}')
    marginal_coverage, ave_length = compute_coverage(y_resid_test, q_lower, q_upper)
    print(f'number of units in each subgroup: {[len(x) for x in test_subgroup_idxs]}')
    print(f'marginal coverage: {marginal_coverage}, average length: {ave_length}, average coverage in {K} subgroups: {np.nanmean(coverage_subgroups)}')
    return marginal_coverage, ave_length, np.nanmean(coverage_subgroups), \
        coverage_subgroups, q_lower, q_upper, test_subgroup_idxs, (X_train, X_test, y_resid_test)

if __name__ == '__main__': 
    parser = ArgumentParser()
    parser.add_argument('--grand_seed', default=42, type=int, help='meta level seed')
    parser.add_argument('-n', '--n_experiments', default=2, type=int, help='# of experiments to run')
    parser.add_argument('--alpha', default=0.1, type=float, help='level of confidence intervals produced')
    parser.add_argument('-na','--num_alphas', default=20, type=int, help='# of alphas for TCP')
    parser.add_argument('-k', '--K', default=10, type=int, help='# of clusters')
    parser.add_argument('--base_path', type=str, default='./data/real_data/')
    parser.add_argument('--save', type=strtobool, default=True)
    parser.add_argument('--save_subgroup_ids', type=strtobool, default=False)
    parser.add_argument('--fixed', type=strtobool, default=False, help='set to True if you want to fix the train set.')
    parser.add_argument('-d','--datasets', nargs='+', help='list of datasets', required=True) 
    parser.add_argument('-m','--methods', nargs='+', help='list of methods', required=True) 

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
    seeds = np.random.randint(0, 100, size=n_experiments)
    print(seeds)
    kmeans_seed = np.random.randint(0, 100, size=1)[0]
    datasets = args.datasets 
    methods  = args.methods
    exp_results = []

    counter = 0
    for i in range(n_experiments): 
        for dataset in datasets:
            if args.fixed: 
                print('FIXED')
                params = {'calib_seed': seeds[i], 'train_seed': grand_seed, 'test_size': 0.15, 'kmeans_seed': kmeans_seed}
            else: 
                params = {'calib_seed': seeds[i], 'train_seed': seeds[i], 'test_size': 0.15, 'kmeans_seed': kmeans_seed}
            
            real_world_datasets[dataset]['params'] = params

            for method in methods: 
                print(f'Running Experiment Configuration - [experiment {i+1}, {dataset}, {method}]')
                marginal_coverage, average_length, \
                    subgroup_cov, coverage_subgroups, \
                    q_lower, q_upper, test_subgroup_idxs, Xy_train_test = run_experiment(model=method, 
                                                data_params=real_world_datasets[dataset],
                                                K=args.K,
                                                alpha=args.alpha,
                                                num_alphas=args.num_alphas, 
                                                save_subgroup_ids=args.save_subgroup_ids)
                X_train, X_test, y_resid_test = Xy_train_test
                if args.save_subgroup_ids: 
                    ts = np.array(test_subgroup_idxs, dtype=object)
                    np.savez(f'./results/FIGS_test_subgroup_idxs_{args.K}_clusters_{datasets[0]}', ts)
                    np.save(f'./results/GRAD_TABLE_{args.K}_clusters_{datasets[0]}_X_train', X_train)
                    np.save(f'./results/GRAD_TABLE_{args.K}_clusters_{datasets[0]}_X_test', X_test)
                    np.save(f'./results/GRAD_TABLE_{args.K}_clusters_{datasets[0]}_y_resid_test', y_resid_test)
                    continue
                result = {'exp_num': i, 
                          'dataset': dataset, 
                          'model': method, 
                          'marginal_coverage': marginal_coverage, 
                          'average_length': average_length,
                          'subgroup_coverage_metric2': subgroup_cov, 
                          'coverage_subgroups': coverage_subgroups, 
                          'q_lower': list(q_lower.squeeze()), 
                          'q_upper': list(q_upper.squeeze()), 
                          'test_subgroup_idxs': list(test_subgroup_idxs), 
                          'y_resid_test': list(y_resid_test.squeeze())}
                exp_results.append(result)
                
                counter += 1
                if (counter % 2) == 0 and args.save: 
                    print('saving intermediate results!')
                    R_sub = pd.DataFrame(exp_results)
                    R_sub.to_csv(f'./results/GRAD_FIGS_real_world_results_{datasets[0]}_{n_experiments}_runs_{methods[0]}_{args.K}_{args.num_alphas}.csv', index=False)

    
    R = pd.DataFrame(exp_results)
    print(R)
    if args.save: 
        R.to_csv(f'./results/GRAD_FIGS_real_world_results_{datasets[0]}_{n_experiments}_runs_{methods[0]}_{args.K}_{args.num_alphas}.csv', index=False)

