''' 
    Part of this code is taken from: https://github.com/Shai128/oqr. 
'''

import torch 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import Namespace
import pandas as pd
import sys
import helper

def get_dataset(name, base_path): 

    ''' 
    (Function taken from above Git repo.)

    Load a dataset
    
    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. 'path/to/datasets/directory/'
    
    Returns
    -------
    X : features (nXp)
    y : labels (n)
    
	'''
    if name == 'meps_19':
        df = pd.read_csv(base_path + 'meps_19_reg.csv')
        column_names = df.columns
        response_name = 'UTILIZATION_reg'
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != 'Unnamed: 0']
        
        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                   'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                   'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                   'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                   'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                   'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                   'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                   'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                   'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                   'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                   'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                   'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                   'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                   'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                   'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                   'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                   'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                   'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                   'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                   'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                   'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                   'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                   'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                   'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                   'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                   'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                   'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']
        
        y = df[response_name].values
        X = df[col_names].values
        
    if name == 'meps_20':
        df = pd.read_csv(base_path + 'meps_20_reg.csv')
        column_names = df.columns
        response_name = 'UTILIZATION_reg'
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != 'Unnamed: 0']
        
        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                   'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                   'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                   'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                   'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                   'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                   'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                   'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                   'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                   'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                   'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                   'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                   'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                   'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                   'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                   'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                   'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                   'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                   'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                   'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                   'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                   'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                   'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                   'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                   'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                   'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                   'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']
        
        y = df[response_name].values
        X = df[col_names].values
        
    if name == 'meps_21':
        df = pd.read_csv(base_path + 'meps_21_reg.csv')
        column_names = df.columns
        response_name = 'UTILIZATION_reg'
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != 'Unnamed: 0']
        
        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
                   'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                   'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                   'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                   'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                   'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                   'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                   'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                   'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                   'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                   'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                   'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                   'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                   'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                   'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                   'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                   'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                   'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                   'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                   'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                   'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                   'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                   'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                   'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                   'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                   'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                   'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']
        
        y = df[response_name].values
        X = df[col_names].values
        
    if name == 'facebook_1':
        df = pd.read_csv(base_path + 'facebook/Features_Variant_1.csv')        
        y = df.iloc[:,53].values
        X = df.iloc[:,0:53].values        
    
    if name == 'facebook_2':
        df = pd.read_csv(base_path + 'facebook/Features_Variant_2.csv')        
        y = df.iloc[:,53].values
        X = df.iloc[:,0:53].values 
        
    if name == 'bio':
        #https://github.com/joefavergel/TertiaryPhysicochemicalProperties/blob/master/RMSD-ProteinTertiaryStructures.ipynb
        df = pd.read_csv(base_path + 'CASP.csv')        
        y = df.iloc[:,0].values
        X = df.iloc[:,1:].values        
    
    if name == 'blog_data':
        # https://github.com/xinbinhuang/feature-selection_blogfeedback
        df = pd.read_csv(base_path + 'blogData_train.csv', header=None)
        X = df.iloc[:,0:280].values
        y = df.iloc[:,-1].values
    
    
    UCI_datasets = ['kin8nm', 'naval']

    if name in UCI_datasets:
        data_dir = 'UCI_Datasets/'
        data = np.loadtxt(base_path+data_dir+name+'.txt')
        X = data[:, :-1]
        y = data[:, -1]
        
    try:
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
    except Exception:
        raise Exception('invalid dataset')
    
    return X, y

def get_scaled_dataset(dataset_name, 
                       dataset_base_path, 
                       params={
                           'seed': 42, 
                           'test_size': 0.15
                       }): 
    log_transform_datasets = ['facebook_1', 'facebook_2', 'blog_data', 'bio']
    if 'bio' not in dataset_name:
        need_scaling = 'scaled' in dataset_name
        dataset_name = dataset_name.replace('scaled_', '')
    else:
        need_scaling = False
    X, y = get_dataset(dataset_name, dataset_base_path)
    if dataset_name in log_transform_datasets or need_scaling:
        y = np.log(y - min(y) + 1)
    data_size = len(X)  
    np.random.seed(params['seed'])
    idx = np.random.permutation(len(X))[:data_size]
    X = X[idx]
    y = y[idx]
    y = y.reshape(-1, 1)

    return scale_data(X, y, params)

def scale_data(X, y, params={}): 
    X_train, X_te, y_train, y_te = train_test_split(
        X, y, test_size=params['test_size'], random_state=params['seed'])
    X_tr, X_ca, y_tr, y_ca = train_test_split(
        X_train, y_train, test_size=0.5, random_state=params['seed'])

    s_tr_X = StandardScaler().fit(X_tr)
    s_tr_y = StandardScaler().fit(y_tr)

    X_tr = torch.Tensor(s_tr_X.transform(X_tr))
    X_ca = torch.Tensor(s_tr_X.transform(X_ca))
    X_te = torch.Tensor(s_tr_X.transform(X_te))

    y_tr = torch.Tensor(s_tr_y.transform(y_tr))
    y_ca = torch.Tensor(s_tr_y.transform(y_ca))
    y_te = torch.Tensor(s_tr_y.transform(y_te))
    y_all = torch.Tensor(s_tr_y.transform(y))

    X_train = torch.cat([X_tr, X_ca], dim=0)
    y_train = torch.cat([y_tr, y_ca], dim=0)
    out_namespace = Namespace(X_tr=X_tr, X_ca=X_ca, X_te=X_te,
                              y_tr=y_tr, y_ca=y_ca, y_te=y_te, y_al=y_all,
                              X_train=X_train, y_train=y_train)

    return out_namespace

if __name__ == '__main__': 
    params   = {'seed': 42, 'test_size': 0.15}
    data_out = get_scaled_dataset('meps_22', dataset_base_path='./real_data/', params=params)
    print(data_out.X_train[:5])
    print(data_out.y_train[:5])
    print(data_out.X_te[:5])
    print(data_out.y_te[:5])
    print(data_out.X_train.shape)
    print(data_out.y_train.shape)
    print(data_out.X_te.shape)
    print(data_out.y_te.shape)