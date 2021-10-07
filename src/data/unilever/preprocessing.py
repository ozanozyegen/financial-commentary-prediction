from data.helpers import undersampler
import numpy as np
import pandas as pd
import os, pickle, random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def sampling(X, y, config):
    """ Samples the data 
    Arguments:
        config.NUM_ROUND(int): Specifies random_state of split
    """
    split_state = config['NUM_ROUND']*10
    train_x, test_x, train_y, test_y = train_test_split(X, y,
                                        test_size=0.2,
                                        random_state=split_state,
                                        stratify=y)
    
    if config.get('DOWNSAMPLE'):
        train_x, train_y = undersampler(train_x, train_y)
    if config.get('UPSAMPLE'):
        resampler = RandomOverSampler(random_state=101)
        train_x, train_y = resampler.fit_resample(train_x, train_y)

    return train_x, train_y, test_x, test_y

def preprocess_variance_data(config,
        DATA_PATH='data/processed/topic_modeling_data.pickle',):
    """ Preprocessing for the variance dataset """
    label_type = config['label_type']
    np.random.seed(101)
    df_ts = pickle.load(open(DATA_PATH, 'rb'))
    labels = {'Promo':0.0,'Phasing':1.0,'POS':2.0,'Other':3.0,'NoComm':4.0}

    info_cols = [str(i) for i in range(1,14)]
    df_ts["standard"] = df_ts["truth"].replace(to_replace=labels)
    if config['NUM_CLASSES'] == 4: # Remove NoComm
        X = df_ts[df_ts[label_type]!=4][info_cols].values
        y = df_ts[df_ts[label_type]!=4][label_type].values
        labels = ['Promo', 'Phasing', 'POS', 'Other']
    elif config['NUM_CLASSES'] == 5:
        X = df_ts[info_cols].values
        y = df_ts[label_type].values
        labels = ['Promo', 'Phasing', 'POS', 'Other', 'NoComm']

    train_x, train_y, test_x, test_y = sampling(X, y, config)
    
    dataset = {'train_x':train_x, 'train_y':train_y, 
                'test_x':test_x, 'test_y':test_y}
    dataset_params = dict(
        LABELS = labels,
    )
    return dataset_params, dataset

def preprocess_rule_based_data(config,
        DATA_PATH='data/processed/rule_based_data_5.pickle'):
    """ Preprocessing for the rule based dataset """
    X, y = pickle.load(open(DATA_PATH, 'rb'))
    y = np.argmax(y,1)
    labels = ['Promo', 'Phasing', 'POS', 'Other', 'NoComm']
    
    if config['NUM_CLASSES'] == 4:
        nocomm_mask = [i!=4 for i in y]
        X = X[nocomm_mask]
        y = y[nocomm_mask]
        labels = ['Promo', 'Phasing', 'POS', 'Other']
    
    train_x, train_y, test_x, test_y = sampling(X, y, config)
    dataset = {'train_x':train_x, 'train_y':train_y, 
                'test_x':test_x, 'test_y':test_y}
    dataset_params = dict(
        LABELS = labels,
    )
    return dataset_params, dataset    
