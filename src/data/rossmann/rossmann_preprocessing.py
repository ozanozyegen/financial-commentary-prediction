"""
    Preprocessing code for the rossman dataset
    Generates data/processed/rossman_lstm_data.pickle file
"""
from os import replace
import numpy as np
import pandas as pd
from scipy import stats
from math import ceil
from tqdm import tqdm
import pickle
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from data.helpers import week_of_month, multivariate_data, undersampler
from data.rossmann.gen_labels import label_generators
from sklearn.model_selection import StratifiedKFold

def gen_covariates(times, num_covariates=3):
    covariates = np.zeros((times.shape[0], num_covariates))
    for i, input_time in enumerate(times):
        covariates[i, 0] = input_time.weekday()
        covariates[i, 1] = input_time.month
        covariates[i, 2] = week_of_month(input_time)
    for i in range(num_covariates):
        covariates[:, i] = stats.zscore(covariates[:,i])
    return covariates[:, :num_covariates]

def moving_avg(x: pd.Series, window=7):
    return x.rolling(window).mean()

def exp_moving_avg(x: pd.Series, com=0.5):
    return x.ewm(com=com).mean()

def load_dataset_file(path='data/raw/rossmann-store-sales/train.csv'):
    data = pd.read_csv(path, parse_dates=True, index_col='Date') 
    data['StateHoliday'] = data['StateHoliday'].map({'0':0, 'a':1, 'b':2, 'c':3})
    data = data.astype(float)
    data['StateHoliday'].fillna(0, inplace=True)
    stores = data['Store'].unique()
    return data, stores

def generate_store_dfs(data, selected_stores):
    for selected_store in selected_stores:
        store = data.loc[data['Store'] == selected_store].sort_index()
        if store.shape[0] != 942 or selected_store == 988.0:
            continue
        yield store

def get_altered_series(series, config, converter=None):
    """ Alters time series feature to different formats
    """
    if converter is None: return series
    if converter == 'MA': # MA causes NaNs which should be handled later
        return series - moving_avg(series, config.get('MA_WINDOW', 7))
    elif converter == 'EMA':
        return series - exp_moving_avg(series, config.get('EMA_COM', 0.5))
    else:
        raise NotImplementedError()

def generate_rossmann_classification(config, selected_stores=None, univariate=True):
    """
    Arguments:
        selected_stores(list): Generate series only using the selected_stores
        univariate(bool): Returns univariate input train_x and train_y
        config.NUM_ROUND(int): Specifies random_state of split
    """
    data, stores = load_dataset_file()    
    np.random.seed(0)
    if not selected_stores:
        selected_stores = np.random.choice(stores, size=config['NUM_SERIES'], replace=False)
    
    train_start = '2013-01-01'
    get_date_idx = lambda date: np.where(store.index == date)[0][0]

    X_all, y_all = [], []
    for store_count, store in enumerate(generate_store_dfs(data, selected_stores)):
        covariates_df = store.drop('Sales', 1)
        time_covariates = gen_covariates(store.index)
        covariates_df['weekday'] = time_covariates[:, 0]
        covariates_df['month'] = time_covariates[:, 1]
        covariates_df['weekofmonth'] = time_covariates[:, 2]

        mod_series = get_altered_series(store['Sales'], config, converter=config.get('CONVERTER', None))
        data_np = np.concatenate((mod_series.values.reshape(-1,1), 
                                covariates_df.values.astype(np.float)), axis=1)

        history_size, target_size, stride = config['HISTORY_SIZE'], config['TARGET_SIZE'], config['STRIDE']
        X, _ = multivariate_data(data_np, target=data_np[:, 0], start_index=get_date_idx(train_start), 
                end_index=None, history_size=history_size, target_size=target_size, stride=stride)
        # Convert labels
        label_generator = label_generators[config['LABEL_GENERATOR']]
        y = label_generator(X, config)

        X_all.append(X);y_all.append(y);
    
    print('Packing dataset')
    concat = lambda x: np.concatenate(x, axis=0)
    X, y = concat(X_all), concat(y_all)
    if univariate:
        X = X[:,:,0]
    elif config['KEEP_WEEKDAYS']:
        X = X[:,:,[0,2]] # Keep Sales and Weekdays
    # Create K folds
    folds = []
    skf = StratifiedKFold(shuffle=True, random_state=config['NUM_ROUND']*10)
    for train_index, test_index in skf.split(X, y):
        train_x, train_y = X[train_index], y[train_index]
        test_x, test_y = X[test_index], y[test_index]

        if config.get('DOWNSAMPLE'): # Downsample majority class
            train_x, train_y = undersampler(train_x, train_y)
        if config.get('UPSAMPLE'): # Upsample minority class
            resampler = RandomOverSampler(random_state=0)
            train_x, train_y = resampler.fit_resample(train_x, train_y)
        folds.append({'train_x':train_x, 'train_y':train_y, 
                      'test_x':test_x, 'test_y':test_y})

    dataset_params = dict()
    dataset = {'folds': folds, 
        'train_x':train_x, 'train_y':train_y, 'test_x':test_x, 'test_y':test_y}
    return dataset_params, dataset

if __name__ == "__main__":
    config = {
        'HISTORY_SIZE':14,
        'STRIDE':1,
        'NUM_SERIES':100,
        'CONVERTER': 'EMA',
        'LABEL_HISTORY': 14,
        'LABEL_GENERATOR': 'MULTI_CLASS_PROMO',

        'TARGET_SIZE':5, # Not important
    }
    generate_rossmann_classification(config)