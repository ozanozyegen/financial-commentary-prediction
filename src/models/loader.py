from models.fcn import create_fcn
from models.lstm import create_lstm
from models.Model import KerasModel, SkModel
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf

def model_loader(config):
    model_name = config['MODEL_NAME']
    tf.random.set_seed(0)
    if model_name == 'lstm':
        return KerasModel(config, create_lstm(config))
    elif model_name == 'fcn':
        return KerasModel(config, create_fcn(config))
    elif model_name == 'gbr':
        return SkModel(config, GradientBoostingClassifier(n_estimators=20, 
            learning_rate=0.1, max_features=12, max_depth=20))
    elif model_name == 'xgb':
        from xgboost import XGBClassifier
        return SkModel(config, XGBClassifier(max_depth=20, 
            verbosity=0, random_state=101))
    elif model_name == 'knn':
        from sktime.classification.distance_based._time_series_neighbors import KNeighborsTimeSeriesClassifier
        return SkModel(config, KNeighborsTimeSeriesClassifier(n_neighbors=1, 
            metric='dtw'), expand_inp=True)
