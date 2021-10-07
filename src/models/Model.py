from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, \
    classification_report, precision_recall_fscore_support, f1_score, \
    accuracy_score
import numpy as np
import os, pickle

import wandb
from visualization.cam import cam_graph

from tensorflow.python.keras.callbacks import ModelCheckpoint

class Model(ABC):
    def __init__(self, config:dict, model) -> None:
        super().__init__()
        self.config = config
        self.model = model
    
    @abstractmethod
    def train(self,):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def report_results(self, dataset, cat_names:list):
        pred_y = self.predict(dataset['test_x'])
        print(classification_report(dataset['test_y'], pred_y,
            labels=list(range(len(cat_names))), target_names=cat_names)),
        cm = confusion_matrix(dataset['test_y'], pred_y)
        cmd = ConfusionMatrixDisplay(cm, display_labels=cat_names)
        cmd.plot()
    
    def log_results(self, dataset, logger):
        pred_y = self.predict(dataset['test_x'])
        logger.log({
            'acc': accuracy_score(dataset['test_y'], pred_y),
            'f1': f1_score(dataset['test_y'], pred_y, average='macro'),
            'conf_matrix': confusion_matrix(dataset['test_y'], pred_y)
        })
        if self.config['MODEL_NAME'] == 'fcn':
            results = cam_graph(self.model, dataset['test_x'], dataset['test_y'],
                self.config['LABELS'], self.config)
            np.save(os.path.join(logger.run.dir, 'cam.npy'), results)

    def report_dict(self, dataset:dict, average='macro'):
        """
        Returns:
            (precision, recall, f1, support): support is None
        """
        pred_y = self.predict(dataset['test_x'])
        return precision_recall_fscore_support(dataset['test_y'], pred_y, average=average)

    @abstractmethod
    def save(self, SAVE_DIR):
        pass

    @abstractmethod
    def load(self, SAVE_DIR):
        pass

class KerasModel(Model):
    """ Wrapper for Keras models """
    def __init__(self, config: dict, model: tf.keras.Model):
        super().__init__(config, model)

    def train(self, dataset:dict):
        """ 
        Arguments:
            dataset(dict): Dict with train_x, train_y, test_x, test_y keys
        """
        callbacks = [EarlyStopping(patience=500, restore_best_weights=True)]
        self.model.fit(dataset['train_x'], dataset['train_y'], 
            validation_data=(dataset['test_x'], dataset['test_y']),
            epochs=self.config['EPOCHS'], callbacks=callbacks,
            batch_size=self.config['BATCH_SIZE'])

    def predict(self, X):
        return self.model.predict(X).argmax(axis=1)
    
    def predict_proba(self, X):
        return self.model.predict(X)

    def save(self, SAVE_DIR):
        pickle.dump(self.config, open(os.path.join(SAVE_DIR, 'config.pickle'), 'wb'))
        self.model.save(os.path.join(SAVE_DIR, 'model.h5'))

    def load(self, SAVE_DIR):
        self.config = pickle.load(open(os.path.join(SAVE_DIR, 'config.pickle'), 'rb'))
        self.model = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'model.h5'))
    
        
class SkModel(Model):
    """ Wrapper for Sklearn Models """
    def __init__(self, config: dict, model, expand_inp=False):
        super().__init__(config, model)
        self.expand_inp = expand_inp

    def train(self, dataset:dict):
        """ 
        Arguments:
            dataset(dict): Dict with train_x, train_y keys
        """
        X = dataset['train_x']
        if self.expand_inp:
            X = self.conv_2d_to_3d(X)
        self.model.fit(X, dataset['train_y'])

    def predict(self, X):
        if self.expand_inp:
            X = self.conv_2d_to_3d(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.expand_inp:
            X = self.conv_2d_to_3d(X)
        return self.model.predict_proba(X)

    @staticmethod
    def conv_2d_to_3d(x: np.ndarray):
        return x[:,:,np.newaxis]

    def save(self, SAVE_DIR):
        pickle.dump(self.config, open(os.path.join(SAVE_DIR, 'config.pickle'), 'wb'))
        pickle.dump(self.model, open(os.path.join(SAVE_DIR, 'model.h5'), 'wb'))

    def load(self, SAVE_DIR):
        self.config = pickle.load(open(os.path.join(SAVE_DIR, 'config.pickle'), 'rb'))
        self.model = pickle.load(open(os.path.join(SAVE_DIR, 'model.h5'), 'rb'))
    