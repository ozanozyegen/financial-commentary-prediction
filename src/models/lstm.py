import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, LSTM, Dense
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.models import Model
from deprecated import deprecated

@deprecated("Use hyperparameter supporting create_lstm func instead")
def create_basic_lstm(config):
    inp = Input(shape=(config['HISTORY_SIZE']))
    res = Reshape((config['HISTORY_SIZE'],1))(inp)
    hidden = LSTM(100, dropout=0.01)(res)
    out = Dense(config['NUM_CLASSES'], activation='softmax')(hidden)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
        metrics='accuracy')
    return model

def create_lstm(config):
    inp = Input(shape=(config['HISTORY_SIZE']))
    x = Reshape((config['HISTORY_SIZE'],1))(inp)
    for _ in range(config['NUM_LAYERS']):
        x = LSTM(units=config['NUM_UNITS'], dropout=config['DROPOUT_RATE'],
            return_sequences=True)(x)
    x = LSTM(units=config['NUM_UNITS'], dropout=config['DROPOUT_RATE'],
            return_sequences=False)(x)
    out = Dense(config['NUM_CLASSES'], activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
        metrics='accuracy')
    return model

    
