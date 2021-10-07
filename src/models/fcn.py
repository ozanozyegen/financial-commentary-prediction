import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, LSTM, Dense, Conv1D,\
    BatchNormalization, Dropout, Activation, GlobalAveragePooling1D
from tensorflow.python.keras.models import Model

def create_fcn(config):
    inp = Input(shape=(config['HISTORY_SIZE']))
    res = Reshape((config['HISTORY_SIZE'],1))(inp)
    x = Conv1D(filters=config['N_FILTERS'][0],
        kernel_size=config['KERNEL_SIZES'][0],
        padding=config['PADDING'])(res)
    if config['BATCH_NORM']:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(config['DROPOUT_RATE'])(x)
    x = Conv1D(filters=config['N_FILTERS'][1],
        kernel_size=config['KERNEL_SIZES'][1],
        padding=config['PADDING'])(x)
    if config['BATCH_NORM']:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(config['DROPOUT_RATE'])(x)
    x = Conv1D(filters=config['N_FILTERS'][2],
        kernel_size=config['KERNEL_SIZES'][2],
        padding=config['PADDING'])(x)
    if config['BATCH_NORM']:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(config['NUM_CLASSES'], 
        activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
        metrics='accuracy')
    return model