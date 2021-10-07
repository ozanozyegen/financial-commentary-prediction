from configs.defaults import default_fcn_config

fcn_hyper = dict(
    N_FILTERS = [
        [4,4,4],
        [8,8,8],
        [16,16,16]
    ],
    PADDING = ['SAME'],
    KERNEL_SIZES = [
        [3,5,7],
        [5,7,9],
        [7,9,11]
    ],
    BATCH_NORM = [True, False],
    DROPOUT_RATE = [0, 0.2, 0.5]
)

lstm_hyper = dict(
    NUM_LAYERS = [0,1,2],
    NUM_UNITS = [50, 100, 150],
    DROPOUT_RATE = [0, 0.01, 0.02, 0.1]
)

hyper_model_config_loader = dict(
    lstm = lstm_hyper,
    fcn = fcn_hyper
)