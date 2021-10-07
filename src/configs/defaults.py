from configs.helpers import get_permutations

class Globs:
    PROJECT_NAME = 'Unilever'
    model_names = ['lstm', 'fcn', 'knn', 'gbr', 'xgb']

default_train_config = dict(
    BATCH_SIZE = 26,
    EPOCHS = 1000,
)

default_fcn_config = dict(
    N_FILTERS = [8,8,8],
    PADDING = 'SAME',
    KERNEL_SIZES = [5,7,9],
    BATCH_NORM = True,
    DROPOUT_RATE = 0.2
)

default_lstm_config = dict(
    NUM_LAYERS = 0,
    NUM_UNITS = 100,
    DROPOUT_RATE = 0.01,
)

model_config_loader = dict(
    lstm = {**default_lstm_config, **default_train_config},
    fcn = {**default_fcn_config, **default_train_config},
    knn = dict(),
    gbr = dict(),
    xgb = dict()
)

_unilever_config = dict(
    HISTORY_SIZE = 13,
)

_variance_data_4_configs = dict(
    NUM_CLASSES = [4],
    label_type = ['LDA', 'GSDMM', 'GPUDMM', 'standard'],
    UPSAMPLE=[True],
    DOWNSAMPLE=[False]
)

_variance_data_5_configs = dict(
    NUM_CLASSES = [5],
    label_type = ['LDA', 'GSDMM', 'GPUDMM', 'standard'],
    UPSAMPLE=[True],
    DOWNSAMPLE=[True]
)

variance_data_configs = get_permutations(_variance_data_4_configs)

# variance_data_hyper_configs = _variance_data_4_configs.copy()
# variance_data_hyper_configs.update(dict(label_type=['standard']))
# variance_data_hyper_configs = get_permutations(variance_data_hyper_configs)

_rule_based_4_configs = dict(
    NUM_CLASSES = 4,
    UPSAMPLE=True,
    DOWNSAMPLE=False,
    **_unilever_config
)

_rule_based_5_configs = dict(
    NUM_CLASSES = 5,
    UPSAMPLE=True,
    DOWNSAMPLE=True,
    **_unilever_config
)

rule_based_configs = [_rule_based_4_configs]

rossmann_config = dict(
    HISTORY_SIZE = 14,
    STRIDE = 1,
    NUM_SERIES = 100,
    CONVERTER = 'EMA',
    LABEL_HISTORY = 14,
    LABEL_GENERATOR = 'MULTI_CLASS_PROMO',
    LABELS = ['Low','Mid', 'High'],
    DOWNSAMPLE = True,
    KEEP_WEEKDAYS = True,
    TARGET_SIZE = 5, # Not important
    NUM_CLASSES = 3,
)