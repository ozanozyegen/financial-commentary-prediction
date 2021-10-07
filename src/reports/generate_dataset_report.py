import numpy as np
from configs.defaults import (rossmann_config, rule_based_configs,
                              variance_data_configs)
from data.loaders import dataset_loader

datasets = dict(
    rule_based=rule_based_configs[0],
    variance_data=variance_data_configs[-1],
    rossmann=rossmann_config,
)

for dataset_name, config in datasets.items():
    config.update({'NUM_ROUND':0})
    config.update({'UPSAMPLE':False, 'DOWNSAMPLE':False})
    dataset_params, dataset = dataset_loader[dataset_name](config)
    print(dataset_name)
    train_counts = np.unique(dataset['train_y'], return_counts=True)
    test_counts = np.unique(dataset['test_y'], return_counts=True)
    print(train_counts[1]+test_counts[1])
    print(train_counts)
    print(test_counts)