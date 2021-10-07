""" Create configs for hyperparameter tuning on rule data """
from configs.defaults import Globs, \
    model_config_loader, rule_based_configs, _unilever_config,\
    variance_data_configs
from configs.hyper_models import hyper_model_config_loader
from configs.helpers import get_permutations
from configs.hyper_best import best_lstm_config, best_fcn_config
from train.train_unilever import batch_train, train_all

def setup_rule_hyper_configs():
    """ Setup hyperparameter tuning of NN models for rule based data """
    configs = []
    for config_rule in rule_based_configs:
        for model_name in ['lstm', 'fcn']:
            model_config = model_config_loader[model_name]
            hyper_model_configs = get_permutations(hyper_model_config_loader[model_name])

            for hyper_model_config in hyper_model_configs:
                config = config_rule.copy()
                config.update({'MODEL_NAME':model_name,
                                'DATASET':'rule_based',
                                **_unilever_config})
                config.update(model_config)
                config.update(hyper_model_config)
                config.update({"hyper_config":str(hyper_model_config)}) # For grouping later
                config['NUM_ROUNDS'] = 10
                configs.append(config)
    return configs

def setup_rule_based_data_configs():
    """ Setup config files for various rule based data exps """
    configs = []
    for config_rule in rule_based_configs:
        for model_name in ['knn', 'gbr', 'xgb']:
            config = config_rule.copy()
            config.update({'MODEL_NAME': model_name,
                            'DATASET': 'rule_based',
                            **_unilever_config})
            config.update(model_config_loader[model_name])
            config['NUM_ROUNDS'] = 10
            configs.append(config)
    return configs

def setup_rossmann_data_configs():
    """ Setup config files for the Rossmann data exps """
    from configs.defaults import rossmann_config
    configs = []
    for model_name in Globs.model_names:
        config = dict()
        config.update({'MODEL_NAME': model_name,
                        'DATASET': 'rossmann',
                        **rossmann_config})
        config.update(model_config_loader[model_name])  
        if model_name == 'lstm':
            config.update(best_lstm_config)
        elif model_name == 'fcn':
            config.update(best_fcn_config)
        config['NUM_ROUNDS'] = 10
        config.update({'BATCH_SIZE':128,
                        'EPOCHS':100})
        configs.append(config)
    return configs

if __name__ == "__main__":
    batch_train(setup_rule_based_data_configs(), chunk_size=5)
    batch_train(setup_rule_hyper_configs(), wandb_tags=['hyper'], batch_experiments=True, chunk_size=5)
    train_all(setup_rossmann_data_configs())
