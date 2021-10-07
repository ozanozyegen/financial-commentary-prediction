import wandb
from helpers.gpu_selection import auto_gpu_selection
from helpers.wandb_common import wandb_save, convert_wandb_config_to_dict
from helpers.parallelize import run_in_separate_process
from models.loader import model_loader
from data.loaders import dataset_loader
from configs.defaults import Globs 

def batch_train(configs, wandb_tags=[],
    batch_experiments=True, chunk_size=100):
    """ Create batches of experiments to prevent wandb crashes
    """
    if len(configs)>100 and batch_experiments:
       for i in range(0, len(configs), chunk_size):
           chunk_configs = configs[i:i+chunk_size]
           run_in_separate_process(train_all, (chunk_configs, wandb_tags))
    else:
        train_all(configs, wandb_tags)

def get_round_configs(configs):
    for config in configs:
        if config['NUM_ROUNDS']:
            for num_round in range(config['NUM_ROUNDS']):
                config.update({'NUM_ROUND':num_round})
                yield config
        else:
            yield config

def train_all(configs, wandb_tags=[]):
    """ Runs multiple experiments sequentially
    Arguments:
        configs(list): list of config dictionaries
        batch_experiments(bool): batch experiments to prevent wandb crash
    """
    for config in get_round_configs(configs):
        print(config)
        train(config, wandb_tags)

def train(config, wandb_tags=[]):
    auto_gpu_selection()
    wandb_save(True, True)
    run = wandb.init(project=Globs.PROJECT_NAME, config=config, 
        tags=wandb_tags, reinit=True)
    with run:
        config = wandb.config
        print(config)
        dataset_params, dataset = dataset_loader[config.DATASET](config)
        config.update(dataset_params)
        model = model_loader(convert_wandb_config_to_dict(config))
        print(dataset['train_x'].shape, dataset['train_y'].shape,
            dataset['test_x'].shape, dataset['test_y'].shape)

        model.train(dataset)
        # model.save(wandb.run.dir)
        model.log_results(dataset, wandb)
