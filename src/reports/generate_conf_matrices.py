# %%
import wandb
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from helpers.wandb_common import get_wandb_df
from configs.defaults import Globs
# %%
api = wandb.Api()
df = get_wandb_df(Globs.PROJECT_NAME)
df = df[df.tags.apply(lambda x: 'best' in x or 'hyper' not in x)]

# %%
queries = [
    "MODEL_NAME == 'fcn' & DATASET == 'rossmann'",
    "MODEL_NAME == 'fcn' & DATASET == 'rule_based' & NUM_CLASSES == 4",
    "MODEL_NAME == 'fcn' & DATASET == 'variance_data' & NUM_CLASSES == 4 & label_type == 'standard'",
    "MODEL_NAME == 'lstm' & DATASET == 'rossmann'",
    "MODEL_NAME == 'lstm' & DATASET == 'rule_based' & NUM_CLASSES == 4",
    "MODEL_NAME == 'lstm' & DATASET == 'variance_data' & NUM_CLASSES == 4 & label_type == 'standard'",
]
# %%
configs, conf_matrices = [], []
for query in queries:
    df_query = df.query(query).sort_values(by='f1', ascending=False)
    if len(df_query) > 0:
        model_id = df_query.iloc[0]['id']
        run = api.run(f"oozyegen/{Globs.PROJECT_NAME}/{model_id}")
        configs.append( run.config.copy() )
        conf_matrices.append( run.summary['conf_matrix'].copy() )

# %%
SAVE_DIR = 'reports/figures/conf_matrix/'
for config, conf_matrix in zip(configs, conf_matrices):
    if config['DATASET'] == 'rossmann':
        labels = ['Low', 'Mid', 'High']
    else:
        labels = ['Promo', 'Phasing', 'POS', 'Other']
    df_cm = pd.DataFrame(conf_matrix, index=labels,
        columns=labels)
    sns.set(font_scale=2)
    plt.figure(figsize=(10,7))
    plt.yticks(va='center')
    sns_plot = sns.heatmap(df_cm, annot=True, fmt='g', 
                           annot_kws={'fontsize': 20},
                           cbar=False)
    sns_plot.set_ylabel('True Labels')
    plt.savefig(os.path.join(SAVE_DIR, \
        f'{config["DATASET"]}_{config["MODEL_NAME"]}_conf_matrix.png'), dpi=400)
    