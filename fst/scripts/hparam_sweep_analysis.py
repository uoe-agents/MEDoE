import wandb
import pandas as pd
import numpy as np
from fst.utils.util import lod_to_dol

api = wandb.Api()

USERNAME = "WANDB_USERNAME"
PROJECT = "PROJECT"
GROUP = "GROUP"

MODE = "auc"

if MODE == "auc":
    result_key = "AUCgap"
    result_col_name = "AUC gap"
else:
    result_key = "Return/team/mean"
    result_col_name = "Return"

def process(run):
    cfg = run.config
    summary = run.summary
    return {
            "lr": cfg["agents"]["lr"]["actor"],
            "ent coef": cfg["agents"]["ent_coef"],
            "n steps": cfg["agents"]["n_steps"],
            "epochs": cfg["agents"]["n_epochs"],
            "seed": cfg["seed"],
            "AUCgap": summary.get(result_key, 2),
        }

HPARAM_COLS = [
    "agents.lr.actor",
    "agents.ent_coef",
    "agents.n_steps",
    "agents.n_epochs",
    ]
AGG_COLS = [
    "seed"
    ]
RESULT_COLS = [
        result_col_name
    ]

runs = api.runs(
        f"{USERNAME}/{PROJECT}",
        filters={"group": GROUP},
        )
data = pd.DataFrame(lod_to_dol([process(r) for r in runs]))
data_without_seed = data.drop(columns=['seed'])
average_by_group = data_without_seed.groupby(data_without_seed.columns.drop(result_col_name).tolist())[result_col_name].mean()
sorted_average_by_group = average_by_group.sort_values(ascending=True)
print(sorted_average_by_group)
