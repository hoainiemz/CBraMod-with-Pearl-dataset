from ray import tune
from hyperparameter_optimization import eval
from trainable import setup_seed
from ray.air.integrations.wandb import WandbLoggerCallback
import random
from math import exp

import numpy as np
import torch

if __name__ == '__main__':
    search_space = {
        "lr_log": tune.grid_search([np.float64(-8.2803610349)]),
        "--dropout": tune.grid_search([np.float64(0.18471500678443833)]),
        "weight_decay_log": tune.grid_search([np.float64(-6.33955014653)]),
        "--clip_value": tune.grid_search([np.float64(1.289063037557857)]),
        "--seed": tune.grid_search([42, 123, 2021, 3407, 9999]) 
    }

    # use_wandb = True
    # callbacks = []
    # if use_wandb:
    #     callbacks.append(
    #         WandbLoggerCallback(
    #             project="finetune_test_seed",
    #             group="exp-001",
    #             log_config=True
    #         )
    #     )
        
    tuner = tune.Tuner(
        tune.with_resources(eval, {"gpu": 1, "cpu": 4}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="kappa", mode="max",
        ),
        # run_config=tune.RunConfig(
        #     name="finetune_test_seed",
        #     callbacks=callbacks
        # ),
    )

    results = tuner.fit()