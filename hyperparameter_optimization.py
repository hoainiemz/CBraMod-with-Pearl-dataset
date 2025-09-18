import argparse
import random
from math import exp

import wandb
import numpy as np
import torch

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from trainable import eval

if __name__ == '__main__':
    search_space = {
        "lr_log": tune.uniform(-13.8155, -7.6009)
    }
    
    asha = ASHAScheduler(
        time_attr="epoch",        # dùng "epoch" từ tune.report
        max_t=50,                 # đúng yêu cầu 50 epochs
        grace_period=15,           # không cắt quá sớm
        reduction_factor=2
    )
    
    bo = BayesOptSearch(
        metric="kappa",
        mode="max",
        random_search_steps=8,    # khởi động bằng vài điểm random
        random_state=42
    )
    
    tuner = tune.Tuner(
        tune.with_resources(eval, {"gpu": 1, "cpu": 4}),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="kappa", mode="max",
            scheduler=asha,
            search_alg=bo,
            num_samples=50,         # số trial (điều chỉnh theo tài nguyên)
        )
    )
    results = tuner.fit()