import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from optuna_trainable import eval

storage = "sqlite:///optuna_rf_dashboard.db"

def objective(trial: optuna.Trial) -> float:
    # LẤY MẪU TRỰC TIẾP THEO THANG LOG (log=True)
    lr = trial.suggest_float("lr", 1e-6, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-2, log=True)
    dropout = trial.suggest_float("--dropout", 0.0, 0.2)
    clip_value = trial.suggest_float("--clip_value", 0.5, 2.0)

    config = {
        "--lr": lr,
        "--weight_decay": weight_decay,
        "--dropout": dropout,
        "--clip_value": clip_value,
    }

    # Nếu bạn có metric theo epoch, có thể report + prune trong vòng lặp train
    score = eval(config, trial)['kappa']
    return score

study = optuna.create_study(
    storage=storage,
    direction="maximize",
    sampler=TPESampler(seed=42),
    pruner=SuccessiveHalvingPruner(
        min_resource=1,      # số step/epoch tối thiểu
        reduction_factor=2,  # mỗi lần cắt giảm còn 1/2 trial
        min_early_stopping_rate=0
    ),
    load_if_exists=True
)

study.optimize(objective, n_trials=100)

print("Best value:", study.best_value)
print("Best params:", study.best_params)