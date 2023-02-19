import os
import numpy as np
import pandas as pd
from ray import tune
from autotabular.common import get_logger
from .constants import (
    MIN_N_ESTIMATORS,
    MAX_N_ESTIMATORS,
    TRAIN_LOG_FILE,
    FEATURE_IMPORTANCE_FILE,
    SEARCH_HISTORY_FILE,
    LEARNING_CURVE_FILE
)

logger = get_logger()


def build_train_settings(args):
    if args.log_file_name is None:
        os.makedirs(args.output_dir, exist_ok=True)
        args["log_file_name"] = os.path.join(args.output_dir, TRAIN_LOG_FILE)

    train_settings = {}
    params_list = ['task',
                   'seed',
                   'time_budget',
                   'metric',
                   'estimator_list',
                   'log_file_name',
                   'fit_kwargs']

    for param_key in params_list:
        param = args.get(param_key)
        if param is not None:
            if param_key == 'fit_kwargs':
                if isinstance(param, dict):
                    train_settings.update(param)
            else:
                train_settings[param_key] = param
    return train_settings


def search_space_transform(hp: str, domain: list):
    if (hp == "n_estimators") or \
            (hp == "max_leaves") or (hp == "num_leaves"):
        lower = max(MIN_N_ESTIMATORS, domain[0])
        upper = max(lower, min(MAX_N_ESTIMATORS, domain[1]))  # upper must be larger than lower
        return {
                "domain": tune.lograndint(lower=lower, upper=upper),
                "init_value": lower,
                "low_cost_init_value": lower,
            }
    elif hp == "learning_rate":
        return {
                "domain": tune.loguniform(lower=domain[0], upper=domain[1]),
                "init_value": 0.1,
            }
    elif hp == "colsample_bytree":
        assert domain[1] <= 1.0, \
            f"Error, 'colsample_bytree' should no bigger than 1.0"
        return {
                "domain": tune.uniform(lower=domain[0], upper=domain[1]),
                "init_value": 1.0,
            }
    elif (hp == "reg_alpha") or (hp == "reg_lambda"):
        return {
                "domain": tune.loguniform(lower=domain[0], upper=domain[1]),
                "init_value": domain[0],
            }
    else:
        return None


def get_search_space(hp_domain: dict):
    custom_hp = {
        "xgboost": {},
        "lgbm": {},
    }
    for hp, domain in hp_domain.items():
        hp_search_space = search_space_transform(hp, domain)
        if (hp == "max_leaves") or (hp == "num_leaves"):
            custom_hp["xgboost"]["max_leaves"] = hp_search_space
            custom_hp["lgbm"]["num_leaves"] = hp_search_space
        elif hp_search_space is not None:
            custom_hp["xgboost"][hp] = hp_search_space
            custom_hp["lgbm"][hp] = hp_search_space
    return custom_hp


def save_feature_importance(args, feature_importances):
    save_file_path = os.path.join(args.output_dir, FEATURE_IMPORTANCE_FILE)
    fea_imp = pd.DataFrame([args.feature_cols, feature_importances]) \
        .transpose().sort_values(by=1, ascending=False).rename(columns={0: "feature", 1: "importances"})
    fea_imp.to_csv(save_file_path, index=0)


def save_learning_curve(args, train_settings: dict, larger_better_loss_list: list):
    from flaml.automl.data import get_output_from_log
    import matplotlib.pyplot as plt

    time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = \
        get_output_from_log(filename=train_settings["log_file_name"], time_budget=args.time_budget)

    best_valid_eval = np.array(best_valid_loss_history)
    if train_settings["metric"] in larger_better_loss_list:
        best_valid_eval = 1 - best_valid_eval
    search_history_pd = pd.DataFrame([time_history, list(best_valid_eval)]) \
        .transpose().rename(columns={0: "Wall Time (s)", 1: train_settings["metric"]})
    search_history_path = os.path.join(args.output_dir, SEARCH_HISTORY_FILE)
    search_history_pd.to_csv(search_history_path, index=0)
    fig = plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Wall Time (s)")
    plt.ylabel("Validation {}".format(train_settings["metric"]))
    plt.step(time_history, best_valid_eval, where="post")

    learning_curve_path = os.path.join(args.output_dir, LEARNING_CURVE_FILE)
    fig.savefig(learning_curve_path)
