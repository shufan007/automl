# -*- coding: utf-8 -*-
"""
# @Time : 2022.07.27
# @Author : shuangxi.fan
# @Description : tune flow on timeseries task
"""
import os
import time
import math
import random
import numpy as np
import json
import uuid
import torch
import torch.nn as nn

from ray import air, tune
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from ray.train.torch import prepare_data_loader, prepare_model
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig

# from ray.tune.integration.torch import DistributedTrainableCreator
# from functools import partial
# import flaml

from common import dotdict, get_logger, save_checkpoint, save_torchscript, save_searched_model
from common.dl_utils import save_checkpoint
# from trainer_interface_def import load_data, get_model, run_train, run_evaluate
# from forecasting.interface.trainer_interface import load_data, get_model, run_train, run_evaluate, load_model

import gc
gc.collect()

logger = get_logger()
# NCCL_TIMEOUT_S = 120


def acquire_device(args):
    devices = None
    args.device_ids = []
    if torch.cuda.is_available():
        devices = os.getenv("CUDA_VISIBLE_DEVICES")
        logger.info(f"CUDA_VISIBLE_DEVICES: {devices}")
    if devices is not None:
        device_ids = devices.replace(' ', '').split(',')
        # device_ids = [int(i) for i in device_ids]
        args.device_ids = list(range(0, len(device_ids)))
        # gpu = device_ids[0]
        gpu = 0
        device = torch.device('cuda:{}'.format(gpu))
        logger.info('Use GPU: cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
        logger.info('Use CPU')
    args.device = device
    return args


def model_parallel(args, model):
    args = acquire_device(args)
    model = model.to(args.device)
    """
    unc:`ray.train.torch.get_device` and :func:`ray.train.torch.prepare_model`
    """

    if not args.is_tune:
        args.distributed = False
    if args.distributed:
        """
        # model = prepare_model(model) # which will wrap it in DistributedDataParallel
        model = nn.parallel.DistributedDataParallel(model,
                                                    # device_ids=self.args.device_ids,
                                                    find_unused_parameters=True
                                                    )        
        """
        model = nn.DataParallel(model)

    elif len(args.device_ids) > 1:
        model = nn.DataParallel(model)

    # model = prepare_model(model)

    return model


def report_final(model, args, val_value):
    """

    :param model:
    :param args:
    :param val_value:
    :return:
    """
    config_str = ''
    if args.config is not None:
        config_str = ','.join([item[0] + '=' + str(item[1]) for item in args.config.items()])
    checkpoint_dir = "model_" + config_str

    os.makedirs(checkpoint_dir, exist_ok=True)
    suffix = str(uuid.uuid1())
    save_model = model
    if args.distributed:
        save_model = model.module
    save_checkpoint(save_model, checkpoint_dir, suffix)
    checkpoint = Checkpoint.from_directory(checkpoint_dir)

    logger.info(f">>> report_final: args.metric:{args.metric}, "
                f"val_value:{val_value}, checkpoint:{checkpoint}")
    session.report({args.metric: val_value}, checkpoint=checkpoint)


def train_func(config, args, train_loader=None, val_loader=None):
    """
    Trainable function
    :param config:
    :param args:
    :param train_loader:
    :param val_loader:
    :return:
    """
    logger.info(f">>> training with config: {config} ...")
    args = dotdict(args)
    args.config = config
    if config is not None:
        args.update(config)

    model = args.trainer_module.get_model(args)
    model = model_parallel(args, model)
    # model = prepare_model(model)

    # Prepares DataLoader for distributed execution
    """
    if train_loader is not None:
        train_loader = prepare_data_loader(train_loader)
    if val_loader is not None:
        val_loader = prepare_data_loader(val_loader)
    """

    args.trainer_module.run_train(model, args, train_loader)

    val_value = args.trainer_module.run_evaluate(model, args, val_loader)

    logger.info(f"train_func: val_value: {val_value}")

    """ save checkpoint """
    report_final(model, args, val_value)


def get_search_space(search_space_args):
    """
    transform search_space_args to tune search space
    :param search_space_args:
    :return: search_space
    """
    exponent_base = 2
    search_space = {}
    if search_space_args is None:
        return search_space

    for k, attr in search_space_args.items():
        value = attr["value"]
        if attr["htype"] == "choice":
            search_space[k] = tune.choice(value)
        elif attr["htype"] == "uniform":
            search_space[k] = tune.uniform(value[0], value[1])
        elif attr["htype"] == "randint":
            search_space[k] = tune.randint(value[0], value[1])
        elif attr["htype"] == "lograndint":
            search_space[k] = tune.lograndint(value[0], value[1])
        elif attr["htype"] == "loguniform":
            search_space[k] = tune.loguniform(value[0], value[1])
        elif attr["htype"] == "quniform":
            search_space[k] = tune.quniform(value[0], value[1], value[2])
        elif attr["htype"] == "qloguniform":
            search_space[k] = tune.qloguniform(value[0], value[1], value[2])
        elif attr["htype"] == "qrandint":
            search_space[k] = tune.qrandint(value[0], value[1], value[2])
        elif attr["htype"] == "qlograndint":
            search_space[k] = tune.qlograndint(value[0], value[1], value[2])
        elif attr["htype"] == "exponent":
            lower = int(math.log(value[0], exponent_base))
            upper = int(math.log(value[1], exponent_base))
            space_list = [exponent_base ** i for i in range(lower, upper+1)]
            search_space[k] = tune.choice(space_list)
    return search_space


def tune_model_evaluate(args, best_result, val_loader):
    checkpoint_dir = best_result.checkpoint.to_directory()
    logger.info(">>> Evaluate best model on Val data...")
    logger.info(f"Best model path: {checkpoint_dir}")

    logger.info('>>> Start validating... ')
    args.update(best_result.config)
    args.distributed = False

    model = args.trainer_module.get_model(args)
    model = args.trainer_module.load_model(model, checkpoint_dir=checkpoint_dir)

    val_loss = args.trainer_module.run_evaluate(model, args, val_loader)
    logger.info(f"Best model val_loss: {val_loss}")


def save_tune_result(args, best_result):
    logger.info(f"Best trial config: {best_result.config}")
    logger.info(f"Best result metrics: {best_result.metrics}")
    logger.info(f"Best result log_dir: {best_result.log_dir}")

    """ save best trial params """
    json_save_path = os.path.join(args.output_path, "best_trial_params.json")
    with open(json_save_path, 'w', encoding='utf-8') as file:
        json.dump(best_result.config, file, ensure_ascii=False)

    checkpoint_dir = best_result.checkpoint.to_directory()

    logger.info(">>> Evaluate best model on Val data...")
    logger.info(f"Best model path: {checkpoint_dir}")

    logger.info('>>> Start validating... ')
    args.update(best_result.config)
    args.distributed = False

    model = args.trainer_module.get_model(args)
    model = args.trainer_module.load_model(model, checkpoint_dir=checkpoint_dir)

    logger.info(f">>> Model archive: save model to {args.output_path}")
    save_searched_model(args, save_path=args.output_path, checkpoint_dir=checkpoint_dir, model=model)


def automl_tune(args):
    """
    automl tune pipline
    """
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    logger.info(">>> prepare dataset  ...")
    train_loader, val_loader = args.trainer_module.load_data(args)

    if not args.is_tune:
        args.num_samples = 1

    logger.info("Distributed Trainable Creator...")
    """ Checkpointing and synchronization: set local_dir and sync_config for Checkpoints"""
    sync_config = None
    if args.local_dir is not None:
        # if shared storage
        if args.local_dir.find('/nfs/') >= 0:
            sync_config = tune.SyncConfig(
                syncer=None  # Disable syncing
            )

    # Note: parameter args must be pass as type of dict
    logger.info("Generate search space...")
    search_space = get_search_space(args.search_space)

    logger.info(">>> Start tuning...")
    # -- debug --
    if args.verbose:
        os.environ['RAY_PICKLE_VERBOSE_DEBUG'] = "1"
        # os.environ['TUNE_DISABLE_AUTO_CALLBACK_SYNCER'] = "1"

    logger.info(f">>> training with args.local_dir: {args.local_dir}, "
                f"args.num_training_workers {args.num_training_workers}, "
                f"args.num_samples: {args.num_samples}")

    trainable = tune.with_resources(
        tune.with_parameters(train_func,
                             args=dict(args),
                             train_loader=train_loader,
                             val_loader=val_loader
                             ),
        resources=args.resources_per_trial)

    tune_config = tune.TuneConfig(num_samples=args.num_samples,
                                  time_budget_s=args.time_budget_s,
                                  scheduler=ASHAScheduler(metric=args.metric, mode=args.metric_mode),
                                  max_concurrent_trials=args.num_training_workers,
                                  )
    run_config = air.RunConfig(name="experiment",
                               local_dir=args.local_dir,
                               sync_config=sync_config,
                               )

    start_time = time.time()

    tuner = tune.Tuner(trainable=trainable,
                       param_space=search_space,
                       run_config=run_config,
                       tune_config=tune_config,
                       )

    tune_result = tuner.fit()

    # logger.info(f"#trials={len(result.trials)}")
    logger.info(f"time={time.time() - start_time}")

    best_result = tune_result.get_best_result(metric=args.metric, mode=args.metric_mode, scope="all")

    save_tune_result(args, best_result)

    tune_model_evaluate(args, best_result, val_loader)

