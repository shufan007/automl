# -*- coding: utf-8 -*-
"""
# @Time : 2022.07.27
# @Author : shuangxi.fan
# @Description : tune flow on timeseries task
"""
import os, sys
import torch

from common import get_logger, get_suffix_file
from trainer import Trainer, args_prepare
from data_provider import get_data_reader, get_dataset, data_provider

import gc
gc.collect()

logger = get_logger()


def user_args_preprocess(args):
    """
    user defined args preprocess
    :param args:
    :return: args, args after preprocess
    """
    args = args_prepare(args)
    return args


def load_data(args, train_data_path=None, val_data_path=None):
    logger.info(">>> prepare dataset  ...")
    data_reader = get_data_reader(args, data_path=args.train_data)
    train_dataset = get_dataset(args, data_path=args.train_data, data_reader=data_reader, flag='train')
    if args.val_data is not None:
        val_dataset = get_dataset(args, data_path=args.val_data)
    elif args.split_ratio > 0:
        val_dataset = get_dataset(args, data_path=args.train_data, data_reader=data_reader, flag='val')

    train_loader = data_provider(args, data_set=train_dataset)
    val_loader = data_provider(args, data_set=val_dataset)
    return train_loader, val_loader


def get_model(args):
    trainer = Trainer(args)
    return trainer.model


def run_train(model, args, train_loader):
    trainer = Trainer(args)
    trainer.train(model, args, train_loader)


def run_evaluate(model, args, val_loader):
    trainer = Trainer(args)
    eval_value = trainer.eval(model, args, val_loader)
    return eval_value


def load_model(model, checkpoint_dir=None, model_type='checkpoint'):
    if checkpoint_dir is not None:
        logger.info(f'loading model from {checkpoint_dir}')
        if model_type == 'checkpoint':
            # path = os.path.join(checkpoint_dir, "checkpoint.pth")
            model_name = get_suffix_file(checkpoint_dir, '.pth')
            path = os.path.join(checkpoint_dir, model_name)
            logger.info(f'model path: {path}')
            model.load_state_dict(torch.load(path))
        else:
            path = os.path.join(checkpoint_dir, "model.torchscript")
            logger.info(f'model path: {path}')
            model = torch.jit.load(path)
    return model
