# -*- coding: utf-8 -*-
"""
# @Time : 2022.10.21
# @Author : shuangxi.fan
# @Description : automl search interface definition
"""


def user_args_preprocess(args):
    """
    user defined args preprocess
    :param args:
    :return: args, args after preprocess
    """
    return args


def load_data(args, train_data_path=None, val_data_path=None):
    """

    :param args:
    :param train_data_path:
    :param val_data_path:
    :return: train_loader, val_loader
    """
    pass


def get_model(args):
    """

    :param args:
    :return: model
    """
    pass


def run_train(model, args, train_loader):
    """

    :param model:
    :param args:
    :param train_loader:
    :return: model
    """
    pass


def run_evaluate(model, args, val_loader):
    """

    :param model:
    :param args:
    :param val_loader:
    :return: eval_value
    """
    pass

