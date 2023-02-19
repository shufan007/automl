# -*- coding: utf-8 -*-
"""
# @Time : 2022.07.27
# @Author : shuangxi.fan
# @Description : tune flow on timeseries task
"""
import os
from common import dotdict, get_logger
from data_provider.data_factory import get_data_reader, get_dataset, get_model_input_args, data_provider

MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import gc
gc.collect()

logger = get_logger()


def default_args_check(args):
    """
    default_args_check, go to UserGuides for details of the args usage
    :param args:
    :return: args
    """
    choice_args_dict = {
        "data_format": ['table', 'array-table', 'textfile', 'tensor'],
        "freq": ['s', 't', 'h', 'd', 'b', 'w', 'm'],
        "model": ['FEDformer', 'Autoformer', 'Informer', 'Transformer'],
        "embed": ['timeF', 'fixed', 'learned']
    }

    default_args_dict = {
        "data_format": 'textfile',
        "date_col": "date",
        "freq": 'h',
        "seq_len": 24,
        "pred_len": 1,
        "model": 'Autoformer',
        "d_model": 512,
        "n_heads": 8,
        "enc_layers": 2,
        "dec_layers": 1,
        "d_ffn": 2048,
        "moving_avg": 25,
        "dropout": 0.05,
        "embed": 'timeF',
        "train_epochs": 10,
        "patience": 3,
        "batch_size": 32,
        "learning_rate": 0.0001,
        "loss": "mse",
        "verbose": False,
        "sample_step": 1
    }

    for k, v in choice_args_dict.items():
        if k in args and args[k] is not None:
            assert args[k] in v, f"Error, param: {k}, should be one of {v}!"

    for k, v in default_args_dict.items():
        if k not in args:
            args[k] = v

    # logger.info(f">>> default_args_check: {args}")

    return args


def set_hidden_args(args):
    """
    set default args
    :param args:
    :return:
    """
    # supplementary config for FEDformer model
    args.version = 'Fourier'  # for FEDformer, default='Fourier', choices=['Fourier', 'Wavelets']
    args.mode_select = 'random'  # default='random', choices=['random', 'low']
    args.modes = 64   # 'modes to be selected random 64'
    args.L = 3  # 'ignore level'
    args.base = 'legendre'  # mwt base
    args.cross_activation = 'tanh'  # mwt cross atention activation function tanh or softmax

    """model define config"""
    args.factor = 1     # attn factor
    args.distil = True  # for Informer, whether to use distilling in encoder

    args.activation = 'gelu'   # activation
    args.output_attention = False   # whether to output attention in encoder
    args.lr_adj = 'type1'      # 'adjust learning rate', choices=['type1', 'type2', 'type3', 'type4']
    # args.embed = 'timeF'       # 'time features encoding, options:[timeF, fixed, learned]'
    # args.patience = 3    # early stopping patience

    """optimization config"""
    args.use_amp = False   # use automatic mixed precision training

    return args


def set_model_definition_path(args):
    model_definition_path = ['models', 'layers', 'utils/model_load_example.py']
    args.model_definition_path = [os.path.join(MODULE_DIR, i) for i in model_definition_path]
    return args


def data_format_infer(args):
    data_format_list = ['table', 'array-table', 'textfile', 'tensor']
    if args.data_format is not None:
        assert args.data_format in data_format_list, \
            f"Error, data_format: {args.data_format}, should be one of {data_format_list}!"
    else:
        """
        TODO: infer data_format
        """
        args.data_format = 'textfile'
    return args


def data_args_infer(args):
    if args.label_col is not None:
        if args.label_col.find('[') >= 0:
            args.label_col = eval(args.label_col)
        if type(args.label_col) is not list:
            args.label_col = [args.label_col]
    else:
        args.label_col = []

    if args.features_col is not None:
        if type(args.label_col) is not list:
            if args.features_col.find('[') >= 0:
                args.features_col = eval(args.features_col)

        if type(args.features_col) is not list:
            args.features_col = [args.features_col]

        for col in args.label_col:
            if args.features_col.__contains__(col):
                args.features_col.remove(col)
    else:
        args.features_col = []

    if args.data_format in ['table', 'textfile']:
        if len(args.label_col) == 0:
            args.label_col = args.features_col
            args.features_col = []

        if len(args.label_col) > 0:
            args.enc_in = len(args.features_col) + len(args.label_col)
            args.dec_in = len(args.features_col) + len(args.label_col)
            args.c_out = len(args.features_col) + len(args.label_col)
            args.d_out = len(args.label_col)

        if (args.label_len is None) or (args.label_len > args.seq_len):
            args.label_len = args.seq_len // 2

    logger.info(f">>> data_args_infer: enc_in:{args.enc_in}, dec_in:{args.dec_in},"
                f"c_out:{args.c_out}, d_out:{args.d_out}, label_len:{args.label_len}")
    return args


def model_args_infer(args):
    """
    inference model params from data
    :param args:
    :return:
    """
    logger.info(">>> inference model params from data  ...")
    logger.info(f"Load data from {args.train_data} ...")
    data_reader = get_data_reader(args, data_path=args.train_data)

    logger.info(">>> Infer and update model input args ...")
    if args.dec_in is None:
        args = get_model_input_args(args, data_reader)

    logger.info(f">>> model_args_infer: mark_in:{args.mark_in}, enc_in:{args.enc_in},"
                f"c_out:{args.c_out}, d_out:{args.d_out}, label_len:{args.label_len},"
                f"seq_len:{args.seq_len}, pred_len:{args.pred_len}")

    return args


def get_resource_constraints(search_space_args):
    """
    get resource_constraints
    :param search_space_args:
    :return: resource_constraints
    """
    resource_key, min_resource, max_resource = None, None, None
    config_order = ['train_epochs', 'd_model', 'n_heads', 'd_ffn', 'enc_layers', 'dec_layers']
    for k in config_order:
        if k in search_space_args.keys():
            resource_key, min_resource, max_resource = k, search_space_args[k][0], search_space_args[k][1]

    low_cost_partial_config = None
    if resource_key:
        low_cost_partial_config = {resource_key: min_resource}
    return low_cost_partial_config, min_resource, max_resource


def args_prepare(args):
    """
    user defined args preprocess
    :param args:
    :return: args, args after preprocess
    """
    args = default_args_check(args)
    args = set_hidden_args(args)
    args = set_model_definition_path(args)
    args = data_format_infer(args)
    args = data_args_infer(args)
    args = model_args_infer(args)

    return args

