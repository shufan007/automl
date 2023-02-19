# -*- coding: utf-8 -*-
"""
# @Time : 2022.10.24
# @Author : shuangxi.fan
# @Description : entrance for automl tune flow
"""
import os
import sys
import json
import torch
import importlib

import gc
gc.collect()

from common import dotdict, get_logger

logger = get_logger()


def set_startup_args(args):
    """ env config """
    args.remote = False  # True, submit job outside the Ray cluster, False, submit inside the ray cluster

    """ tune config """
    args.distributed = False  # if distributed mod
    args.is_tune = True  # if tune mod
    return args


def node_list_parser(args):
    if args.node_list is not None:
        args.node_list = eval(args.node_list)
    elif args.host_file and os.path.exists(args.host_file):
        with open(args.host_file, "r") as f:
            lines = f.readlines()
            args.node_list = [line.split(' ')[0].strip() for line in lines]
            logger.info(f"node_list in host_file: {args.node_list}")
    return args


def distributed_args_infer(args):
    """
    tune config, DistributedTrainableCreator config
    set num_workers, workers_per_host, backend according to num_gpus_per_worker
     - num_gpus_per_worker: Number of GPU resources to reserve per training worker.
     - num_training_workers: Number of training workers to include in world
     - num_workers_per_host: Number of workers to colocate per host.
     - backend: One of “gloo”, “nccl”

    num_workers: type=int,default=1, 调参训练trail的并行进程数。
    num_gpus_per_worker: type=int, default=0, 每个进程分配的gpu数，如果没有gpu，则设为0
    workers_per_host: type=int, default=1, 每个节点分配的进程数。
        例如：有2个节点，每个节点有4 GPUs，如设置num_gpus_per_worker=1，则'workers_per_host'最大可设置为4，此时'num_workers'最大可设置为8

    :param args:
    :return: args
    """
    n_node = 1
    if args.node_list is not None:
        n_node = len(args.node_list)

    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        logger.info(f"n_gpu: {n_gpu}")
    total_gpu = n_node * n_gpu
    logger.info(f"total_gpu: {total_gpu}")

    if args.num_gpus_per_worker > n_gpu:
        args.num_gpus_per_worker = n_gpu

    if args.num_training_workers is None:
        if args.num_gpus_per_worker > 0:
            args.num_training_workers = total_gpu // args.num_gpus_per_worker
        else:
            args.num_training_workers = n_node
    else:
        args.num_training_workers = max(args.num_training_workers, n_node)

    if n_node <= 1:
        args.workers_per_host = None
    else:
        args.workers_per_host = min(n_gpu // args.num_gpus_per_worker, args.num_training_workers)

    args.backend = "nccl" if args.num_gpus_per_worker > 0 else "gloo"

    logger.info(f"workers_per_host: {args.workers_per_host}, "
                f"num_gpus_per_worker: {args.num_gpus_per_worker}, "
                f"num_training_workers: {args.num_training_workers}, "
                f"backend: {args.backend}")

    return args


def args_check_trial_resources(args):
    if args.resources_per_trial is not None:
        args.resources_per_trial = eval(args.resources_per_trial)
        for k, v in args.resources_per_trial.items():
            args.resources_per_trial[k] = float(v)
    else:
        args.resources_per_trial = {"cpu": 1, "gpu": 1}
    return args


def args_check_search_space(args):
    """
    check and normalize the search_space params, with dict as:
    {"key1":{"htype":"choice","value":["c1","c2"]},
    "key2":{"htype":"uniform","value":[lower_bond,upper_bond]},
    "key3":{"htype":"quniform","value":[lower_bond,upper_bond,q]},
        ...
      }
    :param args:
    :return:
    """
    htype_list = ["choice", "uniform", "quniform", "loguniform", "qloguniform",
                   "randint", "qrandint", "lograndint", "qlograndint", "exponent"]
    if args.search_space is None:
        args.is_tune = False
        return args

    args.is_tune = True
    args.search_space = json.loads(args.search_space)
    logger.info(f"search_space: {args.search_space}")
    for k, attr in args.search_space.items():
        htype = attr["htype"]
        assert htype in htype_list, \
            f"Error, htype should be one of {htype_list}, but provided {htype} !"
        value = attr["value"]
        if len(value) == 0:
            args.search_space.pop(k)
        elif htype != "choice":
            if len(value) == 1:
                value = [value[0], value[0]]
                args.search_space[k]["value"] = value
            elif len(value) > 1:
                assert value[1] >= value[0], \
                    f"Error, search space args, {k}:{value}, the lower should no bigger than the upper!"
            if htype.find("q") == 0:
                diff = value[1] - value[0]
                if len(value) >= 3:
                    q = value[2]
                else:
                    if diff > 0:
                        q = diff
                    else:
                        q = value[0]
                    args.search_space[k]["value"].append(q)
                assert diff >= q, \
                    f"Error, search space args, {k}:{value}, the quantized value should no bigger than diff between lower and upper!"

    return args


def args_check_path(args):
    args.local_dir = os.path.join(args.output_path, "temp")
    args.output_path = os.path.join(args.output_path, "model")
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    return args


def set_hadoop_config(args):
    if args.hadoop_config is not None:
        args.hadoop_config = eval(args.hadoop_config)
    HADOOP_USER_NAME = os.getenv('HADOOP_USER_NAME')
    HADOOP_USER_PASSWORD = os.getenv('HADOOP_USER_PASSWORD')
    if not HADOOP_USER_NAME or (not HADOOP_USER_PASSWORD):
        if args.hadoop_config:
            os.environ['HADOOP_USER_NAME'] = args.hadoop_config['HADOOP_USER_NAME']
            os.environ['HADOOP_USER_PASSWORD'] = args.hadoop_config['HADOOP_USER_PASSWORD']
        else:
            logger.warn("If input data is hdfs file or hive table, "
                        "the environment variables HADOOP_USER_NAME and HADOOP_USER_PASSWORD must be set!")
    return args


def load_trainer_interface(args):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.task_type == "user_task":
        """
        TODO: download user code
        """
        package_name = ""
        module_name = "." + args.trainer_interface.split(".")[0]
    else:
        package_name = args.task_type
        module_name = ".trainer_interface"

    package_path = os.path.join(base_path, package_name)
    sys.path.append(package_path)
    os.environ['PYTHONPATH'] = package_path
    # logger.info(f"=>1 sys.path: {sys.path}")
    trainer_module = importlib.import_module(module_name, package=package_name)
    args.trainer_module = trainer_module
    return args


def prepare_env(args):

    # args = node_list_parser(args)
    args = set_startup_args(args)

    args = args_check_trial_resources(args)
    args = args_check_search_space(args)
    args = args_check_path(args)

    args = set_hadoop_config(args)
    args = distributed_args_infer(args)

    """ user_args_preprocess """
    if args.params is not None:
        args.params = json.loads(args.params)
        logger.info(f"user defined params: {args.params}")
        args.update(args.params)

    args = load_trainer_interface(args)

    # logger.info(f"=>2 sys.path: {sys.path}")
    args = args.trainer_module.user_args_preprocess(args)

    return args



