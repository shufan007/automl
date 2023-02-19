# -*- coding: utf-8 -*-
"""
# @Time : 2022.10.21
# @Author : shuangxi.fan
# @Description : entrance for automl tune flow
"""
import os
import sys
import json
import torch
import gc

gc.collect()

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURR_DIR)
# print("BASE_DIR: ", BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(CURR_DIR)


from common import dotdict, get_logger
from common import RayCluster

from interface.args_prepare import set_default_args, set_model_definition_path, data_args_infer, model_args_infer

from interface.interface import automl_tune

logger = get_logger()


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
    if args.search_space is None:
        args.is_tune = False
    else:
        args.is_tune = True
        args.search_space = json.loads(args.search_space)
        for k, v in args.search_space.items():
            if len(v) == 0:
                args.search_space.pop(k)
            elif k not in ["model"]:
                if len(v) == 1:
                    args.search_space[k] = [v[0], v[0]]
                elif len(v) > 1:
                    assert v[1] >= v[0], "Error, in search space args, the lower bond should no bigger than upper bond!"
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


def args_prepare(args):
    args = set_default_args(args)
    args = set_model_definition_path(args)
    args = data_args_infer(args)
    args = model_args_infer(args)

    args = args_check_trial_resources(args)
    args = args_check_search_space(args)
    args = args_check_path(args)

    args = set_hadoop_config(args)
    args = distributed_args_infer(args)
    args = data_args_infer(args)

    return args


def task_args_parser(argv, usage=None):
    """
    :param argv:
    :return:
    """
    import argparse

    parser = argparse.ArgumentParser(prog='main', usage=usage,
                                     description='Autoformer & Transformer family for Time Series Forecasting')

    # env config
    parser.add_argument('--node_list', type=str, help="""[optional] List of node ip address:
                        for run distributed Ray applications on some local nodes available on premise.
                        the first node will be choose to be the head node.""")  # external shield
    parser.add_argument('--host_file', type=str,
                        help="[optional] same as node_list, host file path, List of node ip address.")
    parser.add_argument('--hadoop_config', type=str,
                        help="""[optional] hadoop_config: params dict like '{"HADOOP_USER_NAME":"xxx", "HADOOP_USER_PASSWORD":"xxx"}' for hadoop access""")
    # tune config
    parser.add_argument('--seed', type=int, default=0, help='seed, default:0')
    parser.add_argument('--time_budget_s', type=int, default=60,
                        help='constrains the wall-clock time (seconds) used by the AutoML process. default: 60')
    parser.add_argument('--num_samples', type=int, default=500, help='maximal number of trials')

    # tune config, DistributedTrainableCreator config
    parser.add_argument('--resources_per_trial', type=str,
                        help='resources dict like: {"cpu":1, "gpu":1}, number of cpus, gpus for each trial; 0.5 means two training jobs can share one gpu')
    parser.add_argument("--num_training_workers", type=int, help="Number of training workers to include in world.")
    parser.add_argument("--num_gpus_per_worker", type=int, default=0, help="Sets number of gpus each worker uses.")

    parser.add_argument('--train_data', type=str, required=True, help='path or tableName of train data')
    parser.add_argument('--split_ratio', type=float, default=0,
                        help='[optional] split rate of val data, split from train data')
    parser.add_argument('--val_data', type=str, help='[optional] path or tableName of val data')
    parser.add_argument('--output_path', type=str, default='temp/',
                        help='[optional] output path for model(or checkpoint) to save.'
                             'you should always try to use shared storage if possible when training on a multi node cluster.'
                             'eg: /nfs/volume-807-1/temp/models/test01')

    parser.add_argument('--metric', type=str, help='metric name')
    parser.add_argument('--metric_mode', type=str, default='min', choices=['min', 'max'],
                        help='metric mode, options:[min, max]')
    # search space
    parser.add_argument('--search_space', type=str,
                        help='dict like: {"model":["FEDformer","Autoformer"],"d_model":[64,512],"n_heads":[4,8],"d_ffn":[128,2048],"learning_rate":[0.00001,0.001]}, '
                             'search space for parameters with lower and upper')

    parser.add_argument('--label_col', type=str, help='label column or column list')
    parser.add_argument('--features_col', type=str, help='[optional] features column list')

    parser.add_argument('--num_workers', type=int, default=10, help='num workers of data loader')

    # data loader
    parser.add_argument('--data_format', type=str, default='textfile',
                        choices=['table', 'array-table', 'textfile', 'tensor'],
                        help='train data format:'
                             ' "table": hive table (with 2 dimensions)'
                             '"textfile": text file(s) path of tabular data (with 2 dimensions), '
                             ' "array-table": array table (with 3 dimensions), DataFrame with dtypes: '
                             '[("features", "array<array<float>>"), ("target", "array<array<float>>")]'
                             '"tensor": tensors dict like {"features":X, "target":y} or list like [X, y] with a 3 dimensions '
                             ' X: [# samples x # sequence length x # variables] and target named as "target"')

    parser.add_argument('--date_col', type=str, default='date', help='date column')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # basic config
    parser.add_argument('--model', type=str, choices=['FEDformer', 'Autoformer', 'Informer', 'Transformer'],
                        default='Autoformer',
                        help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

    # model define
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--enc_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ffn', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', choices=['timeF', 'fixed', 'learned'],
                        help='time features encoding, options:[timeF, fixed, learned]')  # external shield

    # optimization
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')  # external shield
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')

    # for debug
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="[for debug], print detailed debug logs")  # external shield
    parser.add_argument("--sample_step", type=int, default=1,
                        help=' [for debug], sampling train batch data when training, jump by sample_step, for speed up the testing procedure')  # external shield

    args = parser.parse_args(argv)

    args = dotdict(vars(args))

    return args


def task_manager(argv, usage=None):
    """
    job manager: job args parser, job submit
    :param argv:
    :param usage:
    :return:
    """
    args = task_args_parser(argv, usage=usage)
    logger.info(f"Args in experiment: {args} ")

    """ node_list parser"""
    args = node_list_parser(args)

    """ Startup Ray"""
    if args.node_list and len(args.node_list) > 1:
        ray_op = RayCluster()
        args.node_list, args.remote = ray_op.startup(node_list=args.node_list, remote=args.remote)
    logger.info(f"node_list: {args.node_list} ")

    """# args prepare"""
    args = args_prepare(args)

    """# run task"""
    automl_tune(args)

    """# Cleanup Ray"""
    if args.node_list and len(args.node_list) > 1:
        ray_op = RayCluster()
        ray_op.clean_up(args.node_list, args.remote)


def main(argv):
    usage = '''
    example:

    python3 ./forcasting/run.py \
      --seed 1234 \
      --time_budget_s 300 \
      --num_gpus_per_worker 1 \
      --train_data datasets/ETDataset/ETT-small/ETTh1.csv \
      --split_ratio 0.2 \
      --output_path models/ts_model \
      --search_space '{"model":["FEDformer","Autoformer"],"d_model":[64,512],"n_heads":[4,8],"d_ffn":[128,2048],"learning_rate":[0.00001,0.001]}' \
      --label_col OT \
      --date_col date \
      --features_col "['HUFL','HULL','MUFL','MULL','LUFL','LULL']" \
      --freq h \
      --seq_len 96 \
      --pred_len 24 \
      --enc_layers 2 \
      --dec_layers 1 \
      --moving_avg 25 \
      --train_epochs 2 \
      --patience 2 \
      --batch_size 32 \
      --loss mse 

    or

    sh run.sh ./forcasting/run.py --host_file /etc/HOROVOD_HOSTFILE ...
    '''

    task_manager(argv, usage)


if __name__ == "__main__":
    main(sys.argv[1:])

