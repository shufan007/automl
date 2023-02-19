# -*- coding: utf-8 -*-
"""
# @Time : 2022.10.21
# @Author : shuangxi.fan
# @Description : entrance for automl tune flow
"""
import os
import sys
import gc
gc.collect()

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURR_DIR)
# print("BASE_DIR: ", BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(CURR_DIR)

from common import dotdict, get_logger
from common import RayCluster

from prepare_env import node_list_parser, prepare_env
from tuner import automl_tune

logger = get_logger()


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
    # user code path
    parser.add_argument('--task_type', type=str, default='forecasting', choices=['forecasting', 'user_task'],
                        help='task type, options: [forecasting, user_task]')
    parser.add_argument('--code_path', type=str, help='user defined code path')
    parser.add_argument('--trainer_interface', type=str, help='file name of the trainer interface')

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

    parser.add_argument('--metric', type=str, default='val_loss', help='metric name')
    parser.add_argument('--metric_mode', type=str, default='min', choices=['min', 'max'],
                        help='metric mode, options:[min, max]')
    # search space
    parser.add_argument('--search_space', type=str,
                        help='dict like: {"model":{"htype":"choice","value":["FEDformer","Autoformer"]},'
                             '"d_model":{"htype":"exponent","value":[64,512]},'
                             '"n_heads":{"htype":"qrandint","value":[4,12,4]},'
                             '"d_ffn":{"htype":"exponent","value":[128,2048]},'
                             '"learning_rate":{"htype":"loguniform","value":[0.00001,0.001]}}'
                             'search space for parameters with lower and upper')

    parser.add_argument('--label_col', type=str, help='label column or column list')
    parser.add_argument('--features_col', type=str, help='[optional] features column list')

    parser.add_argument('--num_workers', type=int, default=10, help='num workers of data loader')

    # user defined params
    parser.add_argument('--params', type=str, help='user defined params, with type of json string')

    """ parse_args """
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
    args = prepare_env(args)

    """# run task"""
    automl_tune(args)

    """# Cleanup Ray"""
    if args.node_list and len(args.node_list) > 1:
        ray_op = RayCluster()
        ray_op.clean_up(args.node_list, args.remote)


def main(argv):
    usage = '''
    example:
    python3 ./autosearch/run.py \
      --seed 1234 \
      --task_type forecasting \
      --time_budget_s 300 \
      --num_gpus_per_worker 1 \
      --train_data ETDataset/ETT-small/ETTh1.csv \
      --split_ratio 0.2 \
      --output_path /nfs/volume-807-1/fanshuangxi/models/ts_model \
      --search_space '{"model":{"htype":"choice","value":["FEDformer","Autoformer"]},"d_model":{"htype":"exponent","value":[64,512]},"n_heads":{"htype":"qrandint","value":[4,12,4]},"d_ffn":{"htype":"exponent","value":[128,2048]},"learning_rate":{"htype":"loguniform","value":[0.00001,0.001]}}' \
      --params '{"label_col":"OT","date_col":"date","features_col":["HUFL","HULL","MUFL","MULL","LUFL","LULL"],"freq":"h","seq_len":96,"pred_len":24}'
    or
    sh run.sh ./forcasting/run.py --host_file /etc/HOROVOD_HOSTFILE ...
    '''

    task_manager(argv, usage)


if __name__ == "__main__":
    main(sys.argv[1:])

