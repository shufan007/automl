# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms

import ray
from ray import tune
from ray.tune.integration.torch import (
    DistributedTrainableCreator,
    distributed_checkpoint_dir,
)

from functools import partial
import numpy as np
import flaml
import os
import sys

NCCL_TIMEOUT_S = 60

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("BASE_DIR: ", BASE_DIR)
sys.path.append(BASE_DIR)

from ray_cluster_manager import RayCluster

logger = logging.getLogger(__name__)


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


"""
Network Specification
"""
class Net(nn.Module):

    def __init__(self, l1=120, l2=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""
Data
"""
def load_data(data_dir="data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset


def train_cifar(config, data_dir, isTune=True, distributed=True, checkpoint_dir=None):
    # load data
    print("data loading...")
    trainset, testset = load_data(data_dir)

    # whether to distributed training
    if distributed:
        rank, world_size = get_dist_info()
        print("rank:{}, world_size:{}".format(rank, world_size))

    if "l1" not in config:
        print("* warning: ", config)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    """
    if use_cuda:
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count: ", torch.cuda.device_count())
            # net = nn.DataParallel(net)      
    """

    net = Net(2 ** config["l1"], 2 ** config["l2"])
    net.to(device)

    print("device:", device)
    # print("net: ", net)

    if distributed:
        """
        TODO: device_ids
        """
        net = DistributedDataParallel(net)
        # net = DistributedDataParallel(net, device_ids=[0])
    else:
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count: ", torch.cuda.device_count())
            net = nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        if os.path.exists(checkpoint):
            model_state, optimizer_state = torch.load(checkpoint)
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    """ dataset"""
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    # whether to distributed training
    train_sampler = None
    val_sampler = None

    if distributed:
        # train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank)
        train_sampler = DistributedSampler(train_subset)
        val_sampler = DistributedSampler(val_subset)

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(2 ** config["batch_size"]),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(2 ** config["batch_size"]),
        sampler=val_sampler,
        shuffle=(val_sampler is None),
        num_workers=4)

    """
    batch_size = 4
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(2 ** batch_size),
        shuffle=True,
        num_workers=4)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(2 ** batch_size),
        shuffle=True,
        num_workers=4) 
    """

    for epoch in range(int(round(config["num_epochs"]))):  # loop over the dataset multiple times

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        if not isTune:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path)
            continue

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        # with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
        with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
        print("val loss: %.3f, accuracy: %.3f" % (val_loss / val_steps, correct / total))

    print("Finished Training")


def _test_accuracy(net, testset, batch_size=4, device="cpu"):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    i = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.float(), labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            i += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d] correct: %.3f" % (i + 1, correct / total))

    return correct / total


def run_ddp_tune(data_dir, num_workers, num_gpus_per_worker, workers_per_node=None):
    # gpu_ids is used to calculate iter when resuming checkpoint
    rank, world_size = get_dist_info()
    print("rank:{}, world_size:{}".format(rank, world_size))

    # train_cifar(config, trainset, isTune=True, distributed=True, checkpoint_dir=None)
    print("DistributedTrainableCreator...")

    trainable_cls = DistributedTrainableCreator(
        partial(train_cifar, data_dir=data_dir),
        num_workers=num_workers,
        num_gpus_per_worker=num_gpus_per_worker,
        num_workers_per_host=workers_per_node,
        backend="nccl",
        timeout_s=NCCL_TIMEOUT_S,
    )

    """
    Search space
    """
    max_num_epoch = 2
    config = {
        "l1": tune.randint(2, 4),  # log transformed with base 2
        "l2": tune.randint(2, 6),  # log transformed with base 2
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_epochs": tune.loguniform(1, max_num_epoch),
        "batch_size": tune.randint(2, 5)  # log transformed with base 2
    }

    print("start tuning...")

    analysis = tune.run(
        trainable_cls,
        config=config,
        # resources_per_trial=None,
        num_samples=4,
        stop={"training_iteration": 2},
        metric="accuracy",
        mode="max",
    )

    print("Best hyperparameters found were: ", analysis.best_config)


"""
Hyperparameter Optimization
"""
def automl_tune(args):
    """
    #time_budget_s    # time budget in seconds
    # resources_per_trial  # resources dict like: {"cpu": 1, "gpu": gpus_per_trial}, number of cpus, gpus for each trial;
        gpus_per_trial   # number of gpus for each trial; 0.5 means two training jobs can share one gpu
    #num_samples      # maximal number of trials
    """
    rank, world_size = get_dist_info()
    print("rank:{}, world_size:{}".format(rank, world_size))

    data_dir = args["train_data"]
    time_budget_s = args["time_budget"]
    num_samples = args["num_samples"]
    output_path = "./logs"
    if args["output_path"]:
        output_path = args["output_path"]

    # resources_per_trial = args["resources_per_trial"]

    print("DistributedTrainableCreator...")
    # train_cifar(config, trainset, isTune=True, distributed=True, checkpoint_dir=None)

    trainable_cls = DistributedTrainableCreator(
        partial(train_cifar, data_dir=data_dir),
        num_workers=args["num_workers"],
        num_gpus_per_worker=args["num_gpus_per_worker"],
        num_workers_per_host=args["workers_per_node"],
        backend="nccl",
        timeout_s=NCCL_TIMEOUT_S,
    )

    """
    Search space
    """
    max_num_epoch = 2
    config = {
        "l1": tune.randint(2, 9),  # log transformed with base 2
        "l2": tune.randint(2, 9),  # log transformed with base 2
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_epochs": tune.loguniform(1, max_num_epoch),
        "batch_size": tune.randint(1, 5)  # log transformed with base 2
    }

    np.random.seed(7654321)

    # resources_per_trial={"cpu": 1, "gpu": 1},

    import time
    start_time = time.time()
    result = flaml.tune.run(
        tune.with_parameters(trainable_cls),
        config=config,
        metric="loss",
        mode="min",
        low_cost_partial_config={"num_epochs": 1},
        max_resource=max_num_epoch,
        min_resource=1,
        scheduler="asha",  # need to use tune.report to report intermediate results in train_cifar
        # resources_per_trial=resources_per_trial,
        local_dir=output_path,
        num_samples=num_samples,
        time_budget_s=time_budget_s,
        use_ray=True)

    print(f"#trials={len(result.trials)}")
    print(f"time={time.time() - start_time}")
    best_trial = result.get_best_trial("loss", "min", "all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.metric_analysis["loss"]["min"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.metric_analysis["accuracy"]["max"]))

    best_trained_model = Net(2 ** best_trial.config["l1"],
                             2 ** best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)

    """
    TODO: issue: load_state_dict 
    """
    # best_trained_model.load_state_dict(model_state)
    best_trained_model.load_state_dict(model_state, strict=False)

    trainset, testset = load_data(data_dir)
    test_acc = _test_accuracy(best_trained_model, testset=testset, batch_size=128, device=device)
    # test_acc = _test_accuracy(best_trained_model, testset, device)
    print("Best trial test set accuracy: {}".format(test_acc))


def task_arg_parser(argv, usage=None):
    """
    :param argv:
    :return:

    TODO: add param 'n_concurrent_trials'
    """
    import argparse

    parser = argparse.ArgumentParser(prog='main', usage=usage)

    parser.add_argument('--node_list', type=str, help="""[optional] List of node ip address:
                        for run distributed Ray applications on some local nodes available on premise.
                        the first node will be choose to be the head node.
                        """)
    parser.add_argument('--host_file', type=str, help="""[optional] same as node_list, host file path, List of node ip address.
                        """)
    parser.add_argument('--remote', action="store_true", default=False,
                        help="True, submit job outside the Ray cluster, False, submit inside the ray cluster")

    parser.add_argument('--train_data', type=str, help='path of train data')
    parser.add_argument('--test_data', type=str, help='[optional] path of test data')
    parser.add_argument('--output_path', type=str, help='[optional] output path for model to save')
    parser.add_argument('--time_budget', type=int, default=60,
                        help='constrains the wall-clock time (seconds) used by the AutoML process. default: 60')
    parser.add_argument('--resources_per_trial', type=str,
                        help='resources dict like: {"cpu": 1, "gpu": gpus_per_trial}, number of cpus, gpus for each trial; 0.5 means two training jobs can share one gpu')
    parser.add_argument('--num_samples', type=int, help='maximal number of trials')

    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--num-gpus-per-worker",
        type=int,
        default=0,
        help="Sets number of gpus each worker uses.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=False,
        help="enables multi-node tuning",
    )
    parser.add_argument(
        "--workers-per-node",
        type=int,
        help="Forces workers to be colocated on machines if set.",
    )

    args = parser.parse_args(argv)
    args = vars(args)
    if args['node_list']:
        args['node_list'] = eval(args['node_list'])
    elif args['host_file'] and os.path.exists(args['host_file']):
        with open(args['host_file'], "r") as f:
            lines = f.readlines()
            args['node_list'] = [line.split(' ')[0].strip() for line in lines]
            print("node_list in host_file: ", args['node_list'])

    if args['resources_per_trial']:
        args['resources_per_trial'] = eval(args['resources_per_trial'])
        for k, v in args['resources_per_trial'].items():
            args['resources_per_trial'][k] = float(v)

    return args


def task_manager(argv, usage=None):
    """
    job manager: job args parser, job submit
    :param argv:
    :param usage:
    :return:
    """
    args = task_arg_parser(argv, usage=usage)
    print("parse args: '{}'".format(args))

    """
    ********
    TODO:
    check and adjust --num-workers --num-gpus-per-worker
    accronding gpu number: torch.cuda.device_count()
    *********
    """

    try:
        # Startup Ray
        if args['node_list'] and len(args['node_list']) > 1:
            ray_op = RayCluster()
            node_list, remote = ray_op.startup(node_list=args['node_list'], remote=args['remote'])
            args['node_list'] = node_list
            args['remote'] = remote

        print("node_list: ", args['node_list'])
        if args['node_list'] == None or len(args['node_list']) <= 1:
            args["workers_per_node"] = None
            # print("workers_per_node: ", args["workers_per_node"])

        # run task
        automl_tune(args)

    except Exception as e:
        print("except:", e)
    finally:
        # Cleanup Ray
        if args['node_list'] and len(args['node_list']) > 1:
            ray_op = RayCluster()
            ray_op.clean_up(args['node_list'], args['remote'])


def main(argv):
    usage = '''
    example1:
        python3 ./examples/ddt_tune_pytorch_cifar10.py --node_list '["10.186.2.241"]' --train_data datasets/cifar-10 --output_path test.outputs --time_budget 300 --resources_per_trial '{"cpu":1,"gpu":1}' --num_samples 500 --num-workers 2 --num-gpus-per-worker 1
        or
        sh run.sh ./examples/ddt_tune_pytorch_cifar10.py --host_file /etc/HOROVOD_HOSTFILE --train_data datasets/cifar-10 --output_path fanshuangxi/test.outputs --time_budget 300 --resources_per_trial '{"cpu":1,"gpu":0.5}' --num_samples 500 --num-workers 2 --num-gpus-per-worker 1
            '''
    print("argv:", argv)
    task_manager(argv, usage)


if __name__ == "__main__":
    main(sys.argv[1:])
