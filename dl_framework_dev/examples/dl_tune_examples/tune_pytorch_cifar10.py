# -*- coding: utf-8 -*-
"""
# @Time : 2022.06.16
# @Author : shuangxi.fan
# @Description : example of tune pytorch on cifar10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms

import numpy as np
import flaml
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("BASE_DIR: ", BASE_DIR)
sys.path.append(BASE_DIR)

from ray_cluster_manager import RayCluster

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

"""
Training
"""
from ray import tune

def train_cifar(config, trainset, checkpoint_dir=None):
    if "l1" not in config:
        print("* warning: ", config)

    net = Net(2 ** config["l1"], 2 ** config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            print("torch.cuda.device_count: ", torch.cuda.device_count())
            net = nn.DataParallel(net)

    net.to(device)
    print("torch.cuda.is_available: ", torch.cuda.is_available())
    print("device: ", device)
    print("net: ", net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(2 ** config["batch_size"]),
        shuffle=True,
        num_workers=4)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(2 ** config["batch_size"]),
        shuffle=True,
        num_workers=4)

    for epoch in range(int(round(config["num_epochs"]))):  # loop over the dataset multiple times
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

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


def _test_accuracy(net, testset, device="cpu"):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

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
    data_dir = args["train_data"]
    time_budget_s = args["time_budget"]
    num_samples = args["num_samples"]
    output_path="./logs"
    if args["output_path"]:
        output_path = args["output_path"]
    resources_per_trial = args["resources_per_trial"]

    # load data
    # Download data for all trials before starting the run
    trainset, testset = load_data(data_dir)

    """
    Search space
    """
    max_num_epoch = 10
    config = {
        "l1": tune.randint(2, 9),  # log transformed with base 2
        "l2": tune.randint(2, 9),  # log transformed with base 2
        "lr": tune.loguniform(1e-4, 1e-1),
        "num_epochs": tune.loguniform(1, max_num_epoch),
        "batch_size": tune.randint(1, 5)  # log transformed with base 2
    }

    np.random.seed(7654321)

    import time
    start_time = time.time()
    result = flaml.tune.run(
        tune.with_parameters(train_cifar, trainset=trainset),
        config=config,
        metric="loss",
        mode="min",
        low_cost_partial_config={"num_epochs": 1},
        max_resource=max_num_epoch,
        min_resource=1,
        scheduler="asha",  # need to use tune.report to report intermediate results in train_cifar
        resources_per_trial=resources_per_trial,
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
    best_trained_model.load_state_dict(model_state)

    test_acc = _test_accuracy(best_trained_model, testset, device)
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
    parser.add_argument('--remote', type=str, default='False', help="True, submit job outside the Ray cluster, False, submit inside the ray cluster")
    parser.add_argument('--train_data', type=str, help='path of train data')
    parser.add_argument('--test_data', type=str, help='[optional] path of test data')
    parser.add_argument('--output_path', type=str, help='[optional] output path for model to save')
    parser.add_argument('--time_budget', type=int, default=60,
                        help='constrains the wall-clock time (seconds) used by the AutoML process. default: 60')
    parser.add_argument('--resources_per_trial', type=str,
                        help='resources dict like: {"cpu": 1, "gpu": gpus_per_trial}, number of cpus, gpus for each trial; 0.5 means two training jobs can share one gpu')
    parser.add_argument('--num_samples', type=int, help='maximal number of trials')

    args = parser.parse_args(argv)
    args = vars(args)
    if args['node_list']:
        args['node_list'] = eval(args['node_list'])
    elif args['host_file']:
        with open(args['host_file'], "r") as f:
            lines = f.readlines()
            args['node_list'] = [line.split(' ')[0].strip() for line in lines]
            print("node_list in host_file: ", args['node_list'])
    if args['remote']:
        assert (args['remote'].lower() in ['true', 'false', '0', '1'])
        if (args['remote'].lower() == 'true') or (args['remote'] == '1'):
            args['remote'] = True
        else:
            args['remote'] = False

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

    try:
        # Startup Ray
        if args['node_list'] and len(args['node_list']) > 1:
            ray_op = RayCluster()
            node_list, remote = ray_op.startup(node_list=args['node_list'], remote=args['remote'])
            args['node_list'] = node_list
            args['remote'] = remote

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
        python3 ./examples/tune_pytorch_cifar10.py --node_list '["10.186.2.241"]' --train_data datasets/cifar-10 --output_path test.outputs --time_budget 300 --resources_per_trial '{"cpu":1,"gpu":1}' --num_samples 500
        or
        sh run.sh ./examples/tune_pytorch_cifar10.py --host_file /etc/HOROVOD_HOSTFILE --train_data cifar-10 --output_path test.outputs --time_budget 300 --resources_per_trial '{"cpu":1,"gpu":0.5}' --num_samples 500
            '''
    print("argv:", argv)
    task_manager(argv, usage)


if __name__ == "__main__":
    main(sys.argv[1:])

