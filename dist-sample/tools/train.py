import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F

from deepctr.datasets import build_dataloader
from deepctr.datasets.criteo_dataset import CriteoDataset
from deepctr.models import DeepFM
from deepctr.utils import distributed_utils
from deepctr.utils.logging import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a ctr model')
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--feature_cols', type=str, help='feature column names')
    parser.add_argument('--sparse_cols', type=str, help='sparse feature column names')
    parser.add_argument('--label_cols', type=str, help='label column names')
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--log_iter_interval', default=10, type=int)

    args = parser.parse_args()
    return args


def main():
    # parse args
    args = parse_args()

    # whether to distributed training
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        distributed_utils.init_dist(args.launcher, 'nccl')
        # gpu_ids is used to calculate iter when resuming checkpoint
        rank, world_size = distributed_utils.get_dist_info()

    # create logger
    logger = get_logger("deepctr")

    # load data
    cols_kwargs = dict()
    cols_kwargs['feature_cols'] = args.feature_cols.split(',')
    cols_kwargs['sparse_cols'] = args.sparse_cols.split(',')
    cols_kwargs['label_cols'] = args.label_cols.split(',')
    criteo_dataset = CriteoDataset(args.data_path, 'orc', **cols_kwargs)

    train_set, train_loader, train_sampler = build_dataloader(
        dataset=criteo_dataset,
        batch_size=args.batch_size,
        dist=distributed,
        num_workers=args.num_workers,
        training=True)

    # model
    model = DeepFM(cols_kwargs['feature_cols'], cols_kwargs['sparse_cols'], criteo_dataset.num_embeddings).cuda()

    # optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank % torch.cuda.device_count()])

    # train
    for epoch in range(0, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        for step, (x_train, y_train) in enumerate(train_loader, 1):
            x = x_train.cuda()
            y = y_train.cuda().float()
            y_pred = model(x).squeeze()
            loss = F.binary_cross_entropy(y_pred, y.squeeze(), reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_iter_interval == 0:
                try:
                    cur_lr = float(optimizer.lr)
                except:
                    cur_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    'Train - Epoch [%d/%d] Iter [%d/%d] lr: %f, loss: %f' % (
                    epoch, args.epochs, step, len(train_loader),
                    cur_lr, loss.cpu().item()))

        lr_scheduler.step()


if __name__ == '__main__':
    main()
