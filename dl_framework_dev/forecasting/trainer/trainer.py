# -*- coding: utf-8 -*-
"""
# @Time : 2022.07.27
# @Author : shuangxi.fan
# @Description : example of tune pytorch on ts task
"""
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from trainer.base_trainer import BaseTrainer
from utils.tools import adjust_learning_rate

from common import get_logger, get_suffix_file

warnings.filterwarnings('ignore')

logger = get_logger()

class Trainer(BaseTrainer):
    def __init__(self, args):
        super(Trainer, self).__init__(args)
        self.device = self._acquire_device()
        # self.model = self._build_model().to(self.device)
        self.model = self._build_model()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        # model = model.to(self.device)
        return model

    def _select_optimizer(self, model, args):
        model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def eval(self, model, args, val_loader):
        criterion = self._select_criterion()
        total_loss = []
        model.eval()
        with torch.no_grad():
            for i, batch_in in enumerate(val_loader):
                if (i + 1) % args.sample_step:
                    continue

                if len(batch_in) == 4:
                    [batch_x, batch_y, batch_x_mark, batch_y_mark] = batch_in
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                else:
                    [batch_x, batch_y] = batch_in
                    batch_x_mark, batch_y_mark = None, None

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, batch_x_mark, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, batch_y_mark)[0]
                # outputs = outputs[:, -self.args.pred_len:, :]
                # batch_y = batch_y.to(self.device)
                if args.verbose and (not (i + 1) % args.sample_step) and i < args.sample_step:
                    logger.info(f"batch_x.shape: {batch_x.shape}")
                    # logger.info(f"batch_x[0]:  {batch_x[0]}")
                    logger.info(f"batch_x_mark.shape: {batch_x_mark.shape}")
                    logger.info(f"batch_y_mark.shape: {batch_y_mark.shape}")
                    logger.info(f"batch_y.shape: {batch_y.shape}")
                    # logger.info(f"batch_y[0]: {batch_y[0]}")
                    logger.info(f"outputs.shape: {outputs.shape}")
                    # logger.info(f"outputs[0]: {outputs[0]}")

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        model.train()
        return total_loss

    def train(self, model, args, train_loader, val_loader=None):

        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = self._select_optimizer(model, args)
        criterion = self._select_criterion()

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            for i, batch_in in enumerate(train_loader):
                if (i + 1) % args.sample_step:
                    continue
                if len(batch_in) == 4:
                    [batch_x, batch_y, batch_x_mark, batch_y_mark] = batch_in
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                else:
                    [batch_x, batch_y] = batch_in
                    batch_x_mark, batch_y_mark = None, None
                if args.verbose and (not (i + 1) % args.sample_step) and i < args.sample_step:
                    logger.info(f"batch_x.shape: {batch_x.shape}")
                    logger.info(f"batch_y.shape: {batch_y.shape}")

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, batch_x_mark, batch_y_mark)[0]

                        batch_y = batch_y.to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = model(batch_x, batch_x_mark, batch_y_mark)[0]
                    if args.verbose and (not (i + 1) % args.sample_step) and i < args.sample_step:
                        logger.info(f"outputs.shape: {outputs.shape}")
                        logger.info(f"batch_y.shape: {batch_y.shape}")
                    # outputs = outputs[:, -self.args.pred_len:, :]
                    batch_y = batch_y.to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            adjust_learning_rate(model_optim, epoch + 1, args)

            train_loss = np.average(train_loss)
            logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))


