import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')

from common import get_logger
logger = get_logger()


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}  # for args.lr_adj == 'type1'
    if args.lr_adj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lr_adj =='type3':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lr_adj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.9 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logger.info('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.score_update = False
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.score_update = True
            if self.verbose:
                logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
                self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.score_update = False
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.score_update = True
            if self.verbose:
                logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
                self.val_loss_min = val_loss
            self.counter = 0


def save_checkpoint(model, checkpoint_dir, suffix=None):
    model_name = 'checkpoint.pth'
    if suffix is not None:
        model_name = 'checkpoint' + '_' + suffix + '.pth'

    model_path = checkpoint_dir + '/' + model_name
    torch.save(model.state_dict(), model_path)
    logger.info(f'Save model to {model_path}')


class StandardScaler(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
