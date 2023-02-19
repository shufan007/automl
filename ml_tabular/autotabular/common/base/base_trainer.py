# Copyright (c) DiDi Group. All rights reserved.
from typing import Optional
from torch.utils.data import Dataset


class BaseTrainer(object):
    """
    BaseTrainer is a simple but feature-complete training and eval loop for AutoML on tabular dataset
    """
    def __init__(self,
                 args,
                 train_dataset: Optional[Dataset] = None,
                 val_dataset: Optional[Dataset] = None,
                 task_name: str = None):
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = None
        self.task_name = task_name

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def save_model(self, output_dir: str = None):
        raise NotImplementedError
