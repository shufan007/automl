# Copyright (c) DiDi Group. All rights reserved.
from typing import List, Callable

import torch
from torch.utils.data import Dataset

from .table import HdfsTableReader, HiveTableReader
from autotabular.common import get_logger

logger = get_logger()


class TabularDataset(Dataset):
    """Defines a Dataset of columns stored in Parquet, ORC, or CSV format, or from hive table."""

    def __init__(self,
                 path: str,
                 format: str = 'parquet',
                 feature_cols: List[str] = None,
                 label_cols: List[str] = None,
                 transform: Callable = None):
        """
        Create a TabularDataset in memory given a local or cloud path, file format, and field list.

        Args:
            path (str): Path to the data file.

            format (str): The format of the data file. One of "parquet", "orc",
                "csv" (case-insensitive) or 'table'.
        """
        logger.info(f"data loading ...")

        if format == 'table':
            table_path, data_format = HiveTableReader(format).get_table_path(path)
            self.pd_frame = HdfsTableReader(data_format)(table_path)
        else:
            self.pd_frame = HdfsTableReader(format)(path)

        logger.info(f"data shape:{self.pd_frame.shape}")

        self.feature_cols = feature_cols
        self.label_cols = label_cols

        if transform:
            # inplace operation
            self.pd_frame = transform(self.pd_frame)

    def to_ndarray(self):
        """
        Get features and labels dataframe as training input.

        Returns:
            Dataframe of features and labels
        """
        assert self.feature_cols is not None
        features = self.pd_frame[self.feature_cols].values
        if self.label_cols is not None:
            labels = self.pd_frame[self.label_cols].values
            return features, labels
        else:
            return features

    def process(self, row_data):
        """
        Process row data and generate features and labels in pytorch tensor format

        Args:
            row_data (pandas.Series): Row data from data frame

        Returns:
            Tensor or Tuple[Tensor]: Features and labels at training stage, otherwise only features
        """
        assert self.feature_cols is not None
        features = row_data.loc[self.feature_cols].to_numpy()

        if self.label_cols:
            labels = row_data.loc[self.label_cols].to_numpy()

            features = torch.from_numpy(features).double()
            labels = torch.from_numpy(labels).double()
            return features, labels
        else:
            features = torch.from_numpy(features).double()
            return features

    def __len__(self):
        return self.pd_frame.shape[0]

    def __getitem__(self, index):
        row = self.pd_frame.iloc[index]
        tensors = self.process(row)
        return tensors
