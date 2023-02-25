import dmlite.torch as dmlite
from dmlite.common.table import TableReader


class CriteoDataset(dmlite.TabularDataset):
    def __init__(self, path, format, is_training=True, **kwargs):
        self.data_frame = TableReader(path, format).read()
        self.is_training = is_training
        self.kwargs = kwargs

        self.num_embeddings = self.preprocess_sparse_features()

    def preprocess_sparse_features(self):
        sparse_cols = self.kwargs['sparse_cols']

        num_embeddings = 0
        for col in sparse_cols:
            last_num_embeddings = num_embeddings
            num_embeddings += int(self.data_frame[col].max() + 1)
            self.data_frame[col] += last_num_embeddings

        return num_embeddings
