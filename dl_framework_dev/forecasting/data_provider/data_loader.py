
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from common import table_data_load, get_logger
import warnings

warnings.filterwarnings('ignore')

logger = get_logger()


class TabularReader(object):
    """
    read tabular data, and do some preprocess
    """
    def __init__(self, data_path, data_format='textfile', label_col=[], date_col='date',
                 features_col=[], scale=True, time_enc=0, freq='h', size=None, cond_str: str = None):

        self.data_path = data_path
        self.data_format = data_format
        self.label_col = label_col
        self.date_col = date_col
        self.features_col = features_col
        self.scale = scale
        self.time_enc = time_enc
        self.freq = freq
        self.cond_str = cond_str
        self.mark_in = None
        self.enc_in = None
        self.d_out = None  # final out put dim

        self.data = None
        self.data_stamp = None

        # size [seq_len, label_len, pred_len]
        if size is not None:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        else:
            logger.warn("seq_len, label_len, pred_len not provided !")
            self.seq_len = 1
            self.label_len = 1
            self.pred_len = 1

        self.__read_data__()

        self.dec_in = self.enc_in
        self.c_out = self.dec_in

    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv(self.data_path)
        df_raw = table_data_load(self.data_path, self.data_format, cond_str=self.cond_str)
        '''
        df_raw.columns: ['date', ...(other features), label_col feature]
        '''
        raw_cols = list(df_raw.columns)

        if len(self.features_col) > 0:
            for col in self.features_col:
                if col not in raw_cols:
                    logger.warn(f"features_col column '{col}' not in data columns {raw_cols} !")
                    self.features_col.remove(col)

        if len(self.label_col) > 0:
            for col in self.label_col:
                if col not in raw_cols:
                    logger.warn(f"label_col column '{col}' not in data columns {raw_cols} !")
                    self.label_col.remove(col)

        cols = self.features_col
        """
        if len(self.features_col) > 0:
            cols = self.features_col
        else:
            cols = copy.deepcopy(raw_cols)        
        """

        if (self.date_col is not None) and (self.date_col in raw_cols):
            if self.date_col not in cols:
                cols = [self.date_col] + cols
            else:
                cols.remove(self.date_col)
                cols = [self.date_col] + cols

        for col in self.label_col:
            if cols.__contains__(col):
                cols.remove(col)

        if len(self.label_col) > 0:
            self.d_out = len(self.label_col)
        else:
            self.d_out = len(cols)
        logger.info(f">>>>>>> Inferred args : {self.label_col}, features: {cols}, d_out: {self.d_out} ")
        df_raw = df_raw[cols + self.label_col]
        cols_data = list(df_raw.columns)
        if self.date_col in cols_data:
            cols_data.remove(self.date_col)

        df_data = df_raw[cols_data]
        if self.scale:
            self.scaler.fit(df_data.values)
            self.data = self.scaler.transform(df_data.values)
        else:
            self.data = df_data.values

        data_stamp = None
        if self.date_col in df_raw.columns:
            df_stamp = df_raw[[self.date_col]]
            df_stamp[self.date_col] = pd.to_datetime(df_stamp[self.date_col])
            if self.time_enc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop([self.date_col], 1).values
            elif self.time_enc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp[self.date_col].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

        self.enc_in = self.data.shape[1]
        if self.data_stamp is not None:
            self.mark_in = data_stamp.shape[1]

        self.data_stamp = data_stamp


class DatasetTabular(Dataset):
    def __init__(self, data_reader: TabularReader, flag='train', split_ratio=0):
        self.data = data_reader.data
        self.data_stamp = data_reader.data_stamp
        self.d_out = data_reader.d_out

        self.seq_len = data_reader.seq_len
        self.label_len = data_reader.label_len
        self.pred_len = data_reader.pred_len

        # init
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.split_ratio = split_ratio
        self.__read_data__()

    def __read_data__(self):

        num_train = int(len(self.data) * (1 - self.split_ratio))
        num_val = int(len(self.data) * self.split_ratio)
        num_test = len(self.data) - num_train - num_val
        border1s = [0, num_train - self.seq_len, len(self.data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(self.data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = self.data[border1:border2]
        self.data_y = self.data[border1:border2][..., -self.d_out:]
        self.data_stamp = self.data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        if self.data_stamp is None:
            return [seq_x, seq_y]
        else:
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            return [seq_x, seq_y, seq_x_mark, seq_y_mark]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class ArrayTableReader(object):
    """
    read array table data, and do some preprocess
    """
    def __init__(self, data_path, data_format='array-table', label_col='target',
                 features_col='features', cond_str: str = None):
        self.data_path = data_path
        self.data_format = data_format
        if isinstance(label_col, list):
            label_col = label_col[0]

        self.label_col = label_col
        self.features_col = features_col
        self.scale = False
        self.cond_str = cond_str

        self.label_len = None
        self.mark_in = None
        self.enc_in = None
        self.d_out = None
        self.X = None
        self.y = None

        self.__read_data__()

        self.dec_in = self.enc_in
        self.c_out = self.dec_in

    def __read_data__(self):
        self.scaler = StandardScaler()  # StandardScaler expected dimensions <= 2.

        if self.data_format == 'array-table':
            raw_data = table_data_load(self.data_path, self.data_format, label_col=self.label_col,
                                       features_col=self.features_col, cond_str=self.cond_str)
        elif self.data_format == 'tensor':
            raw_data = torch.load(self.data_path)

        if (type(raw_data) not in [pd.core.frame.DataFrame, dict, list]) or (len(raw_data) != 2):
            raise Exception(""" train data format Error, data format should be 
                          ' "table": hive table (with 2 dimensions)'
                          '"textfile": text file(s) path of tabular data (with 2 dimensions), '
                          ' "array-table": array table (with 3 dimensions), 
                                DataFrame with dtypes: [("features", "array<array<float>>"), ("target", "array<array<float>>")]'
                          '"tensor": tensors dict like {"features":X, "target":y} or list like [X, y] 
                                with a 3 dimensions X: [# samples x # sequence length x # variables] and target named as "target"'
                          """)
        if type(raw_data) == pd.core.frame.DataFrame:
            X = raw_data[self.features_col].values
            y = raw_data[self.label_col].values
        elif type(raw_data) == list:
            X = raw_data[0]
            y = raw_data[1]
        else:
            keys = list(raw_data.keys())
            if self.label_col in keys:
                y = raw_data[self.label_col]
                keys.remove(self.label_col)
            else:
                y = raw_data[keys[-1]]
                keys.remove(keys[-1])
            X = raw_data[keys[-1]]

        [self.data_len, self.seq_len, n_features] = X.shape
        self.enc_in = n_features
        y_shape = y.shape

        if (self.label_len is None) or (self.label_len > self.seq_len):
            self.label_len = self.seq_len // 2

        if len(y_shape) == 1:
            self.pred_len = 1
            self.d_out = 1
        else:
            self.pred_len = y_shape[1]
            self.d_out = y_shape[-1]

        if self.scale:
            self.scaler.fit(X)
            self.X = self.scaler.transform(X)
        else:
            self.X = X
        self.y = y


class DatasetArray(Dataset):
    def __init__(self, data_reader: ArrayTableReader, flag='train', split_ratio=0):
        self.X = data_reader.X
        self.y = data_reader.y
        self.data_len = data_reader.data_len
        self.d_out = data_reader.d_out

        self.seq_len = data_reader.seq_len
        self.label_len = data_reader.label_len
        self.pred_len = data_reader.pred_len
        # init
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.split_ratio = split_ratio
        self.__read_data__()

    def __read_data__(self):
        num_train = int(self.data_len * (1 - self.split_ratio))
        num_val = int(self.data_len * self.split_ratio)
        num_test = self.data_len - num_train - num_val
        border1s = [0, num_train, self.data_len - num_test]
        border2s = [num_train, num_train + num_val, self.data_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = self.X[border1:border2]
        self.data_y = self.y[border1:border2]

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        return [seq_x, seq_y]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
