
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_provider.data_loader import (
    TabularReader,
    ArrayTableReader,
    DatasetTabular,
    DatasetArray)

from common import get_logger

logger = get_logger()

data_dict = {
    'textfile': DatasetTabular,
    'table': DatasetTabular,
    'array-table': DatasetArray,
    'tensor': DatasetArray,
}


def get_data_reader(args, data_path, cond_str=None):
    time_enc = 0 if args.embed != 'timeF' else 1
    data_reader = None
    if args.data_format in ['textfile', 'table']:
        data_reader = TabularReader(
                                data_path=data_path,
                                data_format=args.data_format,
                                label_col=args.label_col,
                                date_col=args.date_col,
                                features_col=args.features_col,
                                time_enc=time_enc,
                                freq=args.freq,  # args.detail_freq for FEDformer
                                size=[args.seq_len, args.label_len, args.pred_len],
                                cond_str=cond_str
                            )
    elif args.data_format in ['array-table', 'tensor']:
        data_reader = ArrayTableReader(
                                data_path=data_path,
                                data_format=args.data_format,
                                label_col=args.label_col,
                                features_col=args.features_col,
                                cond_str=cond_str
                            )
    else:
        logger.error(f"got an error data_format:{args.data_format} ")

    return data_reader


def get_dataset(args, data_path=None, data_reader=None, flag='train'):
    if data_reader is None:
        data_reader = get_data_reader(args, data_path)
    Data = data_dict[args.data_format]
    # if flag == 'pred': Data = DatasetPred
    data_set = Data(
        data_reader=data_reader,
        split_ratio=args.split_ratio,
        flag=flag,
    )
    print(flag, len(data_set))

    return data_set


def get_model_input_args(args, data_reader=None):
    """
    :param args:
    :param data_reader:
    :return:
    """
    if data_reader is None:
        data_reader = get_data_reader(args, args.train_data, cond_str='limit 1')

    args.mark_in = data_reader.mark_in
    args.enc_in = data_reader.enc_in
    args.d_out = data_reader.d_out
    if not args.dec_in:
        args.dec_in = data_reader.dec_in
    if not args.c_out:
        args.c_out = data_reader.c_out

    args.seq_len = data_reader.seq_len
    args.label_len = data_reader.label_len
    args.pred_len = data_reader.pred_len

    return args


def data_provider(args, data_set=None, data_path=None, flag='train'):
    if data_set is None:
        data_set = get_dataset(args, data_path, flag=flag)

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_sampler = None
    if args.distributed:
        data_sampler = DistributedSampler(data_set)

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        sampler=data_sampler,
        shuffle=(data_sampler is None) and shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_loader
