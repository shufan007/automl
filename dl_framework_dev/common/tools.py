"""
# @Time : 2022.04.15
# @Author : shuangxi.fan
# @Description : tools
"""

import os
import shutil
import logging
import torch.distributed as dist
logger_initialized = dict()

import time
from functools import wraps


def log(filename=None):
    logging.basicConfig(filename=filename,
                        level=logging.INFO,
                        filemode='w',
                        format='%(asctime)s - %(levelname)s: [line:%(lineno)d]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    # create logger
    logger = logging.getLogger('Automl')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: [line:%(lineno)d]: %(message)s')

    if filename:
        fh = logging.FileHandler(filename)
        fh.setFormatter(formatter)
        # add handler for logger
        if not logger.handlers:
            logger.addHandler(fh)
    """
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch)    
    """
    return logger


def get_logger(name='Automl', log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger.propagate = False

    logger_initialized[name] = True

    return logger



def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwds):
        start = time.time()
        data = func(*args, **kwds)
        print('function:{} {} '.format(func.__name__, calculate_cost_time(start)))
        return data

    return wrapper


def calculate_cost_time(start_time):
    """
    time cost
    :return:
    """
    end_time = time.time()
    cost_time = end_time - start_time
    if cost_time < 60:
        return "Time cost:{} seconds".format(round(cost_time, 3))
    elif cost_time < 3600:
        minute_num = cost_time // 60
        second_num = cost_time % 60
        return "Time cost:{} minutes {} seconds".format(int(minute_num), round(second_num, 3))
    else:
        hour_num = cost_time // (60 * 60)
        minute_left_time = (cost_time - hour_num * 60 * 60)
        minute_num = minute_left_time // 60
        second_left_time = cost_time - minute_left_time - minute_num * 60
        return "Time cost:{} hours {} minutes {} seconds".\
            format(int(hour_num), int(minute_num), round(second_left_time, 3))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def copy_dir(src, dst, ignore_pattern=None):
    if os.path.exists(dst):
        if os.path.isdir(src):
            shutil.rmtree(dst, ignore_errors=False)
    if os.path.isdir(src):
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(ignore_pattern))
    elif os.path.isfile(src):
        shutil.copy(src, dst)


def get_suffix_file(path, suffix=None):
    file_list = os.listdir(path)
    if len(file_list) == 0:
        return None

    if suffix is not None:
        return [file for file in file_list if file.find(suffix) >= 0][0]
    else:
        return file_list[0]

