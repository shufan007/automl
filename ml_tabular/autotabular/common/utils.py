
import os
import shutil

import time
from functools import wraps

from autotabular.common import get_logger

logger = get_logger()

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwds):
        start = time.time()
        data = func(*args, **kwds)
        logger.info('function:{} {} '.format(func.__name__, calculate_cost_time(start)))
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


def get_luban_node_resources():
    """
    get cpu and memory resource of luban node
    :return: n_cpus, memory (unit:G)
    """
    cgroup_file = "/proc/self/cgroup"
    fs_cgroup_path = "/sys/fs/cgroup"

    cpu_key = "cpu"
    cpu_info_file = "cpu.cfs_quota_us"
    memory_key = "memory"
    memory_info_file = "memory.limit_in_bytes"

    def extract_contents(match_key, cgroup_info, file_name):
        import re
        pattern = re.compile("(?:%s:(.*)\n?)" % (match_key), re.M)
        search_group = pattern.findall(cgroup_info)
        matched = search_group[0] if len(search_group) > 0 else None
        if matched is None:
            logger.warn('pattern not match for key:{}'.format(match_key))
            return None
        extract_str = None
        file_path = os.path.join(fs_cgroup_path, match_key + os.path.join(matched, file_name))
        if os.path.exists(file_path):
            with open(file_path, 'r') as fd:
                extract_str = fd.read().strip()
        else:
            logger.warn('file:{} not exist.'.format(file_path))
        return extract_str

    memory = None
    n_cpus = None
    if os.path.exists(cgroup_file):
        with open(cgroup_file, 'r') as fd:
            cgroup_info = fd.read()
            cpu_extract = extract_contents(cpu_key, cgroup_info, cpu_info_file)
            if cpu_extract is not None:
                n_cpus = round(int(cpu_extract) / 100000)
            memory_extract = extract_contents(memory_key, cgroup_info, memory_info_file)
            if memory_extract is not None:
                memory = round(int(memory_extract) / (1024 * 1024 * 1024))
    else:
        logger.warn('cgroup file:{} not exist.'.format(cgroup_file))

    return n_cpus, memory