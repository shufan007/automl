# Copyright (c) DiDi Group. All rights reserved.
import time
import socket
from collections import Counter

import ray

from autotabular.common import MultiNodeRunner, get_logger

logger = get_logger()


if __name__ == '__main__':
    runner = MultiNodeRunner(hostfile='/etc/HOROVOD_HOSTFILE')
    runner.start()

    @ray.remote
    def f():
        time.sleep(0.001)
        # Return IP address.
        return socket.gethostbyname(socket.gethostname())

    object_ids = [f.remote() for _ in range(10000)]
    ip_addresses = ray.get(object_ids)
    print('Tasks executed')
    for ip_address, num_tasks in Counter(ip_addresses).items():
        print('{} tasks on {}'.format(num_tasks, ip_address))

    runner.stop()
