# Copyright (c) DiDi Group. All rights reserved.
import os
import subprocess

import ray
from .node import Node, NodeList
from .constants import PDSH_MAX_FAN_OUT, PDSH_MAX_RETRIES, OMPI_MCA_PLM_RSH_AGENT
from autotabular.common import get_logger

logger = get_logger()


class MultiNodeRunner(object):
    """
    A runner to execute commands on an array of machines. This runner
    is inspired by Nezha (https://github.com/zhuzilin/NeZha).
    """

    def __init__(self, host: str = None, hostfile: str = None, ray_port: int = 6379, node_resources=None):
        self.ray_port = ray_port
        self.resources_cmd = self.get_resources_cmd(node_resources)
        self.ray_path = "ray"
        # cannot accept host and hostfile at the same time
        if host and hostfile:
            logger.error("Error: hostfile and host are mutually exclusive, only one is required")

        # check if nodes is given
        self.nodes = None
        if host:
            self.nodes = self.parse_hostname(host)
        elif hostfile:
            self.nodes = self.parse_hostfile(hostfile)

        # split nodes into head and worker nodes
        self.head_node = None
        self.worker_nodes = None
        if self.nodes:
            self.head_node = self.nodes.get_node(0)
            if len(self.nodes) > 1:
                self.worker_nodes = [self.nodes.get_node(i) for i in range(1, len(self.nodes))]

        # set pdsh environment
        os.environ['PDSH_RCMD_TYPE'] = 'ssh'
        self.exec_remote_cmd = ['pdsh', '-S', '-f', str(PDSH_MAX_FAN_OUT), '-w']

        if OMPI_MCA_PLM_RSH_AGENT in os.environ and os.environ[OMPI_MCA_PLM_RSH_AGENT] is not None:
            self.exec_remote_cmd = [os.environ[OMPI_MCA_PLM_RSH_AGENT]]

        self.env = os.environ.copy()

    def parse_hostname(self, host: str) -> NodeList:
        nodes = NodeList()
        node_names = host.split(',')
        for node_name in node_names:
            node_name = node_name.strip()
            node = Node(name=node_name)

            if nodes.has(node_name):
                logger.warning(f"WARNING: found duplicate node {node_name} in the nodes")
                continue

            nodes.append(node)

        nodes = nodes if len(nodes) > 0 else None
        return nodes

    def parse_hostfile(self, hostfile: str) -> NodeList:
        """
        Parse the hostfile to obtain a list of nodes.

        A hostfile should look like:
        worker-0 slots=4
        worker-1 slots=4
        worker-2 slots=4

        Args:
            hostfile (str): the path to the hostfile
        """
        if not os.path.isfile(hostfile):
            logger.error(f"Error: Unable to find the hostfile, no such file: {hostfile}")
            return None

        nodes = NodeList()
        with open(hostfile, 'r') as fd:
            for line in fd.readlines():
                if line == '':
                    # skip empty lines
                    continue

                # build the HostInfo object
                node_name = line.strip().split()[0]
                node = Node(name=node_name)

                if nodes.has(node_name):
                    logger.warning(f"WARNING: found duplicate node {node_name} in the nodes")
                    continue

                nodes.append(node)

        nodes = nodes if len(nodes) > 0 else None
        return nodes

    def get_resources_cmd(self, node_resources):
        """
        get resources cmd like: [--num-cpus=4, --memory=34359738368, --object-store-memory=10000000000]
        :param node_resources: params dict for the node resources,like:
                                '{"cpu":4,"memory":32,"object_store_memory":10}'
                                cpu: INTEGER, cpu limit of one node
                                memory: memory limit of one node, unit:GBi
                                object_store_memory: ray object store memory, unit:GBi
        :return:
        """
        resources_cmd = []
        if node_resources.cpu is not None:
            resources_cmd.append(f"--num-cpus={node_resources.cpu}")
        if node_resources.memory is not None:
            mem = int(node_resources.memory) * (1024*1024*1024)
            resources_cmd.append(f"--memory={mem}")
        if node_resources.object_store_memory is not None:
            mem = int(node_resources.object_store_memory) * (1024*1024*1024)
            resources_cmd.append(f"--object-store-memory={mem}")
        return resources_cmd

    def run_cmd(self, cmd):
        current_retries = 0
        while current_retries < PDSH_MAX_RETRIES:
            ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
            current_retries += 1
            if ret.returncode == 0:
                return True

        logger.error('Failed to execute ray cmd {}'.format(cmd))
        return False

    def start_head(self):
        cmd = ['ray', 'start', '--head', f'--port={self.ray_port}']
        cmd += self.resources_cmd
        status = self.run_cmd(cmd=cmd)
        return status

    def stop_head(self):
        cmd = ['ray', 'stop']
        status = self.run_cmd(cmd=cmd)
        return status

    def start_workers(self):
        assert self.head_node is not None
        finished_nodes = []
        for node in self.worker_nodes:
            cmd = self.exec_remote_cmd + [node.name]
            cmd += [r"{}".format(self.ray_path), 'start', f'--address={self.head_node.name}:{self.ray_port}']
            cmd += self.resources_cmd
            if self.run_cmd(cmd=cmd):
                finished_nodes.append(node)
        return finished_nodes

    def stop_workers(self):
        finished_nodes = []
        for node in self.worker_nodes:
            cmd = self.exec_remote_cmd + [node.name]
            cmd += [r"{}".format(self.ray_path), 'stop']
            if self.run_cmd(cmd=cmd):
                finished_nodes.append(node)
        return finished_nodes

    def start(self):
        if self.nodes:
            if self.start_head():
                logger.info('Running ray cluster on head node: {}'.format(self.head_node))
            else:
                logger.error('Start ray head node: {} failed'.format(self.head_node))
                return False

            if self.worker_nodes:
                alive_worker_nodes = self.start_workers()
                logger.info('Running ray cluster on worker nodes: {}'.format(alive_worker_nodes))

            ray.init(address='auto')
            logger.info(
                'This cluster consists of {} nodes in total {} resources in total '.format(len(ray.nodes()),
                                                                                           ray.cluster_resources()))
        else:
            logger.info('Running ray cluster on localhost')

        return True

    def stop(self):
        # stop alive nodes
        status = True
        if self.nodes:
            if self.worker_nodes:
                stopped_worker_nodes = self.stop_workers()
                if len(stopped_worker_nodes) == len(self.worker_nodes):
                    logger.info('Stopped ray cluster on worker nodes: {} successfully'.format(stopped_worker_nodes))
                else:
                    logger.error('Stopped ray cluster on worker nodes: {} failed'.format(stopped_worker_nodes))
                    status = False

            if self.stop_head():
                logger.info('Stopped ray cluster on head node: {} successfully'.format(self.head_node))
            else:
                logger.error('Stopped ray cluster on head node: {} failed'.format(self.head_node))
                status = False
        else:
            logger.info('Stopped ray cluster on localhost successfully')

        return status
