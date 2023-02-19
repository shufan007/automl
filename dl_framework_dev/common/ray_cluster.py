import os
import sys
import subprocess
from .tools import get_logger

logger = get_logger()

RAY_CLUSTER_POT = 6379
TIMEOUT = 30

class RayCluster(object):
    """
    Local On Premise Cluster manage:
    Setup Ray Cluster based on List of nodes.
    You would use this mode if you want to run distributed Ray applications on some local nodes available on premise.
    This assumes that you have a list of machines and that the nodes in the cluster can communicate with each other.
    It also assumes that Ray is installed on each machine.
    """

    def __init__(self):
        self.ray_path = subprocess.run(['which', 'ray'], check=True, stdout=subprocess.PIPE) \
            .stdout.decode('utf-8').strip()

    def ray_cmd_exec_each_node(self, node_ips, cmd_args, remote=True):
        """
        execute ray cmd on each node.
        :param node_ips: the ip list of the nodes.
        :param cmd_args: execute command args
        :param remote: execute command at remote node or current node.
        :return: success_ips, ip list of node for which success execute cmd.
        """
        subp_list = []
        for ip in node_ips:
            logger.info(" execute ray cmd {} on node: {}".format(cmd_args, ip))
            if remote:
                cmd = ["ssh", ip, r"{}".format(self.ray_path), *cmd_args]
            else:
                cmd = ['ray', *cmd_args]
            subp = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            subp_list.append(subp)
        for subp in subp_list:
            subp.wait(TIMEOUT)
        success_ips = []
        for i in range(len(node_ips)):
            if subp_list[i].poll() != 0:
                logger.error("Error: when execute ray cmd {}  at {}: {}".format(cmd_args, node_ips[i], subp_list[i].stderr))
            else:
                success_ips.append(node_ips[i])
        return success_ips

    def ray_status_check_each_node(self, node_ips, remote=True):
        """
        ray status check on each node.
        :param node_ips:
        :param remote: execute command at remote node or current node.
        :return: success_ips
        """
        success_ips = self.ray_cmd_exec_each_node(node_ips, ['status'], remote=remote)
        logger.info("ray activating at: {}".format(success_ips))
        return success_ips

    def start_ray_head(self, head_ip, remote=False):
        """
        start ray head
        :param head_ip:
        :param remote:
        :return:
        """
        logger.info("start ray head on node: {}".format(head_ip))
        cmd_args = ['start', '--head', r'--port={}'.format(RAY_CLUSTER_POT)]
        success_ips = self.ray_cmd_exec_each_node([head_ip], cmd_args, remote=remote)
        ret_code = 0
        if len(success_ips) == 0:
            logger.warning("Failed to start ray head at: {}".format(head_ip))
            ret_code = 1
        else:
            logger.info("ray head start success at: {}".format(head_ip))
        return ret_code

    def start_ray_workers(self, head_ip, worker_ips):
        """
        Start ray on each worker nodes.
        :param head_ip: the ip of the head node.
        :param worker_ips: the ip list of the worker nodes.
        :return: worker nodes for which success startup ray.
        """
        cmd_args = ['start', r"--address='{}:{}'".format(head_ip, RAY_CLUSTER_POT)]
        success_ips = self.ray_cmd_exec_each_node(worker_ips, cmd_args, remote=True)
        logger.info("ray worker start success at: {}".format(success_ips))
        return success_ips

    def setup_ray_cluster_remote(self, node_list, traverse=False):
        """
        Setup ray cluster remote
        If failed start ray head in first node, try the next, until success or run out all nodes. [drop]
        :param node_list: node list
        :param traverse: If traverse to try next node when failed start ray, until success or run out all nodes.
                         'True': traverse to try next node failed start ray.
                         'False': just try the first node.
        :return: node list for which success startup ray.
        """
        if len(node_list) < 1:
            return node_list
        logger.info("install ray cluster ...")

        if traverse:
            # If failed start ray head in first node, try the next,until success or run out all nodes.
            ret_code = 1
            while (ret_code != 0) and (len(node_list) > 1):
                head_ip = node_list[0]
                ret_code = self.start_ray_head(head_ip, remote=True)
                node_list = node_list[1:]
        else:
            head_ip = node_list[0]
            ret_code = self.start_ray_head(head_ip, remote=True)

        if ret_code != 0:
            logger.info("** Failed to install Ray cluster!")
            return node_list
        else:
            logger.info("ray head start success at: {}".format(head_ip))

        node_list = [head_ip] + node_list
        print("start ray workers...")
        worker_ips = node_list[1:]
        success_ips = self.start_ray_workers(head_ip, worker_ips)
        node_list = [head_ip] + success_ips
        return node_list

    def setup_ray_cluster_on_headnode(self, node_list, traverse=False):
        """
        install ray cluster on head node.
        If failed start ray head in current node, turn to remote mode, try start ray head on the next node. [drop]
        :param node_list: node list
        :param traverse: If traverse to try next node when failed start ray, until success or run out all nodes.
                         'True': traverse to try next node failed start ray.
                         'False': just try the first node.
        :return: node list for which success startup ray.
        """
        if len(node_list) <= 1:
            return node_list

        print("install ray cluster ...")
        head_ip = node_list[0]
        ret_code = self.start_ray_head(head_ip, remote=False)
        if ret_code != 0:
            if traverse:
                """ try start ray head on the next node [drop] """
                node_list = node_list[1:]
                if len(node_list) > 1:
                    node_list = self.setup_ray_cluster_remote(node_list)
                    return node_list
                return [head_ip]
            else:
                return []

        logger.info("start ray workers...")
        worker_ips = node_list[1:]
        success_ips = self.start_ray_workers(head_ip, worker_ips)
        node_list = [head_ip] + success_ips
        return node_list

    def setup(self, node_list, remote):
        """
        Setup ray cluster: setup remote or on the head node.
        :param node_list: ip list of local nodes, the first will choose to be the head node.
        :param remote: if the submit mode is remote or not,
                True: means submit job outside the ray cluster, False: submit job on the head of the ray cluster
        :return: node list for which success startup ray.
        """
        if len(node_list) <= 1:
            return node_list
        if remote:
            node_list = self.setup_ray_cluster_remote(node_list)
        else:
            node_list = self.setup_ray_cluster_on_headnode(node_list)
        return node_list

    def stop_ray_each_node(self, node_ips, remote=True):
        """
        Stop ray on each node.
        :param node_ips:
        :param remote: execute command at remote node or current node.
        :return: success_ips
        """
        success_ips = self.ray_cmd_exec_each_node(node_ips, ['stop'], remote=remote)
        logger.info("ray stop success at: {}".format(success_ips))
        return success_ips

    def clean_up(self, node_list, remote):
        """
        stop ray cluster
        :param node_list: ip list of local nodes, the first will choose to be the head node.
        :param remote: if the submit mode is remote or not,
                True: means submit job outside the ray cluster, False: submit job on the head of the ray cluster
        stop ray cluster: stop ray on head node or remote
        :return:
        """
        if len(node_list) <= 0:
            return
        if remote:
            self.stop_ray_each_node(node_list, remote=True)
        else:
            logger.info("stop ray workers ...")
            worker_ips = node_list[1:]
            self.stop_ray_each_node(worker_ips, remote=True)
            logger.info("stop ray head")
            head_ip = node_list[0]
            self.stop_ray_each_node([head_ip], remote=False)

    def startup(self, node_list, remote):
        """
        ray startup entrance
        :param node_list: ip list of local nodes, the first will choose to be the head node.
        :param remote: if the submit mode is remote or not,
                True: means submit job outside the ray cluster, False: submit job on the head of the ray cluster
        :return:
                node_list: available node list
                remote: the parma may change
        """
        if node_list and len(node_list) > 1:
            node_list_success = self.setup(node_list, remote)
            if len(node_list_success) > 1:
                import ray
                if remote:
                    ray.init(address='ray://{}:10001'.format(node_list_success[0]))
                else:
                    if node_list_success[0] != node_list[0]:
                        remote = True
                        ray.init(address='ray://{}:10001'.format(node_list_success[0]))
                    else:
                        ray.init(address='auto')
                node_list = node_list_success
        return node_list, remote


def arg_parser(argv):
    """
    :param argv:
    :return:
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog='main',
        usage='''
        usage:
          --ray setup
            setup ray cluster, start ray head on first node, then start ray worker on the rest nodes.
            example:
            python3 ./ray_operator.py --ray setup --node_list '["172.17.126.125","172.17.126.126","172.17.126.124"]' --remote 0
          --ray cleanup
            stop raycluster
            example:
            python3 ./ray_operator.py --ray cleanup --node_list '["172.17.126.125","172.17.126.126","172.17.126.124"]' --remote 0            
          --ray startup
            setup and init raycluster
            example:
            python3 ./ray_operator.py --ray startup --node_list '["172.17.126.125","172.17.126.126","172.17.126.124"]' --remote 0            
            ''')
    parser.add_argument('--ray', type=str, help=" cmd for ray operator: 'setup', 'cleanup' or 'startup' ")
    parser.add_argument('--node_list', type=str, help="""[optional] List of node ip address:
                        for run distributed Ray applications on some local nodes available on premise.
                        the first node will be choose to be the head node.
                        """)
    parser.add_argument('--host_file', type=str, help="""[optional] same as node_list, host file path, List of node ip address.
                        """)
    parser.add_argument('--remote', type=str, default='False', help="True, submit job outside the Ray cluster, False, submit inside the ray cluster")

    args = parser.parse_args(argv)
    args = vars(args)

    if args['node_list']:
        args['node_list'] = eval(args['node_list'])
    elif args['host_file']:
        with open(args['host_file'], "r") as f:
            lines = f.readlines()
            args['node_list'] = [line.split(' ')[0].strip() for line in lines]
            logger.info("node_list in host_file: ", args['node_list'])

    if args['remote']:
        assert (args['remote'].lower() in ['true', 'false', '0', '1'])
        if (args['remote'].lower() == 'true') or (args['remote'] == '1'):
            args['remote'] = True
        else:
            args['remote'] = False

    return args


def main(argv):
    """
    NOTE: 目前 submit job 仍存在json解析问题，ray submit提交时，解析命令行报错。
         先采用在命令窗口手动提交 ray job submit
    :param argv:
    :return:
    """
    logger.info("argv:", argv)
    args = arg_parser(argv)
    logger.info("parse args: '{}'".format(args))

    ray_op = RayCluster()

    if args["ray"] == "setup":
        """setup ray cluster """
        ray_op.startup(node_list=args['node_list'], remote=args['remote'])

    elif args["ray"] == "cleanup":
        """clean ray cluster """
        ray_op.clean_up(node_list=args['node_list'], remote=args['remote'])
    elif args["ray"] == "startup":
        """start up ray cluster """
        ray_op.startup(node_list=args['node_list'], remote=args['remote'])

if __name__ == "__main__":
    main(sys.argv[1:])
