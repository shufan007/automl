# Copyright (c) DiDi Group. All rights reserved.

class Node(object):
    """
    A data class to store host connection-related data.
    Args:
        name (str): name or IP address of the node
        is_head (bool): whether used as head node
    """

    def __init__(self, name: str, is_head: bool = False):
        self.name = name
        self.is_head = is_head

    def __str__(self):
        return f'name: {self.name}'

    def __repr__(self):
        return self.__str__()


class NodeList(object):
    """
    A data class to store a list of Node objects.
    """

    def __init__(self):
        self.node_list = []

    def append(self, node: Node) -> None:
        """
        Add an HostInfo object to the list.
        Args:
            node (Node): node information
        """

        self.node_list.append(node)

    def remove(self, node_name: str) -> None:
        """
        Add an Node object to the list.
        Args:
            node_name (str): the name of the node
        """

        node = self.get_node(node_name)
        self.node_list.remove(node)

    def get_node(self, node_name: str) -> Node:
        """
        Return the HostInfo object which matches with the hostname.
        Args:
            node_name (str): the name of the node
        Returns:
            node (Node): the Node object which matches with the node name
        """

        for node in self.node_list:
            if node.name == node_name:
                return node

        raise Exception(f"Node name: {node_name} is not found")

    def get_node(self, index: int) -> Node:
        """
        Return the HostInfo object which matches with the hostname.
        Args:
            index (int): the index of the node
        Returns:
            node (Node): the Node object which matches with the node index
        """

        return self.node_list[index]

        raise Exception(f"Index: {str(index)} is out of range")

    def has(self, node_name: str) -> bool:
        """
        Check if the hostname has been added.
        Args:
            node_name (str): the name of the node
        Returns:
            bool: True if added, False otherwise
        """
        for node in self.node_list:
            if node.name == node_name:
                return True
        return False

    def __iter__(self):
        return iter(self.node_list)

    def __len__(self):
        return len(self.node_list)
