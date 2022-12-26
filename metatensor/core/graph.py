class Graph:
    """
    Base class: computational graph
    """
    def __init__(self):
        self.nodes = []
        self.name_scope = None

    def add_node(self, node):
        """
        Add nodes to graph.
        """
        self.nodes.append(node)

    def clear_jacobi(self):
        """
        Clear jacobi of all nodes.
        """
        for node in self.nodes:
            node.clear_jacobi()
    
    def reset_value(self):
        """
        Clear value of all nodes.
        """
        for node in self.nodes:
            node.reset_value(False)

    def node_count(self):
        """
        Return the number of nodes.
        """
        return len(self.nodes)

    def draw(self, ax=None):
        # TODO
        try:
            import network as nx
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            import numpy as np
        except:
            raise Exception("Import module failed.")
        

# default graph
default_graph = Graph()