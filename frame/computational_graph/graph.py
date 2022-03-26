class Graph:
    """
    Base class: computational graph
    """
    def __init__(self):
        """
        init graph
        """
        self.nodes = [] # all nodes in this graph
        self.name_scope = None

    def add_node(self, node):
        """
        add nodes to graph
        """
        self.nodes.append(node)

    def clear_jacobi(self):
        """
        clear jacobi of each node
        """
        for node in self.nodes:
            node.clear_jacobi()
    
    def reset_value(self):
        """
        clear value of each node
        """
        for node in self.nodes:
            # do not recursively clear the value of it descendants
            node.reset_value(False)

    def node_count(self):
        """
        return the number of nodes
        """
        return len(self.nodes)

    def draw(self, ax=None):
        try:
            import network as nx
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            import numpy as np
        except:
            raise Exception("Import module failed.")
        

# initialize graph
default_graph = Graph()