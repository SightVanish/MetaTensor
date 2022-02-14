"""
Base class
computing graph
"""
class Graph:
    """
    Base class
    computing graph
    """
    def __init__(self):
        self.nodes = [] # all nodes in this graph
        self.name_scope = None

    def add_node(self, node):
        self.nodes.append(node)

    def clear_jacobi(self):
        for node in self.nodes:
            node.clear_jacobi()
    
    def


# initialize graph
default_graph = Graph()